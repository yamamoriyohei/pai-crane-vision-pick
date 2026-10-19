"""お手本デモの自動収集.

1 エピソードの流れ:
  1. キューブ（と任意で妨害物の青キューブ）をランダムな位置に瞬間移動
  2. ros2 bag record を開始（1 エピソード = 1 bag。講義と同じ形式）
  3. 正解のキューブ位置からピック＆プレースの軌道を作り、20Hz で関節指令を流す
  4. 記録停止 → キューブがトレイに入ったか（成功か）をシミュレータで判定
  5. 成功した bag だけを学習用ディレクトリに残す（失敗は <out>_failed/ に移して後で分析）

使い方（Gazebo を起動した状態で別ターミナル）:
  ros2 run crane_vision_pick collect_demos --out ~/dataset/rosbags/vision_pick --episodes 60
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import rclpy

from .expert import add_grasp_args, apply_grasp_args, describe_grasp, PlanError, TaskConfig, plan_pick_and_place, sample_cube_pose
from .ros_io import ARM_TOPIC, GRIPPER_TOPIC, ArmIO
from .sim_utils import SimError, get_model_pose, set_model_pose

IMAGE_TOPIC = '/camera/image_small'
RECORD_TOPICS = ['/joint_states', IMAGE_TOPIC, ARM_TOPIC, GRIPPER_TOPIC]
PARK_BLUE = (2.0, 2.0, 0.03)     # 妨害物を使わないときの退避位置（ワールド座標・カメラの外）


# ------------------------------------------------------------------ helpers
def in_tray(cube_xyz, cfg: TaskConfig):
    tx, ty = cfg.tray_xy
    yaw = math.radians(cfg.tray_theta_deg)
    dx, dy = cube_xyz[0] - tx, cube_xyz[1] - ty
    lx = math.cos(yaw) * dx + math.sin(yaw) * dy
    ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return abs(lx) < cfg.tray_half_size and abs(ly) < cfg.tray_half_size and cube_xyz[2] < 0.045


def sample_distractor(red_xy, cfg, rng, min_angle_deg=22.0, min_dist=0.08):
    """把持動作の通り道（ロボットからキューブへの扇形）を避けて青キューブを置く."""
    th_red = math.atan2(red_xy[1], red_xy[0])
    for _ in range(200):
        x, y, yaw = sample_cube_pose(cfg, rng)
        if abs(math.degrees(math.atan2(y, x) - th_red)) < min_angle_deg:
            continue
        if math.hypot(x - red_xy[0], y - red_xy[1]) < min_dist:
            continue
        return x, y, yaw
    return None


def place_objects(io, red, blue, retries=3):
    """キューブを置いて、本当にその位置に落ち着いたかを確認する."""
    for _ in range(retries):
        set_model_pose('red_cube', red[0], red[1], yaw=red[2])
        if blue is None:
            set_model_pose('blue_cube', *PARK_BLUE, frame='world')
        else:
            set_model_pose('blue_cube', blue[0], blue[1], yaw=blue[2])
        io.sleep_sim(0.8)
        x, y, z, _ = get_model_pose('red_cube')
        if math.hypot(x - red[0], y - red[1]) < 0.01 and abs(z - 0.025) < 0.01:
            return x, y, z
    raise SimError(f'cube did not settle at target (got {x:.3f},{y:.3f},{z:.3f})')


class BagRecorder:
    def __init__(self, io, path, topics=RECORD_TOPICS):
        self.io, self.path, self.topics = io, Path(path), topics
        self.proc = None

    def start(self, timeout=30.0, strict=True):
        """記録を開始し、全トピックの購読が始まるまで待つ（記録の頭が欠けるのを防ぐ）.

        strict=False のときは、時間内に購読を確認できなくても記録プロセスが生きていれば続行する
        （評価用。重い負荷でグラフ情報の更新が遅れることがあるため）。
        """
        base = {t: self.io.count_subscribers(t) for t in self.topics}
        cmd = ['ros2', 'bag', 'record', '--use-sim-time', '-o', str(self.path), *self.topics]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     start_new_session=True)
        end = time.monotonic() + timeout
        while True:
            waiting = [t for t in self.topics if self.io.count_subscribers(t) <= base[t]]
            if not waiting:
                return True
            self.io.spin_for(0.05)
            if self.proc.poll() is not None:
                raise RuntimeError('ros2 bag record exited: ' + self.proc.stderr.read().decode())
            if time.monotonic() > end:
                if not strict:
                    self.io.get_logger().warn(f'bag の購読を確認できないまま続行します（未確認: {waiting}）')
                    return False
                self.stop()
                raise RuntimeError(f'ros2 bag record did not subscribe in time (waiting: {waiting})')

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.proc = None
        # metadata.yaml の書き出しを待つ
        for _ in range(50):
            if (self.path / 'metadata.yaml').exists():
                return
            time.sleep(0.1)


def next_index(*dirs):
    idx = -1
    for d in dirs:
        for p in Path(d).glob('ep_*'):
            try:
                idx = max(idx, int(p.name.split('_')[1]))
            except (IndexError, ValueError):
                pass
    return idx + 1


# --------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='/workspace/dataset/rosbags/vision_pick' if os.path.isdir('/workspace')
                    else os.path.expanduser('~/dataset/rosbags/vision_pick'))
    ap.add_argument('--episodes', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--distractor', action='store_true', help='青い妨害キューブも置く（発展課題）')
    ap.add_argument('--dry-run', action='store_true', help='記録せずにお手本の動作だけ確認する')
    ap.add_argument('--min-image-hz', type=float, default=8.0, help='これを下回る画像周波数のエピソードは失敗扱い')
    ap.add_argument('--max-image-gap', type=float, default=0.35, help='これ以上画像が抜けたエピソードは失敗扱い [s]')
    add_grasp_args(ap)
    args, _ = ap.parse_known_args(argv)

    cfg = apply_grasp_args(TaskConfig(), args)
    out = Path(args.out).expanduser()
    failed_dir = out.parent / (out.name + '_failed')
    out.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    start_idx = next_index(out, failed_dir)
    rng = np.random.default_rng(args.seed + start_idx)   # 続きから収集しても同じ配置を繰り返さない

    rclpy.init()
    io = ArmIO('demo_collector')
    # 収集中の品質ゲート（第3回）: 記録中に届いた画像の撮影時刻（シミュレーション時間）を数える
    img_stamps, watching = [], [False]

    def on_image(msg):
        if watching[0]:
            img_stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    from sensor_msgs.msg import Image
    io.create_subscription(Image, IMAGE_TOPIC, on_image, 10)
    io.wait_for_clock()
    io.get_logger().info('把持設定: ' + describe_grasp(cfg))
    io.go_home(3.0)

    log_path = out.parent / (out.name + '_log.csv')
    new_log = not log_path.exists()
    logf = open(log_path, 'a', newline='')
    log = csv.writer(logf)
    if new_log:
        log.writerow(['episode', 'success', 'cube_x', 'cube_y', 'cube_yaw', 'final_x', 'final_y', 'final_z',
                      'blue_x', 'blue_y', 'n_cmds', 'wall_sec', 'sim_sec', 'task_success', 'image_hz',
                      'image_max_gap', 'reason'])

    n_ok = 0
    n_err = 0
    rec = None
    try:
        for k in range(args.episodes):
            ep = start_idx + k
            name = f'ep_{ep:04d}'
            red = sample_cube_pose(cfg, rng)
            blue = sample_distractor(red[:2], cfg, rng) if args.distractor else None
            try:
                cmds = plan_pick_and_place(red[:2], cfg)
            except PlanError as e:
                io.get_logger().warn(f'{name}: skip ({e})')
                continue
            try:
                actual = place_objects(io, red, blue)
                n_err = 0
            except SimError as e:
                n_err += 1
                io.get_logger().error(f'{name}: {e}')
                if n_err >= 3:
                    raise
                continue

            rec = None if args.dry_run else BagRecorder(io, out / name)
            if rec and rec.path.exists():
                shutil.rmtree(rec.path)        # 前回 Ctrl+C で中断した残骸
            w0, s0 = time.monotonic(), io.now()
            if rec:
                rec.start()
            img_stamps.clear()
            watching[0] = True
            io.stream(cmds)
            io.stream([cmds[-1]] * 6)          # 最後に 0.3 s だけ静止（長く止まると「止まる」を学習する）
            watching[0] = False
            recorded = rec is not None
            if rec:
                rec.stop()
                rec = None
            wall, sim = time.monotonic() - w0, io.now() - s0

            io.sleep_sim(0.5)
            final = get_model_pose('red_cube')
            task_ok = in_tray(final, cfg)
            st = np.array(img_stamps)
            img_hz = (len(st) - 1) / max(st[-1] - st[0], 1e-6) if len(st) > 1 else 0.0
            img_gap = float(np.diff(st).max()) if len(st) > 1 else float('inf')
            data_ok = (not recorded) or (img_hz >= args.min_image_hz and img_gap <= args.max_image_gap)
            ok = task_ok and data_ok
            reason = '' if ok else ('task' if not task_ok else
                                    f'image_drop({img_hz:.1f}Hz,gap{img_gap:.2f}s)')
            n_ok += ok
            meta = dict(episode=ep, success=bool(ok), seed=args.seed, cube_target=red, cube_initial=actual,
                        cube_final=final[:3], blue=blue, tray_xy=cfg.tray_xy, n_cmds=len(cmds),
                        wall_sec=wall, sim_sec=sim, grasp=describe_grasp(cfg), real_time_factor=sim / max(wall, 1e-6),
                        task_success=bool(task_ok), image_hz=img_hz, image_max_gap=img_gap, reason=reason)
            if recorded:
                (out / name / 'episode_meta.json').write_text(json.dumps(meta, indent=2))
                if not ok:
                    shutil.move(str(out / name), str(failed_dir / name))
            log.writerow([ep, int(ok), *np.round(red, 4), *np.round(final[:3], 4),
                          *(np.round(blue[:2], 4) if blue else ['', '']), len(cmds),
                          round(wall, 2), round(sim, 2), int(task_ok), round(img_hz, 2),
                          round(img_gap, 3) if np.isfinite(img_gap) else '', reason])
            logf.flush()
            io.get_logger().info(
                f'{name}: {"SUCCESS" if ok else "FAIL " + reason}  cube=({red[0]:.3f},{red[1]:.3f})  '
                f'final=({final[0]:.3f},{final[1]:.3f},{final[2]:.3f})  RTF={sim / max(wall, 1e-6):.2f}  '
                f'画像 {img_hz:.1f}Hz  [{n_ok}/{k + 1}]')
            if recorded and not data_ok:
                io.get_logger().warn('画像が欠けています。重い処理（インストール・画面収録など）を止めると改善します')
            io.go_home(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if rec:
            rec.stop()
        logf.close()
        io.get_logger().info(f'done: {n_ok} successful episodes saved in {out}')
        io.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
