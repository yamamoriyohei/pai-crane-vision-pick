"""学習した方策（またはお手本）の成功率を測る.

学習に使っていない乱数シードでキューブを置き、方策ノード（training/scripts/infer_vision.py）を
/policy/enable で ON/OFF しながら N 回試行する。結果は CSV に保存する。

  # 方策の評価（infer_vision.py を別ターミナルで起動しておく）
  ros2 run crane_vision_pick evaluate --trials 20 --out ~/results/act_vision.csv
  # お手本の成功率（上限の目安）
  ros2 run crane_vision_pick evaluate --mode expert --trials 20 --out ~/results/expert.csv
  # 動画用に各試行の bag も残す
  ros2 run crane_vision_pick evaluate --trials 5 --record-dir ~/dataset/eval_bags
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import rclpy
from std_msgs.msg import Bool, String

from .collect_demos import BagRecorder, in_tray, place_objects, sample_distractor
from .expert import add_grasp_args, apply_grasp_args, describe_grasp, TaskConfig, plan_pick_and_place, sample_cube_pose
from .kinematics import HOME_Q
from .ros_io import ArmIO
from .sim_utils import SimError, get_model_pose


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['policy', 'expert'], default='policy')
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--seed', type=int, default=10_000, help='学習データと重ならないシード')
    ap.add_argument('--timeout', type=float, default=25.0, help='1 試行の制限時間（シミュレーション秒）')
    ap.add_argument('--distractor', action='store_true')
    ap.add_argument('--out', default='/workspace/results/eval.csv' if os.path.isdir('/workspace')
                    else os.path.expanduser('~/results/eval.csv'))
    ap.add_argument('--record-dir', default='', help='指定すると各試行の bag を保存（動画づくり用）')
    ap.add_argument('--label', default='', help='CSV に書く条件名（例: act_vision / act_state_only）')
    ap.add_argument('--start-trial', type=int, default=0,
                    help='途中から再開（それより前の試行は配置だけ計算して飛ばす。キューブ位置は同じになる）')
    add_grasp_args(ap)
    args, _ = ap.parse_known_args(argv)

    cfg = apply_grasp_args(TaskConfig(), args)
    rng = np.random.default_rng(args.seed)
    rclpy.init()
    io = ArmIO('policy_evaluator')
    enable_pub = io.create_publisher(Bool, '/policy/enable', 10)
    status_pub = io.create_publisher(String, '/eval/status', 10)   # 動画のテロップ用
    io.wait_for_clock()
    if args.mode == 'expert':
        io.get_logger().info('把持設定: ' + describe_grasp(cfg))
    io.go_home(3.0)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rec = None
    try:
        for trial in range(args.trials):
            red = sample_cube_pose(cfg, rng)
            blue = sample_distractor(red[:2], cfg, rng) if args.distractor else None
            if trial < args.start_trial:
                continue                           # 乱数だけ進めて、同じ配置の列を保つ
            for attempt in range(3):
                try:
                    place_objects(io, red, blue)
                    break
                except SimError as e:
                    io.get_logger().warn(f'trial {trial}: 配置に失敗、やり直します ({e})')
                    io.sleep_sim(1.0)
            else:
                raise SimError(f'trial {trial}: キューブを置けませんでした。'
                               f'シミュレータを再起動して --start-trial {trial} で再開してください')
            rec = None
            if args.record_dir:
                bag = Path(args.record_dir).expanduser() / f'{args.label or args.mode}_s{args.seed}_{trial:03d}'
                if bag.exists():
                    shutil.rmtree(bag)             # 同じ条件で撮り直したときは上書き
                bag.parent.mkdir(parents=True, exist_ok=True)
                rec = BagRecorder(io, bag, topics=['/joint_states', '/camera/image_small', '/eval/status'])
                try:
                    rec.start(strict=False)        # 録画は動画用のおまけ。失敗しても評価は続ける
                except RuntimeError as e:
                    io.get_logger().warn(f'trial {trial}: 録画なしで続行します ({e})')
                    rec.stop()
                    rec = None
            status_pub.publish(String(data=json.dumps({'trial': trial, 'state': 'running'})))

            t0, w0 = io.now(), time.monotonic()
            finished_early = False
            if args.mode == 'expert':
                io.stream(plan_pick_and_place(red[:2], cfg))
            else:
                last_check = t0
                while io.now() - t0 < args.timeout:
                    enable_pub.publish(Bool(data=True))       # 方策ノードは 1 秒途切れると自動停止する
                    io.sleep_sim(0.2)
                    # 8 s 以降、2 s ごとに「トレイに入って腕が戻ったか」を見て早期終了
                    if io.now() - t0 > 8.0 and io.now() - last_check > 2.0:
                        last_check = io.now()
                        cube = get_model_pose('red_cube')
                        arm_home = io.joint_pos is not None and np.abs(io.joint_pos[:4] - HOME_Q).max() < 0.2
                        if in_tray(cube, cfg) and arm_home:
                            finished_early = True
                            break
                for _ in range(3):
                    enable_pub.publish(Bool(data=False))
                    io.spin_for(0.05)
            sim_sec, wall_sec = io.now() - t0, time.monotonic() - w0

            io.sleep_sim(0.5)
            final = get_model_pose('red_cube')
            ok = in_tray(final, cfg)
            status_pub.publish(String(data=json.dumps(
                {'trial': trial, 'state': 'success' if ok else 'fail'})))
            io.spin_for(0.3)
            if rec:
                rec.stop()
                rec = None
            dist = float(np.hypot(final[0] - red[0], final[1] - red[1]))
            row = dict(label=args.label or args.mode, trial=trial, success=int(ok),
                       cube_x=round(red[0], 4), cube_y=round(red[1], 4),
                       final_x=round(final[0], 4), final_y=round(final[1], 4), final_z=round(final[2], 4),
                       moved_cm=round(dist * 100, 1), early=int(finished_early),
                       sim_sec=round(sim_sec, 1), wall_sec=round(wall_sec, 1))
            rows.append(row)
            append_row(out, row)                   # 1 試行ごとに保存（途中で止めても結果が残る）
            n = sum(r['success'] for r in rows)
            io.get_logger().info(f'trial {trial}: {"SUCCESS" if ok else "FAIL"}  '
                                 f'cube=({red[0]:.3f},{red[1]:.3f})  moved={dist * 100:.1f}cm  '
                                 f'[{n}/{len(rows)} = {100 * n / len(rows):.0f}%]')
            io.go_home(2.0, gripper=-0.5)          # 開いたまま戻ってキューブを落とす
            io.go_home(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if rows:
            n = sum(r['success'] for r in rows)
            print(f'success rate: {n}/{len(rows)} = {100 * n / len(rows):.1f}%  → {out}', flush=True)
        try:                                       # Ctrl+C で rclpy が先に終了していても落ちないように
            enable_pub.publish(Bool(data=False))
        except Exception:
            pass
        if rec:
            rec.stop()
        try:
            io.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()


def append_row(out: Path, row: dict):
    write_header = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


if __name__ == '__main__':
    main()
