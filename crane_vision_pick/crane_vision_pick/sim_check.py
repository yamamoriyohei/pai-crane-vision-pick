"""環境チェック（第1週の最初にこれを通す）.

第3回講義の「まず測る」に従って、収集を始める前に次を確認する:
  1. /clock が進むか、実時間比 (RTF) はいくつか（Mac の Docker では 1 を下回ることが多い）
  2. /joint_states に 5 関節がそろっているか
  3. カメラ /image_raw と縮小画像 /camera/image_small の解像度・エンコード・周波数
  4. 画像の中に赤いキューブが写っているか（簡単な色しきい値で重心を出す）
  5. ign の set_pose / pose 取得が動くか（キューブを動かして読み戻す）
  6. --move を付けると、お手本で 1 回ピック＆プレースして成功判定まで行う
結果のスナップショットを ~/sim_check.png に保存する。

  ros2 run crane_vision_pick sim_check            # 確認だけ
  ros2 run crane_vision_pick sim_check --move     # お手本も 1 回動かす
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import time
import zlib

import numpy as np
import rclpy
from sensor_msgs.msg import Image

from .collect_demos import in_tray, place_objects
from .expert import TaskConfig, add_grasp_args, apply_grasp_args, plan_pick_and_place
from .kinematics import fk
from .ros_io import ArmIO
from .sim_utils import get_model_pose


def save_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(png)


def to_rgb(msg):
    ch = 3 if msg.encoding in ('rgb8', 'bgr8') else 4
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch)[:, :, :3]
    return img[:, :, ::-1] if msg.encoding.startswith('bgr') else img


def red_centroid(rgb):
    r, g, b = (rgb[:, :, i].astype(int) for i in range(3))
    mask = (r > 120) & (g < 70) & (b < 70)
    if mask.sum() < 5:
        return None, 0
    ys, xs = np.nonzero(mask)
    return (float(xs.mean()), float(ys.mean())), int(mask.sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--move', action='store_true', help='お手本で 1 回ピック＆プレース')
    ap.add_argument('--probe', action='store_true', help='把持テスト: フェーズごとにキューブ位置・グリッパ角を表示')
    ap.add_argument('--repeat', type=int, default=1, help='--move/--probe を何回繰り返すか')
    ap.add_argument('--cube', type=float, nargs=2, default=[0.22, 0.06], metavar=('X', 'Y'))
    ap.add_argument('--cube-yaw-deg', type=float, default=0.0, help='キューブの向き（腕の向きからのずれ）')
    add_grasp_args(ap)
    ap.add_argument('--skip-camera', action='store_true')
    args, _ = ap.parse_known_args(argv)
    if args.probe:
        args.move = True

    rclpy.init()
    io = ArmIO('sim_check')
    stats = {}

    def make_cb(key):
        def cb(msg):
            s = stats.setdefault(key, {'n': 0, 'first_wall': time.monotonic(), 'first_sim': io.now()})
            s['n'] += 1
            s['last_wall'], s['last_sim'], s['msg'] = time.monotonic(), io.now(), msg
        return cb
    io.create_subscription(Image, '/image_raw', make_cb('raw'), 2)
    io.create_subscription(Image, '/camera/image_small', make_cb('small'), 2)

    ok_all = True

    def report(ok, text):
        nonlocal ok_all
        ok_all &= ok
        print(('  [OK]   ' if ok else '  [NG]   ') + text)

    print('\n=== sim_check ===')
    try:
        io.wait_for_clock(timeout=60)
    except RuntimeError as e:
        report(False, str(e))
        return
    w0, s0 = time.monotonic(), io.now()
    io.spin_for(5.0)
    rtf = (io.now() - s0) / (time.monotonic() - w0)
    report(rtf > 0.05, f'/clock が進んでいる  RTF(実時間比) = {rtf:.2f}')
    io.go_home(3.0)
    report(io.joint_pos is not None, f'/joint_states 5関節 = {np.round(io.joint_pos, 3) if io.joint_pos is not None else None}')

    for key, topic in (() if args.skip_camera else (('raw', '/image_raw'), ('small', '/camera/image_small'))):
        s = stats.get(key)
        if not s or s['n'] < 2:
            report(False, f'{topic} が来ていない（use_camera:=true で起動？ image_downsampler は動いている？）')
            continue
        m = s['msg']
        hz_wall = (s['n'] - 1) / max(s['last_wall'] - s['first_wall'], 1e-6)
        hz_sim = (s['n'] - 1) / max(s['last_sim'] - s['first_sim'], 1e-6)
        mb = len(m.data) * hz_wall / 1e6
        report(True, f'{topic}: {m.width}x{m.height} {m.encoding}  {hz_sim:.1f} Hz(sim) / {hz_wall:.1f} Hz(wall)'
                     f'  ≈ {mb:.2f} MB/s')

    try:
        cfg = apply_grasp_args(TaskConfig(), args)
        cx, cy = args.cube
        target = (cx, cy, math.atan2(cy, cx) + math.radians(args.cube_yaw_deg))
        actual = place_objects(io, target, None)
        report(True, f'set_pose / pose 取得 OK: キューブ = ({actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f})')
    except Exception as e:  # noqa: BLE001
        report(False, f'ign でキューブを動かせない: {e}')
        actual = None

    io.spin_for(1.5)
    s = None if args.skip_camera else (stats.get('small') or stats.get('raw'))
    if s:
        rgb = to_rgb(s['msg'])
        c, n = red_centroid(rgb)
        report(c is not None, f'画像中の赤いキューブ: 重心={None if c is None else tuple(round(v) for v in c)}  画素数={n}')
        path = os.path.expanduser('~/sim_check.png')
        save_png(path, np.ascontiguousarray(rgb))
        print(f'         スナップショット: {path}')

    if args.move and actual is not None:
        print(f'         把持設定: grip_offset={cfg.grip_offset:.3f}  lateral={cfg.lateral_offset:+.3f}  '
              f'z_grasp={cfg.z_grasp:.3f}  close={math.degrees(cfg.gripper_close):.0f}deg  '
              f'cube_yaw={args.cube_yaw_deg:+.0f}deg')
        n_ok = 0
        for rep in range(args.repeat):
            if rep > 0:
                actual = place_objects(io, target, None)
            cmds, phases = plan_pick_and_place(actual[:2], cfg, return_phases=True)
            w0, s0 = time.monotonic(), io.now()
            start = 0
            for name, end in phases:
                io.stream(cmds[start:end])
                start = end
                if args.probe:
                    io.spin_for(0.1)
                    cube = get_model_pose('red_cube')
                    q = io.joint_pos
                    tip_cmd = fk(cmds[end - 1, :4])[0]
                    tip_act = fk(q[:4])[0]
                    err = (tip_act - tip_cmd) * 1000
                    print(f'         {name:8s} キューブ=({cube[0]:+.3f},{cube[1]:+.3f},{cube[2]:.3f})  '
                          f'グリッパ 指令 {math.degrees(cmds[end - 1, 4]):+5.1f}° / 実際 {math.degrees(q[4]):+5.1f}°  '
                          f'手先の追従誤差 [mm] x{err[0]:+.0f} y{err[1]:+.0f} z{err[2]:+.0f}')
            io.sleep_sim(0.8)
            final = get_model_pose('red_cube')
            ok = in_tray(final, cfg)
            n_ok += ok
            report(ok, f'お手本のピック＆プレース: 最終位置=({final[0]:.3f},{final[1]:.3f},{final[2]:.3f})'
                       f'  所要 {io.now() - s0:.1f}s(sim) / {time.monotonic() - w0:.1f}s(wall)')
            io.go_home(2.0, gripper=cfg.gripper_open)
            io.go_home(0.5)
        if args.repeat > 1:
            print(f'         成功 {n_ok}/{args.repeat}')

    print('=== ' + ('ALL OK' if ok_all else 'NG あり（上の [NG] を確認）') + ' ===\n')
    io.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
