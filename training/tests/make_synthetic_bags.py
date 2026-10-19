#!/usr/bin/env python3
"""テスト用: collect_demos と同じトピック構成の rosbag を、Gazebo なしで合成する.

お手本の関節指令（crane_vision_pick.expert）を 20Hz で書き、関節状態は 1 ステップ遅れで追従させ、
カメラ画像には赤いキューブの位置に赤い四角を描く。変換・学習スクリプトの動作確認に使う。

  python make_synthetic_bags.py --out /tmp/synth_bags --episodes 6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / 'crane_vision_pick'))
from crane_vision_pick.expert import TaskConfig, plan_pick_and_place, sample_cube_pose  # noqa: E402
from crane_vision_pick.kinematics import ALL_JOINTS, ARM_JOINTS, GRIPPER_JOINT  # noqa: E402

ts = get_typestore(Stores.ROS2_HUMBLE)
JointState = ts.types['sensor_msgs/msg/JointState']
Image = ts.types['sensor_msgs/msg/Image']
Header = ts.types['std_msgs/msg/Header']
Time = ts.types['builtin_interfaces/msg/Time']
Duration = ts.types['builtin_interfaces/msg/Duration']
JT = ts.types['trajectory_msgs/msg/JointTrajectory']
JTP = ts.types['trajectory_msgs/msg/JointTrajectoryPoint']

H, W = 240, 320


def stamp(t):
    return Header(stamp=Time(sec=int(t), nanosec=int((t % 1) * 1e9)), frame_id='')


def render(cube_xy):
    """カメラ（ロボット横の真上, 画像の右 = -y, 上 = +x）から見た簡易画像."""
    img = np.full((H, W, 3), (200, 185, 160), np.uint8)
    cx, cy = 0.106, 0.1025
    m_per_px = 2 * 0.435 * math.tan(math.radians(35)) / W
    u = int(W / 2 - (cube_xy[1] - cy) / m_per_px)
    v = int(H / 2 - (cube_xy[0] - cx) / m_per_px)
    s = int(0.025 / m_per_px)
    img[max(0, v - s):v + s, max(0, u - s):u + s] = (215, 15, 15)
    return img


def jt_msg(names, pos):
    return JT(header=stamp(0), joint_names=list(names),
              points=[JTP(positions=np.asarray(pos, float), velocities=np.zeros(0), accelerations=np.zeros(0),
                          effort=np.zeros(0), time_from_start=Duration(sec=0, nanosec=100_000_000))])


def write_episode(path, cube, cfg, t0=100.0, drop_images_after=None):
    cmds = plan_pick_and_place(cube[:2], cfg)
    cmds = np.vstack([cmds, np.repeat(cmds[-1:], 6, 0)])
    with Writer(path, version=8) as w:
        c_js = w.add_connection('/joint_states', JointState.__msgtype__, typestore=ts)
        c_im = w.add_connection('/camera/image_small', Image.__msgtype__, typestore=ts)
        c_arm = w.add_connection('/crane_plus_arm_controller/joint_trajectory', JT.__msgtype__, typestore=ts)
        c_gr = w.add_connection('/crane_plus_gripper_controller/joint_trajectory', JT.__msgtype__, typestore=ts)
        msgs = []
        T = len(cmds) * 0.05
        # 指令 20Hz
        for k, c in enumerate(cmds):
            t = t0 + 0.02 + k * 0.05
            msgs.append((t, c_arm, jt_msg(ARM_JOINTS, c[:4])))
            msgs.append((t, c_gr, jt_msg([GRIPPER_JOINT], c[4:5])))
        # 関節状態 50Hz（並び順はわざと崩す: 名前で並べ替えられるかの確認）
        order = [0, 1, 3, 2, 4]
        for k in range(int(T / 0.02)):
            t = t0 + k * 0.02
            i = min(len(cmds) - 1, max(0, int((t - t0) / 0.05) - 1))
            q = cmds[i]
            msgs.append((t, c_js, JointState(header=stamp(t), name=[ALL_JOINTS[j] for j in order],
                                             position=np.array([q[j] for j in order]),
                                             velocity=np.zeros(5), effort=np.zeros(5))))
        # 画像 10Hz
        for k in range(int(T / 0.1)):
            t = t0 + 0.01 + k * 0.1
            if drop_images_after is not None and t - t0 > drop_images_after:
                break
            img = render(cube[:2])
            msgs.append((t, c_im, Image(header=stamp(t), height=H, width=W, encoding='rgb8', is_bigendian=0,
                                        step=W * 3, data=img.reshape(-1))))
        for t, conn, m in sorted(msgs, key=lambda x: x[0]):
            w.write(conn, int(t * 1e9), ts.serialize_cdr(m, conn.msgtype))
    (Path(path) / 'episode_meta.json').write_text(json.dumps({'cube_initial': list(cube[:3]), 'blue': None}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--episodes', type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = TaskConfig()
    rng = np.random.default_rng(0)
    for e in range(args.episodes):
        write_episode(out / f'ep_{e:04d}', sample_cube_pose(cfg, rng), cfg)
    # 検証で弾かれるべき壊れたエピソード（途中から画像が来ない）
    write_episode(out / f'ep_{args.episodes:04d}', sample_cube_pose(cfg, rng), cfg, drop_images_after=3.0)
    print(f'wrote {args.episodes + 1} bags to {out}')


if __name__ == '__main__':
    main()
