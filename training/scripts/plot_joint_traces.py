#!/usr/bin/env python3
"""評価 bag の関節の動きを条件ごとに重ねて描く（レポート用）.

ROS をインストールしていない環境でも動くように、bag（sqlite3）の JointState を自前で読む。
同じ試行番号（= 同じキューブ配置）の bag を並べると、方策がどこへ手を伸ばしたかが一目で分かる。

  python plot_joint_traces.py --bags-dir ~/physical/pai_work/dataset/eval_bags --trial 0 \
      --labels expert act_vision act_state_only --out joint_traces_trial0.png
"""
from __future__ import annotations

import argparse
import glob
import math
import sqlite3
import struct
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

JOINTS = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4',
          'crane_plus_joint_hand']
COLORS = {'expert': '#8a8985', 'act_vision': '#2a78d6', 'act_state_only': '#eb6834'}
NAMES = {'expert': 'expert (IK)', 'act_vision': 'ACT + camera', 'act_state_only': 'ACT, joints only'}
INK, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
TRAY_THETA = math.radians(-65.0)


def parse_joint_state(b: bytes):
    """sensor_msgs/JointState の CDR（リトルエンディアン）から時刻と関節角を取り出す."""
    o = 4

    def align(n):
        nonlocal o
        o += (-(o - 4)) % n

    def u32():
        nonlocal o
        align(4)
        v = struct.unpack_from('<I', b, o)[0]
        o += 4
        return v

    def string():
        nonlocal o
        n = u32()
        v = b[o:o + n - 1].decode()
        o += n
        return v

    sec, nsec = u32(), u32()
    string()                                   # frame_id
    names = [string() for _ in range(u32())]
    n = u32()
    align(8)
    pos = struct.unpack_from(f'<{n}d', b, o)
    return sec + nsec * 1e-9, dict(zip(names, pos))


def load_bag(path):
    db = glob.glob(str(Path(path) / '*.db3'))[0]
    con = sqlite3.connect(db)
    tid = {name: i for i, name in con.execute('select id, name from topics')}
    rows = con.execute('select data from messages where topic_id=? order by timestamp',
                       (tid['/joint_states'],)).fetchall()
    q = []
    for (d,) in rows:
        t, p = parse_joint_state(d)
        q.append([t] + [p.get(j, np.nan) for j in JOINTS])
    q = np.array(q)
    q[:, 0] -= q[0, 0]
    return q


def motion_start(q, thr=0.02):
    """ホーム姿勢から動き出した時刻（条件ごとに開始のタイミングがずれるのでそろえる）."""
    d = np.abs(q[:, 1:5] - q[0, 1:5]).max(axis=1)
    i = int(np.argmax(d > thr))
    return q[i, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bags-dir', required=True)
    ap.add_argument('--trial', type=int, default=0)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--labels', nargs='+', default=['expert', 'act_vision', 'act_state_only'])
    ap.add_argument('--cube', nargs=2, type=float, default=None, help='キューブの位置 x y [m]（目標の向きの線を引く）')
    ap.add_argument('--out', default='joint_traces.png')
    args = ap.parse_args()

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, facecolor=BG,
                                 gridspec_kw={'height_ratios': [1.6, 1]})
    for lab in args.labels:
        bag = Path(args.bags_dir).expanduser() / f'{lab}_s{args.seed}_{args.trial:03d}'
        if not bag.exists():
            print(f'skip (not found): {bag}')
            continue
        q = load_bag(bag)
        t = q[:, 0] - motion_start(q)
        keep = t >= -0.5
        c = COLORS.get(lab, INK)
        a1.plot(t[keep], np.degrees(q[keep, 1]), color=c, lw=1.8, label=NAMES.get(lab, lab))
        a2.plot(t[keep], np.degrees(q[keep, 5]), color=c, lw=1.8)
    if args.cube:
        th = math.degrees(math.atan2(args.cube[1], args.cube[0]))
        a1.axhline(th, color=MUTED, lw=1, ls='--')
        a1.annotate(f'cube direction {th:.0f}°', xy=(1, th), xycoords=('axes fraction', 'data'),
                    xytext=(0, 3), textcoords='offset points', ha='right', va='bottom', color=MUTED, fontsize=8)
    a1.axhline(math.degrees(TRAY_THETA), color=MUTED, lw=1, ls=':')
    a1.annotate('tray direction', xy=(1, math.degrees(TRAY_THETA)), xycoords=('axes fraction', 'data'),
                xytext=(0, 3), textcoords='offset points', ha='right', va='bottom', color=MUTED, fontsize=8)
    a2.axhline(20, color=MUTED, lw=1, ls=':')
    a2.annotate('closed', xy=(1, 20), xycoords=('axes fraction', 'data'), xytext=(0, 3),
                textcoords='offset points', ha='right', va='bottom', color=MUTED, fontsize=8)
    a1.set_ylabel('base rotation (joint1) [deg]')
    a2.set_ylabel('gripper [deg]')
    a2.set_xlabel('time since the arm started moving [s, sim time]')
    for ax in (a1, a2):
        ax.set_facecolor(BG)
        ax.grid(axis='y', color=GRID, lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    a1.legend(loc='upper right', frameon=False, ncol=3, bbox_to_anchor=(1, 1.16))
    a1.set_title(f'Evaluation trial {args.trial}: where each controller reaches', loc='left', color=INK,
                 fontsize=10, pad=18)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor=BG)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
