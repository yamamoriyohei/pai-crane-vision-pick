#!/usr/bin/env python3
"""評価結果（evaluate.py の CSV）からレポート用の図を作る.

  1. 条件ごとの成功率（棒グラフ, 95% 信頼区間つき）
  2. キューブの初期位置ごとの成否（上から見た図）→ どこで失敗しやすいかが分かる
  3. 学習曲線（train_vision.py の training_curve.csv）

  pixi run python .../plot_results.py ~/results/*.csv --curves ~/checkpoints/train/*/training_curve.csv \
      --out ~/results/figures
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

TRAY = (0.08875, -0.19033, -65.0)   # expert.TaskConfig と同じ
COLORS = {'expert': '#8a8985', 'act_vision': '#2a78d6', 'act_state_only': '#eb6834'}
NAMES = {'expert': 'Expert (IK)', 'act_vision': 'ACT + camera', 'act_state_only': 'ACT, joints only'}
CURVE_LABELS = {'crane_vision_pick_act': 'act_vision', 'crane_state_only_act': 'act_state_only'}
INK, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
OK_C, NG_C = '#2f8f5b', '#d0453a'   # 成功・失敗（形も変えて色覚に頼らない）


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csvs', nargs='+')
    ap.add_argument('--curves', nargs='*', default=[])
    ap.add_argument('--out', default='figures')
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # 同じ条件・同じ試行番号を撮り直したときは、後から書かれた行（最新）を使う
    latest = {}
    for p in args.csvs:
        with open(Path(p).expanduser()) as f:
            for r in csv.DictReader(f):
                latest[(r['label'], int(r['trial']))] = r
    rows = defaultdict(list)
    for (lab, _), r in sorted(latest.items(), key=lambda kv: kv[0][1]):
        rows[lab].append(r)
    order = ['expert', 'act_vision', 'act_state_only']
    labels = [l for l in order if l in rows] + [l for l in rows if l not in order]
    for lab in labels:
        print(f'{lab}: {sum(int(r["success"]) for r in rows[lab])}/{len(rows[lab])}')

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED})

    def tidy(ax, grid_axis='y'):
        ax.set_facecolor(BG)
        if grid_axis:
            ax.grid(axis=grid_axis, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    # 1. 成功率
    fig, ax = plt.subplots(figsize=(1.8 + 1.5 * len(labels), 3.4), facecolor=BG)
    for i, lab in enumerate(labels):
        k, n = sum(int(r['success']) for r in rows[lab]), len(rows[lab])
        lo, hi = wilson(k, n)
        c = COLORS.get(lab, INK)
        ax.bar(i, 100 * k / n, color=c, width=0.58, zorder=2)
        ax.errorbar(i, 100 * k / n, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]],
                    color=INK, capsize=4, lw=1, zorder=3)
        ax.text(i, 100 * hi + 3, f'{100 * k / n:.0f}%  ({k}/{n})', ha='center', va='bottom', color=INK)
    ax.set_xticks(range(len(labels)), [NAMES.get(l, l) for l in labels])
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('success rate [%]')
    ax.set_title('Pick-and-place success on unseen cube positions (bars: 95% Wilson CI)',
                 loc='left', color=INK, fontsize=10)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out / 'success_rate.png', dpi=160, facecolor=BG)
    plt.close(fig)

    # 2. 位置ごとの成否（ロボットの上から見た図。カメラ画像と同じく右が -y）
    tx, ty, tyaw = TRAY
    h = 0.045
    corners = [(+h, +h), (-h, +h), (-h, -h), (+h, -h)]
    cy, sy = math.cos(math.radians(tyaw)), math.sin(math.radians(tyaw))
    tray_xy = [(tx + cy * a - sy * b, ty + sy * a + cy * b) for a, b in corners]
    fig, axes = plt.subplots(1, len(labels), figsize=(3.3 * len(labels), 3.2), squeeze=False, facecolor=BG)
    for ax, lab in zip(axes[0], labels):
        ok = [r for r in rows[lab] if int(r['success'])]
        ng = [r for r in rows[lab] if not int(r['success'])]
        ax.add_patch(Polygon([(y, x) for x, y in tray_xy], closed=True, fc=GRID, ec=MUTED, lw=0.8))
        ax.text(ty, tx, 'tray', ha='center', va='center', color=MUTED, fontsize=8)
        ax.scatter([float(r['cube_y']) for r in ok], [float(r['cube_x']) for r in ok], s=30,
                   color=OK_C, marker='o', label=f'success ({len(ok)})', zorder=3,
                   edgecolors=BG, linewidths=0.8)
        ax.scatter([float(r['cube_y']) for r in ng], [float(r['cube_x']) for r in ng], s=34,
                   color=NG_C, marker='X', label=f'fail ({len(ng)})', zorder=3,
                   edgecolors=BG, linewidths=0.8)
        ax.scatter([0], [0], marker='s', color=INK, s=30)
        ax.annotate('robot', (0, 0), xytext=(0, -11), textcoords='offset points', ha='center',
                    color=MUTED, fontsize=8)
        ax.set_xlim(0.27, -0.27)
        ax.set_ylim(-0.04, 0.30)
        ax.set_aspect('equal')
        ax.set_title(NAMES.get(lab, lab), color=INK, fontsize=10)
        ax.set_xlabel('y [m]')
        ax.legend(loc='upper left', frameon=False, fontsize=8, handletextpad=0.2)
        tidy(ax, grid_axis='both')
    axes[0][0].set_ylabel('x [m]')
    fig.tight_layout()
    fig.savefig(out / 'success_map.png', dpi=160, facecolor=BG)
    plt.close(fig)

    # 3. 学習曲線（学習損失は移動平均、検証損失は 1000 step ごと）
    if args.curves:
        fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=BG)
        for p in args.curves:
            p = Path(p).expanduser()
            steps, l1, vs, vl = [], [], [], []
            with open(p) as f:
                for r in csv.DictReader(f):
                    steps.append(int(r['step']))
                    l1.append(float(r['l1']))
                    if r.get('val_l1'):
                        vs.append(int(r['step']))
                        vl.append(float(r['val_l1']))
            lab = CURVE_LABELS.get(p.parent.name, p.parent.name)
            c = COLORS.get(lab, INK)
            w = 10
            sm = [sum(l1[max(0, i - w + 1):i + 1]) / len(l1[max(0, i - w + 1):i + 1]) for i in range(len(l1))]
            ax.plot(steps, sm, color=c, lw=1.2, alpha=0.55)
            if vs:
                best = min(range(len(vl)), key=vl.__getitem__)
                ax.plot(vs, vl, 'o-', color=c, lw=1.8, ms=4,
                        label=f'{NAMES.get(lab, lab)}  (best val {vl[best]:.3f} @ {vs[best]})')
        ax.set_yscale('log')
        ax.set_xlabel('training step')
        ax.set_ylabel('action L1 loss (normalized)')
        ax.set_title('Training curves: thin = train (moving avg), dots = validation', loc='left',
                     color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        tidy(ax)
        fig.tight_layout()
        fig.savefig(out / 'training_curve.png', dpi=160, facecolor=BG)
        plt.close(fig)
    for lab in labels:
        k, n = sum(int(r['success']) for r in rows[lab]), len(rows[lab])
        print(f'{lab}: {k}/{n} = {100 * k / n:.1f}%')
    print(f'figures saved in {out}')


if __name__ == '__main__':
    main()
