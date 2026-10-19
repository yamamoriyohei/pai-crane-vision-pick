#!/usr/bin/env python3
"""データ検証の結果（変換時の validation_report.json）を図にする（レポート用）.

上段: エピソードごとの画像の実効周波数（シミュレーション時間）
下段: 画像の最大の抜け
検証を通ったものと除外したものを色と形で分ける。

  python plot_data_quality.py <dataset_root>/meta/validation_report.json --out data_quality.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

KEPT, DROPPED = '#2a78d6', '#eb6834'          # 既定パレットの 1・2 番（色覚多様性チェック済み）
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('report')
    ap.add_argument('--out', default='data_quality.png')
    ap.add_argument('--min-hz', type=float, default=7.0)
    ap.add_argument('--max-gap', type=float, default=0.35)
    args = ap.parse_args()
    rows = json.load(open(args.report))
    idx = [int(r['bag'].split('_')[1]) for r in rows]
    ok = [r['ok'] for r in rows]
    hz = [r.get('image_hz', 0) for r in rows]
    gap = [r.get('max_image_gap_s', 0) for r in rows]

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, facecolor='#fcfcfb')
    for ax, vals, thr, lab, unit in ((a1, hz, args.min_hz, 'image rate', 'Hz'),
                                     (a2, gap, args.max_gap, 'largest image gap', 's')):
        ax.set_facecolor('#fcfcfb')
        for keep, color, marker, name in ((True, KEPT, 'o', 'kept'), (False, DROPPED, 'X', 'excluded')):
            xs = [i for i, k in zip(idx, ok) if k == keep]
            ys = [v for v, k in zip(vals, ok) if k == keep]
            ax.scatter(xs, ys, s=28, color=color, marker=marker, label=f'{name} ({len(xs)})',
                       edgecolors='#fcfcfb', linewidths=0.8, zorder=3)
        ax.axhline(thr, color=MUTED, lw=1, ls='--', zorder=2)
        ax.annotate(f'threshold {thr:g} {unit}', xy=(1, thr), xycoords=('axes fraction', 'data'),
                    xytext=(0, 3), textcoords='offset points', ha='right', va='bottom', color=MUTED, fontsize=8)
        ax.set_ylabel(f'{lab} [{unit}]')
        ax.grid(axis='y', color=GRID, lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    a1.legend(loc='upper left', frameon=False)
    a2.set_xlabel('episode')
    a1.set_title(f'Camera stream check before training: {sum(ok)}/{len(ok)} episodes kept',
                 loc='left', color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor=fig.get_facecolor())
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
