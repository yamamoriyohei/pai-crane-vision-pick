#!/usr/bin/env python3
"""方策のデバッグ: データセットの観測を方策に与え、予測した行動と正解（お手本）を比べる.

講義の debug_policy.py を画像入力に対応させ、図を PNG に保存するようにしたもの
（VNC 上で plt.show() が開かない問題を避ける）。
「データ上で合っているのに Gazebo でうまく動かない」なら汎化の問題、
「データ上でもずれている」なら学習の問題、と切り分ける（第5回講義）。

  pixi run python .../debug_policy_vision.py --dataset-root ~/dataset/lerobot_dataset/crane_vision_pick \
      --repo-id crane_vision_pick --policy-path ~/checkpoints/train/crane_vision_pick_act --episode-index 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--repo-id', required=True)
    ap.add_argument('--policy-path', required=True)
    ap.add_argument('--episode-index', type=int, nargs='+', default=[0])
    ap.add_argument('--device', default='auto', help='auto: GPU があれば cuda、なければ cpu')
    ap.add_argument('--out', default='', help='PNG の保存先（既定: policy-path/debug_ep<N>.png）')
    args = ap.parse_args()

    if args.device == 'auto' or (args.device == 'cuda' and not torch.cuda.is_available()):
        if args.device == 'cuda':
            print('[注意] GPU が使えないので CPU で実行します')
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = ACTPolicy.from_pretrained(args.policy_path).to(args.device).eval()
    keys = list(policy.config.input_features)
    print('policy inputs:', keys)
    results = []
    for ep in args.episode_index:
        ds = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=[ep])
        policy.reset()
        preds, gts = [], []
        for i in range(len(ds)):
            s = ds[i]
            obs = {k: s[k].unsqueeze(0).to(args.device) for k in keys}
            with torch.no_grad():
                preds.append(policy.select_action(obs).squeeze(0).cpu().numpy())
            gts.append(s['action'].numpy())
        preds, gts = np.array(preds), np.array(gts)
        err = np.abs(preds - gts)
        results.append((ep, err.mean(0)))
        print(f'episode {ep}: 平均絶対誤差 [rad] ' + '  '.join(f'{n}={e:.3f}' for n, e in zip(NAMES, err.mean(0))))

        fig, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
        t = np.arange(len(gts)) / ds.fps
        for j, ax in enumerate(axes):
            ax.plot(t, gts[:, j], '--', color='0.4', label='GT (expert)')
            ax.plot(t, preds[:, j], color='C3', label='pred (ACT)')
            ax.set_ylabel(f'{NAMES[j]} [rad]')
            ax.grid(alpha=0.3)
        axes[0].legend(loc='upper right')
        axes[-1].set_xlabel('time [s]')
        fig.suptitle(f'Episode {ep}: GT vs predicted actions  (inputs: {", ".join(keys)})', fontsize=10)
        fig.tight_layout()
        out = Path(args.out) if args.out and len(args.episode_index) == 1 else \
            Path(args.policy_path) / f'debug_ep{ep}.png'
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f'  図を保存: {out}')
    if len(results) > 1:
        m = np.mean([r[1] for r in results], axis=0)
        print('全体平均: ' + '  '.join(f'{n}={e:.3f}' for n, e in zip(NAMES, m)))


if __name__ == '__main__':
    main()
