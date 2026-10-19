#!/usr/bin/env python3
"""カメラ画像つき ACT の学習（講義の pai_training/scripts/train.py を拡張）.

主な変更点:
  - input_keys で方策の入力を選べる（画像あり / 画像なしの比較実験を同じスクリプトで）
  - エピソード単位で学習用 / 検証用に分け、検証 L1 誤差を定期的に計算（過学習の確認）
  - 画像の色・明るさのデータ拡張、勾配クリップ、バックボーン別学習率（LeRobot 既定と同じ）
  - 途中チェックポイント保存、学習曲線を CSV に保存（レポートの図に使う）
  - デバイス自動選択（Colab の GPU → cuda、Mac ネイティブ → mps、それ以外 → cpu）

  cd ~/pai_training
  pixi run python ~/pai_ws/src/pai-crane-vision-pick/training/scripts/train_vision.py \
      --config ~/pai_ws/src/pai-crane-vision-pick/training/configs/vision_train.yaml
  # 動作確認だけなら --steps 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_l1(policy, loader, device, max_batches=50):
    policy.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        _, info = policy.forward(to_device(batch, device))
        losses.append(info['l1_loss'])
    policy.train()
    return float(np.mean(losses)) if losses else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--steps', type=int, default=0, help='設定ファイルの training_steps を上書き')
    ap.add_argument('--dataset-root', default='', help='設定ファイルの dataset_root を上書き（Colab 用）')
    ap.add_argument('--output', default='', help='設定ファイルの output_directory を上書き（Colab 用）')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.steps:
        cfg['training_steps'] = args.steps
    if args.dataset_root:
        cfg['dataset_root'] = args.dataset_root
    if args.output:
        cfg['output_directory'] = args.output

    seed = int(cfg.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = pick_device(cfg.get('device', 'auto'))
    root = Path(cfg['dataset_root']).expanduser()
    out = Path(cfg['output_directory']).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    print(f'device: {device}')

    # ---- features: データセットのメタ情報から作り、input_keys で絞る
    meta = LeRobotDatasetMetadata(cfg['repo_id'], root=root)
    all_feats = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in all_feats.items() if f.type is FeatureType.ACTION}
    input_features = {k: all_feats[k] for k in cfg['input_keys']}
    print('input_features :', {k: tuple(v.shape) for k, v in input_features.items()})
    print('output_features:', {k: tuple(v.shape) for k, v in output_features.items()})

    chunk = int(cfg['chunk_size'])
    policy_cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk,
        n_action_steps=int(cfg['n_action_steps']),
        kl_weight=float(cfg.get('kl_weight', 10.0)),
        optimizer_lr=float(cfg['learning_rate']),
        optimizer_lr_backbone=float(cfg['learning_rate']),
        optimizer_weight_decay=float(cfg.get('weight_decay', 1e-4)),
        # ImageNet で事前学習した ResNet18 から始める（初回は torchvision が重みをダウンロード）
        pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1' if cfg.get('pretrained_backbone', True) else None,
    )
    policy = ACTPolicy(policy_cfg, dataset_stats=meta.stats)
    policy.to(device)
    policy.train()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f'ACT parameters: {n_params / 1e6:.1f} M')

    # ---- dataset: エピソード単位で学習 / 検証に分ける（同じエピソードのフレームが両方に入らないように）
    n_ep = meta.total_episodes
    n_val = max(1, int(round(n_ep * float(cfg.get('val_ratio', 0.1))))) if n_ep >= 5 else 0
    train_eps = list(range(n_ep - n_val))
    val_eps = list(range(n_ep - n_val, n_ep))
    delta = {'action': [i / meta.fps for i in range(chunk)]}
    tf = None
    if cfg.get('image_augmentation') and any(f.type is FeatureType.VISUAL for f in input_features.values()):
        tf = ImageTransforms(ImageTransformsConfig(enable=True, max_num_transforms=2))
    # 注意: このコミットの LeRobot は episodes=[...] と delta_timestamps を同時に使うと
    # エピソード番号の対応がずれる不具合があるため、全エピソードを読んでから Subset で分ける
    full_train = LeRobotDataset(cfg['repo_id'], root=root, delta_timestamps=delta, image_transforms=tf)
    ep_from = full_train.episode_data_index['from'].tolist()
    ep_to = full_train.episode_data_index['to'].tolist()

    def frames_of(eps):
        return [i for e in eps for i in range(ep_from[e], ep_to[e])]
    train_ds = torch.utils.data.Subset(full_train, frames_of(train_eps))
    val_ds = None
    if n_val:
        full_val = LeRobotDataset(cfg['repo_id'], root=root, delta_timestamps=delta)   # 拡張なし
        val_ds = torch.utils.data.Subset(full_val, frames_of(val_eps))
    print(f'episodes: train {len(train_eps)} / val {len(val_eps)}   frames: train {len(train_ds)}'
          + (f' / val {len(val_ds)}' if val_ds else ''))

    nw = int(cfg.get('num_workers', 2))
    loader = torch.utils.data.DataLoader(train_ds, batch_size=int(cfg['batch_size']), shuffle=True,
                                         num_workers=nw, pin_memory=device.type == 'cuda', drop_last=True,
                                         persistent_workers=nw > 0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(cfg['batch_size']), shuffle=False,
                                             num_workers=nw) if val_ds else None

    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=policy_cfg.optimizer_lr,
                                  weight_decay=policy_cfg.optimizer_weight_decay)

    use_wandb = bool(cfg.get('use_wandb'))
    if use_wandb:
        import wandb
        wandb.init(project='pai_crane_vision_pick', config=cfg)

    curve_f = open(out / 'training_curve.csv', 'w', newline='')
    curve = csv.writer(curve_f)
    curve.writerow(['step', 'loss', 'l1', 'kld', 'val_l1', 'sec'])
    total = int(cfg['training_steps'])
    step, t0 = 0, time.time()
    clip = float(cfg.get('grad_clip_norm', 10.0))
    log_every, val_every, save_every = (int(cfg.get(k, d)) for k, d in
                                        (('log_every', 100), ('val_every', 1000), ('save_every', 5000)))
    best_val = float('inf')
    while step < total:
        for batch in loader:
            loss, info = policy.forward(to_device(batch, device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            val_l1 = ''
            if val_loader and (step % val_every == 0 or step == total):
                val_l1 = evaluate_l1(policy, val_loader, device)
                if val_l1 < best_val:
                    best_val = val_l1
                    policy.save_pretrained(out / 'best')
            if step % log_every == 0 or step == 1 or val_l1 != '' or step == total:
                el = time.time() - t0
                msg = (f'step {step:6d}/{total}  loss {loss.item():.4f}  l1 {info["l1_loss"]:.4f}  '
                       f'kld {info.get("kld_loss", 0):.4f}')
                if val_l1 != '':
                    msg += f'  val_l1 {val_l1:.4f}'
                print(msg + f'  ({el / step:.3f} s/step, 残り {el / step * (total - step) / 60:.0f} 分)', flush=True)
                curve.writerow([step, round(loss.item(), 5), round(info['l1_loss'], 5),
                                round(info.get('kld_loss', 0), 5), val_l1 if val_l1 == '' else round(val_l1, 5),
                                round(el, 1)])
                curve_f.flush()
                if use_wandb:
                    log = {'loss': loss.item(), **info}
                    if val_l1 != '':
                        log['val_l1'] = val_l1
                    wandb.log(log, step=step)
            if step % save_every == 0 and step < total:
                policy.save_pretrained(out / f'step_{step:06d}')
            if step >= total:
                break

    policy.save_pretrained(out)
    curve_f.close()
    (out / 'train_settings.json').write_text(json.dumps(
        {**cfg, 'device_used': str(device), 'train_episodes': train_eps, 'val_episodes': val_eps,
         'best_val_l1': best_val if best_val != float('inf') else None,
         'minutes': round((time.time() - t0) / 60, 1)}, indent=2, ensure_ascii=False, default=str))
    print(f'policy saved to {out}  (best val は {out / "best"})')
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
