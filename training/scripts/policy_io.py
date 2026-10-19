"""ROS メッセージ ⇔ 方策の入力テンソルの変換（ROS なしでもテストできるよう分離）."""
from __future__ import annotations

import numpy as np
import torch

JOINTS = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4',
          'crane_plus_joint_hand']


def joint_state_to_vec(names, positions):
    """/joint_states → [j1, j2, j3, j4, hand]（学習データと同じ名前順）. 足りなければ None."""
    m = dict(zip(names, positions))
    if not all(j in m for j in JOINTS):
        return None
    return np.array([m[j] for j in JOINTS], dtype=np.float32)


def image_msg_to_rgb(height, width, step, encoding, data):
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}[encoding]
    img = np.frombuffer(bytes(data), dtype=np.uint8).reshape(height, step)[:, :width * ch]
    img = img.reshape(height, width, ch)
    if ch == 1:
        img = np.repeat(img, 3, axis=2)
    img = img[:, :, :3]
    if encoding.startswith('bgr'):
        img = img[:, :, ::-1]
    return img.copy()      # 受信バッファ（読み取り専用）から切り離す


def build_observation(policy_config, state_vec, rgb, device):
    """方策が要求する入力キーだけを作る（画像あり / なし どちらのモデルでも動く）."""
    obs = {}
    for key, ft in policy_config.input_features.items():
        if key in ('observation.state', 'observation.environment_state'):
            obs[key] = torch.from_numpy(state_vec).unsqueeze(0).to(device)
        elif key.startswith('observation.images'):
            if rgb is None:
                raise ValueError('画像がまだ届いていません')
            c, h, w = ft.shape
            if rgb.shape[:2] != (h, w):
                raise ValueError(f'画像サイズ {rgb.shape[:2]} が学習時 {(h, w)} と違います'
                                 '（image_downsampler の factor を学習時と同じにしてください）')
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)   # (3, H, W), [0, 1]
            obs[key] = t.unsqueeze(0).to(device)
    return obs


def uses_images(policy_config):
    return any(k.startswith('observation.images') for k in policy_config.input_features)
