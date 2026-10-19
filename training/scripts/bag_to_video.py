#!/usr/bin/env python3
"""評価時の bag（evaluate --record-dir）から、カメラ映像に字幕を付けた mp4 を作る（提出動画の素材）.

各 bag の /camera/image_small を 2 倍に拡大し、左上に「試行番号・条件・結果」を重ねて
1 本の動画につなげる。結果は /eval/status（evaluate.py が出す）から読む。

  pixi run python .../bag_to_video.py ~/dataset/eval_bags/act_vision_* --out ~/videos/act_vision.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rosbags.rosbag2 import Reader

from rosbag2lerobot_vision import _deserializer, image_to_numpy


def load_font(size):
    for p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=10, help='カメラは 10Hz。20 にすると 2 倍速')
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--image-topic', default='/camera/image_small')
    args = ap.parse_args()

    deser = _deserializer()
    font = load_font(18)
    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(Path(args.out).expanduser(), fps=args.fps, codec='libx264', quality=8,
                                macro_block_size=8)
    for bag in sorted(args.bags):
        frames, status = [], 'running'
        with Reader(bag) as r:
            conns = [c for c in r.connections if c.topic in (args.image_topic, '/eval/status')]
            for c, _t, raw in r.messages(connections=conns):
                m = deser(raw, c.msgtype)
                if c.topic == '/eval/status':
                    status = json.loads(m.data).get('state', status)
                    continue
                frames.append((image_to_numpy(m), status))
        final = status
        label = Path(bag).name
        for i, (img, st) in enumerate(frames):
            pil = Image.fromarray(img).resize((img.shape[1] * args.scale, img.shape[0] * args.scale),
                                              Image.NEAREST)
            d = ImageDraw.Draw(pil)
            shown = final if i >= len(frames) - args.fps else 'running'   # 最後の 1 秒で結果を表示
            color = {'success': (40, 200, 60), 'fail': (230, 50, 50)}.get(shown, (255, 255, 255))
            d.rectangle([0, 0, pil.width, 30], fill=(0, 0, 0))
            d.text((8, 5), f'{label}   t={i / 10:.1f}s   {shown.upper()}', fill=color, font=font)
            writer.append_data(np.asarray(pil))
        print(f'{label}: {len(frames)} frames, result={final}')
    writer.close()
    print(f'saved: {args.out}')


if __name__ == '__main__':
    main()
