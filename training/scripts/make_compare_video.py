#!/usr/bin/env python3
"""評価 bag を左右に並べた比較動画を作る（提出動画用）.

同じ試行番号（= 同じキューブ配置）の bag を 2 本以上並べ、各パネルの上に条件名と結果を出す。
短い方は最後のコマを伸ばしてそろえる。

  python make_compare_video.py eval_bags/act_vision_s10000_000 eval_bags/act_state_only_s10000_000 \
      --titles "ACT + カメラ" "ACT（関節角のみ）" --out compare_trial0.mp4 --fps 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from rosbags.rosbag2 import Reader

from bag_to_video import load_font
from rosbag2lerobot_vision import _deserializer, image_to_numpy

BAR = 44
COL = {'success': (60, 190, 90), 'fail': (235, 80, 70), 'running': (240, 240, 240)}
TXT = {'success': 'SUCCESS', 'fail': 'FAIL', 'running': ''}


def read_bag(bag, image_topic='/camera/image_small'):
    deser = _deserializer()
    frames, status = [], 'running'
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic in (image_topic, '/eval/status')]
        for c, _t, raw in r.messages(connections=conns):
            m = deser(raw, c.msgtype)
            if c.topic == '/eval/status':
                status = json.loads(m.data).get('state', status)
                continue
            frames.append(image_to_numpy(m))
    return frames, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bags', nargs='+')
    ap.add_argument('--titles', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=20, help='カメラは 10Hz なので 20 で 2 倍速')
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--hold', type=float, default=1.5, help='最後に結果を出して止める秒数')
    ap.add_argument('--caption', default='', help='画面下に出す説明（1 行）')
    args = ap.parse_args()
    assert len(args.titles) == len(args.bags)

    clips = [read_bag(b) for b in args.bags]
    n = max(len(f) for f, _ in clips)
    hold = int(args.hold * args.fps)
    font, small = load_font(22), load_font(18)
    h0, w0 = clips[0][0][0].shape[:2]
    pw, ph = w0 * args.scale, h0 * args.scale
    gap = 8
    W = pw * len(clips) + gap * (len(clips) - 1)
    H = ph + BAR + (36 if args.caption else 0)
    H += (-H) % 8                              # 動画コーデックの都合で 8 の倍数に
    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(Path(args.out).expanduser(), fps=args.fps, codec='libx264', quality=8,
                            macro_block_size=8)
    for i in range(n + hold):
        canvas = Image.new('RGB', (W, H), (20, 20, 20))
        d = ImageDraw.Draw(canvas)
        for k, ((frames, final), title) in enumerate(zip(clips, args.titles)):
            j = min(i, len(frames) - 1)
            done = i >= len(frames) - 1
            st = final if done else 'running'
            img = Image.fromarray(frames[j]).resize((pw, ph), Image.BILINEAR)
            x = k * (pw + gap)
            canvas.paste(img, (x, BAR))
            d.text((x + 10, 8), title, fill=(255, 255, 255), font=font)
            if TXT[st]:
                tw = d.textlength(TXT[st], font=font)
                d.text((x + pw - tw - 10, 8), TXT[st], fill=COL[st], font=font)
            d.text((x + 10, BAR + ph - 26), f't = {j / 10:.1f} s', fill=(255, 255, 255), font=small)
        if args.caption:
            d.text((10, BAR + ph + 6), args.caption, fill=(230, 230, 230), font=small)
        wr.append_data(np.asarray(canvas))
    wr.close()
    print(f'saved {args.out}  ({(n + hold) / args.fps:.1f} s)')


if __name__ == '__main__':
    main()
