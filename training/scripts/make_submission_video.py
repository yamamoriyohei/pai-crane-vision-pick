#!/usr/bin/env python3
"""提出用の動画（約 2 分）を、文字カード・図・評価クリップから組み立てる.

ナレーションなし、日本語字幕のみ。各パートを 1280x720 / 30fps の mp4 にしてからつなぐ（ffmpeg が必要）。

  python make_submission_video.py --clips videos/ --figs docs/figures/ --diagram system.png --out submission.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG, INK, MUTED, ACC = (20, 22, 26), (245, 245, 245), (170, 172, 178), (90, 160, 240)
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def card(title, lines, sub=None, accent_line=None):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    y = 150 if sub is None else 120
    if sub:
        d.text((90, y), sub, fill=ACC, font=font(28, True))
        y += 56
    d.text((90, y), title, fill=INK, font=font(52, True))
    y += 100
    for ln in lines:
        d.text((90, y), ln, fill=INK if ln != accent_line else ACC, font=font(32))
        y += 56
    return im


def figure_frame(img_path, caption, title=None):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    top = 20
    if title:
        d.text((40, 22), title, fill=INK, font=font(34, True))
        top = 80
    fig = Image.open(img_path).convert('RGB')
    box_w, box_h = W - 80, H - top - 90
    s = min(box_w / fig.width, box_h / fig.height)
    fig = fig.resize((int(fig.width * s), int(fig.height * s)), Image.LANCZOS)
    pad = Image.new('RGB', (fig.width + 24, fig.height + 24), (252, 252, 251))
    pad.paste(fig, (12, 12))
    im.paste(pad, ((W - pad.width) // 2, top + (box_h - pad.height) // 2))
    d.text((40, H - 70), caption, fill=INK, font=font(28))
    return im


def still_to_mp4(im, sec, out):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        im.save(f.name)
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-loop', '1', '-i', f.name, '-f', 'lavfi', '-i',
                        'anullsrc=r=48000:cl=stereo', '-t', str(sec), '-r', str(FPS),
                        '-vf', 'fade=in:0:8,fade=out:st=%.2f:d=0.25' % (sec - 0.25), '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(out)], check=True)


def clip_to_mp4(src, title, caption, out):
    """1288x560 の比較クリップを上下に文字を足して 1280x720 にする."""
    t = title.replace("'", "’").replace(':', '\\:')
    c = caption.replace("'", "’").replace(':', '\\:')
    vf = (f"scale=1240:-2,pad={W}:{H}:20:84:color=0x14161a,"
          f"drawtext=fontfile={FONT_B}:text='{t}':x=40:y=24:fontsize=34:fontcolor=white,"
          f"drawtext=fontfile={FONT}:text='{c}':x=40:y={H - 66}:fontsize=28:fontcolor=white,fps={FPS}")
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(src), '-f', 'lavfi', '-i',
                    'anullsrc=r=48000:cl=stereo', '-vf', vf, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-shortest', str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', required=True, help='make_compare_video.py の出力フォルダ')
    ap.add_argument('--figs', required=True, help='plot_*.py の図のフォルダ')
    ap.add_argument('--diagram', required=True, help='システム構成図の PNG')
    ap.add_argument('--repo', default='', help='最後に出すコードの URL')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    C, F = Path(a.clips), Path(a.figs)
    work = Path(tempfile.mkdtemp())
    parts = []

    def add_still(im, sec):
        p = work / f'{len(parts):02d}.mp4'
        still_to_mp4(im, sec, p)
        parts.append(p)

    def add_clip(src, title, caption):
        p = work / f'{len(parts):02d}.mp4'
        clip_to_mp4(src, title, caption, p)
        parts.append(p)

    add_still(card('カメラで見て、毎回違う場所のキューブを拾う',
                   ['CRANE+ V2 × LeRobot ACT（Gazebo シミュレーション）', '',
                    'Physical AI 応用1 最終課題'], sub='VISION PICK-AND-PLACE'), 5)
    add_still(card('課題',
                   ['赤いキューブが毎回ランダムな位置に置かれる',
                    '（ロボットからの距離 0.19–0.27 m、方向 −30°〜+60°）',
                    'つかんで右前のトレイに入れ、初期姿勢に戻る',
                    '方策が見られるのは「上方カメラの画像」と「関節角」だけ',
                    '比較: 関節角だけの ACT（講義デモと同じ入力）と、正解位置を使うお手本'],
                   accent_line='方策が見られるのは「上方カメラの画像」と「関節角」だけ'), 9)
    add_still(figure_frame(a.diagram, '正解位置を知るお手本が自動でデモを作り、方策はカメラと関節角だけで動く',
                           'システム構成'), 9)
    add_still(figure_frame(F / 'data_quality.png', '並行処理で画像が抜けた 29 エピソードを自動で除外し、65 エピソードで学習',
                           'データ品質の検査（94 エピソード）'), 8)
    add_clip(C / 'expert_vs_act_trial0.mp4', 'お手本 と 学習した方策（ACT + カメラ）',
             '学習に使っていない配置。方策はキューブの位置を画像から読み取っている（2 倍速）')
    add_still(figure_frame(F / 'success_rate.png', '同じ 20 通りの未学習配置で評価。カメラあり 90%、関節角のみ 0%',
                           '結果：成功率'), 8)
    add_clip(C / 'compare_trial0.mp4', '同じ配置での比較：カメラあり vs 関節角のみ',
             '関節角だけではキューブの場所が分からず、何もない所をつかむ（2 倍速）')
    add_still(figure_frame(F / 'joint_traces_trial0.png',
                           '関節角のみ（橙）は配置に関係なく 75° へ伸ばし、空振りしてトレイ上で停止',
                           'なぜ失敗するのか'), 10)
    add_clip(C / 'act_success_fail.mp4', 'カメラあり ACT の成功例と失敗例',
             '失敗 2 回はどちらも左前の配置。つかみ損ねてキューブを押しやった（2 倍速）')
    add_still(card('わかったこと',
                   ['検証誤差はほぼ同じ（L1 0.062 と 0.064）なのに、成功率は 90% と 0%',
                    '→ オフラインの誤差では方策は評価できない。閉ループで動かして測る',
                    '推論 約 0.8 秒の遅れも、行動チャンクを 5 ステップずつ実行して吸収',
                    '課題: 試行数、実機への移行、つかみ損ねからの立ち直り']), 11)
    add_still(card('コード', [a.repo or '（GitHub の URL）', '', 'ROS 2 Humble / Ignition Gazebo / LeRobot ACT / Google Colab A100']),
              5)

    lst = work / 'list.txt'
    lst.write_text(''.join(f"file '{p}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(lst),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', str(FPS), '-c:a', 'aac', a.out], check=True)
    dur = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', a.out],
                         capture_output=True, text=True).stdout.strip()
    print(f'saved {a.out}  ({float(dur):.1f} s)')


if __name__ == '__main__':
    main()
