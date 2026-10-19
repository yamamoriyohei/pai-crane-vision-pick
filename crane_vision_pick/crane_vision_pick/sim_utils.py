"""Ignition Gazebo (Fortress) をコマンドラインツール `ign` 経由で操作するユーティリティ.

- set_model_pose : /world/<world>/set_pose サービスでモデルを瞬間移動（UserCommands プラグインが必要）
- get_model_pose : /world/<world>/dynamic_pose/info トピックを 1 回だけ読んでモデルの位置を得る

ROS 側に余計なブリッジを足さずに済むよう、ign CLI を subprocess で呼んでいる。
"""
from __future__ import annotations

import math
import re
import subprocess
import time

WORLD = 'default'
# base_link（ロボット土台）のワールド座標。crane_plus_with_table.launch.py の spawn 引数 -z 1.015
BASE_IN_WORLD = (0.0, 0.0, 1.015)
CUBE_HALF = 0.025


class SimError(RuntimeError):
    pass


def _run(cmd, timeout):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise SimError(f'timeout: {" ".join(cmd)}') from e
    if out.returncode != 0:
        raise SimError(f'failed ({out.returncode}): {" ".join(cmd)}\n{out.stderr}')
    return out.stdout


def set_model_pose(name, x, y, z=None, yaw=0.0, world=WORLD, frame='base', retries=4):
    """モデルを置く. frame='base' なら base_link 基準の座標で指定（z 省略でテーブル上に置く）."""
    if frame == 'base':
        bx, by, bz = BASE_IN_WORLD
        x, y = x + bx, y + by
        z = bz + CUBE_HALF + 0.003 if z is None else z + bz
    qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
    req = (f'name: "{name}", position: {{x: {x:.5f}, y: {y:.5f}, z: {z:.5f}}}, '
           f'orientation: {{x: 0, y: 0, z: {qz:.6f}, w: {qw:.6f}}}')
    # 重い負荷のときは応答が空（サービスのタイムアウト）になることがあるので数回やり直す
    out = ''
    for attempt in range(retries):
        try:
            out = _run(['ign', 'service', '-s', f'/world/{world}/set_pose',
                        '--reqtype', 'ignition.msgs.Pose', '--reptype', 'ignition.msgs.Boolean',
                        '--timeout', '5000', '--req', req], timeout=15)
        except SimError as e:
            out = str(e)
        if 'true' in out:
            return
        time.sleep(1.0 + attempt)
    raise SimError(f'set_pose returned: {out.strip()!r} ({retries} attempts)')


# ---------------------------------------------------------------------------
# protobuf テキスト形式の簡易パーサ（ign topic -e の出力用）
# 値が 0 のフィールドは出力に現れないので、欠けていたら 0 とみなす。
_TOKEN = re.compile(r'\s*(?:(\w+)\s*\{|(\})|(\w+)\s*:\s*("(?:[^"\\]|\\.)*"|[^\s{}]+))')


def parse_pb_text(text):
    root = {}
    stack = [root]
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == '':
                break
            pos += 1
            continue
        pos = m.end()
        if m.group(1):                      # "name {"
            child = {}
            stack[-1].setdefault(m.group(1), []).append(child)
            stack.append(child)
        elif m.group(2):                    # "}"
            if len(stack) > 1:
                stack.pop()
        else:                               # "key: value"
            k, v = m.group(3), m.group(4)
            if v.startswith('"'):
                v = v[1:-1]
            else:
                try:
                    v = float(v)
                except ValueError:
                    pass
            stack[-1].setdefault(k, []).append(v)
    return root


def poses_from_pb_text(text):
    """{name: (x, y, z, yaw)}（ワールド座標）."""
    msg = parse_pb_text(text)
    result = {}
    for p in msg.get('pose', []):
        name = p.get('name', [''])[0]
        pos = (p.get('position') or [{}])[0]
        ori = (p.get('orientation') or [{}])[0]
        x, y, z = (float(pos.get(k, [0.0])[0]) for k in 'xyz')
        qx, qy, qz = (float(ori.get(k, [0.0])[0]) for k in 'xyz')
        qw = float(ori.get('w', [1.0 if not ori else 0.0])[0])
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        if name and name not in result:     # 最初に出てくる＝モデル自身（後ろはリンク）
            result[name] = (x, y, z, yaw)
    return result


def get_model_pose(name, world=WORLD, frame='base'):
    """モデルの (x, y, z, yaw). frame='base' なら base_link 基準."""
    last_err = None
    for topic in (f'/world/{world}/dynamic_pose/info', f'/world/{world}/pose/info'):
        try:
            text = _run(['ign', 'topic', '-e', '-n', '1', '-t', topic], timeout=10)
        except SimError as e:
            last_err = e
            continue
        poses = poses_from_pb_text(text)
        if name in poses:
            x, y, z, yaw = poses[name]
            if frame == 'base':
                bx, by, bz = BASE_IN_WORLD
                x, y, z = x - bx, y - by, z - bz
            return x, y, z, yaw
    raise SimError(f'model "{name}" not found in pose topics ({last_err})')
