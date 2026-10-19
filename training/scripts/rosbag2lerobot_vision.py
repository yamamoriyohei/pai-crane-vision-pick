#!/usr/bin/env python3
"""rosbag（1 エピソード = 1 bag）→ LeRobotDataset 変換（カメラ画像つき）.

講義の pai_training/scripts/rosbag2lerobot.py を拡張したもの。主な変更点:
  1. カメラ画像 (/camera/image_small) を observation.images.top として追加（mp4 で保存）
  2. 関節は「名前」で並べ替える（/joint_states の並び順に依存しない）
  3. 時刻は rosbag の記録時刻（collect_demos が --use-sim-time で記録 → シミュレーション時間）を使い、
     fps 間隔で「直前の値を保持」してサンプリング（講義と同じ sample-and-hold）
  4. 変換前にデータ検証（画像の周波数・抜け・長さ）を行い、基準外のエピソードは理由を出して除外
  5. エピソードごとのキューブ位置などを meta/vision_pick_episodes.jsonl に保存（分析用）

  cd ~/pai_training
  pixi run python ~/pai_ws/src/pai-crane-vision-pick/training/scripts/rosbag2lerobot_vision.py \
      --config ~/pai_ws/src/pai-crane-vision-pick/training/configs/vision_conversion.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

ARM_TOPIC = '/crane_plus_arm_controller/joint_trajectory'
GRIPPER_TOPIC = '/crane_plus_gripper_controller/joint_trajectory'
JOINTS = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4',
          'crane_plus_joint_hand']
JOINT_SHORT = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']


# ---------------------------------------------------------------- reading
def _deserializer():
    try:   # rosbags >= 0.10 の推奨 API
        from rosbags.typesys import Stores, get_typestore
        ts = get_typestore(Stores.ROS2_HUMBLE)
        return ts.deserialize_cdr
    except Exception:  # noqa: BLE001
        from rosbags.serde import deserialize_cdr
        return deserialize_cdr


def image_to_numpy(msg):
    """sensor_msgs/Image → (H, W, 3) uint8 RGB."""
    data = np.asarray(msg.data, dtype=np.uint8)
    enc = msg.encoding
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}[enc]
    img = data.reshape(msg.height, msg.step)[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    if ch == 1:
        img = np.repeat(img, 3, axis=2)
    img = img[:, :, :3]
    if enc.startswith('bgr'):
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def load_episode(bag_dir: Path, image_topic: str):
    """bag を読み、トピックごとに (times[ns], values) を返す."""
    from rosbags.rosbag2 import Reader
    deser = _deserializer()
    js_t, js_v, img_t, img_v, arm_t, arm_v, grip_t, grip_v = ([] for _ in range(8))
    with Reader(bag_dir) as reader:
        conns = [c for c in reader.connections
                 if c.topic in ('/joint_states', image_topic, ARM_TOPIC, GRIPPER_TOPIC)]
        for conn, t, raw in reader.messages(connections=conns):
            msg = deser(raw, conn.msgtype)
            if conn.topic == '/joint_states':
                m = dict(zip(msg.name, msg.position))
                if all(j in m for j in JOINTS):
                    js_t.append(t)
                    js_v.append([m[j] for j in JOINTS])
            elif conn.topic == image_topic:
                img_t.append(t)
                img_v.append(image_to_numpy(msg))
            elif conn.topic == ARM_TOPIC and len(msg.points):
                m = dict(zip(msg.joint_names, msg.points[0].positions))
                if all(j in m for j in JOINTS[:4]):
                    arm_t.append(t)
                    arm_v.append([m[j] for j in JOINTS[:4]])
            elif conn.topic == GRIPPER_TOPIC and len(msg.points) and len(msg.points[0].positions):
                grip_t.append(t)
                grip_v.append([float(msg.points[0].positions[0])])
    as_np = lambda t, v, dt=np.float32: (np.asarray(t, dtype=np.int64), np.asarray(v, dtype=dt))  # noqa: E731
    return {
        'joint': as_np(js_t, js_v),
        'image': (np.asarray(img_t, dtype=np.int64), img_v),
        'arm': as_np(arm_t, arm_v),
        'gripper': as_np(grip_t, grip_v),
    }


# ------------------------------------------------------------- validation
def validate(ep, fps, vcfg):
    """データ検証. (問題のリスト, 統計) を返す."""
    issues, stats = [], {}
    for key in ('joint', 'image', 'arm', 'gripper'):
        if len(ep[key][0]) < 2:
            issues.append(f'{key}: メッセージが {len(ep[key][0])} 個しかない')
    if issues:
        return issues, stats
    img_t = ep['image'][0] * 1e-9
    js_t = ep['joint'][0] * 1e-9
    start = max(ep[k][0][0] for k in ('joint', 'image', 'arm', 'gripper')) * 1e-9
    end = min(ep[k][0][-1] for k in ('joint', 'image', 'arm', 'gripper')) * 1e-9
    stats['duration_s'] = round(float(end - start), 3)
    stats['image_hz'] = round(float((len(img_t) - 1) / max(img_t[-1] - img_t[0], 1e-9)), 2)
    stats['max_image_gap_s'] = round(float(np.diff(img_t).max()), 3)
    stats['max_joint_gap_s'] = round(float(np.diff(js_t).max()), 3)
    stats['n_frames'] = int(np.floor((end - start) * fps)) + 1
    if stats['image_hz'] < vcfg.get('min_image_hz', 0):
        issues.append(f"画像の周波数が低い ({stats['image_hz']} Hz)")
    if stats['max_image_gap_s'] > vcfg.get('max_image_gap_s', 1e9):
        issues.append(f"画像が {stats['max_image_gap_s']} s 抜けている")
    if stats['max_joint_gap_s'] > vcfg.get('max_joint_gap_s', 1e9):
        issues.append(f"関節状態が {stats['max_joint_gap_s']} s 抜けている")
    if stats['n_frames'] < vcfg.get('min_frames', 0):
        issues.append(f"短すぎる ({stats['n_frames']} フレーム)")
    # どれかのトピックだけ途中で途切れていないか（末尾が他より 0.5 s 以上早く終わる）
    last = {k: ep[k][0][-1] * 1e-9 for k in ('joint', 'image', 'arm', 'gripper')}
    for k, v in last.items():
        if max(last.values()) - v > 0.5:
            issues.append(f'{k} が {max(last.values()) - v:.1f} s 早く途切れている')
    sizes = {im.shape for im in ep['image'][1]}
    if len(sizes) != 1:
        issues.append(f'画像サイズが途中で変わる: {sizes}')
    return issues, stats


# ---------------------------------------------------------------- resample
def hold_indices(times, targets):
    """各 target 時刻「以前で最も新しい」サンプルの番号（sample-and-hold）."""
    idx = np.searchsorted(times, targets, side='right') - 1
    return np.clip(idx, 0, len(times) - 1)


def resample(ep, fps):
    start = max(ep[k][0][0] for k in ('joint', 'image', 'arm', 'gripper'))
    end = min(ep[k][0][-1] for k in ('joint', 'image', 'arm', 'gripper'))
    n = int(np.floor((end - start) * 1e-9 * fps)) + 1
    targets = start + (np.arange(n) * (1e9 / fps)).astype(np.int64)
    state = ep['joint'][1][hold_indices(ep['joint'][0], targets)]
    action = np.hstack([ep['arm'][1][hold_indices(ep['arm'][0], targets)],
                        ep['gripper'][1][hold_indices(ep['gripper'][0], targets)]])
    img_idx = hold_indices(ep['image'][0], targets)
    # 画像の「古さ」（その時刻から見て何秒前に撮った画像か）も記録しておく（第3回: 時刻の照合）
    img_age = (targets - ep['image'][0][img_idx]) * 1e-9
    return state.astype(np.float32), action.astype(np.float32), img_idx, img_age


def make_features(h, w, image_key):
    vec = lambda: {'dtype': 'float32', 'shape': (5,), 'names': list(JOINT_SHORT)}  # noqa: E731
    return {
        'observation.state': vec(),
        'observation.environment_state': vec(),     # 比較実験（画像なし ACT）用。中身は state と同じ
        image_key: {'dtype': 'video', 'shape': (h, w, 3), 'names': ['height', 'width', 'channels']},
        'action': vec(),
    }


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--overwrite', action='store_true', help='出力先が既にあれば消して作り直す')
    ap.add_argument('--max-episodes', type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fps = int(cfg.get('fps', 10))
    vcfg = cfg.get('validation', {})
    image_key = cfg.get('image_key', 'observation.images.top')

    bag_root = Path(cfg['bag_dir']).expanduser()
    episodes = sorted(p.parent for p in bag_root.rglob('metadata.yaml'))
    # collect_demos のログ（<bag_dir>_log.csv）があれば、失敗したエピソードを除外する
    log_csv = bag_root.parent / (bag_root.name + '_log.csv')
    if log_csv.exists():
        import csv
        failed = {f"ep_{int(r['episode']):04d}" for r in csv.DictReader(open(log_csv)) if r['success'] == '0'}
        skipped = [d.name for d in episodes if d.name in failed]
        episodes = [d for d in episodes if d.name not in failed]
        if skipped:
            print(f'収集ログで失敗だったため除外: {", ".join(skipped)}')
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    if not episodes:
        raise SystemExit(f'bag が見つかりません: {bag_root}')
    print(f'{len(episodes)} 個の bag を検出: {bag_root}')

    # 1 周目: 読み込み + 検証（メモリ節約のため画像は保持しない）
    good, report = [], []
    img_shape = None
    for d in episodes:
        ep = None
        try:
            ep = load_episode(d, cfg['image_topic'])
            issues, stats = validate(ep, fps, vcfg)
            if not issues:
                img_shape = img_shape or ep['image'][1][0].shape
        except Exception as e:  # noqa: BLE001
            issues, stats = [f'読み込みエラー: {e}'], {}
        del ep
        report.append({'bag': d.name, 'ok': not issues, 'issues': issues, **stats})
        mark = 'OK ' if not issues else 'NG '
        print(f'  {mark} {d.name}: {stats}  {"; ".join(issues)}')
        if not issues:
            good.append((d, stats['n_frames']))
    if not good:
        raise SystemExit('検証を通ったエピソードがありません')
    print(f'\n検証 OK: {len(good)}/{len(episodes)} エピソード')

    h, w = img_shape[:2]
    features = make_features(h, w, image_key)
    out = Path(cfg['output_dir']).expanduser()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f'{out} は既にあります（--overwrite で作り直し）')
        shutil.rmtree(out)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset.create(repo_id=cfg['repo_id'], fps=fps, root=out,
                                    robot_type=cfg.get('robot_type', 'crane_plus'),
                                    features=features, use_videos=True)
    task = cfg.get('task_name', 'pick and place')
    max_len = max(n for _, n in good)
    side = []
    for ep_index, (d, _) in enumerate(good):
        ep = load_episode(d, cfg['image_topic'])
        state, action, img_idx, img_age = resample(ep, fps)
        n = len(state)
        total = max_len if cfg.get('pad_to_longest') else n
        for i in range(total):
            j = min(i, n - 1)
            dataset.add_frame({
                'observation.state': state[j],
                'observation.environment_state': state[j],
                image_key: ep['image'][1][img_idx[j]],
                'action': action[j],
                'task': task,
            })
        dataset.save_episode()
        meta_path = d / 'episode_meta.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        side.append({'episode_index': ep_index, 'bag': d.name, 'length': total,
                     'image_age_mean_s': round(float(img_age.mean()), 4),
                     'image_age_max_s': round(float(img_age.max()), 4),
                     'cube_initial': meta.get('cube_initial'), 'blue': meta.get('blue')})
        print(f'  saved episode {ep_index} ({d.name}): {total} frames, '
              f'画像の古さ 平均 {img_age.mean() * 1000:.0f} ms / 最大 {img_age.max() * 1000:.0f} ms')

    with open(out / 'meta' / 'vision_pick_episodes.jsonl', 'w') as f:
        for row in side:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with open(out / 'meta' / 'validation_report.json', 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\nDataset saved to: {out}  ({len(good)} episodes, {fps} fps, image {w}x{h})')


if __name__ == '__main__':
    main()
