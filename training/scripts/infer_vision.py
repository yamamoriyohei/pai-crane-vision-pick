#!/usr/bin/env python3
"""学習した ACT を Gazebo にデプロイする推論ノード（講義の infer.py を拡張）.

- 入力: /joint_states と /camera/image_small（モデルが画像を使う場合）
- 出力: /crane_plus_arm_controller/joint_trajectory, /crane_plus_gripper_controller/joint_trajectory
- /policy/enable (std_msgs/Bool) が True の間だけ動く。False → True で方策の内部状態をリセット
  （evaluate.py が試行ごとに ON/OFF する）。--always-on で常に動かす（手動デモ用）
- シミュレーション時間で fps 周期に動く。推論時間（ms）を計測して表示する（第3回: まず測る）

  cd ~/pai_training
  pixi run python ~/pai_ws/src/pai-crane-vision-pick/training/scripts/infer_vision.py \
      --model-path /home/ubuntu/checkpoints/train/crane_vision_pick_act
注意: テレオペノードや collect_demos は止めておく（指令がぶつかる）
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from policy_io import JOINTS, build_observation, image_msg_to_rgb, joint_state_to_vec, uses_images


class VisionPolicyNode(Node):
    def __init__(self, args):
        super().__init__('crane_vision_act_infer', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.device = torch.device(args.device)
        self.policy = ACTPolicy.from_pretrained(args.model_path)
        if args.n_action_steps:
            self.policy.config.n_action_steps = args.n_action_steps
        self.policy.to(self.device).eval()
        self.policy.reset()
        torch.set_num_threads(max(1, args.threads))
        self.use_images = uses_images(self.policy.config)
        self.tfs = args.time_from_start
        self.always_on = args.always_on
        cfg = self.policy.config
        self.get_logger().info(
            f'loaded {args.model_path}: inputs={list(cfg.input_features)} chunk={cfg.chunk_size} '
            f'n_action_steps={cfg.n_action_steps} device={self.device}')

        self.arm_pub = self.create_publisher(JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        if self.use_images:
            self.create_subscription(Image, args.image_topic, self.on_image, 2)
        self.create_subscription(Bool, '/policy/enable', self.on_enable, 10)

        self.state = None
        self.rgb, self.rgb_stamp = None, 0.0
        self.enabled, self.last_enable = False, -1e9
        self.latencies = []
        self.ticks = 0
        self.create_timer(1.0 / args.fps, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_js(self, msg):
        v = joint_state_to_vec(msg.name, msg.position)
        if v is not None:
            self.state = v

    def on_image(self, msg):
        self.rgb = image_msg_to_rgb(msg.height, msg.width, msg.step, msg.encoding, msg.data)
        self.rgb_stamp = self.now()

    def on_enable(self, msg):
        self.last_enable = self.now()
        if msg.data and not self.enabled:
            self.policy.reset()                    # 新しい試行: 行動キューを空にする
            self.get_logger().info('policy ENABLED')
        if not msg.data and self.enabled:
            self.get_logger().info('policy disabled')
        self.enabled = bool(msg.data)

    def tick(self):
        active = self.always_on or (self.enabled and self.now() - self.last_enable < 1.0)
        if not active:
            if self.enabled:
                self.enabled = False
                self.get_logger().warn('enable が 1 秒以上届かないので停止しました')
            return
        if self.state is None or (self.use_images and self.rgb is None):
            return
        if self.use_images and self.now() - self.rgb_stamp > 0.5:
            self.get_logger().warn(f'画像が {self.now() - self.rgb_stamp:.2f}s 古い', throttle_duration_sec=2.0)
        obs = build_observation(self.policy.config, self.state, self.rgb, self.device)
        will_infer = len(getattr(self.policy, '_action_queue', [])) == 0
        t0 = time.perf_counter()
        with torch.no_grad():
            action = self.policy.select_action(obs).squeeze(0).cpu().numpy()
        if will_infer:
            self.latencies.append((time.perf_counter() - t0) * 1000)
        self.publish(action)
        self.ticks += 1
        if self.ticks % 50 == 0 and self.latencies:
            lat = np.array(self.latencies[-100:])
            self.get_logger().info(f'推論時間: 平均 {lat.mean():.0f} ms / p95 {np.percentile(lat, 95):.0f} ms  '
                                   f'action={np.round(action, 2)}')

    def publish(self, a):
        sec = int(self.tfs)
        nsec = int((self.tfs - sec) * 1e9)
        jt = JointTrajectory()
        jt.joint_names = JOINTS[:4]
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in a[:4]]
        pt.time_from_start.sec, pt.time_from_start.nanosec = sec, nsec
        jt.points.append(pt)
        self.arm_pub.publish(jt)
        jg = JointTrajectory()
        jg.joint_names = [JOINTS[4]]
        pg = JointTrajectoryPoint()
        pg.positions = [float(a[4])]
        pg.time_from_start.sec, pg.time_from_start.nanosec = sec, nsec
        jg.points.append(pg)
        self.grip_pub.publish(jg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--fps', type=float, default=10.0, help='学習データの fps と同じにする')
    ap.add_argument('--n-action-steps', type=int, default=0, help='0 なら学習時の値。小さいほど反応が速いが推論回数が増える')
    ap.add_argument('--time-from-start', type=float, default=0.15, help='各指令の到達時間 [s]')
    ap.add_argument('--image-topic', default='/camera/image_small')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--always-on', action='store_true', help='/policy/enable を待たずに動かす')
    args = ap.parse_args()
    rclpy.init()
    node = VisionPolicyNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
