"""ロボットへの指令・状態取得・シミュレーション時間での待機をまとめたノード.

指令の出し方は講義の physicalai_teleop / pai_training と同じ:
  /crane_plus_arm_controller/joint_trajectory と /crane_plus_gripper_controller/joint_trajectory に
  1 点だけの JointTrajectory（time_from_start = 0.1 s）を 20Hz で出し続ける。
こうしておくと rosbag に記録された指令がそのまま「行動（action）」になる。
"""
from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .kinematics import ALL_JOINTS, ARM_JOINTS, GRIPPER_DEFAULT, GRIPPER_JOINT, HOME_Q

ARM_TOPIC = '/crane_plus_arm_controller/joint_trajectory'
GRIPPER_TOPIC = '/crane_plus_gripper_controller/joint_trajectory'
CONTROL_DT = 0.05   # 20Hz


class ArmIO(Node):
    def __init__(self, name):
        super().__init__(name, parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.arm_pub = self.create_publisher(JointTrajectory, ARM_TOPIC, 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, GRIPPER_TOPIC, 10)
        self.joint_pos = None
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)

    # ------------------------------------------------------------------ state
    def _on_js(self, msg):
        m = dict(zip(msg.name, msg.position))
        if all(j in m for j in ALL_JOINTS):
            self.joint_pos = np.array([m[j] for j in ALL_JOINTS])

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_for(self, wall_sec):
        end = time.monotonic() + wall_sec
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_for_clock(self, timeout=60.0):
        end = time.monotonic() + timeout
        while self.now() <= 0.0 or self.joint_pos is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() > end:
                raise RuntimeError('/clock または /joint_states が来ません（Gazebo は起動していますか？）')

    def sleep_sim(self, sec):
        """シミュレーション時間で sec 秒待つ（Mac で Gazebo が遅くても動作のタイミングが崩れない）."""
        t_end = self.now() + sec
        wall_deadline = time.monotonic() + max(30.0, sec * 50)
        while self.now() < t_end:
            rclpy.spin_once(self, timeout_sec=0.005)
            if time.monotonic() > wall_deadline:
                raise RuntimeError('シミュレーション時間が進みません（Gazebo が一時停止していませんか？）')

    # --------------------------------------------------------------- command
    def send(self, cmd5):
        """cmd5 = [j1, j2, j3, j4, hand]（関節名つきで送るので順番の取り違えが起きない）."""
        jt = JointTrajectory()
        jt.joint_names = list(ARM_JOINTS)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in cmd5[:4]]
        pt.time_from_start.nanosec = int(0.1 * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)
        jg = JointTrajectory()
        jg.joint_names = [GRIPPER_JOINT]
        pg = JointTrajectoryPoint()
        pg.positions = [float(cmd5[4])]
        pg.time_from_start.nanosec = int(0.1 * 1e9)
        jg.points.append(pg)
        self.gripper_pub.publish(jg)

    def stream(self, cmds, dt=CONTROL_DT):
        """(N,5) の指令列をシミュレーション時間 dt ごとに送る."""
        t_next = self.now()
        for c in cmds:
            self.send(c)
            t_next += dt
            while self.now() < t_next:
                rclpy.spin_once(self, timeout_sec=0.002)

    def go_home(self, duration=2.0, gripper=GRIPPER_DEFAULT):
        """現在の姿勢から HOME へ関節空間で滑らかに戻す."""
        self.spin_for(0.2)
        start = self.joint_pos.copy() if self.joint_pos is not None else np.r_[HOME_Q, gripper]
        goal = np.r_[HOME_Q, gripper]
        n = max(1, int(duration / CONTROL_DT))
        s = np.linspace(0, 1, n + 1)[1:]
        s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        cmds = start[None] + (goal - start)[None] * s[:, None]
        self.stream(cmds)
        self.stream([goal] * 10)   # 少し保持して落ち着かせる
