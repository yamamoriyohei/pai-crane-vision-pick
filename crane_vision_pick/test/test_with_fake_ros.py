"""ROS / Gazebo なしで collect_demos・evaluate・sim_check の制御の流れを通すテスト.

rclpy などを最小限の偽物に差し替え、
  - 関節は指令どおりに即座に動く
  - グリッパが閉じたとき把持中心がキューブに近ければキューブがくっつき、開くと真下に落ちる
という簡易な世界で、お手本がキューブをトレイに入れられるか（成功判定まで）を確かめる。
"""
import importlib
import math
import sys
import types

import numpy as np
import pytest

# ----------------------------------------------------------------- fake ROS
SIM = {'t': 0.0, 'q': np.array([0.0, 0.0, 0.0, 0.0, 0.0]), 'cube': [0.22, 0.05, 0.025], 'attached': False,
       'subs': 0}


class _Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Duration:
    def __init__(self):
        self.sec = 0
        self.nanosec = 0


class JointTrajectoryPoint:
    def __init__(self):
        self.positions = []
        self.time_from_start = _Duration()


class JointTrajectory:
    def __init__(self):
        self.joint_names = []
        self.points = []


class _Clock:
    def now(self):
        return types.SimpleNamespace(nanoseconds=int(SIM['t'] * 1e9))


class _Pub:
    def __init__(self, topic):
        self.topic = topic

    def publish(self, msg):
        if 'arm_controller' in self.topic:
            names = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']
            m = dict(zip(msg.joint_names, msg.points[0].positions))
            SIM['q'][:4] = [m[n] for n in names]
        elif 'gripper_controller' in self.topic:
            SIM['q'][4] = msg.points[0].positions[0]
            _update_cube()


def _update_cube():
    from crane_vision_pick.kinematics import fk
    tip = fk(SIM['q'][:4])[0]
    closed = SIM['q'][4] > 0.0
    if SIM['attached'] and not closed:
        SIM['attached'] = False
        SIM['cube'] = [tip[0], tip[1], 0.025]
    elif not SIM['attached'] and closed and np.linalg.norm(tip - np.array(SIM['cube'])) < 0.03:
        SIM['attached'] = True
    if SIM['attached']:
        SIM['cube'] = [tip[0], tip[1], tip[2] - 0.015]


class Node:
    def __init__(self, name, parameter_overrides=None):
        self._subs = []

    def create_publisher(self, typ, topic, qos):
        return _Pub(topic)

    def create_subscription(self, typ, topic, cb, qos):
        self._subs.append((topic, cb))

    def create_timer(self, *a):
        pass

    def get_clock(self):
        return _Clock()

    def count_subscribers(self, topic):
        SIM['subs'] += 1
        return SIM['subs']

    def get_logger(self):
        return types.SimpleNamespace(info=print, warn=print, error=print)

    def destroy_node(self):
        pass


def _spin_once(node, timeout_sec=0.0):
    SIM['t'] += 0.01
    _update_cube()
    for topic, cb in node._subs:
        if topic == '/joint_states':
            cb(_Msg(name=['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4',
                          'crane_plus_joint3', 'crane_plus_joint_hand'],
                    position=[SIM['q'][i] for i in (0, 1, 3, 2, 4)]))


def _install_fakes(monkeypatch):
    rclpy = types.ModuleType('rclpy')
    rclpy.init = lambda *a, **k: None
    rclpy.try_shutdown = lambda *a, **k: None
    rclpy.spin_once = _spin_once
    node_mod = types.ModuleType('rclpy.node')
    node_mod.Node = Node
    param_mod = types.ModuleType('rclpy.parameter')
    param_mod.Parameter = lambda *a, **k: None
    qos_mod = types.ModuleType('rclpy.qos')
    qos_mod.QoSProfile = lambda **k: None
    qos_mod.ReliabilityPolicy = types.SimpleNamespace(RELIABLE=1)
    sensor = types.ModuleType('sensor_msgs.msg')
    sensor.JointState = _Msg
    sensor.Image = _Msg
    traj = types.ModuleType('trajectory_msgs.msg')
    traj.JointTrajectory = JointTrajectory
    traj.JointTrajectoryPoint = JointTrajectoryPoint
    std = types.ModuleType('std_msgs.msg')
    std.Bool = _Msg
    std.String = _Msg
    for name, mod in {'rclpy': rclpy, 'rclpy.node': node_mod, 'rclpy.parameter': param_mod,
                      'rclpy.qos': qos_mod, 'sensor_msgs': types.ModuleType('sensor_msgs'),
                      'sensor_msgs.msg': sensor, 'trajectory_msgs': types.ModuleType('trajectory_msgs'),
                      'trajectory_msgs.msg': traj, 'std_msgs': types.ModuleType('std_msgs'),
                      'std_msgs.msg': std}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for m in [k for k in sys.modules if k.startswith('crane_vision_pick.') and k.split('.')[-1] in
              ('ros_io', 'collect_demos', 'evaluate', 'sim_check')]:
        monkeypatch.delitem(sys.modules, m)

    import crane_vision_pick.sim_utils as su

    def set_pose(name, x, y, z=None, yaw=0.0, world='default', frame='base'):
        if name == 'red_cube':
            SIM['cube'] = [x, y, 0.025]
            SIM['attached'] = False

    def get_pose(name, world='default', frame='base'):
        c = SIM['cube']
        return c[0], c[1], c[2], 0.0
    monkeypatch.setattr(su, 'set_model_pose', set_pose)
    monkeypatch.setattr(su, 'get_model_pose', get_pose)


@pytest.fixture
def fake(monkeypatch):
    _install_fakes(monkeypatch)
    SIM.update(t=0.0, q=np.zeros(5), cube=[0.22, 0.05, 0.025], attached=False, subs=0)
    yield


def test_collect_dry_run(fake, tmp_path):
    cd = importlib.import_module('crane_vision_pick.collect_demos')
    cd.main(['--dry-run', '--episodes', '3', '--out', str(tmp_path / 'bags')])
    log = (tmp_path / 'bags_log.csv').read_text().strip().splitlines()
    assert len(log) == 4
    assert all(line.split(',')[1] == '1' for line in log[1:]), log     # 3 回とも成功


def test_evaluate_expert(fake, tmp_path):
    ev = importlib.import_module('crane_vision_pick.evaluate')
    out = tmp_path / 'eval.csv'
    ev.main(['--mode', 'expert', '--trials', '3', '--out', str(out), '--label', 'expert'])
    rows = out.read_text().strip().splitlines()
    assert len(rows) == 4 and all(r.split(',')[2] == '1' for r in rows[1:]), rows


def test_in_tray_geometry():
    from crane_vision_pick.collect_demos import in_tray
    from crane_vision_pick.expert import TaskConfig
    cfg = TaskConfig()
    tx, ty = cfg.tray_xy
    assert in_tray((tx, ty, 0.025), cfg)
    assert in_tray((tx + 0.03 * math.cos(-1.13), ty + 0.03 * math.sin(-1.13), 0.025), cfg)
    assert not in_tray((tx + 0.08, ty, 0.025), cfg)
    assert not in_tray((tx, ty, 0.12), cfg)            # まだ持ち上げたまま


def test_sim_check_probe(fake, capsys):
    sc = importlib.import_module('crane_vision_pick.sim_check')
    sc.main(['--probe', '--skip-camera', '--lateral', '0.01', '--repeat', '2'])
    out = capsys.readouterr().out
    assert 'descend' in out and 'lift' in out and '成功 2/2' in out, out
