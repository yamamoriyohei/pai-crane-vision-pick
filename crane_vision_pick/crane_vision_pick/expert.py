"""スクリプト化されたお手本（エキスパート）の軌道生成.

シミュレータから得た「正解のキューブ位置」（特権情報）を使って、
ピック＆プレースの関節軌道を 20Hz の指令列として作る。
学習する方策はこの特権情報を見られず、カメラ画像と関節角だけから同じ動きを再現する必要がある。

軌道は「円柱座標 (r, θ, z) で直線補間 → 各点で IK」で作るので、
途中で手先がテーブルに潜り込まない。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .kinematics import (GRIPPER_DEFAULT, GRIPPER_OPEN, HOME_Q,
                         ik_side_grasp)

CONTROL_HZ = 20.0


@dataclass
class TaskConfig:
    # キューブを置く範囲（base_link 基準の円柱座標）
    r_range: tuple = (0.19, 0.27)
    theta_range_deg: tuple = (-30.0, 60.0)
    yaw_noise_deg: float = 10.0
    # トレイ（置き場所）の中心
    tray_r: float = 0.21
    tray_theta_deg: float = -65.0
    tray_half_size: float = 0.045     # 成功判定に使うトレイ内側の半幅 [m]
    # 高さ（把持中心の z, base_link 基準）
    z_grasp: float = 0.04
    z_carry: float = 0.10
    z_release: float = 0.075
    # 把持の調整（sim_check --probe で実機ならぬ実シミュレータに合わせて決める）
    grip_offset: float = 0.13         # link4 原点 → 把持中心 [m]（指の付け根側へ寄せるなら小さく）
    # 把持中心をキューブ中心から横（ロボットから見て左 = 固定指側）へずらす [m]
    # 固定指の内面は中心線から +37mm。キューブ(半幅25mm)を中心で挟もうとすると閉じるときに 12mm 押してしまう
    # （Gazebo で実測: y が 0.060 → 0.072）。8mm 寄せておき、指が降りるときの余裕 4mm を残す
    lateral_offset: float = 0.008
    gripper_open: float = GRIPPER_OPEN
    # 閉じる目標角度。位置制御なので「目標 − 実際の角度」が握る力になる。
    # Gazebo で比較: 10deg → 成功 2/10、20deg → 成功 8/8 以上（同じ配置・シード）
    gripper_close: float = math.radians(20.0)
    # 各フェーズの所要時間 [s]（決定的にして軌道の多峰性を避ける）
    durations: dict = field(default_factory=lambda: {
        'reach': 2.2, 'descend': 1.4, 'close': 0.8, 'lift': 1.4,
        'transfer': 2.4, 'lower': 0.6, 'open': 0.8, 'retreat': 2.0,   # グリッパは遅いので open は長め
    })

    @property
    def tray_xy(self):
        t = math.radians(self.tray_theta_deg)
        return self.tray_r * math.cos(t), self.tray_r * math.sin(t)


def add_grasp_args(ap):
    """把持の調整用オプション（sim_check / collect_demos / evaluate 共通）."""
    ap.add_argument('--lateral', type=float, default=None, help='把持中心の横ずらし [m]（+ = 固定指側）')
    ap.add_argument('--grip-offset', type=float, default=None, help='link4 → 把持中心 [m]（既定 0.13）')
    ap.add_argument('--z-grasp', type=float, default=None, help='把持の高さ [m]（既定 0.04）')
    ap.add_argument('--close-deg', type=float, default=None, help='グリッパを閉じる角度 [deg]（既定 20）')


def apply_grasp_args(cfg, args):
    for k, attr in (('lateral', 'lateral_offset'), ('grip_offset', 'grip_offset'), ('z_grasp', 'z_grasp')):
        if getattr(args, k, None) is not None:
            setattr(cfg, attr, getattr(args, k))
    if getattr(args, 'close_deg', None) is not None:
        cfg.gripper_close = math.radians(args.close_deg)
    return cfg


def describe_grasp(cfg):
    return (f'grip_offset={cfg.grip_offset:.3f} lateral={cfg.lateral_offset:+.3f} z_grasp={cfg.z_grasp:.3f} '
            f'close={math.degrees(cfg.gripper_close):.0f}deg')


def sample_cube_pose(cfg: TaskConfig, rng: np.random.Generator):
    """キューブの (x, y, yaw) をランダムに決める. yaw は把持方向（半径方向）に揃えて少しノイズ."""
    r = rng.uniform(*cfg.r_range)
    th = math.radians(rng.uniform(*cfg.theta_range_deg))
    yaw = th + math.radians(rng.uniform(-cfg.yaw_noise_deg, cfg.yaw_noise_deg))
    return r * math.cos(th), r * math.sin(th), yaw


def _min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def _steps(duration):
    return max(1, int(round(duration * CONTROL_HZ)))


class PlanError(RuntimeError):
    pass


def _cyl(x, y):
    return math.hypot(x, y), math.atan2(y, x)


def _cart_segment(p0, p1, duration, q_prev, grip_offset=None):
    """円柱座標 (r, θ, z) で補間して各点の IK を解く."""
    from .kinematics import GRIP_OFFSET
    grip_offset = GRIP_OFFSET if grip_offset is None else grip_offset
    (r0, t0, z0), (r1, t1, z1) = p0, p1
    out = []
    q = q_prev
    for k in range(1, _steps(duration) + 1):
        s = _min_jerk(k / _steps(duration))
        r, t, z = r0 + (r1 - r0) * s, t0 + (t1 - t0) * s, z0 + (z1 - z0) * s
        res = ik_side_grasp(r * math.cos(t), r * math.sin(t), z, prefer=q, grip_offset=grip_offset)
        if not res.ok:
            raise PlanError(f'IK failed at r={r:.3f} th={math.degrees(t):.1f} z={z:.3f}: {res.reason}')
        q = res.q
        out.append(q)
    return out


def _joint_segment(q0, q1, duration):
    q0, q1 = np.asarray(q0), np.asarray(q1)
    n = _steps(duration)
    return [q0 + (q1 - q0) * _min_jerk(k / n) for k in range(1, n + 1)]


def _grip_ramp(g0, g1, n):
    return [g0 + (g1 - g0) * _min_jerk(k / n) for k in range(1, n + 1)]


def plan_pick_and_place(cube_xy, cfg: TaskConfig, q_start=HOME_Q, return_phases=False):
    """(N, 5) の指令列 [j1, j2, j3, j4, hand] を返す（20Hz）.

    return_phases=True なら (cmds, [(フェーズ名, 終了インデックス), ...]) を返す（把持テスト用）。
    """
    d = cfg.durations
    go = cfg.grip_offset
    g_open, g_close = cfg.gripper_open, cfg.gripper_close
    # 把持中心をキューブ中心から横にずらす（ロボットから見て左 = +lateral）
    th = math.atan2(cube_xy[1], cube_xy[0])
    gx = cube_xy[0] - cfg.lateral_offset * math.sin(th)
    gy = cube_xy[1] + cfg.lateral_offset * math.cos(th)
    rc, tc = _cyl(gx, gy)
    rt, tt = _cyl(*cfg.tray_xy)
    above_cube = (rc, tc, cfg.z_carry)
    at_cube = (rc, tc, cfg.z_grasp)
    above_tray = (rt, tt, cfg.z_carry)
    at_tray = (rt, tt, cfg.z_release)

    res = ik_side_grasp(rc * math.cos(tc), rc * math.sin(tc), cfg.z_carry, prefer=q_start, grip_offset=go)
    if not res.ok:
        raise PlanError(f'cannot reach above cube: {res.reason}')

    arm, grip, phases = [], [], []

    def add(name, qs, g):
        arm.extend(qs)
        grip.extend(g if isinstance(g, list) else [g] * len(qs))
        phases.append((name, len(arm)))

    # 1. home → キューブの上（関節空間で補間）。同時にグリッパを開く
    seg = _joint_segment(q_start, res.q, d['reach'])
    add('reach', seg, _grip_ramp(GRIPPER_DEFAULT, g_open, len(seg)))
    # 2. 真下に降りる
    add('descend', _cart_segment(above_cube, at_cube, d['descend'], arm[-1], go), g_open)
    # 3. 閉じる（腕は静止）
    n = _steps(d['close'])
    add('close', [arm[-1]] * n, _grip_ramp(g_open, g_close, n))
    # 4. 持ち上げ
    add('lift', _cart_segment(at_cube, above_cube, d['lift'], arm[-1], go), g_close)
    # 5. トレイの上へ
    add('transfer', _cart_segment(above_cube, above_tray, d['transfer'], arm[-1], go), g_close)
    # 6. 少し下げる
    add('lower', _cart_segment(above_tray, at_tray, d['lower'], arm[-1], go), g_close)
    # 7. 離す
    n = _steps(d['open'])
    add('open', [arm[-1]] * n, _grip_ramp(g_close, g_open, n))
    # 8. home に戻る
    seg = _joint_segment(arm[-1], q_start, d['retreat'])
    add('retreat', seg, _grip_ramp(g_open, GRIPPER_DEFAULT, len(seg)))

    cmds = np.hstack([np.asarray(arm), np.asarray(grip).reshape(-1, 1)])
    return (cmds, phases) if return_phases else cmds
