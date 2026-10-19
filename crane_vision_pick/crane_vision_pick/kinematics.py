"""CRANE+ V2 の順運動学・逆運動学（解析解）.

数値は crane_plus_description/urdf/crane_plus.xacro（humble ブランチ）から取得:
  joint1: z=0.0454, 軸 +z（旋回）
  joint2: z=0.026,  軸 +y
  joint3: z=0.083,  軸 -y
  joint4: z=0.0935, 軸 -y
  link4 原点 → 把持中心: link4 の z 軸方向に約 0.13 m
    （rt-net/crane_plus の pick_and_place_tf.cpp の GRIPPER_OFFSET と同じ値）

座標はすべて base_link（= crane_plus_base）基準 [m]。
「横向き把持」= link4 の z 軸（指の向き）を水平にして、キューブを横から挟む。
MoveIt を使わずに解析的に解くことで、データ収集を軽く・決定的にしている。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- リンク長 -----------------------------------------------------------------
H_J1 = 0.0454          # base → joint1
H_J2 = 0.026           # joint1 → joint2
L2 = 0.083             # joint2 → joint3
L3 = 0.0935            # joint3 → joint4
GRIP_OFFSET = 0.13     # joint4(link4 原点) → 把持中心
SHOULDER_Z = H_J1 + H_J2

# --- 関節リミット（urdf と同じ式: deg - 150deg） --------------------------------
_D = math.pi / 180.0
_HOME = 150.0 * _D
JOINT_LIMITS = {
    'crane_plus_joint1': (0.0 * _D - _HOME, 300.0 * _D - _HOME),
    'crane_plus_joint2': (45.45 * _D - _HOME, 252.20 * _D - _HOME),
    'crane_plus_joint3': (3.52 * _D - _HOME, 290.62 * _D - _HOME),
    'crane_plus_joint4': (44.57 * _D - _HOME, 251.32 * _D - _HOME),
    'crane_plus_joint_hand': (109.38 * _D - _HOME, 188.27 * _D - _HOME),
}
ARM_JOINTS = ['crane_plus_joint1', 'crane_plus_joint2',
              'crane_plus_joint3', 'crane_plus_joint4']
GRIPPER_JOINT = 'crane_plus_joint_hand'
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]

# 安全マージン（リミットぎりぎりを避ける）
_MARGIN = 0.05


@dataclass
class IKResult:
    q: np.ndarray            # (4,) arm joints [j1, j2, j3, j4]
    ok: bool
    reason: str = ''


def _rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_y(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def fk(q):
    """順運動学. q=[j1,j2,j3,j4] → (把持中心位置(3,), link4 の z 軸方向(3,), joint4 位置(3,)).

    urdf の関節定義をそのまま積み上げる（IK の検証用なので素朴に書く）。
    """
    j1, j2, j3, j4 = q
    R = np.eye(3)
    p = np.zeros(3)
    # joint1
    p = p + R @ np.array([0, 0, H_J1])
    R = R @ _rot_z(j1)
    # joint2 (axis +y)
    p = p + R @ np.array([0, 0, H_J2])
    R = R @ _rot_y(j2)
    # joint3 (axis -y)
    p = p + R @ np.array([0, 0, L2])
    R = R @ _rot_y(-j3)
    # joint4 (axis -y)
    p = p + R @ np.array([0, 0, L3])
    R = R @ _rot_y(-j4)
    wrist = p.copy()
    z_axis = R @ np.array([0, 0, 1.0])
    tip = wrist + GRIP_OFFSET * z_axis
    return tip, z_axis, wrist


def within_limits(q, margin=_MARGIN):
    for name, v in zip(ARM_JOINTS, q):
        lo, hi = JOINT_LIMITS[name]
        if not (lo + margin <= v <= hi - margin):
            return False
    return True


def ik_side_grasp(x, y, z, pitch=math.pi / 2, prefer=None, grip_offset=GRIP_OFFSET):
    """把持中心を (x, y, z) に置き、指の向きを鉛直から pitch [rad] 傾けた姿勢の IK.

    pitch = pi/2 で水平（横から把持）。2 つの肘解のうちリミット内のものを返す。
    prefer が与えられたらそれに近い解を優先する。
    """
    j1 = math.atan2(y, x)
    r = math.hypot(x, y)
    # 手首（joint4）位置 = 把持中心から指の向きに GRIP_OFFSET 戻した点
    rw = r - grip_offset * math.sin(pitch)
    zw = z - grip_offset * math.cos(pitch)
    dx, dz = rw, zw - SHOULDER_Z
    d2 = dx * dx + dz * dz
    c = (d2 - L2 * L2 - L3 * L3) / (2 * L2 * L3)
    if abs(c) > 1.0:
        return IKResult(np.zeros(4), False, f'unreachable (c={c:.3f}, r={r:.3f}, z={z:.3f})')
    beta = math.atan2(dx, dz)   # 鉛直からの角度（前方が正）
    cands = []
    for sign in (1.0, -1.0):
        delta = sign * math.acos(c)               # = phi3 - phi2
        phi2 = beta - math.atan2(L3 * math.sin(delta), L2 + L3 * math.cos(delta))
        phi3 = phi2 + delta
        q = np.array([j1, phi2, -delta, phi3 - pitch])
        if within_limits(q):
            cands.append(q)
    if not cands:
        return IKResult(np.zeros(4), False, 'joint limits')
    if prefer is not None:
        cands.sort(key=lambda q: float(np.abs(q - np.asarray(prefer)).sum()))
    return IKResult(cands[0], True)


# 初期姿勢（SRDF の "home" と同じ）。指は水平前方を向く。
HOME_Q = np.array([0.0, -1.16, -2.01, -0.73])

GRIPPER_OPEN = -30.0 * _D    # pick_and_place_tf.cpp と同じ
GRIPPER_CLOSE = 20.0 * _D    # キューブに当たって止まる（押し付け続ける）。10deg では握る力が足りなかった
GRIPPER_DEFAULT = 0.0
