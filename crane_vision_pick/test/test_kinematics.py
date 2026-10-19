import math

import numpy as np

from crane_vision_pick.expert import TaskConfig, plan_pick_and_place, sample_cube_pose
from crane_vision_pick.kinematics import HOME_Q, fk, ik_side_grasp, within_limits


def test_ik_roundtrip():
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(500):
        x, y, z = rng.uniform(0.15, 0.3), rng.uniform(-0.15, 0.2), rng.uniform(0.03, 0.12)
        res = ik_side_grasp(x, y, z)
        if not res.ok:
            continue
        n_ok += 1
        tip, z_axis, _ = fk(res.q)
        assert np.allclose(tip, [x, y, z], atol=1e-6)
        assert abs(z_axis[2]) < 1e-6          # 指が水平
        assert within_limits(res.q)
    assert n_ok > 200


def test_home_is_horizontal():
    _, z_axis, _ = fk(HOME_Q)
    assert abs(z_axis[2]) < 0.02


def test_plans_whole_sampling_region():
    cfg = TaskConfig()
    rng = np.random.default_rng(1)
    for _ in range(300):
        x, y, _ = sample_cube_pose(cfg, rng)
        cmds = plan_pick_and_place((x, y), cfg)
        assert cmds.shape[1] == 5
        # 1 ステップ (50ms) の関節変化が大きすぎない（関節速度制限 2.0 rad/s に対し 1.8 rad/s 以下）
        assert np.abs(np.diff(cmds[:, :4], axis=0)).max() < 0.09
        # 手先が常にテーブル(z=0)より上
        assert min(fk(q)[0][2] for q in cmds[:, :4]) > 0.02
    # 把持位置の FK が狙い通り
    cfg2 = TaskConfig(lateral_offset=0.0)
    cmds = plan_pick_and_place((0.22, 0.05), cfg2)
    tips = np.array([fk(q)[0] for q in cmds[:, :4]])
    i = int(np.argmin(tips[:, 2]))
    assert np.allclose(tips[i, :2], [0.22, 0.05], atol=1e-3)
    assert math.isclose(tips[i, 2], cfg2.z_grasp, abs_tol=1e-3)


def test_lateral_offset_is_perpendicular():
    cfg = TaskConfig(lateral_offset=0.008)
    cube = np.array([0.22, 0.05])
    cmds = plan_pick_and_place(tuple(cube), cfg)
    tips = np.array([fk(q)[0] for q in cmds[:, :4]])
    grasp = tips[int(np.argmin(tips[:, 2])), :2]
    radial = cube / np.linalg.norm(cube)
    left = np.array([-radial[1], radial[0]])
    d = grasp - cube
    assert abs(d @ left - 0.008) < 1e-3 and abs(d @ radial) < 1e-3
