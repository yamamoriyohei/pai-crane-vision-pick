import math

from crane_vision_pick.sim_utils import poses_from_pb_text

SAMPLE = '''
header {
  stamp {
    sec: 12
    nsec: 345000000
  }
  data {
    key: "frame_id"
    value: "default"
  }
}
pose {
  name: "crane_plus"
  id: 9
  position {
    z: 1.015
  }
  orientation {
    w: 1
  }
}
pose {
  name: "red_cube"
  id: 21
  position {
    x: 0.21
    y: -0.05
    z: 1.04
  }
  orientation {
    z: 0.38268343
    w: 0.92387953
  }
}
pose {
  name: "link"
  id: 22
  position {
  }
  orientation {
    w: 1
  }
}
'''


def test_parse_poses():
    poses = poses_from_pb_text(SAMPLE)
    x, y, z, yaw = poses['red_cube']
    assert (x, y, z) == (0.21, -0.05, 1.04)
    assert math.isclose(yaw, math.pi / 4, abs_tol=1e-6)
    # 0 のフィールドが省略されていても読める
    assert poses['crane_plus'][:3] == (0.0, 0.0, 1.015)
    assert poses['link'][:3] == (0.0, 0.0, 0.0)
