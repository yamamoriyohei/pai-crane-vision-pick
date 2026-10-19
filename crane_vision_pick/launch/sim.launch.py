"""ビジョンピック用シミュレーションの起動.

  ros2 launch crane_vision_pick sim.launch.py

中身は講義と同じ crane_plus_with_table.launch.py に
  - world_name: 本パッケージの pick_world.sdf（赤/青キューブとトレイ）
  - use_camera: true（ロボット横のスタンドに俯瞰カメラが付く）
を渡し、画像縮小ノードを追加しただけ。
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    world = os.path.join(get_package_share_directory('crane_vision_pick'), 'worlds', 'pick_world.sdf')
    declare_world = DeclareLaunchArgument('world', default_value=world)
    declare_factor = DeclareLaunchArgument('image_factor', default_value='2',
                                           description='画像の間引き率（2 → 320x240）')
    crane = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('crane_plus_gazebo'), 'launch', 'crane_plus_with_table.launch.py')),
        launch_arguments={'world_name': LaunchConfiguration('world'), 'use_camera': 'true'}.items(),
    )
    downsampler = Node(
        package='crane_vision_pick', executable='image_downsampler', output='screen',
        parameters=[{'factor': LaunchConfiguration('image_factor')}],
    )
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        declare_world, declare_factor, crane, downsampler,
    ])
