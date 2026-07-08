from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    stereo_cam_node = Node(
        package='stereo_slam_camera',
        executable='stereo_cam_node',
        name='stereo_cam_node',
        output='screen',
        parameters=[
            {
                'device': '/dev/video21',
                'full_width': 1280,
                'full_height': 480,
                'fps': 30,
            }
        ]
    )

    return LaunchDescription([
        stereo_cam_node
    ])

