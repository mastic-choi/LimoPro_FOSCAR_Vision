import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as C
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("lane_perception"), "config", "lane_perception.yaml")
    args = {"model": "yolopv2", "backend": "trt", "allow_aspect_mismatch": "false", "timing_csv": "", "publish_overlay": "false"}
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
        + [DeclareLaunchArgument("python", default_value=os.path.expanduser("~/miniconda3/envs/dl_env/bin/python"))]
        + [
            Node(
                package="lane_perception",
                executable="lane_perception_node",
                name="lane_perception_node",
                output="screen",
                prefix=C("python"),  # torch lives in dl_env, not in the system Python
                parameters=[cfg, {k: C(k) for k in args}],
            )
        ]
    )
