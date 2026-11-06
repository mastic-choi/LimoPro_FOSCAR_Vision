"""Camera for lane perception: 1280x720 color only, the resolution the models were trained from.

Uses the driver launch that teleop_launch.py also uses, without changing it.
dabai_dcl.py is not used: over this USB 2.0 connection it has no 640x360 profile and enters a device reset loop.
Depth, IR and point clouds are off by default; 1280x720 color with depth is untested on USB 2.0.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    defaults = {"color_fps": "30", "enable_depth": "false"}
    fixed = {
        "color_width": "1280",
        "color_height": "720",
        "color_format": "MJPG",
        "enable_ir": "false",
        "enable_point_cloud": "false",
        "enable_colored_point_cloud": "false",
    }
    driver = PathJoinSubstitution([FindPackageShare("orbbec_camera"), "launch", "astra_stereo_u3.launch.py"])
    args = dict(fixed)
    args.update({k: LaunchConfiguration(k) for k in defaults})
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()]
        + [IncludeLaunchDescription(PythonLaunchDescriptionSource([driver]), launch_arguments=args.items())]
    )
