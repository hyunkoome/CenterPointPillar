from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    package_name = "centerpoint"
    package_directory = FindPackageShare(package_name).find(package_name)

    return LaunchDescription(
        [
            Node(package=package_name,
                 executable="centerpoint_node",
                 name="centerpoint_node",
                 parameters=[{"config_path": package_directory+"/config/config.yaml"}],
                 output="screen",),
            # SetParameter(name="config_path", value=package_directory+"/config/config.yaml"),
        ]
    )