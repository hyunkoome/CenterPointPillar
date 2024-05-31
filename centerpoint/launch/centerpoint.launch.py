from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = "centerpoint"
    package_directory = FindPackageShare(package_name).find(package_name)
    src_directory = package_directory.replace("/install/"+package_name+"/share/", "/src/")

    return LaunchDescription(
        [
            Node(package=package_name,
                 executable="centerpoint_node",
                 name="centerpoint_node",
                 parameters=[{"config_path": src_directory+"/config/config.yaml",
                              "model_path": src_directory+"/model/model.trt"}],
                 output="screen",),
        ]
    )