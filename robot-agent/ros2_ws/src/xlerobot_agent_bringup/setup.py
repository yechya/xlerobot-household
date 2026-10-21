from glob import glob

from setuptools import find_packages, setup

package_name = "xlerobot_agent_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="XLeRobot Maintainers",
    maintainer_email="robot@example.invalid",
    description="Safety-gated Nav2 bringup for XLeRobot Agent",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "base_bridge = robot_agent.base_bridge_node:main",
            "segmentation_adapter = robot_agent.segmentation_adapter_node:main",
            "rgb_world_model_bridge = robot_agent.rgb_world_model_bridge_node:main",
        ],
    },
)
