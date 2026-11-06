from glob import glob
from setuptools import setup

name = "lane_perception"
setup(
    name=name,
    version="0.1.0",
    packages=[name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + name]),
        ("share/" + name, ["package.xml"]),
        ("share/" + name + "/launch", glob("launch/*.py")),
        ("share/" + name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "lane_perception_node = lane_perception.node:main",
        "lane_viz_node = lane_perception.viz_node:main",
    ]},
)
