from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

description = """MaMuJoCo-PettingZoo - Multi-agent MuJoCo robotics environment with Convenient Wrappers and Utilities. The origin environment can be found at https://github.com/Farama-Foundation/Gymnasium-Robotics."""

setup(
    name="mamujoco_pettingzoo",
    version="1.0.0",
    description="MaMuJoCo-PettingZoo - Multi-agent MuJoCo robotics environment with Convenient Wrappers and Utilities.",
    long_description=description,
    license="MIT License",
    keywords="Multi-agent MuJoCo, PettingZoo, Multi-Agent Reinforcement Learning",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "pettingzoo>=1.24.3",
        "gymnasium-robotics @ git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git",
        "mujoco",
        "pre-commit",
    ],
)
