from os.path import join, dirname, realpath
from setuptools import setup
import sys

setup(
    name='anchored_rl',
    py_modules=['anchored_rl'],
    version='0.1',
    install_requires=[
        'gymnasium[classical_control]',
        'numpy',
        'tensorflow',
        'tqdm'
    ],
    description="deep RL actor critic methods with anchors",
    author="Bassel El Mabsout"
)
