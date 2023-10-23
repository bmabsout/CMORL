from os.path import join, dirname, realpath
from setuptools import setup, find_packages
import sys

setup(
    name='cmorl',
    py_modules=['cmorl'],
    packages=find_packages(),
    version='0.1',
    install_requires=[
        'gymnasium[classical_control]',
        'numpy',
        'tensorflow',
        'tqdm'
    ],
    description="deep RL actor critic methods with multi-objective composition",
    author="Bassel El Mabsout"
)
