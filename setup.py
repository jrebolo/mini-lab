#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

setup(
    name='mini_lab',
    author='João Vieira',
    description='Mini Laboratory for machine learning experiments.',
    license='MIT',
    version='0.1.0',
    packages=['minilab','minilab.models', 'minilab.train'],
    python_requires='>=3.8',
)