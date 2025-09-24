#!/usr/bin/env python3
"""Setup script for SimEx package."""

from setuptools import setup, find_packages
import os

# Read the README file
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Systematic exploration tool for simulation and modeling"

# Read requirements
try:
    with open(os.path.join(current_directory, 'requirements.txt'), encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    requirements = [
        'matplotlib>=3.8.0',
        'numpy>=1.26.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.2.2',
    ]

setup(
    name='simex',
    version='1.0.0',
    author='SimEx Team',
    author_email='simex@silab.group',
    description='Systematic exploration tool for simulation and modeling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SiLab-group/SimEx',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'flake8>=4.0',
            'jupyter>=1.0',
            'notebook>=6.0',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'simex-run=examples.simex_run:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
