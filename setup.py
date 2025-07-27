#  Copyright (c) 2025.7.9, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: setup.py
#  Environment: Python 3.12

from setuptools import setup

setup(
    name='BM4Ckit',
    version='0.9a',
    packages=['utils', 'BatchMC', 'BatchMD', 'BatchOptim', 'BatchOptim.TS', 'BatchOptim._utils', 'BatchOptim.minimize',
              'BatchGenerate', 'Preprocessing', 'TrainingMethod', 'BatchStructures'],
    package_dir={'': 'BM4Ckit'},
    url='https://github.com/TrinitroCat/BM4Ckit',
    license='MIT',
    author='Pu Pengxin, Song xin',
    author_email='',
    description='Batched Machine-learning for Catalysis kit',
    python_requires='>= 3.10',
    install_requires=[
        'joblib>=1.4.2',
        'numpy>=1.26.4',
        'PyYAML>=6.0.1',
        'torch>=2.4.1'
    ],
    extras_require={
        'dgl': ['dgl', ],
        'pyg': ['torch_geometric', ]
        }
)
