#  Copyright (c) 2025.7.9, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: setup.py
#  Environment: Python 3.12

from setuptools import setup, find_packages

setup(
    name='BUCToolkit',
    version='1.0b',
    #packages=['utils', 'BatchMC', 'BatchMD', 'BatchOptim', 'BatchOptim.TS', 'BatchOptim._utils', 'BatchOptim.minimize',
    #          'BatchGenerate', 'Preprocessing', 'TrainingMethod', 'BatchStructures', 'CLI'],
    packages=find_packages(),
    url='https://github.com/TrinitroCat/BUCToolkit',
    license='MIT',
    author='Pu Pengxin, Song xin',
    author_email='',
    description='Batch-upscaled Catalysis Toolkit, which can run PyTorch-based deep learning model training, predictions, '
                'batched structure optimization, batched molecular dynamics with/without constraints, and batched Monte Carlo simulations.',
    python_requires='>= 3.11',
    entry_points={
        "console_scripts": [
            "buctoolkit = BUCToolkit.CLI.main:main",
            "bckt = BUCToolkit.CLI.main:main",
        ],
        "buctoolkit.cli": [
            "bckt_cli = BUCToolkit.CLI.__init__:run_base_cli",
        ],
    },
    install_requires=[
        'joblib~=1.4.2',
        'numpy~=1.26.4',
        'PyYAML~=6.0.1',
        'torch~=2.4.1'
    ],
    extras_require={
        'dgl': ['dgl', ],
        'pyg': ['torch_geometric~=2.6.1', ],
        'prompt-toolkit': ['prompt-toolkit~=3.0.43', ],
        }
)
