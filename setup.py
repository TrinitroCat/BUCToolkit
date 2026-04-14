#  Copyright (c) 2025.7.9, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: setup.py
#  Environment: Python 3.12

from setuptools import setup, find_packages

setup(
    name='BUCToolkit',
    version='1.0b1',
    #packages=['utils', 'BatchMC', 'BatchMD', 'BatchOptim', 'BatchOptim.TS', 'BatchOptim._utils', 'BatchOptim.minimize',
    #          'BatchGenerate', 'Preprocessing', 'api', 'BatchStructures', 'cli'],
    packages=find_packages(),
    url='https://github.com/TrinitroCat/BUCToolkit',
    license='MIT',
    author='Pu Pengxin, Song xin',
    author_email='',
    include_package_data=True,
    package_data={"": ["README.md"]},
    description='Batch-upscaled Catalysis Toolkit (BUCToolkit) is an ai4science software package of computational chemistry, '
                'which can apply PyTorch-based deep-learning models (of molecular or crystal potentials) to perform '
                'training, predictions, batched structure optimization, batched molecular dynamics with/without constraints, '
                'and batched Monte Carlo simulations. Various tools for handling catalyst structure files are also included.',
    python_requires='>= 3.11',
    entry_points={
        "console_scripts": [
            "buctoolkit = BUCToolkit.cli.main:main",
            "bckt = BUCToolkit.cli.main:main",
        ],
        "buctoolkit.cli": [
            "bckt_cli = BUCToolkit.cli.__init__:run_base_cli",
        ],
    },
    install_requires=[
        'joblib>=1.4.2',
        'numpy>=1.26.4',
        'PyYAML>=6.0.1',
        'torch>=2.4.1'
    ],
    extras_require={
        'dgl': ['dgl', ],
        'pyg': ['torch_geometric>=2.6.1', ],
        'prompt-toolkit': ['prompt-toolkit>=3.0.43', ],
        }
)
