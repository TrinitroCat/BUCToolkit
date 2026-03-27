"""
The Traditional Style Main Function/Program that runs tasks by a single command line with
input files and args.
"""
#  Copyright (c) 2026.3.27, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: main.py
#  Environment: Python 3.12


import argparse

import BUCToolkit as bt


class Main:
    """
    Usage Convention:
        buctoolkit -i xxx.inp -o xxx.oup
        bctk
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="BUCToolkit",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument()

    def main(self):
        pass
