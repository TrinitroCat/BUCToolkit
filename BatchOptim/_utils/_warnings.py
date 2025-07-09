

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _warnings.py
#  Environment: Python 3.12

class FaildToConvergeWarning(Warning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)