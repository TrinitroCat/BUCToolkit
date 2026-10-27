

#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: _warnings.py
#  Environment: Python 3.12

class NotConvergeWarning(Warning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
