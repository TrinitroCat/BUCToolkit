"""
Configs of Printing Numpy Arrays Format.

e.g., numpy.array2string(Array, **FLOAT_ARRAY_FORMAT)
"""
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _print_formatter.py
#  Environment: Python 3.12

import numpy as np

FLOAT_ARRAY_FORMAT = {
    'max_line_width': 256,
    'precision': 8,
    'separator': '  ',
    'sign': ' ',
    'floatmode': 'fixed',
    'threshold': 21_4748_3647,
    'formatter': {'float_kind': lambda x: f'{x:0< 16.8f}', 'int_kind': lambda x: f'{x:> 16d}'}
}

INT_ARRAY_FORMAT = {
    'max_line_width': 120,
    'separator': '  ',
    'sign': ' ',
    'threshold': 21_4748_3647,
    'formatter': {'int_kind': lambda x: f'{x:> 16d}'}
}

STRING_ARRAY_FORMAT = {
    'max_line_width': 120,
    'separator': '  ',
    'sign': ' ',
    'threshold': 21_4748_3647,
    'formatter': {'str_kind': lambda x: f'{x}'}
}

SCIENTIFIC_ARRAY_FORMAT = {
    'max_line_width': 256,
    'precision': 8,
    'separator': '  ',
    'sign': ' ',
    'floatmode': 'fixed',
    'threshold': 21_4748_3647,
    'formatter': {'float_kind': lambda x: f'{x:< .8e}', 'int_kind': lambda x: f'{x:> 16d}'}
}

GLOBAL_SCIENTIFIC_ARRAY_FORMAT = {
    'precision': 8,
    'sign': ' ',
    'floatmode': 'fixed',
    'threshold': 21_4748_3647,
    'formatter': {'float_kind': lambda x: f'{x:< .8e}', 'int_kind': lambda x: f'{x:> 16d}'}
}

def __as_formatted_str(a: float, ): return f'{a:0< 14.8f}'
AS_PRINT_COORDS = np.vectorize(__as_formatted_str, otypes='U')
