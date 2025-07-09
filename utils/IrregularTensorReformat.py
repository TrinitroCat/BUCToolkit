""" regularize Sequence of Tensors by padding """
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: IrregularTensorReformat.py
#  Environment: Python 3.12

from typing import List, Tuple

import torch as th


class IrregularTensorReformat:
    """
    Regularize List[Tensor] into one Tensor by padding, and then recover them into List[Tensor].
    The input tensors can only be irregular at 1st dim.
    """
    def __init__(self, ):
        self._length_record = None
        self.__shape_check = None

    def regularize(self, X: List[th.Tensor], padding_value:float=0.) -> Tuple[th.Tensor, th.Tensor]:
        """
        Regularize the List of Tensors.
        Args:
            X: the input sequence of Tensors.
            padding_value: the padding_value.

        Returns:
            X_r: Tensor, the regularized tensor.
            mask: Tensor, the mask of paddings. The part of `False` is paddings.
        """
        if th.any(th.isnan(th.cat(X, dim=0))):
            raise ValueError('Occurred NaN in input `X`.')
        self._length_record = [len(xi) for xi in X]
        # padding
        X = th.nn.utils.rnn.pad_sequence(X, True, padding_value=padding_value)
        pad_mask = ~th.isnan(X)
        # shape check
        self.__shape_check = len(X)

        return X, pad_mask

    def recover(self, X:th.Tensor) -> List[th.Tensor]:
        """
        Recover the regularized tensor into List[Tensor]
        Args:
            X: the Tensor that have same shape of X
        Returns:
            List[Tensor], the recovered Tensors.
        """
        if self._length_record is None:
            raise RuntimeError('`X` has not regularized yet.')
        elif len(X) != self.__shape_check:
            raise ValueError(f'Wrong shape of `X`. The length of `X` should be {self.__shape_check}, but occurred {len(X)}.')

        X = [X[i, :self._length_record[i]] for i in range(self.__shape_check)]

        return X
