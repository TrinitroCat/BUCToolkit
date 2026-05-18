#  Copyright (c) 2026.5.18, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: model_wrappers.py
#  Environment: Python 3.12

import BUCToolkit as bt
from BUCToolkit.utils.function_utils import _BaseWrapper, compare_tensors
#from BUCToolkit.BatchStructures import Batch


class Model_Wrapper_pyg(_BaseWrapper):

    __slots__ = ('_model', 'forces', 'X', )

    def __init__(self, model, pos_attr_name='pos',) -> None:
        """
        A format transformer for converting Tensor X into PygData.pos
        Wrap the model(graph, ...) into f(X)

        Args:
            model: An instantiate nn.Module

        Methods:
            Energy: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['energy'].
            Grad: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['forces'].

        """
        super().__init__(model)
        self.pos_attr_name = pos_attr_name
        self.X = None
        #if check_module('torch_geometric') is None:
        #    ImportError('The method is unavailable because the `torch-geometric` cannot be imported.')
        pass

    def Energy(self, X, graph):
        self.X = X
        if hasattr(graph, 'pos'):
            graph.pos = self.X.reshape(-1,3).contiguous()
        if hasattr(graph, 'positions'):
            graph.positions = self.X.reshape(-1,3).contiguous()
        y = self._model(graph)
        energy = y['energy']
        self.forces = y['forces']
        return energy

    def Grad(self, X, graph):
        origin_shape = X.shape
        if (self.X is None) or (not compare_tensors(X, self.X)):
            self.forces = None
        if self.forces is None:
            self.X = X
            if hasattr(graph, 'pos'):
                graph.pos = self.X.reshape(-1, 3).contiguous()
            if hasattr(graph, 'positions'):
                graph.positions = self.X.reshape(-1, 3).contiguous()
            return - ((self._model(graph))['forces']).reshape(origin_shape)
        else:
            force = self.forces
            self.forces = None
            return - force.reshape(origin_shape).contiguous()
