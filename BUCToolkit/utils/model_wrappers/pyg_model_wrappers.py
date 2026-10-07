#  Copyright (c) 2026.5.18, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: model_wrappers.py
#  Environment: Python 3.12

import torch as th
from BUCToolkit.utils.function_utils import _BaseWrapper, compare_tensors
from BUCToolkit.BatchStructures import Batch
from BUCToolkit.utils._CheckModules import check_module


class Model_Wrapper_pyg(_BaseWrapper):

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


class Model_Wrapper_pyg_only_X(_BaseWrapper):
    """Adapt a PyG model when one fixed graph is supplied at construction."""

    def __init__(self, model, graph: Batch) -> None:
        super().__init__(model)
        self.graph = graph
        self.X = None

    def Energy(self, X):
        self.X = X
        self.graph.pos = X.squeeze(0).reshape(-1, 3).contiguous()
        result = self._model(self.graph)
        self.forces = result['forces']
        return th.sum(result['energy']).unsqueeze(0)

    def Grad(self, X):
        origin_shape = X.shape
        if self.X is None or not compare_tensors(X, self.X):
            self.forces = None
        if self.forces is None:
            self.Energy(X)
        force = self.forces
        self.forces = None
        return -force.reshape(origin_shape).contiguous()


class Model_Wrapper_regularBatch_pyg(_BaseWrapper):
    """Adapt a single-structure PyG graph to a regular coordinate batch."""

    def __init__(self, model, **kwargs) -> None:
        super().__init__(model)
        pyg_module = check_module('torch_geometric.data')
        self.pygBatch = pyg_module.Batch if pyg_module is not None else Batch
        self.X = None

    def Energy(self, X: th.Tensor, graph: Batch):
        self.X = X
        if graph.batch_size == 1:
            graph = self.pygBatch.from_data_list([graph] * X.size(0), exclude_keys=['batch', 'ptr'])
        graph.pos = X.flatten(0, 1)
        result = self._model(graph)
        self.forces = result['forces']
        return result['energy']

    def Grad(self, X, graph: Batch):
        if self.X is None or not compare_tensors(X, self.X):
            self.forces = None
        if self.forces is None:
            self.Energy(X, graph)
        force = self.forces
        self.forces = None
        return -force.reshape_as(X)
