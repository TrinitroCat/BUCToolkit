"""Deprecated DGL model adapter retained for source compatibility."""

import torch as th

from BUCToolkit.utils._CheckModules import check_module
from BUCToolkit.utils.function_utils import _BaseWrapper, compare_tensors


class Model_Wrapper_dgl(_BaseWrapper):
    """Adapt a DGL calculator to the Energy/Grad wrapper convention."""

    def __init__(self, model) -> None:
        super().__init__(model)
        if check_module('dgl') is None:
            raise ImportError('`Model_Wrapper_dgl` requires package `dgl`.')
        self.X = None

    def Energy(self, X, graph, return_format: str = 'origin'):
        self.X = X
        graph.nodes['atom'].data['pos'] = X.squeeze(0)
        result = self._model(graph)
        self.forces = result['forces']
        energy = result['energy']
        if return_format == 'sum':
            energy = th.sum(energy).unsqueeze(0)
        return energy

    def Grad(self, X, graph):
        if self.X is None or not compare_tensors(X, self.X):
            self.forces = None
        if self.forces is None:
            self.X = X
            graph.nodes['atom'].data['pos'] = X.squeeze(0)
            return -self._model(graph)['forces'].unsqueeze(0)
        force = self.forces
        self.forces = None
        return -force.unsqueeze(0)

