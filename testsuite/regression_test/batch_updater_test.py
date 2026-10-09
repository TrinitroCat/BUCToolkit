"""Focused compatibility tests for graph-batch reallocation."""

import unittest

import torch as th

from BUCToolkit.api._io import PygBatchUpdater as ApiPygBatchUpdater
from BUCToolkit.BatchStructures.batch import Batch, _Batch
from BUCToolkit.BatchStructures.data import Data, _Data
from BUCToolkit.utils.batch_updaters import (
    PygBatchUpdater as UtilsPygBatchUpdater,
)


class PygBatchUpdaterTest(unittest.TestCase):
    """Check public graph aliases and batch reconstruction."""

    updater_types = (ApiPygBatchUpdater, UtilsPygBatchUpdater)

    @staticmethod
    def _call_updater(updater, batch):
        mask = th.tensor([True, False])
        return updater(mask, (batch,), {}, (batch,), {})

    def test_public_graph_classes_select_one_compatible_implementation(self):
        """Data and Batch aliases select PyG together or the fallback pair."""
        try:
            from torch_geometric.data import Batch as PygBatch
            from torch_geometric.data import Data as PygData
        except ImportError:
            self.assertIs(Data, _Data)
            self.assertIs(Batch, _Batch)
        else:
            self.assertIs(Data, PygData)
            self.assertIs(Batch, PygBatch)

    def test_updaters_rebuild_the_public_batch_implementation(self):
        """Both updater locations use the centrally selected Batch alias."""
        batch = Batch.from_data_list([
            Data(pos=th.zeros((2, 3))),
            Data(pos=th.ones((3, 3))),
        ])

        for updater_type in self.updater_types:
            with self.subTest(updater=updater_type.__module__):
                updater = updater_type()
                func_args, _, grad_func_args, _ = self._call_updater(
                    updater, batch
                )

                self.assertIsInstance(func_args[0], Batch)
                self.assertIs(func_args[0], grad_func_args[0])
                self.assertEqual(func_args[0].num_graphs, 1)
                self.assertTrue(th.equal(func_args[0].pos, batch[0].pos))


if __name__ == '__main__':
    unittest.main()
