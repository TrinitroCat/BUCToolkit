import unittest
from importlib.util import find_spec

import numpy as np
import torch as th

from BUCToolkit.BatchStructures import Batch, Data
from BUCToolkit.utils.model_wrappers import (
    MACEDataAdapter,
    MACEWrapper,
)
from BUCToolkit.api.DataLoaders import MACEDataLoader


HAS_MACE = find_spec('mace') is not None and find_spec('e3nn') is not None


@unittest.skipUnless(HAS_MACE, 'MACE and e3nn are required')
class MACEModelWrapperTest(unittest.TestCase):
    @staticmethod
    def _structure(offset: float = 0.0) -> Data:
        return Data(
            pos=th.tensor([[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]]),
            atomic_numbers=th.tensor([1, 8]),
            cell=th.eye(3).view(1, 3, 3) * 8.0,
            pbc=th.tensor([[False, False, False]]),
            idx=f'structure_{offset}',
        )

    @staticmethod
    def _model_config():
        return {
            'model_type': 'ScaleShiftMACE',
            'interaction': 'RealAgnosticResidualInteractionBlock',
            'interaction_first': 'RealAgnosticInteractionBlock',
            'hidden_irreps': '8x0e + 8x1o',
            'MLP_irreps': '8x0e',
            'atomic_numbers': [1, 8],
            'r_max': 3.0,
            'heads': ['Default'],
            'default_head': 'Default',
            'gate': 'silu',
            'num_bessel': 4,
            'num_polynomial_cutoff': 3,
            'max_ell': 1,
            'num_interactions': 2,
            'num_elements': 2,
            'atomic_energies': [[0.0, 0.0]],
            'avg_num_neighbors': 1.0,
            'correlation': 2,
            'atomic_inter_scale': 1.0,
            'atomic_inter_shift': 0.0,
        }

    @staticmethod
    def _loader_config():
        return {
            'atomic_numbers': [1, 8],
            'r_max': 3.0,
            'heads': ['Default'],
            'default_head': 'Default',
        }

    def test_adapter_converts_fallback_data_and_batch(self):
        adapter = MACEDataAdapter(
            atomic_numbers=[1, 8], r_max=3.0, heads=['Default']
        )
        atomic_data = adapter.to_atomic_data(self._structure())
        self.assertEqual(tuple(atomic_data.positions.shape), (2, 3))
        self.assertEqual(tuple(atomic_data.node_attrs.shape), (2, 2))
        self.assertEqual(atomic_data.idx, 'structure_0.0')

        generic_batch = Batch.from_data_list([
            self._structure(), self._structure(offset=0.2)
        ])
        mace_batch = adapter.to_batch(generic_batch)
        self.assertEqual(mace_batch.num_graphs, 2)
        self.assertEqual(tuple(mace_batch.positions.shape), (4, 3))

    def test_multiple_heads_require_explicit_default(self):
        with self.assertRaisesRegex(ValueError, 'default_head'):
            MACEDataAdapter(
                atomic_numbers=[1, 8], r_max=3.0, heads=['head_a', 'head_b']
            )

    def test_mace_data_loader_uses_mace_batch(self):
        adapter = MACEDataAdapter(
            atomic_numbers=[1, 8], r_max=3.0, heads=['Default']
        )
        atomic_data = adapter.to_atomic_data_list([
            self._structure(), self._structure(offset=0.2)
        ])
        loader = MACEDataLoader(
            {
                'data': atomic_data,
                'labels': {
                    'energy': [0.0, 0.0],
                    'forces': [np.zeros((2, 3)), np.zeros((2, 3))],
                },
            },
            batch_size=2,
            shuffle=False,
            model_config=self._loader_config(),
        )

        mace_batch, labels = next(iter(loader))

        self.assertIs(adapter.to_batch(mace_batch), mace_batch)
        self.assertEqual(mace_batch.batch_size, 2)
        self.assertEqual(mace_batch.idx, ['structure_0.0', 'structure_0.2'])
        self.assertEqual(mace_batch.num_graphs, 2)
        self.assertEqual(tuple(mace_batch.positions.shape), (4, 3))
        self.assertEqual(tuple(labels['energy'].shape), (2,))
        self.assertEqual(tuple(labels['forces'].shape), (4, 3))

    def test_mace_data_loader_prefetches_raw_structures(self):
        raw = [self._structure(offset=0.2 * i) for i in range(5)]
        labels = {
            'energy': [float(i) for i in range(5)],
            'forces': [np.zeros((2, 3)) for _ in range(5)],
        }
        loader = MACEDataLoader(
            {'data': raw, 'labels': labels},
            batch_size=2,
            shuffle=False,
            model_config=self._loader_config(),
            buffer_size=4,
        )
        batches = list(loader)

        self.assertEqual([batch.batch_size for batch, _ in batches], [2, 2, 1])
        self.assertEqual(
            [index for batch, _ in batches for index in batch.idx],
            [self._structure(offset=0.2 * i).idx for i in range(5)],
        )
        self.assertEqual(tuple(batches[0][1]['energy'].shape), (2,))
        self.assertEqual(tuple(batches[0][1]['forces'].shape), (4, 3))
        self.assertEqual(tuple(batches[-1][1]['energy'].shape), (1,))
        self.assertEqual(tuple(batches[-1][1]['forces'].shape), (2, 3))

        adapter = MACEDataAdapter(**self._loader_config())
        expected = adapter.to_batch(Batch.from_data_list(raw[:2]))
        first_batch = batches[0][0]
        self.assertTrue(th.equal(first_batch.positions, expected.positions))
        self.assertTrue(th.equal(first_batch.edge_index, expected.edge_index))
        self.assertTrue(th.equal(first_batch.shifts, expected.shifts))
        self.assertTrue(th.equal(first_batch.unit_shifts, expected.unit_shifts))
        self.assertTrue(first_batch.batch.equal(expected.batch))
        self.assertTrue(loader._closed)

    def test_mace_data_loader_prefetch_lifecycle(self):
        raw = [self._structure(offset=0.2 * i) for i in range(4)]
        loader = MACEDataLoader(
            {
                'data': raw,
                'labels': {
                    'energy': [0.0] * 4,
                    'forces': [np.zeros((2, 3))] * 4,
                },
            },
            batch_size=2,
            shuffle=False,
            model_config=self._loader_config(),
            buffer_size=4,
        )
        self.assertIsNone(loader._prefetch_queue)

        next(iter(loader))
        self.assertIsNotNone(loader._prefetch_queue)
        self.assertEqual(loader._prefetch_queue.maxsize, 4)
        self.assertEqual(loader._prefetch_thread.name, 'MACEDataLoaderPrefetch')

        loader.close()
        self.assertTrue(loader._closed)
        self.assertFalse(loader._prefetch_thread.is_alive())
        with self.assertRaises(StopIteration):
            next(loader)

    def test_mace_data_loader_prefetch_error_after_complete_batch(self):
        valid = [self._structure(offset=0.2 * i) for i in range(2)]
        invalid = Data(
            atomic_numbers=th.tensor([1, 8]),
            cell=th.eye(3).view(1, 3, 3) * 8.0,
            pbc=th.tensor([[False, False, False]]),
            idx='structure_invalid',
        )
        loader = MACEDataLoader(
            {
                'data': valid + [invalid],
                'labels': {
                    'energy': [0.0, 0.0, 0.0],
                    'forces': [np.zeros((2, 3))] * 3,
                },
            },
            batch_size=2,
            shuffle=False,
            model_config=self._loader_config(),
            buffer_size=4,
        )

        batch, labels = next(iter(loader))
        self.assertEqual(batch.idx, ['structure_0.0', 'structure_0.2'])
        self.assertEqual(tuple(labels['energy'].shape), (2,))

        with self.assertRaises(ValueError):
            next(loader)
        self.assertTrue(loader._closed)

    def test_mace_data_loader_prefetch_shuffle_aligns_labels(self):
        raw = [self._structure(offset=0.2 * i) for i in range(6)]
        loader = MACEDataLoader(
            {
                'data': raw,
                'labels': {'energy': [float(i) for i in range(6)]},
            },
            batch_size=2,
            shuffle=True,
            model_config=self._loader_config(),
            buffer_size=4,
        )
        seen = []
        for batch, labels in loader:
            for index, energy in zip(batch.idx, labels['energy'].tolist()):
                offset = float(index.replace('structure_', ''))
                seen.append((index, energy))
                self.assertAlmostEqual(energy, offset / 0.2, places=5)
        self.assertEqual(len(seen), 6)

    def test_mace_data_loader_requires_model_config(self):
        with self.assertRaisesRegex(ValueError, 'model_config'):
            MACEDataLoader(
                {'data': [], 'labels': {'energy': []}},
                batch_size=2,
                shuffle=False,
            )

    def test_wrapper_uses_original_mace_state_dict_keys(self):
        wrapper = MACEWrapper(**self._model_config())
        wrapper_keys = set(wrapper.state_dict())
        model_keys = set(wrapper.mace_model.state_dict())
        self.assertEqual(wrapper_keys, model_keys)
        self.assertFalse(any(key.startswith('mace_model.') for key in wrapper_keys))

        output = wrapper(self._structure())
        self.assertEqual(tuple(output['energy'].shape), (1,))
        self.assertEqual(tuple(output['forces'].shape), (2, 3))


if __name__ == '__main__':
    unittest.main()
