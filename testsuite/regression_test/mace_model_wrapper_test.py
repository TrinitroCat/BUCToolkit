import unittest

import numpy as np
import torch as th

from BUCToolkit.BatchStructures import Batch, Data
from BUCToolkit.utils.model_wrappers import (
    MACEDataAdapter,
    MACEWrapper,
)
from BUCToolkit.api.DataLoaders import MACEDataLoader


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
        )

        mace_batch, labels = next(iter(loader))

        self.assertIs(adapter.to_batch(mace_batch), mace_batch)
        self.assertEqual(mace_batch.batch_size, 2)
        self.assertEqual(mace_batch.idx, ['structure_0.0', 'structure_0.2'])
        self.assertEqual(mace_batch.num_graphs, 2)
        self.assertEqual(tuple(mace_batch.positions.shape), (4, 3))
        self.assertEqual(tuple(labels['energy'].shape), (2,))
        self.assertEqual(tuple(labels['forces'].shape), (4, 3))

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
