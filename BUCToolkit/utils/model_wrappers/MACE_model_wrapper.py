#  Copyright (c) 2026.5.18, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: MACE_model_wrapper.py
#  Environment: Python 3.12

import random
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Literal, Self, Tuple

import numpy as np
import torch as th
from torch import nn


_MACE_COMPONENTS: Dict[str, Any] | None = None


def _require_mace() -> Dict[str, Any]:
    """Import and cache the optional MACE dependency when it is first used."""
    global _MACE_COMPONENTS
    if _MACE_COMPONENTS is None:
        try:
            safe_globals = getattr(th.serialization, 'safe_globals', None)
            if safe_globals is None:
                from e3nn import o3
            else:
                # e3nn 0.4.4 imports a package-owned constants.pt while importing
                # e3nn.o3. That file contains precomputed tensor tables plus a
                # built-in ``slice`` object. PyTorch 2.6 changed torch.load's
                # default to weights_only=True, so the old e3nn import now fails
                # before MACE itself can be imported because ``slice`` is not in
                # the default safe-global set. The file is a trusted, installed
                # dependency rather than a user checkpoint; allowlist exactly
                # that built-in type only for this import operation. Keep this
                # context local: do not set weights_only=False and do not add a
                # permanent process-wide allowlist, both of which would weaken
                # the checkpoint security boundary that this wrapper preserves.
                with safe_globals([slice]):
                    from e3nn import o3
            from mace import modules
            from mace.data import AtomicData, Configuration
            from mace.tools import AtomicNumberTable
            from mace.tools.torch_geometric import Batch
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                '`MACEWrapper` requires an official MACE installation and its dependencies.'
            ) from exc
        _MACE_COMPONENTS = {
            'AtomicData': AtomicData,
            'AtomicNumberTable': AtomicNumberTable,
            'Batch': Batch,
            'Configuration': Configuration,
            'modules': modules,
            'o3': o3,
        }
    return _MACE_COMPONENTS


def _resolve_registered_value(registry: Mapping[str, Any], name: str, argument_name: str) -> Any:
    if not isinstance(name, str):
        raise TypeError(f'`{argument_name}` must be a string, but got {type(name).__name__}.')
    if name not in registry:
        supported_names = ', '.join(sorted(registry))
        raise ValueError(
            f'Unsupported `{argument_name}` value {name!r}. Supported values: {supported_names}.'
        )
    return registry[name]


class MACEDataAdapter:
    """Convert generic PyG-compatible atomic structures into MACE data objects.

    Args:
        atomic_numbers: Ordered atomic numbers used by the MACE element table.
        r_max: Neighbor-list cutoff in Angstrom.
        heads: Ordered MACE head names. Defaults to ``['Default']``.
        default_head: Head assigned when an input structure has no ``head`` attribute.
            It is required when more than one head is configured.

    Notes:
        Converting data constructs the MACE neighbor graph on CPU. Static training data
        should be converted once before the epoch loop instead of in every forward call.
    """

    def __init__(
            self,
            atomic_numbers: Sequence[int],
            r_max: float,
            heads: Sequence[str] | None = None,
            default_head: str | None = None,
    ) -> None:
        components = _require_mace()
        normalized_atomic_numbers = [int(atomic_number) for atomic_number in atomic_numbers]
        if len(normalized_atomic_numbers) == 0:
            raise ValueError('`atomic_numbers` must contain at least one atomic number.')
        if len(set(normalized_atomic_numbers)) != len(normalized_atomic_numbers):
            raise ValueError('`atomic_numbers` must not contain duplicate values.')

        normalized_heads = ['Default'] if heads is None else [str(head) for head in heads]
        if len(normalized_heads) == 0:
            raise ValueError('`heads` must contain at least one head name.')
        if default_head is None:
            if len(normalized_heads) == 1:
                default_head = normalized_heads[0]
            elif 'Default' in normalized_heads:
                default_head = 'Default'
            else:
                raise ValueError(
                    '`default_head` is required when multiple MACE heads are configured.'
                )
        if default_head not in normalized_heads:
            raise ValueError(
                f'`default_head` {default_head!r} is not present in `heads` {normalized_heads!r}.'
            )

        self.atomic_numbers = normalized_atomic_numbers
        self.r_max = float(r_max)
        self.heads = normalized_heads
        self.default_head = default_head
        self._atomic_data_class = components['AtomicData']
        self._batch_class = components['Batch']
        self._configuration_class = components['Configuration']
        self._z_table = components['AtomicNumberTable'](normalized_atomic_numbers)

    @staticmethod
    def _get_value(data: Any, names: Sequence[str], required: bool = False) -> Any:
        for name in names:
            if isinstance(data, Mapping):
                value = data.get(name)
            else:
                value = getattr(data, name, None)
            if value is not None:
                return value
        if required:
            joined_names = ' or '.join(f'`{name}`' for name in names)
            raise ValueError(f'MACE input data requires the attribute {joined_names}.')
        return None

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, th.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _resolve_head(self, data: Any) -> str:
        head = self._get_value(data, ('head',))
        if head is None:
            return self.default_head
        if isinstance(head, th.Tensor):
            if head.numel() != 1:
                raise ValueError(f'`head` must be scalar, but got shape {tuple(head.shape)}.')
            head = head.item()
        if isinstance(head, (int, np.integer)):
            try:
                return self.heads[int(head)]
            except IndexError as exc:
                raise ValueError(
                    f'`head` index {head} is outside the configured {len(self.heads)} heads.'
                ) from exc
        if not isinstance(head, str):
            raise TypeError(f'`head` must be a string or integer index, but got {type(head).__name__}.')
        if head not in self.heads:
            raise ValueError(f'Input head {head!r} is not present in configured heads {self.heads!r}.')
        return head

    def to_atomic_data(self, data: Any) -> Any:
        """Convert one generic structure to ``mace.data.AtomicData``.

        Args:
            data: A mapping or Data-like object containing ``pos``/``positions``,
                ``atomic_numbers``, and optional ``cell``, ``pbc``, and ``head``.

        Returns:
            A MACE ``AtomicData`` object on the same device as the input positions.

        Raises:
            TypeError: If the input does not expose the required Data-like interface.
            ValueError: If required attributes or shapes are invalid.
        """
        if isinstance(data, self._atomic_data_class):
            return data

        positions = self._get_value(data, ('pos', 'positions'), required=True)
        atomic_numbers = self._get_value(data, ('atomic_numbers',), required=True)
        cell = self._get_value(data, ('cell',))
        pbc = self._get_value(data, ('pbc',))
        structure_idx = self._get_value(data, ('idx',))

        positions_array = self._to_numpy(positions)
        atomic_numbers_array = self._to_numpy(atomic_numbers).astype(np.int64, copy=False)
        if positions_array.ndim != 2 or positions_array.shape[1] != 3:
            raise ValueError(f'Positions must have shape (n_atom, 3), but got {positions_array.shape}.')
        if atomic_numbers_array.ndim != 1 or len(atomic_numbers_array) != len(positions_array):
            raise ValueError(
                '`atomic_numbers` must have shape (n_atom,) matching positions, '
                f'but got {atomic_numbers_array.shape} and {positions_array.shape}.'
            )

        cell_array = None
        if cell is not None:
            cell_array = self._to_numpy(cell)
            if cell_array.shape == (1, 3, 3):
                cell_array = cell_array[0]
            if cell_array.shape != (3, 3):
                raise ValueError(f'`cell` must have shape (3, 3) or (1, 3, 3), but got {cell_array.shape}.')

        pbc_tuple = None
        if pbc is not None:
            pbc_array = self._to_numpy(pbc)
            if pbc_array.shape == (1, 3):
                pbc_array = pbc_array[0]
            if pbc_array.shape != (3,):
                raise ValueError(f'`pbc` must have shape (3,) or (1, 3), but got {pbc_array.shape}.')
            pbc_tuple = tuple(bool(value) for value in pbc_array)

        configuration = self._configuration_class(
            atomic_numbers=atomic_numbers_array,
            positions=positions_array,
            properties={},
            property_weights={},
            cell=cell_array,
            pbc=pbc_tuple,
            head=self._resolve_head(data),
        )
        atomic_data = self._atomic_data_class.from_config(
            configuration,
            z_table=self._z_table,
            cutoff=self.r_max,
            heads=self.heads,
        )
        if structure_idx is not None:
            atomic_data.idx = structure_idx
        if isinstance(positions, th.Tensor):
            atomic_data = atomic_data.to(positions.device)
        return atomic_data

    def to_atomic_data_list(self, data: Sequence[Any]) -> list[Any]:
        """Convert a sequence of structures once for repeated training use.

        Args:
            data: Sequence of generic Data-like atomic structures.

        Returns:
            MACE ``AtomicData`` objects in the original order.
        """
        return [self.to_atomic_data(structure) for structure in data]

    def to_batch(self, data: Any) -> Any:
        """Convert a single structure or generic Batch to a MACE Batch.

        Args:
            data: One Data-like structure or a Batch exposing ``to_data_list``.

        Returns:
            A MACE Batch using MACE's own concatenation rules.

        Raises:
            TypeError: If a batch-like input cannot be separated into structures.
        """
        if isinstance(data, self._batch_class):
            return data
        if isinstance(data, self._atomic_data_class):
            data_list = [data]
        elif hasattr(data, 'to_data_list') and callable(data.to_data_list):
            data_list = data.to_data_list()
        else:
            data_list = [data]
        if len(data_list) == 0:
            raise ValueError('Cannot convert an empty data batch to MACE format.')

        positions = self._get_value(data_list[0], ('pos', 'positions'))
        mace_batch = self._batch_class.from_data_list(
            [self.to_atomic_data(structure) for structure in data_list]
        )
        if isinstance(positions, th.Tensor):
            mace_batch = mace_batch.to(positions.device)
        return mace_batch


class MACEWrapper(nn.Module):
    """Expose an official MACE model through standard Python configuration values.

    Args:
        model_type: Official MACE model class name. Supported values are ``MACE``
            and ``ScaleShiftMACE``.
        interaction: Interaction class name from ``mace.modules.interaction_classes``.
        interaction_first: First interaction class name from the same registry.
        hidden_irreps: Hidden irreducible representations as a string.
        MLP_irreps: Readout irreducible representations as a string.
        atomic_numbers: Ordered atomic numbers used by the model.
        r_max: Neighbor-list cutoff in Angstrom.
        heads: Ordered MACE head names.
        default_head: Head used for generic input data without a head attribute.
        gate: Gate name from ``mace.modules.gate_dict``.
        readout: Readout class name from ``mace.modules.readout_classes``.
        model_config: Remaining official MACE constructor arguments, represented by
            standard Python scalars, lists, dictionaries, and strings.

    Notes:
        The wrapped model is reconstructed from trusted code and explicit configuration.
        ``state_dict`` and ``load_state_dict`` delegate to the MACE model so checkpoints
        contain only the original MACE parameter keys.
    """

    def __init__(
            self,
            interaction: str,
            interaction_first: str,
            hidden_irreps: str,
            MLP_irreps: str,
            atomic_numbers: Sequence[int],
            r_max: float,
            model_type: str = 'ScaleShiftMACE',
            heads: Sequence[str] | None = None,
            default_head: str | None = None,
            gate: str = 'None',
            readout: str = 'NonLinearReadoutBlock',
            edge_irreps: str | None = None,
            **model_config: Any,
    ) -> None:
        super().__init__()
        components = _require_mace()
        modules = components['modules']
        o3 = components['o3']

        model_classes = {
            'MACE': modules.MACE,
            'ScaleShiftMACE': modules.ScaleShiftMACE,
        }
        model_class = _resolve_registered_value(model_classes, model_type, 'model_type')
        interaction_class = _resolve_registered_value(
            modules.interaction_classes, interaction, 'interaction'
        )
        interaction_first_class = _resolve_registered_value(
            modules.interaction_classes, interaction_first, 'interaction_first'
        )
        gate_function = _resolve_registered_value(modules.gate_dict, gate, 'gate')
        readout_class = _resolve_registered_value(modules.readout_classes, readout, 'readout')

        normalized_atomic_numbers = [int(atomic_number) for atomic_number in atomic_numbers]
        normalized_heads = ['Default'] if heads is None else [str(head) for head in heads]
        if 'atomic_energies' in model_config:
            model_config['atomic_energies'] = np.asarray(model_config['atomic_energies'])
        if 'num_elements' in model_config:
            n_elements = int(model_config['num_elements'])
            if n_elements != len(normalized_atomic_numbers):
                raise ValueError(
                    f'`num_elements` is {n_elements}, but `atomic_numbers` contains '
                    f'{len(normalized_atomic_numbers)} elements.'
                )

        official_config = dict(model_config)
        official_config.update({
            'r_max': float(r_max),
            'interaction_cls': interaction_class,
            'interaction_cls_first': interaction_first_class,
            'hidden_irreps': o3.Irreps(hidden_irreps),
            'MLP_irreps': o3.Irreps(MLP_irreps),
            'atomic_numbers': normalized_atomic_numbers,
            'heads': normalized_heads,
            'gate': gate_function,
            'readout_cls': readout_class,
            'edge_irreps': o3.Irreps(edge_irreps) if edge_irreps is not None else None,
        })

        self.data_adapter = MACEDataAdapter(
            atomic_numbers=normalized_atomic_numbers,
            r_max=r_max,
            heads=normalized_heads,
            default_head=default_head,
        )
        self.mace_model = model_class(**official_config)

    def forward(self, data: Any, **forward_config: Any) -> Dict[str, Any]:
        mace_batch = self.data_adapter.to_batch(data)
        mace_input = mace_batch.to_dict()
        forward_config.setdefault('training', self.training)
        # MACE obtains forces with autograd even during evaluation, while BUCToolkit
        # deliberately evaluates validation batches inside a no-grad context.
        with th.enable_grad():
            return self.mace_model(mace_input, **forward_config)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.mace_model.state_dict(*args, **kwargs)

    def load_state_dict(
            self,
            state_dict: Mapping[str, Any],
            strict: bool = True,
            assign: bool = False,
    ) -> Any:
        return self.mace_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def parameters(self, recurse: bool = True):
        return self.mace_model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
        return self.mace_model.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )


__all__ = ['MACEDataAdapter', 'MACEWrapper']
