"""Uncertainty-aware and on-the-fly model wrappers.

The wrappers in this module keep the numerical ``Energy``/``Grad`` protocol
small.  Training options remain owned by the normal BUCToolkit input file and
``Trainer`` implementation.
"""

from __future__ import annotations

import copy
import json
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch as th

from BUCToolkit.BatchStructures import BatchStructures
from BUCToolkit.Preprocessing.load_files import ExtXyz2Feat, OUTCAR2Feat
from BUCToolkit.Preprocessing.preprocessing import CreatePygData
from BUCToolkit.utils._Element_info import ATOMIC_NUMBER
from BUCToolkit.utils.function_utils import _BaseWrapper, compare_tensors


ReferenceReader = Literal['BS', 'OUTCAR', 'extxyz'] | Callable[[str], BatchStructures]

__all__ = ['ModelWithUncertaintyWrapper', 'OnTheFlyModelWrapper']


class ModelWithUncertaintyWrapper(_BaseWrapper, ABC):
    """Base wrapper that adds an uncertainty contract to a model.

    Subclasses implement :meth:`uncertainty`; the wrapped model must return a
    mapping containing ``energy`` and ``forces``.  Forces use the same physical
    sign convention as the existing PyG wrappers, while :meth:`Grad` returns
    the mathematical energy gradient required by ``BatchMD``.
    """

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        if not callable(model):
            raise TypeError(f'`model` must be callable, but got {type(model).__name__}.')
        self._prediction_cache: tuple[Any, th.Tensor, th.Tensor, th.Tensor] | None = None
        self._raw_prediction: Mapping | None = None

    def _predict(self, X: th.Tensor, *args: Any, **kwargs: Any) -> tuple[th.Tensor, th.Tensor]:
        result = self._model(X, *args, **kwargs)
        if not isinstance(result, Mapping) or 'energy' not in result or 'forces' not in result:
            raise TypeError('The wrapped model must return a mapping with `energy` and `forces`.')
        energy = result['energy']
        forces = result['forces']
        self._raw_prediction = result
        if not isinstance(energy, th.Tensor) or not isinstance(forces, th.Tensor):
            raise TypeError('Model `energy` and `forces` outputs must be torch.Tensor objects.')
        return energy, forces

    @abstractmethod
    def uncertainty(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Return a scalar or per-structure uncertainty tensor."""

    def evaluate(self, X: th.Tensor, *args: Any, **kwargs: Any) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Evaluate energy, physical forces, and uncertainty once."""
        energy, forces = self._predict(X, *args, **kwargs)
        uncertainty = th.as_tensor(self.uncertainty(X, *args, **kwargs), device=energy.device)
        self._prediction_cache = (X, energy, forces, uncertainty)
        return energy, forces, uncertainty

    def Energy(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Evaluate and return energy while retaining forces for ``Grad``."""
        return self.evaluate(X, *args, **kwargs)[0]

    def Grad(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Return the mathematical energy gradient (negative physical force)."""
        if (self._prediction_cache is not None) and compare_tensors(self._prediction_cache[0], X):
            self.evaluate(X, *args, **kwargs)
        if self._prediction_cache is None: raise RuntimeError(f"BUG: prediction cache is None after evaluation.")
        forces = self._prediction_cache[2]
        self._prediction_cache = None
        return -forces.reshape_as(X).contiguous()

    def to(self, *args: Any, **kwargs: Any) -> 'ModelWithUncertaintyWrapper':
        super().to(*args, **kwargs)
        return self

    def eval(self) -> 'ModelWithUncertaintyWrapper':
        super().eval()
        return self


class OnTheFlyModelWrapper(_BaseWrapper):
    """Use a model until its uncertainty exceeds a threshold, then label data.

    Training configuration is read exclusively from ``input_file``.  At each
    trigger a fresh :class:`~BUCToolkit.api.Trainer` is created, supplied with a
    random reference-data sample and all persisted online samples, and run
    using the model's class and current parameters.
    """

    def __init__(
        self,
        model_wrapper: ModelWithUncertaintyWrapper,
        reference_calculator: Any,
        input_file: str,
        uncertainty_threshold: float,
        split_ratios: Sequence[float] | Mapping[str, float],
        limit_num_to_train: int,
        reference_data_path: str | None = None,
        reference_data_reader: ReferenceReader = 'BS',
        reference_data_sample_ratio: float = 0.0,
        data_dump_path: str | None = None,
        random_seed: int | None = None,
        data_builder: Callable[..., BatchStructures] | None = None,
    ) -> None:
        super().__init__(model_wrapper)
        if not isinstance(model_wrapper, ModelWithUncertaintyWrapper):
            raise TypeError('`model_wrapper` must inherit from ModelWithUncertaintyWrapper.')
        if not isinstance(input_file, str):
            raise TypeError('`input_file` must be a string path.')
        if not os.path.isfile(input_file):
            raise FileNotFoundError(input_file)
        if not np.isfinite(float(uncertainty_threshold)):
            raise ValueError('`uncertainty_threshold` must be finite.')
        if not isinstance(limit_num_to_train, int) or limit_num_to_train <= 0:
            raise ValueError('`limit_num_to_train` must be a positive integer.')
        if not 0.0 <= float(reference_data_sample_ratio) <= 1.0:
            raise ValueError('`reference_data_sample_ratio` must be in [0, 1].')
        self.model_wrapper = model_wrapper
        self.reference_calculator = reference_calculator
        self.input_file = input_file
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.limit_num_to_train = limit_num_to_train
        self.reference_data_path = reference_data_path
        self.reference_data_reader = reference_data_reader
        self.reference_data_sample_ratio = float(reference_data_sample_ratio)
        self.data_dump_path = os.path.abspath(data_dump_path) if data_dump_path is not None else None
        self._rng = random.Random(random_seed)
        self._data_builder = data_builder
        self._round = 1
        self._pending_count = 0
        self._validation_loss = float('inf')
        self._cached_result: tuple[Any, th.Tensor, th.Tensor] | None = None
        self._collected_splits = {name: BatchStructures() for name in self._split_names}
        self._reference_data = self._read_reference_data()
        self._split_names, self._split_ratios = self._normalize_splits(split_ratios)
        if self.data_dump_path is not None:
            os.makedirs(self.data_dump_path, exist_ok=True)
            self._load_persisted_rounds()

    @staticmethod
    def _normalize_splits(
            split_ratios: Sequence[float] | Mapping[str, float]
    ) -> tuple[list[str], list[float]]:
        """ Handle the input format for splitting data """
        if isinstance(split_ratios, Mapping):
            names = [str(name) for name in split_ratios]
            ratios = [float(split_ratios[name]) for name in split_ratios]
        else:
            names = ['train', 'valid', 'test'][:len(split_ratios)]
            ratios = [float(value) for value in split_ratios]
        if len(ratios) < 2 or len(ratios) > 3:
            raise ValueError('`split_ratios` must define two or three splits.')
        if any(value < 0.0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
            raise ValueError('`split_ratios` must be non-negative and sum to one.')
        if len(set(names)) != len(names) or names[0] != 'train' or names[1] not in {'valid', 'validation'}:
            raise ValueError('Split names must start with `train` and `valid`.')
        names[1] = 'valid'
        return names, ratios

    def _read_reference_data(self) -> BatchStructures | None:
        if self.reference_data_path is None:
            return None
        reader = self.reference_data_reader
        if callable(reader):
            data = reader(self.reference_data_path)
        elif reader == 'BS':
            data = BatchStructures.load_from_file(self.reference_data_path)
        elif reader == 'OUTCAR':
            data = OUTCAR2Feat(self.reference_data_path, verbose=0)
            data.read(n_core=1)
        elif reader == 'extxyz':
            data = ExtXyz2Feat(self.reference_data_path, verbose=0)
            data.read(n_core=1)
        else:
            raise ValueError('`reference_data_reader` must be BS, OUTCAR, extxyz, or callable.')
        if not isinstance(data, BatchStructures):
            raise TypeError('Reference reader must return BatchStructures.')
        return data

    def _load_persisted_rounds(self) -> None:
        """
        Load the previous on-the-fly data
        Returns:

        """
        round_dirs = sorted(Path(self.data_dump_path).glob('round[0-9]*'))
        if not round_dirs:
            return
        self._round = max(int(path.name[5:]) for path in round_dirs) + 1
        state_path = Path(self.data_dump_path) / 'state.json'
        if state_path.is_file():
            with state_path.open('r', encoding='utf-8') as state_file:
                state = json.load(state_file)
            self._round = int(state.get('round', self._round))
            self._pending_count = int(state.get('pending_count', 0))
            self._validation_loss = float(state.get('validation_loss', float('inf')))
        else:
            latest_round = round_dirs[-1]
            for split in self._split_names:
                split_path = latest_round / f'{split}.bs'
                if split_path.is_dir():
                    self._pending_count += len(BatchStructures.load_from_file(str(split_path)))

    def _save_state(self) -> None:
        if self.data_dump_path is None:
            return
        state_path = Path(self.data_dump_path) / 'state.json'
        with state_path.open('w', encoding='utf-8') as state_file:
            json.dump(
                {
                    'round': self._round,
                    'pending_count': self._pending_count,
                    'validation_loss': self._validation_loss,
                },
                state_file,
            )

    def _build_default_sample(self, X: th.Tensor, args: tuple[Any, ...], energy: th.Tensor, forces: th.Tensor) -> BatchStructures:
        graph = next((value for value in args if hasattr(value, 'atomic_numbers') and hasattr(value, 'pos')), None)
        if graph is None:
            raise ValueError('A `data_builder` is required when wrapper arguments do not contain a graph.')
        graphs = graph.to_data_list() if hasattr(graph, 'to_data_list') else [graph]
        positions = X.reshape(-1, 3).detach().cpu().numpy()
        sample = BatchStructures()
        offset = 0
        for index, item in enumerate(graphs):
            numbers = item.atomic_numbers.detach().cpu().numpy().astype(np.int64).tolist()
            n_atom = len(numbers)
            coords = positions[offset:offset + n_atom].astype(np.float32)
            symbols = [ATOMIC_NUMBER[int(number)] for number in numbers]
            elements, counts = [], []
            for symbol in symbols:
                if len(elements) > 0 and elements[-1] == symbol:
                    counts[-1] += 1
                else:
                    elements.append(symbol)
                    counts.append(1)
            cell = getattr(item, 'cell', np.eye(3, dtype=np.float32))
            if isinstance(cell, th.Tensor):
                cell = cell.detach().cpu().numpy()
            cell = np.asarray(cell, dtype=np.float32).reshape(3, 3)
            sample._Sample_ids.append(f'otf_{self._round}_{self._pending_count}_{index}')
            sample.Cells.append(cell)
            sample.Coords_type.append('C')
            sample.Coords.append(coords)
            sample.Fixed.append(np.ones_like(coords, dtype=np.int8))
            sample.Elements.append(elements)
            sample.Numbers.append(counts)
            sample.Energies = [] if sample.Energies is None else sample.Energies
            sample.Forces = [] if sample.Forces is None else sample.Forces
            energy_value = energy.reshape(-1)[min(index, energy.numel() - 1)].detach().cpu().item()
            force_value = forces[offset:offset + n_atom].detach().cpu().numpy()
            sample.Energies.append(energy_value)
            sample.Forces.append(force_value)
            offset += n_atom
        return sample

    def _persist_sample(self, sample: BatchStructures, split: str) -> None:
        self._collected_splits[split].append(sample, strict=False)
        if self.data_dump_path is None:
            return
        split_path = Path(self.data_dump_path) / f'round{self._round}' / f'{split}.bs'
        split_path.parent.mkdir(parents=True, exist_ok=True)
        sample.save(str(split_path), mode='a' if split_path.exists() else 'w')

    def _collect(self, X: th.Tensor, args: tuple[Any, ...], energy: th.Tensor, forces: th.Tensor) -> None:
        sample = self._data_builder(X, *args, energy, forces) if self._data_builder is not None else self._build_default_sample(X, args, energy, forces)
        # random split by given ratio
        value = self._rng.random()
        cumulative = 0.0
        split = self._split_names[-1]  # as the backup
        for name, ratio in zip(self._split_names, self._split_ratios):
            cumulative += ratio
            if value < cumulative:
                split = name

        self._persist_sample(sample, split)
        self._pending_count += len(sample)
        self._save_state()

    def _all_collected_data(self, split: str | None = None) -> BatchStructures:
        result = BatchStructures()
        if self.data_dump_path is None:
            return self._collected_splits[split] if split else self._merge_in_memory_splits()
        for round_dir in sorted(Path(self.data_dump_path).glob('round[0-9]*')):
            paths = [round_dir / f'{split}.bs'] if split else [round_dir / f'{name}.bs' for name in self._split_names]
            for path in paths:
                if path.is_dir():
                    result.append(BatchStructures.load_from_file(str(path)), strict=False)
        return result

    def _merge_in_memory_splits(self) -> BatchStructures:
        result = BatchStructures()
        for data in self._collected_splits.values():
            if len(data) > 0:
                result.append(data, strict=False)
        return result

    def _training_data(self) -> tuple[dict[str, Any], dict[str, Any]]:
        collected_train = self._all_collected_data('train')
        collected_valid = self._all_collected_data('valid')
        if self._reference_data is not None and self.reference_data_sample_ratio > 0.0:
            n_sample = int(round(len(self._reference_data) * self.reference_data_sample_ratio))
            n_sample = min(len(self._reference_data), max(1, n_sample))
            indices = self._rng.sample(range(len(self._reference_data)), n_sample)
            reference = self._reference_data[indices]
            collected_train.append(reference, strict=False)
            if len(collected_valid) == 0:
                collected_valid.append(reference, strict=False)
        if len(collected_train) == 0:
            raise RuntimeError('No collected or reference samples are available for training.')
        if len(collected_valid) == 0:
            collected_valid.append(collected_train, strict=False)
        converter = CreatePygData(0)
        train_list = converter.feat2data_list(collected_train, n_core=1)
        valid_list = converter.feat2data_list(collected_valid, n_core=1)
        train_labels = {
            'energy': [value for value in collected_train.Energies],
            'forces': collected_train.Forces,
        }
        valid_labels = {
            'energy': [value for value in collected_valid.Energies],
            'forces': collected_valid.Forces,
        }
        return {'data': train_list, 'labels': train_labels}, {'data': valid_list, 'labels': valid_labels}

    def _train_if_ready(self) -> None:
        if self._pending_count < self.limit_num_to_train:
            return
        train_data, valid_data = self._training_data()
        old_state = copy.deepcopy(self.model_wrapper._model.state_dict())
        # Import lazily: ``DataLoaders`` imports the model-wrapper package for
        # optional MACE support, so importing Trainer at module import time
        # would create a package-initialization cycle.
        from BUCToolkit.api.DataLoaders import PyGDataLoader
        from BUCToolkit.api.Trainer import Trainer
        trainer = Trainer(self.input_file)
        trainer.START = 'from_scratch'
        trainer.set_model_param(old_state)
        trainer.set_dataset(train_data, valid_data)
        trainer.set_dataloader(PyGDataLoader, {'shuffle': True})
        trainer.SAVE_CHK = True
        trainer.CHK_SAVE_POSTFIX = f'{trainer.CHK_SAVE_POSTFIX}_otf_round{self._round}'
        model_class = self.model_wrapper._model.__class__
        trainer.train(model_class)
        checkpoint = Path(trainer.CHK_SAVE_PATH) / f'best_checkpoint{trainer.CHK_SAVE_POSTFIX}.pt'
        if not checkpoint.is_file():
            self._pending_count = 0
            self._round += 1
            self._save_state()
            return
        state = th.load(str(checkpoint), weights_only=True)
        candidate_loss = state.get('val_loss', th.inf)
        if isinstance(candidate_loss, (tuple, list)):
            candidate_loss = candidate_loss[0]
        candidate_loss = float(candidate_loss)
        previous_loss = self._validation_loss
        if np.isfinite(candidate_loss) and candidate_loss < previous_loss:
            self.model_wrapper._model.load_state_dict(state['model_state_dict'])
            self._validation_loss = candidate_loss
        self._pending_count = 0
        self._round += 1
        self._save_state()

    def _evaluate_reference(self, X: th.Tensor, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[th.Tensor, th.Tensor]:
        calculator = self.reference_calculator
        if hasattr(calculator, 'evaluate'):
            energy, forces, _ = calculator.evaluate(X, *args, **kwargs)
            return energy, forces
        if hasattr(calculator, 'Energy') and hasattr(calculator, 'Grad'):
            energy = calculator.Energy(X, *args, **kwargs)
            gradient = calculator.Grad(X, *args, **kwargs)
            return energy, -gradient
        result = calculator(X, *args, **kwargs)
        if not isinstance(result, Mapping) or 'energy' not in result or 'forces' not in result:
            raise TypeError('`reference_calculator` must expose evaluate, Energy/Grad, or return energy/forces.')
        return result['energy'], result['forces']

    def evaluate(self, X: th.Tensor, *args: Any, **kwargs: Any) -> tuple[th.Tensor, th.Tensor]:
        energy, forces, uncertainty = self.model_wrapper.evaluate(X, *args, **kwargs)
        use_reference = bool(th.any(uncertainty > self.uncertainty_threshold).item())
        if use_reference:
            energy, forces = self._evaluate_reference(X, args, kwargs)
            self._collect(X, args, energy, forces)
            self._train_if_ready()
        self._cached_result = (X, energy, forces)
        return energy, forces

    def Energy(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Return model or reference energy according to uncertainty."""
        return self.evaluate(X, *args, **kwargs)[0]

    def Grad(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Return the mathematical gradient for the selected calculator."""
        if self._cached_result is None or not compare_tensors(self._cached_result[0], X):
            self.evaluate(X, *args, **kwargs)
        assert self._cached_result is not None
        forces = self._cached_result[2]
        self._cached_result = None
        return -forces.reshape_as(X).contiguous()

    def to(self, *args: Any, **kwargs: Any) -> 'OnTheFlyModelWrapper':
        self.model_wrapper.to(*args, **kwargs)
        return self

    def eval(self) -> 'OnTheFlyModelWrapper':
        self.model_wrapper.eval()
        return self
