#  Copyright (c) 2026.4.24, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: BaseMotion.py
#  Environment: Python 3.12
import logging
import os
import sys
import warnings
from typing import Any, Tuple, Dict, Callable, List, Set
from itertools import accumulate

import torch as th
import numpy as np

from BUCToolkit.BatchStructures.StructuresIO import structures_io_dumper
from BUCToolkit.utils.setup_loggers import has_any_handler, clear_all_handlers
from BUCToolkit.utils._print_formatter import SCIENTIFIC_ARRAY_FORMAT, STRING_ARRAY_FORMAT, FLOAT_ARRAY_FORMAT
from BUCToolkit.utils._Element_info import DTYPE
from BUCToolkit.Bases.StdContainer import StdContainer

FLOAT_TYPE = os.environ.get('BT_FLOAT_TYPE', 'float32')
FLOAT_TYPE = DTYPE.get(FLOAT_TYPE, th.float32)


class BaseIO:
    """
    Base class for I/O operations.
    """
    def __init__(self, output_file: str|None = None) -> None:
        self.logger = None
        self.log_handler = None
        self.output_file = str(output_file) if output_file is not None else None
        self.dumper = structures_io_dumper(
            path=self.output_file,
            mode='x',
        )  # as the default. One can use `reset

        self._dump_vars: Dict[str, th.Tensor | None] = {}
        self._log_vars: Dict[str, th.Tensor | None] = {}
        self._init_dump_vars: Dict[str, th.Tensor | None] = {}
        self._init_log_vars: Dict[str, th.Tensor | None] = {}
        self._transfer_vars: Set[str] = set()
        self._assigned_vars = None
        self._is_assigned = False

    def init_logger(self, logger_name: str):
        # logging
        if self.logger is None:
            # cut off propagation
            supreme_name = logger_name.split('.')[0]
            top_logger = logging.getLogger(supreme_name)
            top_logger.propagate = False
            # set true logger
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            if not has_any_handler(top_logger):
                self.log_handler = logging.StreamHandler(sys.stdout)
                self.log_handler.setLevel(logging.INFO)
                self.log_handler.setFormatter(formatter)
                top_logger.addHandler(self.log_handler)
        else:
            warnings.warn('Logger has already initialized. Nothing will be done.', RuntimeWarning)

    def reset_logger_handler(self, handler: str|logging.StreamHandler|logging.FileHandler, level: int = logging.INFO):
        """
        Clear all logging handlers including current logger and its ancestors, and reset one.
        Args:
            handler: the new handler.
            level: the logger handler output level.

        Returns:

        """
        clear_all_handlers(self.logger)
        # redirect to supreme logger
        self.logger.setLevel(level)
        top_logger = self.logger
        while top_logger.parent and top_logger.propagate:
            top_logger = top_logger.parent
            top_logger.setLevel(level)

        top_logger.setLevel(level)

        formatter = logging.Formatter('%(message)s')
        if isinstance(handler, logging.Handler):
            self.log_handler = handler
        elif isinstance(handler, str):
            output_path = os.path.dirname(handler)
            # check whether path exists
            if not os.path.isdir(output_path): os.makedirs(output_path)
            # set log handler
            self.log_handler = logging.FileHandler(handler, 'w', delay=True)
        else:
            raise TypeError("handler must be a string path or a logging.Handler instance")

        self.log_handler.setLevel(level)
        self.log_handler.setFormatter(formatter)
        top_logger.addHandler(self.log_handler)

    def reset_dumper(self, dumper: Any) -> None:
        """
        Reset a new dumper and delete the old one.
        Args:
            dumper: A dumper object that satisfied the protocol as `_ArrayDumperPlaceHolder` class.

        Returns:

        """
        if self.output_file is not None:
            self.dumper.close()
            del self.dumper
            self.dumper = dumper
        else:
            self.logger.error(
                "ERROR: No output file specified. Hence, resetting dumper is meaningless.\n"
                "'reset_dumper': Operation REFUSED."
            )

    # ---- dynamic dump / transition hooks (universal across MD, MC, etc.) ----
    def register_dump_vars(self, *args: str):
        """Register state attribute names for binary trajectory output.

        Args:
            *args: Attribute names that will be read from the live
                :class:`StdContainer` and written in the same order at every
                output step.

        Returns:
            None. The active ordered mapping is available through
            :meth:`get_dump_vars`; the union needed for CPU/CUDA transfers is
            rebuilt in ``self._transfer_vars``.

        Raises:
            RuntimeError: If transfer variables have already been assigned for
                the active run. Registration must be completed before buffer
                allocation and iteration start.
        """
        if self._is_assigned: raise RuntimeError("Cannot register any variable after assignment.")
        for _name in args:
            self._dump_vars.setdefault(str(_name), None)
        self._transfer_vars = set(self._dump_vars) | set(self._log_vars)

    def get_dump_vars(self) -> Dict[str, th.Tensor | None]:
        """Return the ordered active dump-variable mapping.

        Returns:
            Mapping from names to late-bound tensors or ``None``. Key
            iteration gives the binary-column order.
        """
        return self._dump_vars

    def register_log_vars(self, *args: str):
        """Register state attribute names for per-step logging.

        Args:
            *args: Attribute names copied from the live
                :class:`StdContainer` and passed to the asynchronous log
                consumer in the same order.

        Returns:
            None. The active ordered mapping is available through
            :meth:`get_log_vars`; the union needed for CPU/CUDA transfers is
            rebuilt in ``self._transfer_vars``.

        Raises:
            RuntimeError: If transfer variables have already been assigned for
                the active run.
        """
        if self._is_assigned: raise RuntimeError("Cannot register any variable after assignment.")
        for _name in args:
            self._log_vars.setdefault(str(_name), None)
        self._transfer_vars = set(self._dump_vars) | set(self._log_vars)

    def get_log_vars(self) -> Dict[str, th.Tensor | None]:
        """Return the active variables selected for per-step logging.

        Dictionary insertion order is the queue and display order used by the
        asynchronous log consumer. Values remain ``None`` until late-bound
        quantities are registered or assigned for the current run.

        Returns:
            Ordered mapping from log names to their registered tensors or
            ``None`` placeholders.
        """
        return self._log_vars

    def get_transfer_vars(self) -> Set[str]:
        """Return the unique names requiring a CPU/CUDA snapshot transfer.

        Returns:
            Set union of :meth:`get_dump_vars` and :meth:`get_log_vars`.
        """
        return self._transfer_vars

    def assign_transfer_vars(self, **kwargs) -> Dict[str, Any]:
        """Bind concrete values to every active transfer name.

        Args:
            **kwargs: Mapping from registered dump/log names to their live
                values. Extra keys are ignored; every active transfer name must
                be present.

        Returns:
            Dictionary restricted to the active transfer names.

        Raises:
            KeyError: If any registered transfer name is absent from ``kwargs``.
        """
        try:
            self._assigned_vars = {_k:kwargs[_k] for _k in self._transfer_vars}
        except KeyError as ke:
            raise KeyError('Some registered variables are not assigned any value.') from ke
        self._is_assigned = True

        return self._assigned_vars

    def purge_register_vars(self):
        """Clear active registration and assignment state.

        The immutable constructor selections and persistent extra-variable
        definitions are intentionally preserved, allowing
        :meth:`reset_register_vars` to rebuild the same ordered configuration
        for a later :meth:`run` call.

        Returns:
            None.
        """
        self._dump_vars = {}
        self._log_vars = {}
        self._transfer_vars = set()
        self._assigned_vars = None
        self._is_assigned = False

    # ------------------------------------------------------------------
    # Shared setup / reset logic (used by _BaseMD, _BaseOpt, _BaseMC, …)
    # ------------------------------------------------------------------

    def _setup_register_vars(
            self,
            dump_quantities: Tuple[str, ...] | List[str],
            log_quantities: Tuple[str, ...] | List[str],
    ):
        """Validate and store the constructor-level quantity selections.

        Args:
            dump_quantities: Ordered names requested as binary trajectory
                columns.
            log_quantities: Ordered names requested as per-step log fields.

        Returns:
            None. Ordered base dictionaries are stored in ``_init_dump_vars``
            and ``_init_log_vars``, then active registrations are built by
            :meth:`reset_register_vars`.

        Raises:
            ValueError: If the subclass defines ``ALLOWED_QUANTITIES`` and a
                requested name is not in that set.
        """
        _allowed = getattr(self, 'ALLOWED_QUANTITIES', None)
        if _allowed is not None:
            for _name, _val in (('dump_quantities', dump_quantities),
                                ('log_quantities',  log_quantities)):
                _unknown = set(_val) - _allowed
                if _unknown:
                    raise ValueError(
                        f'{_name} contains unknown names {_unknown!r}. '
                        f'Allowed: {sorted(_allowed)}'
                    )
        self._init_dump_vars = dict.fromkeys(
            (str(_name) for _name in dump_quantities), None
        )
        self._init_log_vars = dict.fromkeys(
            (str(_name) for _name in log_quantities), None
        )
        self.reset_register_vars()

    def reset_register_vars(self):
        """Restore active registrations from the persistent dictionaries.

        Safe to call at the start of each ``run()`` — only the active
        registries are purged; registered names and values survive.

        Returns:
            None. Dictionary insertion order preserves the requested column and
            log-field order while repeated keys update without duplication.
        """
        self.purge_register_vars()
        self._dump_vars = self._init_dump_vars.copy()
        self._log_vars = self._init_log_vars.copy()
        self._transfer_vars = set(self._dump_vars) | set(self._log_vars)

    def register_extra_print_vars(self, **kwargs: th.Tensor):
        """Register extra per-step log quantities with their initial Tensors.

        Values are stored directly in the persistent and active ordered log
        registries. Re-registering a name replaces its value without changing
        its original position.

        Args:
            **kwargs: Mapping from each extra quantity name to a prototype/live
                tensor defining its initial shape, dtype, and device.

        Returns:
            None.

        Raises:
            TypeError: If an extra value is not a :class:`torch.Tensor`.
            RuntimeError: If transfer variables have already been assigned for
                the active run.
        """
        if self._is_assigned:
            raise RuntimeError("Cannot register any variable after assignment.")
        for _k, _v in kwargs.items():
            _k = str(_k)
            if not isinstance(_v, th.Tensor):
                raise TypeError(
                    f'register_extra_print_vars: {_k!r} must be a Tensor, '
                    f'got {type(_v).__name__}'
                )
            self._init_log_vars[_k] = _v
            self._log_vars[_k] = _v
            if _k in self._init_dump_vars:
                self._init_dump_vars[_k] = _v
                self._dump_vars[_k] = _v
        self._transfer_vars = set(self._dump_vars) | set(self._log_vars)

    def register_extra_dump_vars(self, **kwargs: th.Tensor):
        """Register extra per-step dump quantities with their initial Tensors.

        Mirror of :meth:`register_extra_print_vars` for the binary dump side.
        A var already registered for print may also be registered here.

        Args:
            **kwargs: Mapping from each extra quantity name to a prototype/live
                tensor defining its initial shape, dtype, and device.

        Returns:
            None.

        Raises:
            TypeError: If an extra value is not a :class:`torch.Tensor`.
            RuntimeError: If transfer variables have already been assigned for
                the active run.
        """
        if self._is_assigned:
            raise RuntimeError("Cannot register any variable after assignment.")
        for _k, _v in kwargs.items():
            _k = str(_k)
            if not isinstance(_v, th.Tensor):
                raise TypeError(
                    f'register_extra_dump_vars: {_k!r} must be a Tensor, '
                    f'got {type(_v).__name__}'
                )
            self._init_dump_vars[_k] = _v
            self._dump_vars[_k] = _v
            if _k in self._init_log_vars:
                self._init_log_vars[_k] = _v
                self._log_vars[_k] = _v
        self._transfer_vars = set(self._dump_vars) | set(self._log_vars)

    def set_registered_var_values(self, container: StdContainer) -> None:
        """Copy late-bound registered tensors into one motion-state container.

        Constructor-selected quantities initially use ``None`` placeholders,
        while constraint and other extension quantities may already carry a
        tensor prototype. Only concrete registered values are assigned; normal
        state fields created by the caller remain unchanged.

        Args:
            container: Active :class:`StdContainer` that will be snapshotted by
                the dump and log consumers.

        Returns:
            None. Concrete values from the dump and log registries are assigned
            to same-named attributes of ``container``.
        """
        for _variables in (self._dump_vars, self._log_vars):
            for _name, _value in _variables.items():
                if _value is not None:
                    setattr(container, _name, _value)

    # ------------------------------------------------------------------

    @staticmethod
    def _allocate_cpu_buffers(
            s: StdContainer,
            buffer_names: List[str] | Tuple[str, ...] | Set[str],
            device: th.device,
            require_buffer: bool = True,
    ):
        """Allocate dtype-preserving snapshot buffers for registered fields.

        Args:
            s: Live :class:`StdContainer` whose named tensor attributes define
                the required shapes and dtypes.
            buffer_names: Names to mirror. Every name must identify a tensor
                attribute of ``s``.
            device: Device used for the optional staging container.
            require_buffer: If ``True``, also allocate ``s_buf`` on ``device``
                for protected device-to-device then device-to-host copies. If
                ``False``, return ``None`` for ``s_buf``.

        Returns:
            Tuple ``(s_cpu, s_buf)``. ``s_cpu`` contains pinned CPU tensors;
            ``s_buf`` contains same-shaped tensors on ``device`` or is ``None``.
            Each field preserves the source tensor's dtype, including boolean
            and integer quantities.
        """
        s_cpu = StdContainer()
        s_buf = StdContainer() if require_buffer else None
        for name in buffer_names:
            ref = getattr(s, name)
            # Preserve the registered tensor dtype. The former global
            # FLOAT_TYPE coercion happened to work for coordinates/energies,
            # but corrupted boolean or integer extension quantities such as
            # Monte-Carlo acceptance masks.
            setattr(
                s_cpu,
                name,
                th.empty_like(
                    ref,
                    device='cpu',
                    pin_memory=(device.type == 'cuda'),
                ),
            )
            if require_buffer:
                setattr(s_buf, name, th.empty_like(ref, device=device))
        return s_cpu, s_buf

    @staticmethod
    def _transfer_buffers_D2H(
            s: StdContainer,
            s_buf: StdContainer | None,
            s_cpu: StdContainer,
            trans_names: List[str] | Tuple[str, ...],
            copy_stream: th.cuda.Stream,
            device: th.device,
    ):
        """Start a protected device-to-host snapshot for registered tensors.

        The normal CUDA path is ``s --D2D--> s_buf --D2H--> s_cpu``. The D2D
        staging copy freezes the live values before the simulation mutates
        them again. If ``s_buf`` is ``None``, live tensors are copied directly
        to ``s_cpu`` on ``copy_stream``.

        Args:
            s: Live source :class:`StdContainer` on ``device``.
            s_buf: Optional device staging container with matching attributes.
            s_cpu: Pinned CPU destination container with matching attributes.
            trans_names: Names transferred during this snapshot.
            copy_stream: CUDA stream used for nonblocking D2H copies.
            device: CUDA device whose default stream performs the D2D staging
                copies and must be synchronized with ``copy_stream``.

        Returns:
            None. Copies are enqueued asynchronously; the caller must record
            and synchronize a CUDA event before CPU consumers access ``s_cpu``.
        """
        if s_buf is None:
            buf_list = [getattr(s, name) for name in trans_names]
        else:
            buf_list = [getattr(s_buf, name).copy_(getattr(s, name)) for name in trans_names]
        with th.cuda.stream(copy_stream):
            copy_stream.wait_stream(th.cuda.default_stream(device))
            for i, name in enumerate(trans_names):
                getattr(s_cpu, name).copy_(buf_list[i], non_blocking=True)


class BaseMotion(BaseIO):
    """
    Base class for all processes that atoms move & evolution.
    """

    def __init__(self, output_file: str|None = None):
        super().__init__(output_file)

    @staticmethod
    def handle_motion_mask(
            X,
            fixed_atom_tensor,
    ):
        """
        normalize format of atom_masks
        Returns: the standardized atom_masks

        """
        if fixed_atom_tensor is None:
            fixed_atom_tensor = th.ones_like(X, device=X.device)
        else:
            fixed_atom_tensor = fixed_atom_tensor.broadcast_to(X.shape)
        if fixed_atom_tensor.shape == X.shape:
            atom_masks = fixed_atom_tensor.to(X.device)
        else:
            raise RuntimeError(f'The shape of fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not match X (shape: {X.shape}).')

        return atom_masks

    @staticmethod
    def handle_grad_func(
            grad_func: Callable[[th.Tensor, Any, ...], th.Tensor] | None,
            is_grad_func_contain_y: bool,
            require_grad: bool,
            **kwargs
    ):
        """

        Returns: grad_func_, require_grad, is_grad_func_contain_y

        """
        if grad_func is None:
            is_grad_func_contain_y = True
            require_grad = True
            def grad_func_(x, y, grad_shape=None):
                if grad_shape is None:
                    grad_shape = th.ones_like(y)
                _g = th.autograd.grad(y, x, grad_shape)
                return _g[0]
        else:
            grad_func_ = grad_func

        return grad_func_, require_grad, is_grad_func_contain_y

    @staticmethod
    def handle_dtype_device(dtype, device, *tensors: th.Tensor|Any):
        """
        Move all input tensors to the given dtype and device.

        Args:
            dtype: torch.dtype or None that means keep the input tensor dtypes
            device: torch.device, str, or None that means no device changes
            *tensors: input tensors

        Returns:
            Tuple[th.Tensor]: transformed tensors in the same order
        """
        if isinstance(device, str): device = th.device(device)
        if (dtype is None) and (device is None):
            return tensors
        elif device is None:
            # Keep original device
            out_tensors = tuple(ten.to(dtype) if hasattr(ten, 'to') else ten for ten in tensors)
        elif dtype is None:
            out_tensors = tuple(ten.to(device) if hasattr(ten, 'to') else ten for ten in tensors )
        else:
            out_tensors = tuple(ten.to(device=device, dtype=dtype) if hasattr(ten, 'to') else ten for ten in tensors)
        return out_tensors

    @staticmethod
    def handle_batch_indices(
            batch_indices,
            n_batch,
            device
    ):
        r"""
        Calculating `n_true_batch`, `batch_tensor`, `batch_scatter`, and `batch_slice_indx` from input batch_indices.

        Args:
            batch_indices: input batch_indices that each element is the atom number of each sample: [n_0, n_1, ..., n_N].
            n_batch: the length of 1st dimension of X.
            device: torch device

        Returns: n_true_batch, batch_indices, batch_tensor, batch_scatter, batch_slice_indx;
            n_true_batch: the true batch size
            batch_indices: the batch_indices in List format
            batch_tensor: the batch_indices in torch.Tensor format
            batch_scatter: the batch indices in the form of Tensor[0, 0, 0, ..., 1, 1, ..., N - 1]
            batch_slice_indx: the batch indices in the form of ptr List[0, n_0, n_0 + n_1, ..., \sum n_i]
        """
        if batch_indices is not None:
            if n_batch != 1:
                raise RuntimeError(f'If batch_indices was specified, the 1st dimension of X must be 1 instead of {n_batch}.')
            if isinstance(batch_indices, (th.Tensor, np.ndarray)):
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, (List, Tuple)):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'
            batch_slice_indx = [0] + list(accumulate(batch_indices))  # convert n_atom of each batch into split point of each batch
            batch_tensor = th.as_tensor(batch_indices, device=device)  # the tensor version of batch_indices which is a List.
            batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=device),
                batch_tensor,
                dim=0
            )  # scatter mask of the int tensor with the same shape as X.shape[1], which the data in one batch have one index.
            n_true_batch = len(batch_indices)  # the true batch size for irregular batches
        else:
            n_true_batch = n_batch
            batch_tensor = None
            batch_scatter = None
            batch_slice_indx = None

        return n_true_batch, batch_indices, batch_tensor, batch_scatter, batch_slice_indx

    @staticmethod
    def handle_arrays_print(
            logger: logging.Logger | Any,
            batch_indices: List[int] | None,
            batch_slice_indx: List[int],
            arrays: List[List[th.Tensor]] | Tuple[Tuple[th.Tensor, ...]],
            array_names: List[List[str]] | Tuple[Tuple[str, ...]],
            verbose: int,
            force=False
    ):
        """
        Logging function for printing arrays with corresponding names controlled by the verbosity level.
        Args:
            logger: logger object
            batch_indices: input batch_indices that each element is the atom number of each sample
            batch_slice_indx: the batch_indices in ptr slice format
            arrays: input arrays. Format: [[tensors11, tensors12, ...], [tensors21, ...], ...],
                the i-th List in the outer list corresponds to the i-th verbosity level to log,
                and the tensors in the inner list will be all logged. With an
                irregular batch, only arrays whose second dimension spans all
                atoms are split; batch-leading and other arrays remain intact.
            array_names: input arrays names. Format: [[name11, name12, ...], [name21, ...], ...],
            verbose: verbosity level
            force: whether to use Tensor.numpy(force=True) when printing. If True, data will be copied once.

        Returns:

        """
        if len(arrays) != len(array_names):
            raise ValueError(f'arrays and array_names must have the same length, but got {len(arrays)} and {len(array_names)}.')
        if batch_indices is not None:
            for v_lev in range(len(arrays)):
                if verbose > v_lev + 1:
                    for na, arr in enumerate(arrays[v_lev]):
                        X_np = arr.numpy(force=force)
                        # Check whether the arr satisfy the alignment of batch_indices.
                        #   only the 2nd dim that has the same length of `batch_scatter` would be split.
                        if (
                                arr.ndim > 1
                                and arr.shape[0] != len(batch_indices)
                                and arr.shape[1] == batch_slice_indx[-1]
                        ):
                            X_tup = np.split(
                                X_np, batch_slice_indx[1:-1], axis=1
                            )
                        else:
                            X_tup = (X_np,)
                        logger.info(f" {array_names[v_lev][na]}:\n")
                        X_str = [
                            np.array2string(xi, **FLOAT_ARRAY_FORMAT).translate(str.maketrans('[]', '  '))
                            for xi in X_tup
                        ]
                        for x_str in X_str: logger.info(f'{x_str}\n')
                else:
                    break
        else:
            for v_lev in range(len(arrays)):
                if verbose > v_lev + 1:
                    for na, arr in enumerate(arrays[v_lev]):
                        X_tup = (arr.numpy(force=force),)
                        logger.info(f" {array_names[v_lev][na]}:\n")
                        X_str = [
                            np.array2string(xi, **FLOAT_ARRAY_FORMAT).translate(str.maketrans('[]', '  '))
                            for xi in X_tup
                        ]
                        for x_str in X_str: logger.info(f'{x_str}\n')
                else:
                    break

    def _calc_EF(
            self,
            X: th.Tensor,
            func: Callable[[th.Tensor, Any, ...], th.Tensor],
            func_args: Tuple,
            func_kwargs: Dict,
            grad_func_: Callable[[th.Tensor, Any, ...], th.Tensor],
            grad_func_args: Tuple,
            grad_func_kwargs: Dict,
            require_grad: bool,
            is_grad_func_contain_y: bool,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Calculate the energy and forces. In fact, it is function value y and its NEGATIVE gradient -g.

        Returns: Tuple[th.Tensor, th.Tensor], energy and forces.

        """
        y, g = self._calc_y_grad(
            X,
            func,
            func_args,
            func_kwargs,
            grad_func_,
            grad_func_args,
            grad_func_kwargs,
            require_grad,
            is_grad_func_contain_y,
        )

        return y, g.neg_()

    #@th.compiler.disable
    def _calc_y_grad(
            self,
            X: th.Tensor,
            func: Callable[[th.Tensor, Any, ...], th.Tensor],
            func_args: Tuple,
            func_kwargs: Dict,
            grad_func_: Callable[[th.Tensor, Any, ...], th.Tensor],
            grad_func_args: Tuple,
            grad_func_kwargs: Dict,
            require_grad: bool,
            is_grad_func_contain_y: bool,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Calculate the function value y and the corresponding gradient of the y.

        Returns: Tuple[th.Tensor, th.Tensor], y and the gradient of the y

        """
        with th.set_grad_enabled(require_grad):
            X.requires_grad_(require_grad)
            y = func(X, *func_args, **func_kwargs)
            if is_grad_func_contain_y:
                g = grad_func_(X, y, *grad_func_args, **grad_func_kwargs)
            else:
                g = grad_func_(X, *grad_func_args, **grad_func_kwargs)
        return y, g
