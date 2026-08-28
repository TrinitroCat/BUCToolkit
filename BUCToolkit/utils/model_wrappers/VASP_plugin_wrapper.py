"""BUCToolkit-controlled VASP calculator using the built-in Python plugin.

BUCToolkit remains the controller: its numerical core requests one structure,
then blocks until the long-lived VASP process reports the matching energy and
forces.  The generated plugin contains only the small callback-side bridge so
VASP's embedded Python does not need to import the full BUCToolkit package.
"""

from __future__ import annotations

import base64
import os
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from multiprocessing.connection import Connection, Listener
from typing import Any, Callable

import numpy as np
import torch as th

from BUCToolkit.utils.function_utils import _BaseWrapper


class ConnectionTransport:
    """Transport VASP plugin requests and results over one authenticated connection.

    This class is intentionally independent of wrapper state and numerical
    semantics.  A future mmap transport can implement this same interface
    without changing ``VASP_PluginModel`` or the request/result protocol.

    Args:
        use_network: Use TCP for a potentially remote Slurm job. Otherwise use
            a local Unix-domain socket.
        host: Controller host advertised to a remote plugin. Defaults to the
            fully qualified local hostname.
        socket_dir: Directory for a local Unix-domain socket.
    """

    def __init__(
            self,
            *,
            use_network: bool = False,
            host: str | None = None,
            socket_dir: str | None = None,
    ) -> None:
        self.use_network = bool(use_network)
        self.host = host or socket.getfqdn()
        self.socket_dir = socket_dir or "/tmp"
        self.listener: Listener | None = None
        self.connection: Connection | None = None
        self.family: str | None = None
        self.address: str | tuple[str, int] | None = None
        self.authkey: bytes | None = None
        self._socket_path: str | None = None

    def start(self) -> None:
        """Create the controller-side listener before VASP is launched.

        Returns:
            None. Connection settings are exposed by
            :meth:`plugin_environment` after initialization.
        """
        self.authkey = os.urandom(32)
        if self.use_network:
            self.family = "AF_INET"
            # The listening interface and the address visible from a compute
            # node are different concepts on multi-homed cluster controllers.
            self.listener = Listener(("0.0.0.0", 0), family=self.family, authkey=self.authkey)
            self.address = (self.host, self.listener.address[1])
        else:
            self.family = "AF_UNIX"
            self._socket_path = os.path.join(
                self.socket_dir,
                f"buctoolkit-vasp-{uuid.uuid4().hex}.sock",
            )
            self.listener = Listener(self._socket_path, family=self.family, authkey=self.authkey)
            self.address = self._socket_path

    def plugin_environment(self) -> dict[str, str]:
        """Return connection settings for the generated plugin.

        Returns:
            Environment variables containing the family, address, and encoded
            authentication key.
        """
        if self.listener is None or self.address is None or self.authkey is None:
            raise RuntimeError("ConnectionTransport.start() must be called first.")
        if isinstance(self.address, tuple):
            address = f"{self.address[0]}\0{self.address[1]}"
        else:
            address = self.address
        return {
            "BUCTOOLKIT_VASP_PLUGIN_FAMILY": self.family or "",
            "BUCTOOLKIT_VASP_PLUGIN_ADDRESS": address,
            "BUCTOOLKIT_VASP_PLUGIN_AUTHKEY": base64.b64encode(self.authkey).decode("ascii"),
        }

    def accept(
            self,
            timeout: float | None,
            is_alive: Callable[[], bool] | None = None,
    ) -> Connection:
        """Accept the plugin connection while observing the VASP process.

        Args:
            timeout: Maximum seconds to wait, or ``None`` for no timeout.
            is_alive: Optional process-liveness callback checked while waiting.

        Returns:
            The accepted plugin connection.

        Notes:
            ``Listener.accept`` has no timeout or child-process awareness. The
            daemon helper prevents the main thread from waiting forever when
            VASP exits, is killed, or never loads the plugin.
        """
        if self.listener is None:
            raise RuntimeError("ConnectionTransport.start() must be called first.")
        listener = self.listener
        result: list[Connection] = []
        errors: list[BaseException] = []

        def _accept() -> None:
            try:
                result.append(listener.accept())
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=_accept, daemon=True)
        thread.start()
        elapsed = 0.0
        while thread.is_alive():
            thread.join(0.1)
            elapsed += 0.1
            if is_alive is not None and not is_alive():
                self.close_listener()
                thread.join(1.0)
                raise RuntimeError("VASP exited before the plugin connected.")
            if timeout is not None and elapsed >= timeout:
                self.close_listener()
                thread.join(1.0)
                raise TimeoutError("Timed out waiting for the VASP plugin connection.")
        if errors:
            raise RuntimeError("The VASP plugin listener failed.") from errors[0]
        if len(result) == 0:
            raise RuntimeError("The VASP plugin closed before connecting.")
        self.connection = result[0]
        return self.connection

    def send(self, message: dict[str, Any]) -> None:
        """Send one protocol message to the plugin.

        Args:
            message: Message dictionary containing a string ``type`` field.

        Returns:
            None.
        """
        if self.connection is None:
            raise RuntimeError("The VASP plugin is not connected.")
        self.connection.send(message)

    def receive(self, timeout: float | None = None) -> dict[str, Any]:
        """Receive and validate one protocol message from the plugin.

        Args:
            timeout: Maximum seconds to wait, or ``None`` to block.

        Returns:
            The received message dictionary.
        """
        if self.connection is None:
            raise RuntimeError("The VASP plugin is not connected.")
        if timeout is not None and not self.connection.poll(timeout):
            raise TimeoutError("Timed out waiting for a VASP plugin event.")
        try:
            message = self.connection.recv()
        except EOFError as exc:
            raise ConnectionError("The VASP plugin closed the IPC connection.") from exc
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise RuntimeError("Received an invalid VASP plugin message.")
        return message

    def close_listener(self) -> None:
        """Close the listener without closing an accepted connection."""
        if self.listener is not None:
            self.listener.close()
            self.listener = None

    def close(self) -> None:
        """Close all endpoints and remove the local socket path.

        Returns:
            None. The operation is idempotent.
        """
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None
        self.close_listener()
        if self._socket_path is not None:
            try:
                os.unlink(self._socket_path)
            except FileNotFoundError:
                pass
            self._socket_path = None


_PLUGIN_SOURCE = r'''"""Generated BUCToolkit VASP plugin bridge."""
import base64
import os
import numpy as np
from multiprocessing.connection import Client


def _connect():
    """Connect without importing BUCToolkit inside VASP's Python runtime."""
    family = os.environ["BUCTOOLKIT_VASP_PLUGIN_FAMILY"]
    address_text = os.environ["BUCTOOLKIT_VASP_PLUGIN_ADDRESS"]
    authkey = base64.b64decode(os.environ["BUCTOOLKIT_VASP_PLUGIN_AUTHKEY"])
    if family == "AF_INET":
        host, port = address_text.split("\0", 1)
        address = (host, int(port))
    elif family == "AF_UNIX":
        address = address_text
    else:
        raise RuntimeError(f"Unsupported VASP plugin connection family: {family!r}.")
    return Client(address, family=family, authkey=authkey)


class _Bridge:
    def __init__(self):
        self.connection = _connect()
        self.connection.send({"type": "HELLO", "pid": os.getpid()})
        self.pending = None
        self.tolerance = float(os.environ.get("BUCTOOLKIT_VASP_POSITION_TOLERANCE", "1.e-10"))

    def request(self):
        """Wait for the next controller request at a VASP callback boundary."""
        while True:
            try:
                message = self.connection.recv()
            except EOFError as exc:
                raise RuntimeError("BUCToolkit closed the VASP plugin connection.") from exc
            if not isinstance(message, dict):
                raise RuntimeError("Invalid message received from BUCToolkit.")
            kind = message.get("type")
            if kind == "ACK":
                continue
            if kind == "CLOSE":
                self.connection.send({"type": "GOODBYE"})
                raise SystemExit(0)
            if kind != "STRUCTURE_REQUEST":
                raise RuntimeError(f"Unexpected BUCToolkit message: {kind!r}.")
            return message

    def ensure_request(self):
        """Retain one request until VASP reports that exact structure."""
        if self.pending is None:
            self.pending = self.request()
        return self.pending


_bridge = None


def _get_bridge():
    """Create one connection for the complete lifetime of this VASP process."""
    global _bridge
    if _bridge is None:
        _bridge = _Bridge()
    return _bridge


def structure(constants, additions):
    """Apply a requested structure or publish its completed VASP result."""
    bridge = _get_bridge()
    try:
        request = bridge.ensure_request()
        target_positions = np.asarray(request["positions"], dtype=np.float64)
        current_positions = np.asarray(constants.positions, dtype=np.float64)
        positions_match = np.allclose(
            current_positions,
            target_positions,
            rtol=0.0,
            atol=bridge.tolerance,
        )

        target_cell = request.get("cell")
        cell_match = True
        if target_cell is not None:
            target_cell = np.asarray(target_cell, dtype=np.float64)
            current_cell = np.asarray(constants.lattice_vectors, dtype=np.float64)
            cell_match = np.allclose(current_cell, target_cell, rtol=0.0, atol=bridge.tolerance)

        # VASP additions are displacements. Writing through ``out`` preserves
        # the VASP-owned array object and its underlying memory address.
        if not positions_match:
            np.subtract(target_positions, current_positions, out=additions.positions)
        if target_cell is not None and not cell_match:
            np.subtract(target_cell, current_cell, out=additions.lattice_vectors)

        # Energy and forces belong to the current constants only after VASP has
        # entered the callback with the requested coordinates and lattice.
        if positions_match and cell_match:
            bridge.connection.send({
                "type": "RESULT",
                "evaluation_id": request["evaluation_id"],
                "energy": float(constants.total_energy),
                "forces": np.array(constants.forces, dtype=np.float64, copy=True),
                "stress": np.array(constants.stress, dtype=np.float64, copy=True),
            })
            bridge.pending = None
    except SystemExit:
        raise
    except Exception as exc:
        try:
            bridge.connection.send({"type": "ERROR", "error": repr(exc)})
        except Exception:
            pass
        raise
'''


class VASP_PluginModel(_BaseWrapper):
    """Use a long-lived VASP Python plugin as an energy/force calculator.

    Args:
        input_path: Directory containing the VASP input files and submit script.
        submit_script: Submit script filename relative to ``input_path``.
        use_slurm: Launch with ``sbatch --wait`` instead of direct execution.
        startup_timeout: Seconds allowed for VASP and the plugin to connect.
        evaluation_timeout: Seconds allowed for one VASP evaluation.
        transport_host: Host advertised for network transport.  Unix sockets
            are used for direct jobs; Slurm jobs use an AF_INET connection.
        position_tolerance: Absolute tolerance used to match VASP positions
            with the requested Cartesian Angstrom structure.

    Notes:
        The input files are used in place.  This class does not create step
        directories, copy files, or parse OUTCAR.  The user must enable
        ``PLUGINS/STRUCTURE = T`` in INCAR.
    """

    def __init__(
            self,
            input_path: str,
            submit_script: str,
            use_slurm: bool = False,
            startup_timeout: float = 60.0,
            evaluation_timeout: float = 300.0,
            transport_host: str | None = None,
            position_tolerance: float = 1.e-10,
    ) -> None:
        super().__init__(input_path)
        self.input_path = os.path.abspath(input_path)
        self.submit_script = submit_script
        self._submit_script_path = os.path.abspath(os.path.join(self.input_path, submit_script))
        required_files = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
        for filename in required_files:
            if not os.path.isfile(os.path.join(self.input_path, filename)):
                raise FileNotFoundError(f"{filename} not found at {self.input_path}.")
        if not os.path.isfile(self._submit_script_path):
            raise FileNotFoundError(f"submit_script {submit_script!r} not found at {self.input_path}.")
        if startup_timeout <= 0 or evaluation_timeout <= 0:
            raise ValueError("startup_timeout and evaluation_timeout must be positive.")
        if position_tolerance < 0:
            raise ValueError("position_tolerance must be non-negative.")

        self.use_slurm = bool(use_slurm)
        self.startup_timeout = float(startup_timeout)
        self.evaluation_timeout = float(evaluation_timeout)
        self.position_tolerance = float(position_tolerance)
        self.transport_host = transport_host

        # Section: Persistent VASP session
        # These objects have the same lifetime as the wrapper. In particular,
        # one VASP process serves every evaluation instead of one process per X.
        self._transport: ConnectionTransport | None = None
        self._job: subprocess.Popen | None = None
        self._job_stdout = None
        self._job_stderr = None

        # ``sbatch --wait`` is only a local waiter. Keep the actual Slurm job id
        # so failure cleanup can cancel the remote VASP allocation as well.
        self._slurm_job_id: str | None = None
        self._slurm_stdout_thread: threading.Thread | None = None
        self._slurm_job_id_ready = threading.Event()

        # Section: Wrapper state
        # A failed session is terminal: reconnecting to a partly initialized or
        # unexpectedly dead VASP process could associate results with wrong X.
        self._started = False
        self._closed = False
        self._failed: BaseException | None = None
        self._methods_replaced = False
        self._evaluation_id = 0

        # Energy and Grad normally arrive as consecutive protocol calls. Cache
        # both results from one VASP snapshot to avoid a duplicate SCF cycle.
        self._cached_positions: np.ndarray | None = None
        self._cached_cell: np.ndarray | None = None
        self._cached_energy: float | None = None
        self._cached_forces: np.ndarray | None = None
        self._cached_stress: np.ndarray | None = None
        self._lock = threading.RLock()
        self._plugin_path = os.path.join(self.input_path, "vasp_plugin.py")

    def _write_plugin(self) -> None:
        """Atomically install the generated callback bridge in ``input_path``."""
        if os.path.exists(self._plugin_path):
            try:
                with open(self._plugin_path, "r", encoding="utf-8") as stream:
                    existing = stream.read(128)
            except OSError as exc:
                raise RuntimeError(f"Could not inspect {self._plugin_path}.") from exc
            if "Generated BUCToolkit VASP plugin bridge" not in existing:
                raise FileExistsError(
                    f"Refusing to overwrite an existing user plugin at {self._plugin_path}."
                )
        # Never expose a partially written plugin to a concurrently starting
        # VASP process. A non-generated user plugin is deliberately preserved.
        fd, temporary_path = tempfile.mkstemp(
            prefix=".vasp_plugin.", suffix=".tmp", dir=self.input_path, text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(_PLUGIN_SOURCE)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self._plugin_path)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def _start_job(self) -> None:
        """Launch the persistent direct or Slurm VASP job without waiting."""
        assert self._transport is not None
        environment = os.environ.copy()
        environment.update(self._transport.plugin_environment())
        environment["BUCTOOLKIT_VASP_POSITION_TOLERANCE"] = repr(self.position_tolerance)
        script_name = os.path.basename(self._submit_script_path)
        if self.use_slurm:
            command = ["sbatch", "--wait", "--parsable", f"./{script_name}"]
        else:
            command = [f"./{script_name}"]
        stdout_path = os.path.join(self.input_path, ".buctoolkit_vasp.stdout")
        stderr_path = os.path.join(self.input_path, ".buctoolkit_vasp.stderr")
        self._job_stdout = open(stdout_path, "w", encoding="utf-8")
        self._job_stderr = open(stderr_path, "w", encoding="utf-8")
        try:
            if self.use_slurm:
                self._job = subprocess.Popen(
                    command,
                    cwd=self.input_path,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=self._job_stderr,
                    text=True,
                    bufsize=1,
                )
                slurm_stdout = self._job.stdout

                def _drain_slurm_stdout() -> None:
                    # Draining avoids a full pipe stalling sbatch. Its first
                    # parsable line also owns the id required for ``scancel``.
                    first_line = True
                    if slurm_stdout is None:
                        return
                    for line in slurm_stdout:
                        if first_line:
                            self._slurm_job_id = line.strip().split(";", 1)[0] or None
                            self._slurm_job_id_ready.set()
                            first_line = False
                        self._job_stdout.write(line)
                        self._job_stdout.flush()
                    self._slurm_job_id_ready.set()

                self._slurm_stdout_thread = threading.Thread(target=_drain_slurm_stdout, daemon=True)
                self._slurm_stdout_thread.start()
            else:
                self._job = subprocess.Popen(
                    command,
                    cwd=self.input_path,
                    env=environment,
                    stdout=self._job_stdout,
                    stderr=self._job_stderr,
                    text=True,
                )
        except BaseException:
            self._job_stdout.close()
            self._job_stderr.close()
            self._job_stdout = self._job_stderr = None
            raise

    def _job_status(self) -> tuple[bool, int | None]:
        """Return whether the local process/waiter lives and its exit status."""
        if self._job is None:
            return False, None
        return self._job.poll() is None, self._job.returncode

    def _job_failure(self, context: str) -> RuntimeError:
        """Classify early normal exit, nonzero exit, and signal termination."""
        alive, returncode = self._job_status()
        if alive:
            return RuntimeError(f"VASP {context} while the process is still running.")
        if returncode is None:
            return RuntimeError(f"VASP {context} without an exit status.")
        if returncode < 0:
            return RuntimeError(f"VASP was killed by signal {-returncode} during {context}.")
        if returncode == 0:
            return RuntimeError(f"VASP exited normally before completing {context}.")
        return RuntimeError(f"VASP exited with status {returncode} during {context}.")

    def _bootstrap(self) -> None:
        """Generate the plugin, start VASP, and complete the first handshake."""
        with self._lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("VASP_PluginModel is closed.")
            if self._failed is not None:
                raise RuntimeError("VASP_PluginModel is in a failed state.") from self._failed
            self._write_plugin()
            self._transport = ConnectionTransport(
                use_network=self.use_slurm,
                host=self.transport_host,
                socket_dir="/tmp",
            )
            try:
                # The listener must exist before VASP imports its plugin; the
                # callback connects immediately during its first invocation.
                self._transport.start()
                self._start_job()
                self._transport.accept(
                    self.startup_timeout,
                    is_alive=lambda: self._job_status()[0],
                )
                hello = self._transport.receive(self.startup_timeout)
                if hello.get("type") != "HELLO":
                    raise RuntimeError(f"Unexpected VASP plugin handshake: {hello.get('type')!r}.")
                self._started = True
            except BaseException as exc:
                if self._job is not None and not self._job_status()[0]:
                    exc = self._job_failure("plugin startup")
                self._failed = exc
                self._abort_process()
                self._transport.close()
                raise exc

    @staticmethod
    def _normalize_input(X: th.Tensor, cell: Any = None) -> tuple[np.ndarray, tuple[int, ...], np.ndarray | None]:
        """Normalize one Cartesian structure before retained state is changed."""
        if not isinstance(X, th.Tensor):
            X = th.as_tensor(X)
        if X.ndim == 2 and X.shape[-1] == 3:
            origin_shape = tuple(X.shape)
            positions = X
        elif X.ndim == 3 and X.shape[0] == 1 and X.shape[-1] == 3:
            origin_shape = tuple(X.shape)
            positions = X.squeeze(0)
        else:
            raise ValueError("VASP_PluginModel accepts one structure shaped (n_atom, 3) or (1, n_atom, 3).")
        positions_np = positions.detach().to(device="cpu", dtype=th.float64).numpy().copy()
        cell_np = None
        if cell is not None:
            cell_tensor = th.as_tensor(cell)
            if cell_tensor.shape != (3, 3):
                raise ValueError(f"cell must have shape (3, 3), but got {tuple(cell_tensor.shape)}.")
            cell_np = cell_tensor.detach().to(device="cpu", dtype=th.float64).numpy().copy()
        return positions_np, origin_shape, cell_np

    def _wait_result(self, evaluation_id: int) -> dict[str, Any]:
        """Wait for one matching result while continuously checking VASP."""
        assert self._transport is not None
        deadline = time.monotonic() + self.evaluation_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for VASP evaluation {evaluation_id}.")
            alive, _ = self._job_status()
            if not alive:
                raise self._job_failure(f"evaluation {evaluation_id}")
            try:
                message = self._transport.receive(min(remaining, 0.25))
            except TimeoutError:
                continue
            except ConnectionError as exc:
                if not self._job_status()[0]:
                    raise self._job_failure(f"evaluation {evaluation_id}") from exc
                raise RuntimeError(
                    f"The VASP plugin connection was lost during evaluation {evaluation_id}."
                ) from exc
            kind = message.get("type")
            if kind == "RESULT":
                if message.get("evaluation_id") != evaluation_id:
                    raise RuntimeError("Received a VASP result for the wrong evaluation id.")
                return message
            if kind == "ERROR":
                raise RuntimeError(f"VASP plugin failed: {message.get('error', 'unknown error')}")
            if kind == "GOODBYE":
                raise RuntimeError("VASP plugin exited before returning the requested result.")
            raise RuntimeError(f"Unexpected VASP plugin event: {kind!r}.")

    def _evaluate(self, X: th.Tensor, *, cell: Any = None) -> tuple[float, np.ndarray, np.ndarray, tuple[int, ...], th.device, th.dtype]:
        """Synchronously evaluate one structure through the persistent session."""
        positions, origin_shape, cell_np = self._normalize_input(X, cell)
        device = X.device if isinstance(X, th.Tensor) else th.device("cpu")
        dtype = X.dtype if isinstance(X, th.Tensor) and X.is_floating_point() else th.float64
        with self._lock:
            self._bootstrap()
            if (
                    self._cached_positions is not None
                    and np.array_equal(positions, self._cached_positions)
                    and ((cell_np is None and self._cached_cell is None) or np.array_equal(cell_np, self._cached_cell))
            ):
                # Cache identity is value based because IPC necessarily creates
                # arrays with storage unrelated to the caller's torch tensor.
                assert self._cached_energy is not None and self._cached_forces is not None and self._cached_stress is not None
                return self._cached_energy, self._cached_forces, self._cached_stress, origin_shape, device, dtype

            assert self._transport is not None
            self._evaluation_id += 1
            evaluation_id = self._evaluation_id
            self._transport.send({
                "type": "STRUCTURE_REQUEST",
                "evaluation_id": evaluation_id,
                "positions": positions,
                "cell": cell_np,
            })
            try:
                result = self._wait_result(evaluation_id)
            except BaseException as exc:
                self._failed = exc
                self._abort_process()
                raise
            try:
                energy = float(result["energy"])
                forces = np.asarray(result["forces"], dtype=np.float64).copy()
                stress = np.asarray(result["stress"], dtype=np.float64).copy()
                if forces.shape != positions.shape:
                    raise RuntimeError(f"VASP returned forces with shape {forces.shape}, expected {positions.shape}.")
                self._cached_positions = positions
                self._cached_cell = cell_np
                self._cached_energy = energy
                self._cached_forces = forces
                self._cached_stress = stress
                # ACK releases the plugin-side result generation. The plugin
                # discards ACK before accepting the next structure request.
                self._transport.send({"type": "ACK", "evaluation_id": evaluation_id})
            except BaseException as exc:
                self._failed = exc
                self._abort_process()
                raise
            if not self._methods_replaced:
                # Bootstrap validation has now succeeded. Later protocol calls
                # can bypass the first-call public dispatch methods.
                self.Energy = self._energy_after_start
                self.Grad = self._grad_after_start
                self._methods_replaced = True
            return energy, forces, stress, origin_shape, device, dtype

    def Energy(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Return the VASP total energy for one Cartesian structure.

        Args:
            X: One structure shaped ``(n_atom, 3)`` or ``(1, n_atom, 3)``.
            *args: Reserved positional arguments for the func protocol.
            **kwargs: Optional ``cell`` shaped ``(3, 3)``.

        Returns:
            A tensor with shape ``(1,)`` on the input device.
        """
        cell = kwargs.get("cell", None)
        energy, _, _, _, device, dtype = self._evaluate(X, cell=cell)
        return th.tensor([energy], device=device, dtype=dtype)

    def Grad(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Return the mathematical energy gradient, i.e. negative VASP forces.

        Args:
            X: One structure shaped ``(n_atom, 3)`` or ``(1, n_atom, 3)``.
            *args: Reserved positional arguments for the grad protocol.
            **kwargs: Optional ``cell`` shaped ``(3, 3)``.

        Returns:
            A tensor shaped like ``X`` on the input device.
        """
        cell = kwargs.get("cell", None)
        _, forces, _, origin_shape, device, dtype = self._evaluate(X, cell=cell)
        return th.as_tensor(-forces, device=device, dtype=dtype).reshape(origin_shape).contiguous()

    def _energy_after_start(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Evaluate energy through an already initialized plugin session."""
        cell = kwargs.get("cell", None)
        energy, _, _, _, device, dtype = self._evaluate(X, cell=cell)
        return th.tensor([energy], device=device, dtype=dtype)

    def _grad_after_start(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Evaluate the gradient through an already initialized plugin session."""
        cell = kwargs.get("cell", None)
        _, forces, _, origin_shape, device, dtype = self._evaluate(X, cell=cell)
        return th.as_tensor(-forces, device=device, dtype=dtype).reshape(origin_shape).contiguous()

    def _abort_process(self) -> None:
        """Cancel the remote allocation and stop its local waiter/process."""
        # Killing ``sbatch --wait`` alone does not cancel the scheduled VASP
        # job, so cancel the allocation before terminating the local waiter.
        if self._slurm_stdout_thread is not None and self._slurm_job_id is None:
            self._slurm_job_id_ready.wait(timeout=0.5)
        if self._slurm_job_id is not None:
            try:
                subprocess.run(
                    ["scancel", self._slurm_job_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5.0,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
            self._slurm_job_id = None
        if self._job is not None and self._job.poll() is None:
            self._job.terminate()
            try:
                self._job.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._job.kill()
                self._job.wait()

    def close(self) -> None:
        """Close the plugin session and terminate VASP if necessary.

        Returns:
            None. The operation is idempotent.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._transport is not None and self._transport.connection is not None:
                try:
                    # Prefer callback-boundary shutdown, then enforce bounded
                    # termination below if VASP never invokes the callback.
                    self._transport.send({"type": "CLOSE"})
                    deadline = time.monotonic() + min(self.startup_timeout, 5.0)
                    while time.monotonic() < deadline and self._job_status()[0]:
                        if self._transport.connection.poll(0.1):
                            message = self._transport.receive()
                            if message.get("type") == "GOODBYE":
                                break
                except Exception:
                    pass
            self._abort_process()
            if self._transport is not None:
                self._transport.close()
            for stream in (self._job_stdout, self._job_stderr):
                if stream is not None:
                    stream.close()
            if self._slurm_stdout_thread is not None:
                self._slurm_stdout_thread.join(timeout=1.0)
                self._slurm_stdout_thread = None
            self._job_stdout = self._job_stderr = None

    def __enter__(self) -> "VASP_PluginModel":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
