"""BUCToolkit-controlled VASP Python-plugin wrapper."""

from __future__ import annotations

import os
import select
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from typing import Any, Sequence

import numpy as np
import torch as th

from BUCToolkit.utils.function_utils import _BaseWrapper


# A short fixed cadence keeps graceful VASP shutdown responsive without
# exposing another lifecycle tuning parameter in the public wrapper API.
POLL_FREQ_AFTER_STOPCAR = 5.0


_PLUGIN_SOURCE = r'''"""Generated BUCToolkit/VASP FIFO bridge."""
import os
import select
import stat
from multiprocessing import shared_memory
import numpy as np


# The plugin and controller reconstruct this flat layout independently. The
# FIFO record is the synchronization boundary for the corresponding array.
def _views(a, n):
    """Reconstruct named views over the shared flat float64 array."""
    offset = 3 * n
    positions = a[:offset].reshape(n, 3)
    cell = a[offset:offset + 9].reshape(3, 3)
    offset += 9
    energy = a[offset:offset + 1]
    offset += 1
    forces = a[offset:offset + 3 * n].reshape(n, 3)
    offset += 3 * n
    return positions, cell, energy, forces, a[offset:offset + 9].reshape(3, 3)


class _Bridge:
    """Own plugin-side FIFO and shared-array handles for one VASP process."""

    def __init__(self):
        """Open controller-owned storage and create the two session FIFOs."""
        n = int(os.environ["BUCTOOLKIT_VASP_NUMBER_ATOMS"])
        self.tolerance = float(os.environ["BUCTOOLKIT_VASP_POSITION_TOLERANCE"])
        self.command_path = os.environ["BUCTOOLKIT_VASP_COMMAND_FIFO"]
        self.event_path = os.environ["BUCTOOLKIT_VASP_EVENT_FIFO"]
        # The controller creates the FIFOs before launching VASP so startup
        # cannot wait for the first plugin callback. Keep this fallback for
        # standalone plugin launches, while rejecting an unrelated file.
        for path in (self.command_path, self.event_path):
            try:
                os.mkfifo(path, 0o600)
            except FileExistsError:
                if not stat.S_ISFIFO(os.stat(path).st_mode):
                    raise
        # O_RDWR prevents either side from blocking while opening its FIFO.
        self.command_fd = os.open(self.command_path, os.O_RDWR | os.O_NONBLOCK)
        self.event_fd = os.open(self.event_path, os.O_RDWR | os.O_NONBLOCK)
        size = 6 * n + 19
        if os.environ["BUCTOOLKIT_VASP_STORAGE"] == "shared_memory":
            self.storage = shared_memory.SharedMemory(name=os.environ["BUCTOOLKIT_VASP_DATA_NAME"])
            self.array = np.ndarray((size,), np.float64, buffer=self.storage.buf)
        else:
            self.storage = None
            self.array = np.memmap(os.environ["BUCTOOLKIT_VASP_DATA_PATH"], np.float64, mode="r+", shape=(size,))
        self.positions, self.cell, self.energy, self.forces, self.stress = _views(self.array, n)
        self.buffer = b""
        self.pending = None
        self.awaiting_update = False
        self.closed = False
        self.send(f"READY {os.getpid()}")

    def send(self, text):
        """Write one complete newline-delimited event to the controller."""
        record = (text + "\n").encode("ascii")
        if os.write(self.event_fd, record) != len(record):
            raise RuntimeError("Incomplete VASP plugin event write.")

    def request(self):
        """Block until the controller publishes the next generation request."""
        while b"\n" not in self.buffer:
            select.select([self.command_fd], [], [])
            chunk = os.read(self.command_fd, 4096)
            if len(chunk) == 0:
                raise RuntimeError("BUCToolkit command FIFO was closed.")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        fields = line.decode("ascii").split()
        if fields == ["STOP"]:
            return None
        if len(fields) != 3 or fields[0] != "EVALUATE":
            raise RuntimeError("Invalid BUCToolkit VASP command.")
        return int(fields[1]), fields[2] == "1"

    def close(self):
        """Release plugin-side FIFO descriptors and shared-array handles."""
        if self.closed:
            return
        self.closed = True
        for name in ("command_fd", "event_fd"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)
        self.positions = self.cell = self.energy = self.forces = self.stress = None
        array = self.array
        self.array = None
        if isinstance(array, np.memmap):
            array.flush()
            if array._mmap is not None:
                array._mmap.close()
        if self.storage is not None:
            self.storage.close()
            self.storage = None


_bridge = None


def structure(constants, additions):
    """Update VASP-owned additions in place, then publish real results.

    The callback keeps a request pending across VASP callback invocations. The
    first invocation applies displacements; a later invocation, after VASP has
    accepted them, copies ``constants.total_energy`` and ``constants.forces``
    into shared memory and emits ``RESULT``.
    """
    global _bridge
    if _bridge is None:
        _bridge = _Bridge()
    try:
        while True:
            if _bridge.pending is None:
                _bridge.pending = _bridge.request()
                _bridge.awaiting_update = False
            if _bridge.pending is None:
                _bridge.send("STOPPED")
                _bridge.close()
                return
            generation, has_cell = _bridge.pending
            positions_match = np.allclose(
                constants.positions,
                _bridge.positions,
                rtol=_bridge.tolerance,
                atol=1.e-7,
            )
            cell_match = True
            if has_cell:
                cell_match = np.allclose(
                    constants.lattice_vectors,
                    _bridge.cell,
                    rtol=_bridge.tolerance,
                    atol=1.e-7,
                )
            if not positions_match:
                if generation == 1:
                    raise RuntimeError("VASP initial coordinates differ from the requested structure")
                if _bridge.awaiting_update:
                    raise RuntimeError("VASP coordinates did not reach the requested structure")
                np.subtract(_bridge.positions, constants.positions, out=additions.positions)
            if has_cell and not cell_match:
                if generation == 1:
                    raise RuntimeError("VASP initial cell differs from the requested structure")
                if _bridge.awaiting_update:
                    raise RuntimeError("VASP lattice did not reach the requested structure")
                np.subtract(_bridge.cell, constants.lattice_vectors, out=additions.lattice_vectors)
            if not positions_match or not cell_match:
                _bridge.awaiting_update = True
                return
            _bridge.energy[0] = float(constants.total_energy)
            np.copyto(_bridge.forces, constants.forces)
            np.copyto(_bridge.stress, constants.stress)
            if isinstance(_bridge.array, np.memmap):
                _bridge.array.flush()
            _bridge.send(f"RESULT {generation}")
            _bridge.awaiting_update = False
            # Keep VASP inside this callback until BUCToolkit has either
            # supplied the next structure or requested shutdown. Returning
            # immediately would trigger a redundant SCF on old coordinates.
            _bridge.pending = _bridge.request()
            if _bridge.pending is None:
                _bridge.send("STOPPED")
                _bridge.close()
                return
    except Exception as exc:
        try:
            _bridge.send(f"ERROR {repr(exc).encode('utf-8').hex()}")
        except Exception:
            pass
        _bridge.close()
        raise
'''


def _views(array: np.ndarray, n_atom: int) -> dict[str, np.ndarray]:
    """Return views for the fixed positions/cell/result memory layout."""
    offset = 3 * n_atom
    views = {"positions": array[:offset].reshape(n_atom, 3)}
    views["cell"] = array[offset:offset + 9].reshape(3, 3)
    offset += 9
    views["energy"] = array[offset:offset + 1]
    offset += 1
    views["forces"] = array[offset:offset + 3 * n_atom].reshape(n_atom, 3)
    offset += 3 * n_atom
    views["stress"] = array[offset:offset + 9].reshape(3, 3)
    return views


def _read_poscar(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read Cartesian coordinates and lattice vectors from one POSCAR.

    The check is deliberately limited to the structural fields needed by the
    wrapper.  VASP's optional selective-dynamics flags and trailing comments
    do not affect those fields.
    """
    with open(path, encoding="utf-8") as stream:
        lines = [line.split("!", 1)[0].strip() for line in stream if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"POSCAR is incomplete: {path}")
    scale = float(lines[1].split()[0])
    lattice = np.asarray([[float(value) for value in lines[index].split()[:3]] for index in (2, 3, 4)], dtype=np.float64)
    if scale < 0:
        # VASP interprets a negative scale as the desired cell volume.
        scale = (abs(scale) / abs(np.linalg.det(lattice))) ** (1.0 / 3.0)
    elif scale == 0:
        raise ValueError("POSCAR scale factor must be non-zero")
    lattice *= scale
    # POSCAR 5 format includes element symbols; older files may omit them.
    count_line = 5 if all(token.lstrip("+-").isdigit() for token in lines[5].split()) else 6
    counts = [int(value) for value in lines[count_line].split()]
    if not counts or any(value < 0 for value in counts):
        raise ValueError(f"POSCAR has invalid atom counts: {path}")
    coordinate_line = count_line + 1
    if lines[coordinate_line].lower().startswith("s"):
        coordinate_line += 1
    coordinate_mode = lines[coordinate_line].lower()[0]
    if coordinate_mode not in {"c", "d"}:
        raise ValueError(f"POSCAR has an invalid coordinate mode: {path}")
    n_atom = sum(counts)
    coordinate_rows = [line.split()[:3] for line in lines[coordinate_line + 1:coordinate_line + 1 + n_atom]]
    if len(coordinate_rows) != n_atom:
        raise ValueError(f"POSCAR does not contain {n_atom} coordinate rows: {path}")
    coordinates = np.asarray(coordinate_rows, dtype=np.float64)
    if coordinate_mode == "c":
        coordinates *= scale
    else:
        coordinates = coordinates @ lattice
    return coordinates, lattice


class _Transport(ABC):
    """Common FIFO protocol with backend-specific storage and job control."""

    def __init__(self, input_path: str, command: Sequence[str], session: str) -> None:
        """Initialize paths and common controller-side protocol state.

        Args:
            input_path: VASP working directory shared with the plugin.
            command: Backend command or submission-script arguments.
            session: Unique suffix preventing concurrent sessions from sharing FIFOs.
        """
        self.input_path, self.command = input_path, list(command)
        prefix = os.path.join(input_path, f".buctoolkit-vasp-{session}")
        self.command_path, self.event_path = prefix + ".command.fifo", prefix + ".event.fifo"
        self.command_fd = self.event_fd = None
        self.event_buffer = b""
        self.array = None
        self.views = {}

    @abstractmethod
    def prepare(self, size: int) -> dict[str, str]:
        """Allocate backend storage and return plugin environment variables."""
        raise NotImplementedError

    @abstractmethod
    def launch(self, environment: dict[str, str]) -> None:
        """Start or submit VASP without waiting for completion."""
        raise NotImplementedError

    @abstractmethod
    def status(self) -> tuple[bool, str]:
        """Return active state and a human-readable lifecycle description."""
        raise NotImplementedError

    @abstractmethod
    def terminate(self) -> None:
        """Stop an active local process group or Slurm allocation."""
        raise NotImplementedError

    @abstractmethod
    def active(self) -> bool:
        """Return whether controller cleanup must still terminate the job."""
        raise NotImplementedError

    @abstractmethod
    def close_data(self) -> None:
        """Release and unlink backend-specific shared data."""
        raise NotImplementedError

    def start(self, n_atom: int, environment: dict[str, str], interval: float) -> None:
        """Allocate data, create IPC endpoints, launch VASP, and await READY.

        ``input_path`` remains the child cwd throughout this operation. This
        is what makes BUCToolkit's configured directory and VASP's relative
        INCAR/POSCAR lookup refer to the same files.

        Args:
            n_atom: Atom count defining the fixed shared-array layout.
            environment: Parent environment inherited by VASP and its plugin.
            interval: Seconds between startup liveness checks while waiting for
                the plugin READY event.

        Returns:
            None.

        Raises:
            RuntimeError: If the VASP job terminates during startup.
        """
        child_env = environment.copy()
        child_env.update({
            "BUCTOOLKIT_VASP_NUMBER_ATOMS": str(n_atom),
            "BUCTOOLKIT_VASP_COMMAND_FIFO": self.command_path,
            "BUCTOOLKIT_VASP_EVENT_FIFO": self.event_path,
        })
        child_env.update(self.prepare(6 * n_atom + 19))
        self.views = _views(self.array, n_atom)
        # Open both ends before launch. This removes the old startup gap in
        # which the controller slept until STRUCTURE created the FIFOs.
        for path in (self.command_path, self.event_path):
            try:
                os.mkfifo(path, 0o600)
            except FileExistsError:
                if not stat.S_ISFIFO(os.stat(path).st_mode):
                    raise
        self.command_fd = os.open(self.command_path, os.O_RDWR | os.O_NONBLOCK)
        self.event_fd = os.open(self.event_path, os.O_RDWR | os.O_NONBLOCK)
        self.launch(child_env)

    def assert_alive(self, context: str) -> None:
        """Raise when the VASP owner of a pending request has terminated.

        Args:
            context: Operation included in the resulting diagnostic.

        Returns:
            None.
        """
        alive, message = self.status()
        if not alive:
            raise RuntimeError(f"VASP stopped during {context}: {message}.")

    def send(self, generation: int, has_cell: bool) -> None:
        """Publish request metadata after shared request arrays are ready.

        Args:
            generation: Monotonic request identifier.
            has_cell: Whether the shared cell belongs to this request.

        Returns:
            None.
        """
        if isinstance(self.array, np.memmap):
            self.array.flush()
        record = f"EVALUATE {generation} {int(has_cell)}\n".encode("ascii")
        if os.write(self.command_fd, record) != len(record):
            raise RuntimeError("Incomplete VASP command write.")

    def stop_plugin(self) -> None:
        """Wake a blocked plugin callback before terminating its VASP owner."""
        if self.command_fd is None:
            return
        try:
            os.write(self.command_fd, b"STOP\n")
        except OSError:
            # The plugin may already have exited or closed its FIFO.
            pass

    def receive(self, interval: float, context: str) -> str:
        """Wait for one event, waking every interval to check VASP status.

        Args:
            interval: Seconds between process-liveness checks; this is not an
                evaluation deadline.
            context: Operation included in lifecycle diagnostics.

        Returns:
            One newline-delimited plugin event without its terminator.
        """
        while True:
            if b"\n" in self.event_buffer:
                line, self.event_buffer = self.event_buffer.split(b"\n", 1)
                return line.decode("ascii")
            self.assert_alive(context)
            ready, _, _ = select.select([self.event_fd], [], [], interval)
            if ready:
                chunk = os.read(self.event_fd, 4096)
                if len(chunk) == 0:
                    raise ConnectionError("VASP plugin event FIFO was closed.")
                self.event_buffer += chunk

    def wait_for_exit(self, timeout: float, poll_frequency: float) -> bool:
        """Wait for the backend job to exit without busy polling.

        Args:
            timeout: Maximum wait in seconds for normal VASP shutdown.
            poll_frequency: Seconds between backend liveness checks.

        Returns:
            ``True`` when the backend has exited, otherwise ``False`` after
            ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while self.active():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            select.select([], [], [], min(poll_frequency, remaining))
        return True

    def close(self) -> None:
        """Close descriptors and unlink FIFO/data resources.

        Returns:
            None. Repeated calls are safe after owned handles are cleared.
        """
        for name in ("command_fd", "event_fd"):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)
        self.close_data()
        for path in (self.command_path, self.event_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


class _LocalTransport(_Transport):
    """Direct process backend using named shared memory."""

    def __init__(self, *args: Any) -> None:
        """Initialize local-process state in addition to common transport state."""
        super().__init__(*args)
        self.shared = None
        self.process = None

    def prepare(self, size: int) -> dict[str, str]:
        """Create named shared memory visible to the local child process."""
        self.shared = shared_memory.SharedMemory(create=True, size=size * 8)
        self.array = np.ndarray((size,), np.float64, buffer=self.shared.buf)
        return {"BUCTOOLKIT_VASP_STORAGE": "shared_memory", "BUCTOOLKIT_VASP_DATA_NAME": self.shared.name}

    def launch(self, environment: dict[str, str]) -> None:
        """Launch the submit script in a new process group."""
        self.process = subprocess.Popen(self.command, cwd=self.input_path, env=environment, start_new_session=True)

    def status(self) -> tuple[bool, str]:
        """Classify running, normal, non-zero, and signal exits."""
        if self.process is None:
            return False, "direct process was not started"
        code = self.process.poll()
        if code is None:
            return True, "direct process is running"
        if code < 0:
            return False, f"direct process was killed by signal {-code}"
        if code == 0:
            return False, "direct process exited normally"
        return False, f"direct process exited with status {code}"

    def active(self) -> bool:
        """Return whether the local VASP process is still running."""
        return self.process is not None and self.process.poll() is None

    def terminate(self) -> None:
        """Terminate the local VASP process group with a bounded fallback."""
        if not self.active():
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def close_data(self) -> None:
        """Release and unlink the controller-owned shared-memory segment."""
        if self.shared is not None:
            # Drop NumPy exports before closing SharedMemory; otherwise Python
            # raises BufferError while the array views are still alive.
            self.views.clear()
            self.array = None
            shared = self.shared
            self.shared = None
            try:
                shared.close()
            finally:
                try:
                    shared.unlink()
                except FileNotFoundError:
                    pass


class _SlurmTransport(_Transport):
    """Slurm backend using a shared-work-directory mmap and job ID polling."""

    ACTIVE = {"CONFIGURING", "COMPLETING", "PENDING", "RUNNING", "SIGNALING", "STAGE_OUT", "SUSPENDED"}

    def __init__(self, *args: Any) -> None:
        """Initialize Slurm job tracking and shared mmap storage paths."""
        super().__init__(*args)
        self.data_path = os.path.join(self.input_path, ".buctoolkit-vasp-data-%s.mmap" % uuid.uuid4().hex)
        self.job_id = None
        self.missing = 0

    def prepare(self, size: int) -> dict[str, str]:
        """Create mmap storage in the shared VASP working directory."""
        self.array = np.memmap(self.data_path, np.float64, mode="w+", shape=(size,))
        return {"BUCTOOLKIT_VASP_STORAGE": "mmap", "BUCTOOLKIT_VASP_DATA_PATH": self.data_path}

    def launch(self, environment: dict[str, str]) -> None:
        """Submit the script and retain the Slurm job ID as lifecycle handle."""
        result = subprocess.run(["sbatch", "--parsable", "--export=ALL", *self.command], cwd=self.input_path, env=environment, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
        self.job_id = result.stdout.strip().split(";", 1)[0]
        if not self.job_id:
            raise RuntimeError("sbatch returned no job id")

    @staticmethod
    def _state(value: str) -> str:
        """Normalize Slurm state text by dropping flags and whitespace."""
        return value.strip().split("+", 1)[0].split()[0].upper()

    def status(self) -> tuple[bool, str]:
        """Poll squeue/sacct and classify active, terminal, or missing jobs."""
        if self.job_id is None:
            return False, "Slurm job was not submitted"
        queue = subprocess.run(["squeue", "--noheader", "--jobs", self.job_id, "--format=%T"], capture_output=True, text=True, check=False)
        if queue.returncode:
            raise RuntimeError(queue.stderr.strip())
        if queue.stdout.strip():
            state = self._state(queue.stdout.splitlines()[0])
            return True, f"Slurm job {self.job_id} is {state}"
        accounting = subprocess.run(["sacct", "--noheader", "--allocations", "--jobs", self.job_id, "--format=State,ExitCode", "--parsable2"], capture_output=True, text=True, check=False)
        for line in accounting.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 2 and fields[0]:
                state = self._state(fields[0])
                if state in self.ACTIVE:
                    return True, f"Slurm job {self.job_id} is {state}"
                return False, f"Slurm job {self.job_id} ended as {state} ({fields[1]})"
        self.missing += 1
        return (self.missing < 3, f"Slurm job {self.job_id} disappeared from scheduler")

    def active(self) -> bool:
        """Return whether Slurm still reports this allocation as active."""
        try:
            return self.status()[0]
        except RuntimeError:
            return self.job_id is not None

    def terminate(self) -> None:
        """Cancel the active Slurm allocation, if one is still present."""
        if self.job_id and self.active():
            subprocess.run(["scancel", self.job_id], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close_data(self) -> None:
        """Flush and remove the controller-owned mmap file."""
        if isinstance(self.array, np.memmap):
            self.array.flush()
        self.views.clear()
        self.array = None
        try:
            os.unlink(self.data_path)
        except FileNotFoundError:
            pass


class VASP_PluginModel(_BaseWrapper):
    """Persistent VASP plugin calculator controlled by BUCToolkit.

    ``input_path`` is always VASP's working directory. ``submit_script`` may
    be relative to it or an absolute shared script; both are launched with
    ``cwd=input_path``.

    Args:
        input_path: Directory containing INCAR, POSCAR, KPOINTS and POTCAR.
        submit_script: Relative script in ``input_path`` or absolute shared script.
        use_slurm: Select Slurm+mmap instead of local process+shared memory.
        evaluation_timeout: Seconds between liveness checks while waiting for
            plugin startup or an SCF evaluation result.
        position_tolerance: Relative tolerance for structure matching.

    Returns:
        An object implementing BUCToolkit's Energy/Grad wrapper protocol.
    """

    def __init__(self, input_path: str, submit_script: str | None = None, use_slurm: bool = False, evaluation_timeout: float = 300.0, position_tolerance: float = 5.e-5, command: str | list[str] | None = None):
        """Create a persistent external VASP calculator.

        Args:
            input_path: Directory containing VASP's INCAR, POSCAR, KPOINTS and POTCAR.
            submit_script: Optional executable script used when ``command`` is absent.
            use_slurm: Submit ``submit_script`` with Slurm instead of launching a
            direct child process. ``command`` is invalid in this mode.
            evaluation_timeout: Seconds between liveness checks while waiting
                for plugin startup or an SCF evaluation result.
            position_tolerance: Relative tolerance for structure matching.
            command: Optional command string or argv list, such as
                ``"mpirun -n N vasp_std"``. It is only used for local launches.

        Returns:
            None. The VASP process is started lazily by the first ``Energy`` call.

        Raises:
            FileNotFoundError: If a required VASP input or submit script is missing.
            ValueError: If timeout or tolerance values are invalid.
        """
        # VASP is an external calculator, not a torch/PyG model.  The base
        # initializer is called only to establish its common bookkeeping
        # fields; no model object is wrapped here.
        super().__init__(None)
        self.input_path = os.path.abspath(input_path)
        if use_slurm and command is not None:
            raise ValueError("`command` must not be provided when `use_slurm=True`; provide `submit_script` instead.")
        if use_slurm and submit_script is None:
            raise ValueError("`submit_script` is required when `use_slurm=True`.")
        if command is None and submit_script is None:
            raise ValueError("`submit_script` or `command` must be provided.")
        self.submit_script = None if submit_script is None else os.path.abspath(submit_script if os.path.isabs(submit_script) else os.path.join(self.input_path, submit_script))
        if isinstance(command, str):
            command = command.split()
        elif command is not None and not isinstance(command, list):
            raise TypeError(f"Expected `command` to be str or List, but got {type(command)}")
        if command is not None:
            self.command = [str(item) for item in command]
            if len(self.command) == 0:
                raise ValueError("`command` must not be empty.")
        else:
            self.command = [self.submit_script]
        for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
            if not os.path.isfile(os.path.join(self.input_path, name)):
                raise FileNotFoundError(f"{name} not found in {self.input_path}")
        if self.submit_script is not None and not os.path.isfile(self.submit_script):
            raise FileNotFoundError(f"submit_script not found: {self.submit_script}")
        if evaluation_timeout <= 0 or position_tolerance < 0:
            raise ValueError("evaluation_timeout must be positive; position_tolerance must be non-negative")
        self.use_slurm = bool(use_slurm)
        self.evaluation_timeout = float(evaluation_timeout)
        self.position_tolerance = float(position_tolerance)
        self.transport = None
        self.is_init = False
        self.closed = False
        self.failed = None
        self.n_atom = None
        self._reference_cell = None
        self.generation = 0
        self.cache = None
        self.plugin_path = os.path.join(self.input_path, "vasp_plugin.py")
    def Energy(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Return VASP energy, dispatching through first-call initialization.

        Args:
            X: Cartesian coordinates shaped ``(n_atom, 3)`` or
                ``(1, n_atom, 3)``.
            *args: Reserved func-protocol arguments.
            **kwargs: Optional ``cell`` shaped ``(3, 3)``.

        Returns:
            A one-element tensor containing the VASP total energy.
        """
        if self.is_init:
            return self.__regular_energy(X, *args, **kwargs)
        return self.__init_energy(X, *args, **kwargs)

    @staticmethod
    def _check_INCAR(input_path: str) -> None:
        """Append plugin settings and comment conflicting assignments.

        The marked block is intentionally left in INCAR after termination. A
        repeated initialization recognizes it and avoids accumulating duplicate
        overrides; no backup file or restoration pass is required.
        """
        incar_path = os.path.join(input_path, "INCAR")
        with open(incar_path, encoding="utf-8") as stream:
            lines = stream.readlines()
        if any("BUCTOOLKIT VASP PLUGIN SETTINGS" in line for line in lines):
            return
        rewritten = []
        for line in lines:
            content = line.split("#", 1)[0].split("!", 1)[0]
            key = content.split("=", 1)[0].strip().upper() if "=" in content else ""
            if key in {"IBRION", "PLUGINS/STRUCTURE"} and content.strip():
                rewritten.append("# BUCToolkit disabled original: " + line.rstrip("\n") + "\n")
            else:
                rewritten.append(line)
        rewritten.extend([
            "\n# BUCTOOLKIT VASP PLUGIN SETTINGS\n",
            "# Required by VASP_PluginModel; original assignments remain above.\n",
            "IBRION = 12\n",
            "PLUGINS/STRUCTURE = T\n",
            "# END BUCTOOLKIT VASP PLUGIN SETTINGS\n",
        ])
        with open(incar_path, "w", encoding="utf-8", newline="\n") as stream:
            stream.writelines(rewritten)

    @staticmethod
    def _normalize(X: th.Tensor, cell: Any):
        """Validate and copy one structure before mutating session state.

        Args:
            X: Floating-point coordinates shaped ``(n_atom, 3)`` or
                ``(1, n_atom, 3)``.
            cell: Optional lattice vectors shaped ``(3, 3)``.

        Returns:
            Positions, cell, original shape, device and dtype.

        Raises:
            TypeError: If ``X`` is not a floating-point tensor.
            ValueError: If coordinate or cell shapes are invalid.
        """
        if not isinstance(X, th.Tensor) or not X.is_floating_point():
            raise TypeError("X must be a floating-point torch.Tensor")
        shape = tuple(X.shape)
        if X.ndim == 2 and X.shape[1] == 3:
            positions = X
        elif X.ndim == 3 and X.shape[0] == 1 and X.shape[2] == 3:
            positions = X[0]
        else:
            raise ValueError(f"X must have shape (n_atom, 3) or (1, n_atom, 3), got {shape}")
        pos = positions.detach().cpu().double().numpy().copy()
        cell_np = None if cell is None else th.as_tensor(cell).detach().cpu().double().numpy().copy()
        # PyG stores one structure's cell as (1, 3, 3); the VASP plugin
        # protocol uses the corresponding unbatched (3, 3) lattice matrix.
        if cell_np is not None and cell_np.shape == (1, 3, 3):
            cell_np = cell_np[0]
        if cell_np is not None and cell_np.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3) or (1, 3, 3)")
        return pos, cell_np, shape, X.device, X.dtype

    @staticmethod
    def _cell_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Extract optional cell context without imposing an argument type.

        The func protocol permits arbitrary positional context.  VASP only
        recognizes an explicit ``cell`` keyword or a conventional ``cell`` /
        ``lattice_vectors`` attribute on the first context object.
        """
        if "cell" in kwargs:
            return kwargs["cell"]
        if len(args) > 0:
            context = args[0]
            for name in ("cell", "lattice_vectors"):
                if hasattr(context, name):
                    return getattr(context, name)
        return None

    def _write_plugin(self) -> None:
        """Atomically install the generated plugin in the VASP working directory."""
        if os.path.exists(self.plugin_path):
            with open(self.plugin_path, encoding="utf-8") as stream:
                if "Generated BUCToolkit/VASP FIFO bridge" not in stream.read(120):
                    raise FileExistsError(f"Refusing to overwrite {self.plugin_path}")
        fd, temp = tempfile.mkstemp(dir=self.input_path, prefix=".vasp_plugin.", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_PLUGIN_SOURCE)
        os.replace(temp, self.plugin_path)

    def _check_initial_structure(self, positions: np.ndarray, cell: np.ndarray | None) -> None:
        """Verify that the first requested structure matches VASP's POSCAR.

        BUCToolkit controls the evaluation sequence, but VASP still reads its
        initial geometry from the working directory.  Refusing a mismatch here
        prevents silently evaluating a different structure and avoids starting
        a long-running process for an invalid request.
        """
        poscar_positions, poscar_cell = _read_poscar(os.path.join(self.input_path, "POSCAR"))
        self._reference_cell = poscar_cell
        if positions.shape != poscar_positions.shape:
            raise ValueError(
                "The initial coordinate shape does not match VASP POSCAR: "
                f"{positions.shape} != {poscar_positions.shape}"
            )
        if not np.allclose(
                positions,
                poscar_positions,
                rtol=self.position_tolerance,
                atol=1.e-7,
        ):
            raise ValueError(
                "The initial coordinates do not match VASP POSCAR within "
                f"a relative tolerance of {self.position_tolerance:g}"
            )
        if cell is not None and not np.allclose(
                cell,
                poscar_cell,
                rtol=self.position_tolerance,
                atol=1.e-7,
        ):
            raise ValueError(
                "The initial cell does not match VASP POSCAR within "
                f"a relative tolerance of {self.position_tolerance:g}"
            )

    def _terminate(self, reason):
        """Record a termination reason and stop an active backend job.

        Args:
            reason: Human-readable explanation written to ``timeout.err``.

        Returns:
            None.
        """
        if self.transport is not None and self.transport.active():
            with open(os.path.join(self.input_path, "timeout.err"), "w", encoding="utf-8") as stream:
                stream.write(reason + "\n")
            self.transport.stop_plugin()
            self.transport.terminate()

    def _write_stopcar(self) -> None:
        """Request VASP's normal hard-stop cleanup through ``STOPCAR``.

        ``LABORT`` is read by VASP's electronic-loop abort handler.  It marks
        the run as a hard stop while still allowing the final output stage,
        including ``WAVECAR`` writing, to execute.
        """
        stopcar_path = os.path.join(self.input_path, "STOPCAR")
        with open(stopcar_path, "w", encoding="ascii", newline="\n") as stream:
            stream.write("LABORT = T\n")

    def __init_energy(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Bootstrap IPC/VASP once, then evaluate the first structure.

        Args:
            X: Coordinates accepted by :meth:`Energy`.
            *args: Optional func-protocol context used to locate a cell.
            **kwargs: Optional keyword context, including ``cell``.

        Returns:
            The first VASP energy as a one-element tensor.
        """
        positions, cell, _, _, _ = self._normalize(X, self._cell_from_call(args, kwargs))
        self._check_initial_structure(positions, cell)
        self._check_INCAR(self.input_path)
        self._write_plugin()
        kind = _SlurmTransport if self.use_slurm else _LocalTransport
        self.transport = kind(self.input_path, self.command, uuid.uuid4().hex)
        env = os.environ.copy()
        env["BUCTOOLKIT_VASP_POSITION_TOLERANCE"] = repr(self.position_tolerance)
        try:
            self.transport.start(len(positions), env, self.evaluation_timeout)
            ready = self.transport.receive(self.evaluation_timeout, "plugin startup").split()
            if len(ready) != 2 or ready[0] != "READY":
                raise RuntimeError(f"Unexpected plugin event: {ready!r}")
        except BaseException as exc:
            self.failed = exc
            self._terminate(f"plugin startup failed: {exc}")
            self.transport.close()
            raise
        self.n_atom, self.is_init = len(positions), True
        return self.__regular_energy(X, *args, **kwargs)

    def _evaluate(self, positions: np.ndarray, cell: np.ndarray | None) -> tuple[float, np.ndarray]:
        """Send one generation and copy its synchronized energy/force result.

        Args:
            positions: Cartesian coordinates in shared-array precision.
            cell: Optional lattice vectors for this request.

        Returns:
            A scalar energy and a private copy of the VASP forces.

        Raises:
            RuntimeError: If VASP or the plugin reports an invalid result.
        """
        if self.transport is None or len(positions) != self.n_atom:
            raise ValueError("atom count does not match the persistent VASP session")
        self.generation += 1
        views = self.transport.views
        # VASP's plugin STRUCTURE interface exposes fractional positions,
        # whereas BUCToolkit's public func/grad protocol uses Cartesian ones.
        # Convert only at the IPC boundary and retain Cartesian values in the
        # cache and in the returned tensors.
        lattice = cell if cell is not None else self._reference_cell
        if lattice is None:
            raise RuntimeError("A lattice is required to convert Cartesian positions for VASP.")
        fractional_positions = np.linalg.solve(lattice.T, positions.T).T
        np.copyto(views["positions"], fractional_positions)
        if cell is not None:
            # Both interfaces expose one lattice vector per row.  Only the
            # coordinate solve above changes orientation because the plugin's
            # Cartesian conversion is ``positions @ lattice_vectors.T``.
            np.copyto(views["cell"], cell)
        try:
            self.transport.send(self.generation, cell is not None)
            # ``evaluation_timeout`` is only a liveness-poll interval.  A
            # long SCF cycle must never be treated as a failed evaluation.
            while True:
                try:
                    event = self.transport.receive(
                        self.evaluation_timeout,
                        f"evaluation {self.generation}",
                    ).split()
                    break
                except TimeoutError:
                    self.transport.assert_alive(f"evaluation {self.generation}")
            if event[0] == "ERROR":
                detail = "VASP plugin error"
                if len(event) == 2:
                    try:
                        detail = bytes.fromhex(event[1]).decode("utf-8")
                    except (ValueError, UnicodeDecodeError):
                        pass
                raise RuntimeError(detail)
            if len(event) != 2 or event[0] != "RESULT" or int(event[1]) != self.generation:
                raise RuntimeError(f"Unexpected plugin event: {event!r}")
            energy = float(views["energy"][0])
            forces = np.array(views["forces"], copy=True)
            self.cache = (positions.copy(), None if cell is None else cell.copy(), energy, forces)
            return energy, forces
        except BaseException as exc:
            self.failed = exc
            self._terminate(f"evaluation failed: {exc}")
            self.transport.close()
            raise

    def __regular_energy(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Evaluate energy through an already initialized persistent session."""
        positions, cell, _, device, dtype = self._normalize(X, self._cell_from_call(args, kwargs))
        if self.closed or self.failed is not None:
            raise RuntimeError("VASP_PluginModel is unavailable")
        if self.cache is not None and np.array_equal(self.cache[0], positions) and ((cell is None and self.cache[1] is None) or np.array_equal(cell, self.cache[1])):
            energy = self.cache[2]
        else:
            energy, _ = self._evaluate(positions, cell)
        return th.tensor([energy], device=device, dtype=dtype)

    def Grad(self, X: th.Tensor, *args: Any, **kwargs: Any) -> th.Tensor:
        """Return the mathematical gradient, i.e. negative VASP forces.

        Args:
            X: Coordinates shaped ``(n_atom, 3)`` or ``(1, n_atom, 3)``.
            *args: Reserved grad-protocol arguments.
            **kwargs: Optional ``cell`` shaped ``(3, 3)``.

        Returns:
            A tensor shaped like ``X`` on the input device and dtype.
        """
        positions, cell, shape, device, dtype = self._normalize(X, self._cell_from_call(args, kwargs))
        if not self.is_init:
            self.Energy(X, *args, **kwargs)
        if self.cache is not None and np.array_equal(self.cache[0], positions) and ((cell is None and self.cache[1] is None) or np.array_equal(cell, self.cache[1])):
            forces = self.cache[3]
        else:
            _, forces = self._evaluate(positions, cell)
        return th.as_tensor(-forces, device=device, dtype=dtype).reshape(shape).contiguous()

    def close(self) -> None:
        """Gracefully stop VASP, then release all IPC resources.

        A ``LABORT`` request gives VASP an opportunity to write final files.
        If the process remains active for ``evaluation_timeout`` seconds, the
        existing forced-termination path is used as a safety fallback.
        """
        if self.closed:
            return
        self.closed = True
        try:
            if self.transport is not None:
                try:
                    if self.transport.active():
                        self._write_stopcar()
                        self.transport.stop_plugin()
                        has_exited = self.transport.wait_for_exit(
                            self.evaluation_timeout,
                            POLL_FREQ_AFTER_STOPCAR,
                        )
                        if not has_exited:
                            self._terminate(
                                "VASP did not exit after STOPCAR within "
                                f"{self.evaluation_timeout:g} seconds"
                            )
                except BaseException as exc:
                    self._terminate(f"VASP graceful shutdown failed: {exc}")
                    raise
                finally:
                    self.transport.close()
        finally:
            self.transport = None
            self.is_init = False
            self.n_atom = None
            self.generation = 0
            self.cache = None
            self.failed = None
            self.closed = False

    def __enter__(self) -> "VASP_PluginModel":
        """Return this wrapper for use in a context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release the persistent VASP session on context exit."""
        self.close()
