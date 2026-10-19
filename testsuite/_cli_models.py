"""Model adapters used by the non-interactive CLI smoke tests."""

import torch as th

from testsuite._toy_models import GNNLJDirectionalEAM


class CLIBatchedGNNLJDirectionalEAM(GNNLJDirectionalEAM):
    """Provide the explicit PyG batch assignment expected by the toy model."""

    def forward(self, atomic_numbers_or_graph, pos=None, batch=None, cell=None):
        """Build ``data.batch`` from ``natoms`` before model evaluation."""
        if hasattr(atomic_numbers_or_graph, "pos"):
            data = atomic_numbers_or_graph
            atom_counts = th.as_tensor(
                data.natoms,
                device=data.pos.device,
                dtype=th.long,
            ).reshape(-1)
            if int(atom_counts.sum().item()) != data.pos.shape[0]:
                raise ValueError(
                    "`data.natoms` must sum to the number of rows in `data.pos`."
                )
            data.batch = th.arange(
                atom_counts.numel(),
                device=data.pos.device,
            ).repeat_interleave(atom_counts)

        return super().forward(atomic_numbers_or_graph, pos, batch, cell)
