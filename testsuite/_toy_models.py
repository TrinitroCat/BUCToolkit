""" Toy GNN-LJ potential models for testing BUCToolkit APIs (training, prediction, MD, etc.).

Provides a learnable Lennard-Jones potential where per-pair (σ, ε) parameters
are predicted by a graph neural network from atomic species embeddings.

Graph construction uses `indices_pairwise_dist` with periodic boundary conditions.
"""

#  Copyright (c) 2026, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _toy_models.py
#  Environment: Python 3.12

from typing import Optional, Tuple

import torch as th
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, Tuple

from BUCToolkit.utils.index_ops import indices_pairwise_dist, index_reduce, standardize_cell, index_periodic_geodesic_diffvec


class GNNLJDirectionalEAM(nn.Module):
    r"""
    Learnable Lennard‑Jones potential augmented with a local, direction‑aware
    embedded‑atom method (EAM) term.

    The total energy is:
        E = E_LJ + E_EAM
    where E_LJ is a per‑pair learnable LJ potential with element‑wise baseline
    parameters and a neural gating modulation, and E_EAM is a many‑body embedding
    energy:
        E_EAM = \sum_i F_{Z_i}(\rho_i)
        \rho_i = \sum_{j \in \mathcal{N}(i)} \rho_{Z_j}(r_{ij}, \hat{\mathbf{r}}_{ij})

    The density function \rho is built from an element‑pair baseline amplitude,
    multiplied by a gating factor that depends on the relative displacement
    vector (including direction).  No explicit angular term is used.

    Parameters
    ----------
    max_atomic_number : int
        Maximum atomic number Z for the embedding table.
    embedding_dim : int
        Dimension of the atomic‑species embedding vectors.
    hidden_dim : int
        Hidden dimension of the LJ parameter prediction MLP.
    cutoff : float
        Cutoff radius for neighbour‑graph construction.
    min_dist : float
        Minimum allowed interatomic distance for numerical safety.
    eam_hidden_dim : int
        Hidden dimension of the EAM gating and embedding networks.
    """

    def __init__(
        self,
        max_atomic_number: int = 100,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        cutoff: float = 6.0,
        min_dist: float = 0.5,
        eam_hidden_dim: int = 32,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.min_dist = min_dist

        # --- Shared atomic embeddings ---
        self.embed = nn.Embedding(max_atomic_number + 1, embedding_dim, padding_idx=0)

        # --- Learnable LJ potential components ---
        # Parameter prediction network: [v_i, v_j] -> raw_sigma, raw_epsilon
        self.param_net = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        # Element‑wise reference (baseline) parameters
        self.ref_sigma = nn.Embedding(max_atomic_number + 1, 1, padding_idx=0)
        self.ref_epsilon = nn.Embedding(max_atomic_number + 1, 1, padding_idx=0)

        # --- EAM components ---
        # Baseline amplitude for density contribution (one per element)
        self.ref_rho = nn.Embedding(max_atomic_number + 1, 1, padding_idx=0)

        # Gating network for direction‑aware density:
        # input = [v_i, v_j, \hat{r}_{ij}] -> output = gating scalar
        self.gate_net = nn.Sequential(
            nn.Linear(2 * embedding_dim + 3, eam_hidden_dim),
            nn.SiLU(),
            nn.Linear(eam_hidden_dim, eam_hidden_dim),
            nn.SiLU(),
            nn.Linear(eam_hidden_dim, 1),
        )

        # exp. decay coeff
        self.decay_alpha = nn.Embedding(max_atomic_number + 1, 1, padding_idx=0)

        # Embedding energy network: input = [v_i, \rho_i] -> output = embedding energy
        self.embed_net = nn.Sequential(
            nn.Linear(embedding_dim + 1, eam_hidden_dim),
            nn.SiLU(),
            nn.Linear(eam_hidden_dim, eam_hidden_dim),
            nn.SiLU(),
            nn.Linear(eam_hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    #  Graph building
    # ------------------------------------------------------------------
    @staticmethod
    def _build_graph(
        pos: th.Tensor,
        batch: th.Tensor,
        cell: Optional[th.Tensor] = None,
        cutoff: float = 6.0,
        min_dist: float = 0.5,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Build a neighbour graph and return edge indices, distances, unit
        displacement vectors, and edge batch assignments.

        For periodic systems, distances are minimum‑image geodesic distances
        and the unit vectors are derived from the corresponding minimum‑image
        displacement vectors.

        Parameters
        ----------
        pos : th.Tensor, shape (N, D)
            Cartesian coordinates of all atoms.
        batch : th.Tensor, shape (N,), dtype int64
            Structure index for each atom (0 .. n_batch-1).
        cell : th.Tensor, shape (n_batch, D, D) or None
            Lattice cell matrix; each row is a cell vector.  None for
            non‑periodic systems.
        cutoff : float
            Cutoff radius for neighbour searching.
        min_dist : float
            Lower bound for distance to avoid singularities.

        Returns
        -------
        edge_index : th.Tensor, shape (2, E), dtype int64
            Source and destination indices of each edge.
        distances : th.Tensor, shape (E,)
            Pairwise distances (≥ min_dist).
        unit_vec : th.Tensor, shape (E, D)
            Unit vector pointing from source to destination (minimum image).
        edge_batch : th.Tensor, shape (E,), dtype int64
            Structure index of the edge (source atom batch).
        """
        metric_kwargs = {}
        if cell is not None:
            metric_kwargs["R"] = cell

        # Build neighbour list; distances returned are already minimum‑image
        # when metric='periodic'.
        edge_index, distances, edge_batch = indices_pairwise_dist(
            pos, pos,
            batch, None,
            cutoff,
            metric="periodic" if cell is not None else "euclidean",
            metric_kwargs=metric_kwargs or None,
            exclude_diag=True,
            is_symmetric=True,
            return_values=True,
        )
        distances = th.clamp(distances, min=min_dist)

        src, dst = edge_index[0], edge_index[1]
        # Raw difference vectors
        x_diff = pos[dst] - pos[src]                     # (E, D)

        if cell is not None:
            # Map to nearest image using the lattice cell
            batch_src = batch[src]                       # (E,)
            x_diff = index_periodic_geodesic_diffvec(x_diff, cell, batch_src)

        # Unit vector from minimum‑image displacement
        unit_vec = x_diff / distances.unsqueeze(-1)      # (E, D)

        return edge_index, distances, unit_vec, edge_batch

    # ------------------------------------------------------------------
    #  LJ energy (reproduced from original GNNLJPotential)
    # ------------------------------------------------------------------
    def _lj_energy(
        self,
        atomic_numbers: th.Tensor,
        edge_index: th.Tensor,
        distances: th.Tensor,
        edge_batch: th.Tensor,
        n_batch: int,
    ) -> th.Tensor:
        """
        Compute per‑structure LJ energies.

        Parameters
        ----------
        atomic_numbers : th.Tensor, shape (N,), dtype int64
            Atomic numbers of all atoms.
        edge_index : th.Tensor, shape (2, E), dtype int64
            Edge list.
        distances : th.Tensor, shape (E,)
            Pairwise distances.
        edge_batch : th.Tensor, shape (E,), dtype int64
            Batch index of each edge.
        n_batch : int
            Number of structures.

        Returns
        -------
        energy : th.Tensor, shape (n_batch,)
            LJ energy for each structure.
        """
        v = self.embed(atomic_numbers)                   # (N, embedding_dim)
        src, dst = edge_index[0], edge_index[1]

        # Pair embedding
        v_pair = th.cat([v[src], v[dst]], dim=-1)    # (E, 2*embedding_dim)

        # Predicted modulation factors
        raw = self.param_net(v_pair)                     # (E, 2)
        sigma_pred = F.softplus(raw[:, 0]).clamp(max=10.0) + 1e-3
        epsilon_pred = F.softplus(raw[:, 1]).clamp(max=10.0) + 1e-3

        # Baseline parameters from reference embeddings
        ref_sig_src = F.softplus(self.ref_sigma(atomic_numbers[src])).squeeze(-1) + 1e-3
        ref_sig_dst = F.softplus(self.ref_sigma(atomic_numbers[dst])).squeeze(-1) + 1e-3
        ref_eps_src = F.softplus(self.ref_epsilon(atomic_numbers[src])).squeeze(-1) + 1e-3
        ref_eps_dst = F.softplus(self.ref_epsilon(atomic_numbers[dst])).squeeze(-1) + 1e-3

        # Effective pair parameters (baseline × modulation)
        sigma = sigma_pred * 0.5 * (ref_sig_src + ref_sig_dst)
        epsilon = epsilon_pred * th.sqrt(ref_eps_src * ref_eps_dst)

        # LJ energy per edge
        inv_r = 1.0 / distances
        s_r = (sigma * inv_r).clamp(max=100.0)
        s_r_6 = s_r ** 6
        e_edge = 4.0 * epsilon * (s_r_6 * s_r_6 - s_r_6)   # (E,)

        # Sum to structures
        energy = index_reduce(e_edge, edge_batch, dim=0, ops="sum", out_size=n_batch)
        return energy

    # ------------------------------------------------------------------
    #  EAM energy
    # ------------------------------------------------------------------
    def _eam_energy(
        self,
        atomic_numbers: th.Tensor,
        edge_index: th.Tensor,
        distances: th.Tensor,
        unit_vec: th.Tensor,
        edge_batch: th.Tensor,
        batch: th.Tensor,
        n_batch: int,
    ) -> th.Tensor:
        """
        Compute per‑structure embedding energies (EAM term).

        Parameters
        ----------
        atomic_numbers : th.Tensor, shape (N,), dtype int64
        edge_index : th.Tensor, shape (2, E), dtype int64
        distances : th.Tensor, shape (E,)
        unit_vec : th.Tensor, shape (E, D)
            Minimum‑image unit vectors from source to destination.
        edge_batch : th.Tensor, shape (E,), dtype int64
        batch : th.Tensor, shape (N,), dtype int64
            Structure index for each atom.
        n_batch : int

        Returns
        -------
        energy : th.Tensor, shape (n_batch,)
            Embedding energy for each structure.
        """
        v = self.embed(atomic_numbers)                   # (N, embed_dim)
        src, dst = edge_index[0], edge_index[1]

        # --- Direction‑aware density contribution per edge ---
        # Baseline amplitude from reference embeddings
        ref_rho_src = F.softplus(self.ref_rho(atomic_numbers[src])).squeeze(-1)  # (E,)
        ref_rho_dst = F.softplus(self.ref_rho(atomic_numbers[dst])).squeeze(-1)
        base_amp = ref_rho_src + ref_rho_dst              # element‑pair baseline

        # Gating factor: [v_i, v_j, unit_vec] -> scalar
        gate_in = th.cat([v[src], v[dst], unit_vec], dim=-1)  # (E, 2*embed_dim+3)
        gate = F.softplus(self.gate_net(gate_in)).squeeze(-1) + 1e-6  # (E,)

        # Density contribution
        #   decay coeff
        alpha_src = F.softplus(self.decay_alpha(atomic_numbers[src])).squeeze(-1) + 1e-6
        alpha_dst = F.softplus(self.decay_alpha(atomic_numbers[dst])).squeeze(-1) + 1e-6
        alpha = 0.5 * (alpha_src + alpha_dst)
        rho_edge = base_amp * gate * th.exp(-alpha * distances)  # (E,)

        # --- Aggregate density onto central atoms ---
        # edge_batch gives the batch index of each edge's source atom,
        # but we need an atom‑wise aggregation. We use source index directly.
        n_atoms = atomic_numbers.shape[0]
        rho = index_reduce(rho_edge, edge_batch, dim=0, ops="sum", out_size=n_atoms)
        # Note: edge_batch here is per‑edge batch index of the source atom's structure,
        # which exactly identifies the source atom. (Assumes contiguous ordering.)

        # --- Embedding energy F(rho_i, v_i) ---
        embed_in = th.cat([v, rho.unsqueeze(-1)], dim=-1)   # (N, embed_dim+1)
        e_atom = self.embed_net(embed_in).squeeze(-1)          # (N,)

        energy = index_reduce(e_atom, batch, dim=0, ops="sum", out_size=n_batch)
        return energy

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        atomic_numbers_or_graph,
        pos: Optional[th.Tensor] = None,
        batch: Optional[th.Tensor] = None,
        cell: Optional[th.Tensor] = None,
    ) -> dict:
        r"""
        Compute energies and forces.

        Supports two calling conventions:

        1) PyG Batch object:  ``model(graph)``
           The object must have attributes ``pos``, ``batch``, ``atomic_numbers``
           and optionally ``cell``.

        2) Explicit tensors: ``model(atomic_numbers, pos, batch, cell)``

        Parameters
        ----------
        atomic_numbers_or_graph : th.Tensor, shape (N,)  dtype int64, or PyG Batch
        pos : th.Tensor, shape (N, D), optional
        batch : th.Tensor, shape (N,), dtype int64, optional
        cell : th.Tensor, shape (n_batch, D, D), optional

        Returns
        -------
        dict
            ``{"energy": th.Tensor, shape (n_batch,), "forces": th.Tensor, shape (N, D)}``
        """
        # --- Unpack input ---
        if hasattr(atomic_numbers_or_graph, 'pos'):
            graph = atomic_numbers_or_graph
            atomic_numbers = getattr(graph, 'atomic_numbers',
                                     th.ones(graph.pos.shape[0], dtype=th.long,
                                                device=graph.pos.device))
            pos = graph.pos
            batch = graph.batch
            if cell is None:
                cell = getattr(graph, 'cell', None)
        else:
            atomic_numbers = atomic_numbers_or_graph

        n_batch = int(batch.max().item()) + 1
        device = pos.device

        # --- Single differentiable forward pass ---
        with th.enable_grad():
            pos = pos.detach().requires_grad_(True)
            # standardize the cell vector to remove the rotation gauge freedom degree
            if cell is not None:
                cell, rot_ops = standardize_cell(cell)
                rot_scatter = rot_ops.index_select(0, batch)
                pos = th.einsum('bd, bqd -> bq', pos, rot_scatter)

            # Build graph
            edge_index, distances, unit_vec, edge_batch = self._build_graph(
                pos, batch, cell, self.cutoff, self.min_dist
            )

            # LJ contribution
            e_lj = self._lj_energy(atomic_numbers, edge_index, distances,
                                   edge_batch, n_batch)

            # EAM contribution
            e_eam = self._eam_energy(atomic_numbers, edge_index, distances,
                                     unit_vec, edge_batch, batch, n_batch)

            total_energy = e_lj + e_eam

            # Forces from autograd
            grad = th.autograd.grad(
                total_energy.sum(),
                pos,
                create_graph=self.training
            )[0]
            forces = -grad

        return {"energy": total_energy, "forces": forces}
