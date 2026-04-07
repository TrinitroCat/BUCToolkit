import torch as th
import torch.nn as nn
from typing import List, Union, Tuple
import itertools
from torch_geometric.data import Data, Batch


class HarmonicLatticePotential(nn.Module):
    r"""
    Harmonic lattice potential for cubic spring networks.

    For each edge (i, j) with distance d = ||pos_i - pos_j||, the energy is:
        E_edge = 0.5 * k * (d - r0)^2
    Total energy is the sum over all edges. Forces are derived as:
        F_i += -k * (d - r0) * (dr / d)
        F_j += +k * (d - r0) * (dr / d)

    Supports both single graphs and batched graphs (using `th_geometric.Batch`).
    For batched input, the `data` object must contain a `batch` attribute indicating
    which graph each node belongs to. The output `energy` will be a 1D tensor of shape
    (num_graphs,), and `forces` will be a tensor of shape (N, 3) concatenated across graphs.

    Args:
        k (float): Spring constant.
        r0 (float): Equilibrium bond length.
        eps (float): Small constant to avoid division by zero.
    """
    def __init__(self, k: float = 1.0, r0: float = 1.0, eps: float = 1e-20):
        super().__init__()
        self.k = k
        self.r0 = r0
        self.eps = eps

    def forward(self, data) -> dict:
        r"""
        Compute total potential energy and forces.

        Args:
            data: A `torch_geometric.data.Data` or `Batch` object containing:
                - pos: Node positions, shape (N, 3)
                - edge_index: Graph connectivity, shape (2, E)
                - batch (optional): Node assignment to graphs, shape (N,)

        Returns:
            dict with keys:
                - 'energy': scalar for a single graph, or (num_graphs,) tensor for batched input.
                - 'forces': (N, 3) tensor of forces on each node.
        """
        pos = data.pos                     # (N, 3)
        edge_index = data.edge_index       # (2, E)
        i, j = edge_index[0], edge_index[1]

        dr = pos[i] - pos[j]               # (E, 3)
        d = th.linalg.norm(dr, dim=1)          # (E,)

        delta = d - self.r0
        edge_energy = 0.5 * self.k * delta ** 2   # (E,)

        # Determine graph assignment for edges
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch             # (N,)
            num_graphs = batch.max().item() + 1
            edge_batch = batch[i]          # (E,) – each edge belongs to the graph of its source node
            # Sum edge energies per graph
            energy = th.zeros(num_graphs, device=pos.device)
            energy.index_add_(0, edge_batch, edge_energy)
        else:
            # Single graph
            energy = edge_energy.sum()

        # Forces
        force_mag = self.k * delta         # (E,)
        direction = dr / (d + self.eps).unsqueeze(1)   # (E, 3)
        force_contrib = force_mag.unsqueeze(1) * direction  # (E, 3)

        forces = th.zeros_like(pos)     # (N, 3)
        forces.index_add_(0, i, -force_contrib)
        forces.index_add_(0, j, force_contrib)

        energy = energy.view(-1)       # ensure shape (num_graphs,)

        return {'energy': energy, 'forces': forces}


class SimpleSpringPotential(nn.Module):
    """
    A Simple Cubic Grids Spring Model. N^3 independent harmonics.
    """

    def __init__(self, pos_init, k: float = 1.0):
        super().__init__()
        self.register_buffer('pos_init', pos_init)
        self.k = k

    def forward(self, data):
        dr = data.pos - self.pos_init  # (N,3)
        # 势能：0.5 * k * sum(dr^2)
        num_graphs = data.batch.max().item() + 1
        energy = th.zeros(num_graphs, device=data.pos.device)
        energy.index_add_(0, data.batch, 0.5 * self.k * th.sum(dr ** 2, dim=-1))
        # 力：-k * dr
        forces = -self.k * dr

        return {'energy': energy, 'forces': forces}


def build_cubic_lattice_data(
        n: int,
        spacing: float = 1.0,
        perturb_std: float = 0.0
):
    r"""
    Generate a single cubic lattice graph.

    Nodes are placed on a regular 3D grid with side length `n` (n^3 nodes).
    Edges connect nearest neighbours along x, y, z directions (no periodic boundaries).
    Positions can be optionally perturbed by Gaussian noise.

    Args:
        n (int): Number of grid points along each dimension.
        spacing (float): Equilibrium bond length (distance between neighbours).
        perturb_std (float): Standard deviation of random displacement added to each node.

    Returns:
        Data object with `pos` and `edge_index`.
    """
    # Node coordinates in equilibrium
    coords = list(itertools.product(range(n), repeat=3))
    pos0 = th.tensor(coords, dtype=th.float) * spacing  # (n^3, 3)

    # Add random perturbation if requested
    if perturb_std > 0:
        pos = th.add(pos0, th.randn_like(pos0), alpha=perturb_std)
    else:
        pos = pos0.clone()

    # Build edges: only positive directions (dx,dy,dz) = (1,0,0), (0,1,0), (0,0,1)
    idx_map = {node: i for i, node in enumerate(coords)}
    edges = []
    for (x, y, z), idx in idx_map.items():
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            nb = (x + dx, y + dy, z + dz)
            if nb in idx_map:
                edges.append([idx, idx_map[nb]])

    edge_index = th.tensor(edges, dtype=th.long).t().contiguous()
    return Data(pos=pos, pos0=pos0, edge_index=edge_index)


def build_cubic_lattice_batch(
        sizes: List[int],
        spacing: float = 1.0,
        perturb_std: float = 0.0
) -> Batch:
    r"""
    Generate a batch of cubic lattices of different sizes.

    Each lattice is independent and will be concatenated into a single `Batch` object.
    The returned batch has `pos`, `edge_index`, and `batch` attributes.

    Args:
        sizes (List[int]): List of side lengths (n) for each lattice.
        spacing (float): Equilibrium bond length (shared across all lattices).
        perturb_std (float): Standard deviation of random displacement (shared).

    Returns:
        Batch object containing all graphs.
    """
    data_list = [build_cubic_lattice_data(n, spacing, perturb_std) for n in sizes]
    return Batch.from_data_list(data_list)


# Example usage
if __name__ == "__main__":
    # Single graph: 3x3x3 lattice
    data_single = build_cubic_lattice_data(3, spacing=1.0, perturb_std=0.05)
    model = HarmonicLatticePotential(k=10.0, r0=1.0)
    out_single = model(data_single)
    print("Single graph energy:", out_single['energy'].item())
    print("Single graph forces shape:", out_single['forces'].shape)

    # Batch: three lattices of sizes 2, 3, 4
    batch_data = build_cubic_lattice_batch([8, 5, 10], spacing=1.0, perturb_std=0.05)
    out_batch = model(batch_data)
    print("Batch energies:", out_batch['energy'])
    print("Batch forces shape:", out_batch['forces'].shape)