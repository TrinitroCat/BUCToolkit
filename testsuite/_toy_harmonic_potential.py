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
        dr = data.pos - data.pos0  # (N,3)
        # pot：0.5 * k * sum(dr^2)
        num_graphs = data.batch.max().item() + 1
        energy = th.zeros(num_graphs, device=data.pos.device)
        energy.index_add_(0, data.batch, 0.5 * self.k * th.sum(dr ** 2, dim=-1))
        # forces：-k * dr
        forces = -self.k * dr

        return {'energy': energy, 'forces': forces}


class FreeParticles(nn.Module):
    """
    Particles move totally free without any potentials and Forces.
    """

    def __init__(self, pos_init, k: float = 1.0):
        super().__init__()
        self.register_buffer('pos_init', pos_init)
        self.k = k
        self.energy = None
        self.forces = None

    def forward(self, data):
        if self.energy is None:
            num_graphs = data.batch.max().item() + 1
            self.energy = th.zeros(num_graphs, device=data.pos.device)
            self.forces = th.zeros_like(data.pos)
        self.energy = self.energy.to(data.pos.device)
        self.forces = self.forces.to(data.pos.device)

        return {'energy': self.energy, 'forces': self.forces}


class LennardJonesCluster(nn.Module):
    """
    Lennard-Jones potential for a cluster of atoms.
    Supports per-graph neighbor calculation via a simple distance matrix.
    """
    def __init__(self, epsilon=1.0, sigma=1.0, cutoff=None):
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

    def forward(self, data):
        with th.enable_grad():
            pos = data.pos
            batch = data.batch
            device = pos.device
            num_graphs = batch.max().item() + 1

            pos.requires_grad_(True)
            energy = th.zeros(num_graphs, device=device)
            for g in range(num_graphs):
                mask = (batch == g)
                pos_g = pos[mask]
                n_atoms = pos_g.shape[0]
                if n_atoms < 2:
                    continue
                # 计算距离矩阵
                diff = pos_g.unsqueeze(1) - pos_g.unsqueeze(0)  # (n, n, 3)
                dist = th.norm(diff, dim=2)  # (n, n)
                # 避免 i==j 和重复计算
                mask_upper = th.triu(th.ones_like(dist), diagonal=1).bool()
                dist_pairs = dist[mask_upper]

                if self.cutoff is not None:
                    dist_pairs = th.where(dist_pairs < self.cutoff, dist_pairs, th.tensor(float('inf'), device=device))

                inv_r6 = (self.sigma / dist_pairs) ** 6
                e_pair = 4 * self.epsilon * (inv_r6 ** 2 - inv_r6)
                # 忽略无穷远
                e_pair = th.nan_to_num(e_pair, nan=0.0, posinf=0.0)
                energy[g] = e_pair.sum()

            # 自动微分求力
            forces = -th.autograd.grad(
                outputs=energy.sum(),
                inputs=pos,
                create_graph=self.training,
            )[0]

            return {'energy': energy, 'forces': forces}

class MullerBrownPotential(nn.Module):
    """
    二维 Mueller-Brown 势，平移到以所有 pos0 的平均位置为中心。
    势能定义在 xy 平面，z 方向自由（能量恒为0）。
    """
    def __init__(self, center_at=(0.0, 0.0)):
        super().__init__()
        # Müller-Brown 参数
        self.register_buffer('A', th.tensor([-200.0, -100.0, -170.0, 15.0]))
        self.register_buffer('a', th.tensor([-1.0, -1.0, -6.5, 0.7]))
        self.register_buffer('b', th.tensor([0.0, 0.0, 11.0, 0.6]))
        self.register_buffer('c', th.tensor([-10.0, -10.0, -6.5, 0.7]))
        self.register_buffer('x0', th.tensor([1.0, 0.0, -0.5, -1.0]))
        self.register_buffer('y0', th.tensor([0.0, 0.5, 1.5, 1.0]))
        self.center_at = center_at   # 势能面上期望与 pos0 平均位置重合的点

    def forward(self, data):
        pos = data.pos
        pos0 = data.pos0
        batch = data.batch
        device = pos.device

        with th.enable_grad():
            pos.requires_grad_(True)
            center_pos0 = pos0.mean(dim=0)  # (3,)

            # 平移量：使得 center_pos0 对应势能面上的 center_at
            shift_x = center_pos0[0] - self.center_at[0]
            shift_y = center_pos0[1] - self.center_at[1]

            # 计算相对坐标（减去平移量，使势能面移动）
            x = pos[:, 0] - shift_x
            y = pos[:, 1] - shift_y

            e_per_atom = th.zeros(pos.shape[0], device=device)
            for i in range(4):
                dx = x - self.x0[i].to(device)
                dy = y - self.y0[i].to(device)
                e_per_atom += self.A[i].to(device) * th.exp(
                    self.a[i].to(device) * dx**2 +
                    self.b[i].to(device) * dx * dy +
                    self.c[i].to(device) * dy**2
                )

            # z 方向无贡献（已隐含在 e_per_atom 中不依赖 z）

            num_graphs = batch.max().item() + 1
            energy = th.zeros(num_graphs, device=device)
            energy.index_add_(0, batch, e_per_atom)

            forces = -th.autograd.grad(
                outputs=energy.sum(),
                inputs=pos,
                create_graph=self.training,
            )[0]
        # 理论 z 方向力为零，手动置零保证数值稳定
        forces[:, 2] = 0.0

        return {'energy': energy, 'forces': forces}


class DoubleWellPotential(nn.Module):
    """
    每原子独立的双势阱（沿局部 x 方向），其余方向为简谐势。
    势能形式：U(dx,dy,dz) = a * (dx - d)^2 * (dx + d)^2 + 0.5*k_y*dy^2 + 0.5*k_z*dz^2
    其中 dx = x - x0，dy = y - y0，dz = z - z0。
    - 若 well_offset = d > 0，极小值位于 dx = ±d，pos0 为鞍点。
    - 若 well_offset = 0，极小值位于 dx = 0（即 pos0 本身）。
    """
    def __init__(self, well_offset=1.0, a=1.0, k_y=1.0, k_z=1.0):
        super().__init__()
        self.d = well_offset          # 势阱偏离中心的距离
        self.a = a
        self.k_y = k_y
        self.k_z = k_z

    def forward(self, data):
        pos = data.pos
        pos0 = data.pos0
        batch = data.batch

        with th.enable_grad():
            pos.requires_grad_(True)
            # 相对位移
            dr = pos - pos0                # (N, 3)
            dx, dy, dz = dr[:, 0], dr[:, 1], dr[:, 2]

            # 双势阱项：a * (dx - d)^2 * (dx + d)^2
            dw = self.a * ((dx - self.d) ** 2) * ((dx + self.d) ** 2)
            # 简谐项
            harm = 0.5 * self.k_y * dy**2 + 0.5 * self.k_z * dz**2

            e_per_atom = dw + harm

            # 按图聚合能量
            num_graphs = batch.max().item() + 1
            energy = th.zeros(num_graphs, device=pos.device)
            energy.index_add_(0, batch, e_per_atom)

            # 显式启用梯度以计算力
            forces = -th.autograd.grad(
                outputs=energy.sum(),
                inputs=pos,
                create_graph=self.training,
            )[0]

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