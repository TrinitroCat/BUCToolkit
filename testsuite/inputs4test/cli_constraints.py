"""Constraint functions used by the non-interactive CLI smoke tests."""

import torch


def first_pair_distance(coordinates: torch.Tensor) -> torch.Tensor:
    """Return the distance between the first two atoms.

    Args:
        coordinates: Cartesian coordinates with shape ``(n_atom, 3)``.

    Returns:
        A one-element tensor containing the interatomic distance.

    Raises:
        ValueError: If fewer than two atoms are provided.
    """
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError(
            "`coordinates` must have shape (n_atom, 3) with at least two atoms."
        )
    return torch.linalg.norm(coordinates[1] - coordinates[0]).reshape(1)


def first_pair_distance_target(time_now: torch.Tensor) -> torch.Tensor:
    """Return one constant target while retaining a differentiable time input."""
    return (2.0 + 0.0 * time_now).reshape(1, 1)
