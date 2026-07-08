from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .config import REUSE_PAIR_BY_FACTOR

SQRT3 = math.sqrt(3.0)


def axial_to_cartesian(q: int, r: int, cell_radius: float) -> np.ndarray:
    x = cell_radius * SQRT3 * (q + 0.5 * r)
    y = cell_radius * 1.5 * r
    return np.array([x, y], dtype=float)


def reuse_distance(reuse_factor: int, cell_radius: float) -> float:
    return math.sqrt(3.0 * reuse_factor) * cell_radius


@lru_cache(maxsize=None)
def _cochannel_interferers_cached(
    reuse_factor: int,
    cell_radius: float,
    count: int,
    lattice_span: int,
) -> tuple[tuple[float, float], ...]:
    if reuse_factor not in REUSE_PAIR_BY_FACTOR:
        raise ValueError(f"Unsupported reuse factor: {reuse_factor}")

    i, j = REUSE_PAIR_BY_FACTOR[reuse_factor]
    basis_a = np.array([i, j], dtype=int)
    basis_b = np.array([-j, i + j], dtype=int)
    points: dict[tuple[float, float], np.ndarray] = {}

    for a in range(-lattice_span, lattice_span + 1):
        for b in range(-lattice_span, lattice_span + 1):
            if a == 0 and b == 0:
                continue
            axial = a * basis_a + b * basis_b
            point = axial_to_cartesian(int(axial[0]), int(axial[1]), cell_radius)
            key = tuple(np.round(point, 9))
            points[key] = point

    ordered = sorted(points.values(), key=lambda point: float(np.linalg.norm(point)))
    return tuple(tuple(point) for point in ordered[:count])


def cochannel_interferers(
    reuse_factor: int,
    cell_radius: float,
    count: int = 6,
    lattice_span: int = 4,
) -> np.ndarray:
    return np.array(
        _cochannel_interferers_cached(reuse_factor, cell_radius, count, lattice_span),
        dtype=float,
    )


def perturb_site_positions(
    site_positions: np.ndarray,
    jitter_radius_m: float,
    rng: np.random.Generator,
    sample_count: int | None = None,
) -> np.ndarray:
    sites = np.asarray(site_positions, dtype=float)
    if jitter_radius_m <= 0.0:
        if sample_count is None:
            return sites.copy()
        return np.broadcast_to(sites, (sample_count, *sites.shape)).copy()

    if sample_count is None:
        radii = jitter_radius_m * np.sqrt(rng.random(size=sites.shape[0]))
        angles = rng.uniform(0.0, 2.0 * math.pi, size=sites.shape[0])
    else:
        radii = jitter_radius_m * np.sqrt(rng.random(size=(sample_count, sites.shape[0])))
        angles = rng.uniform(0.0, 2.0 * math.pi, size=(sample_count, sites.shape[0]))

    offsets = np.stack((radii * np.cos(angles), radii * np.sin(angles)), axis=-1)
    return sites + offsets


def hexagon_vertices(cell_radius: float, center: np.ndarray | None = None) -> np.ndarray:
    angles = np.deg2rad([90.0, 30.0, -30.0, -90.0, -150.0, 150.0])
    vertices = np.column_stack(
        (cell_radius * np.cos(angles), cell_radius * np.sin(angles))
    )
    if center is not None:
        vertices = vertices + np.asarray(center, dtype=float)
    return vertices


def edge_user_point(cell_radius: float) -> np.ndarray:
    return np.array([0.0, cell_radius], dtype=float)


def sample_points_in_hexagon(
    count: int,
    cell_radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x_limit = SQRT3 * cell_radius / 2.0
    accepted: list[np.ndarray] = []
    total = 0

    while total < count:
        batch = max(512, 2 * (count - total))
        x = rng.uniform(-x_limit, x_limit, size=batch)
        y = rng.uniform(-cell_radius, cell_radius, size=batch)
        mask = np.abs(y) <= (cell_radius - np.abs(x) / SQRT3)
        points = np.column_stack((x[mask], y[mask]))
        if points.size == 0:
            continue
        accepted.append(points)
        total += len(points)

    stacked = np.vstack(accepted)
    return stacked[:count]
