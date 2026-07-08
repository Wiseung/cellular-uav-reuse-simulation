from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def _wrap_angle_deg(angle_deg: np.ndarray) -> np.ndarray:
    return np.abs((angle_deg + 180.0) % 360.0 - 180.0)


def sector_gain_db(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    users = np.asarray(user_points, dtype=float)
    sites = np.asarray(site_positions, dtype=float)
    if sites.ndim == 2:
        offsets = users[:, None, :] - sites[None, :, :]
    elif sites.ndim == 3:
        offsets = users[:, None, :] - sites
    else:
        raise ValueError("site_positions must have shape (sites, 2) or (samples, sites, 2)")

    rx_height = np.asarray(rx_height_m, dtype=float)
    if rx_height.ndim == 0:
        rx_height = np.full(users.shape[0], float(rx_height))
    elif rx_height.shape != (users.shape[0],):
        raise ValueError("rx_height_m must be a scalar or a vector matching user_points")

    horizontal_distance = np.sqrt(np.sum(offsets * offsets, axis=-1))
    azimuth_deg = np.degrees(np.arctan2(offsets[..., 1], offsets[..., 0]))
    sector_azimuths = np.asarray(config.sector_azimuths_deg, dtype=float)
    horizontal_offset_deg = _wrap_angle_deg(azimuth_deg[..., None] - sector_azimuths)
    horizontal_attenuation_db = np.minimum(
        12.0 * (horizontal_offset_deg / config.horizontal_beamwidth_deg) ** 2,
        config.horizontal_max_attenuation_db,
    )

    elevation_deg = np.degrees(
        np.arctan2(
            rx_height[:, None] - config.base_station_height_m,
            np.maximum(horizontal_distance, 1.0),
        )
    )
    vertical_offset_deg = np.abs(elevation_deg + config.total_downtilt_deg)
    vertical_attenuation_db = np.minimum(
        12.0 * (vertical_offset_deg / config.vertical_beamwidth_deg) ** 2,
        config.vertical_max_attenuation_db,
    )

    total_attenuation_db = np.minimum(
        horizontal_attenuation_db + vertical_attenuation_db[..., None],
        config.max_pattern_attenuation_db,
    )
    return -total_attenuation_db


def sector_gain_linear(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    return np.power(
        10.0,
        sector_gain_db(site_positions, user_points, rx_height_m, config) / 10.0,
    )


def best_sector_gain_db(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    return np.max(
        sector_gain_db(site_positions, user_points, rx_height_m, config),
        axis=-1,
    )


def best_sector_gain_linear(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    return np.max(
        sector_gain_linear(site_positions, user_points, rx_height_m, config),
        axis=-1,
    )
