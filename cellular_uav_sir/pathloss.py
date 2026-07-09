from __future__ import annotations

import math

import numpy as np

from .config import SimulationConfig

THREE_GPP_UMA_DECAY_M = 63.0
THREE_GPP_UMA_DISTANCE_BREAK_M = 18.0


def three_dimensional_distance(
    horizontal_distance_m: np.ndarray,
    tx_height_m: float,
    rx_height_m: float | np.ndarray,
) -> np.ndarray:
    horizontal_distance = np.asarray(horizontal_distance_m, dtype=float)
    rx_height = np.asarray(rx_height_m, dtype=float)
    if rx_height.ndim == 1 and horizontal_distance.ndim > 1:
        rx_height = rx_height[:, None]
    return np.sqrt(horizontal_distance**2 + (tx_height_m - rx_height) ** 2)


def uma_los_probability(
    distance_2d_m: float | np.ndarray,
    user_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    distance = np.maximum(np.asarray(distance_2d_m, dtype=float), 1.0)
    height = np.asarray(user_height_m, dtype=float)
    if height.ndim == 1 and distance.ndim > 1:
        height = height[:, None]
    capped_height = np.clip(
        height,
        config.ground_terminal_height_m,
        config.uma_max_user_height_m,
    )
    c_prime = np.zeros_like(capped_height, dtype=float)
    elevated_mask = capped_height > 13.0
    c_prime[elevated_mask] = np.power(
        (capped_height[elevated_mask] - 13.0) / 10.0,
        1.5,
    )
    base_probability = (
        THREE_GPP_UMA_DISTANCE_BREAK_M / distance
        + np.exp(-distance / THREE_GPP_UMA_DECAY_M)
        * (1.0 - THREE_GPP_UMA_DISTANCE_BREAK_M / distance)
    )
    height_gain = 1.0 + c_prime * 1.25 * np.power(distance / 100.0, 3.0) * np.exp(
        -distance / 150.0
    )
    probability = np.where(distance <= THREE_GPP_UMA_DISTANCE_BREAK_M, 1.0, base_probability * height_gain)
    return np.clip(probability, 0.0, 1.0)


def itu_statistical_los_probability(
    distance_2d_m: float | np.ndarray,
    tx_height_m: float,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    distance = np.asarray(distance_2d_m, dtype=float)
    if distance.size == 0:
        return np.ones_like(distance, dtype=float)
    rx_height = np.asarray(rx_height_m, dtype=float)
    if rx_height.ndim == 1 and distance.ndim > 1:
        rx_height = rx_height[:, None]
    distance_km = np.maximum(distance / 1000.0, 0.0)
    building_crossings = np.floor(
        distance_km * math.sqrt(config.itu_alpha * config.itu_beta_buildings_per_km2)
    ).astype(int)
    probability = np.ones_like(distance_km, dtype=float)
    max_crossings = int(np.max(building_crossings))

    if max_crossings <= 0:
        return probability

    building_index = np.arange(max_crossings, dtype=float)
    building_count = np.maximum(building_crossings, 1)[..., None]
    path_fraction = (building_index + 0.5) / building_count
    los_height = tx_height_m + path_fraction * (rx_height[..., None] - tx_height_m)
    non_block_probability = 1.0 - np.exp(
        -(los_height**2) / (2.0 * config.itu_gamma_m**2)
    )
    valid_mask = building_index < building_crossings[..., None]
    return np.prod(np.where(valid_mask, non_block_probability, 1.0), axis=-1)


def hybrid_los_probability(
    distance_2d_m: float | np.ndarray,
    user_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    distance = np.asarray(distance_2d_m, dtype=float)
    height = np.asarray(user_height_m, dtype=float)
    if height.ndim == 1 and distance.ndim > 1:
        height = height[:, None]
    p_3gpp = uma_los_probability(distance, height, config)
    p_itu = itu_statistical_los_probability(
        distance,
        tx_height_m=config.base_station_height_m,
        rx_height_m=height,
        config=config,
    )
    blend = np.clip(
        (height - config.uma_max_user_height_m)
        / max(config.itu_blend_top_height_m - config.uma_max_user_height_m, 1.0),
        0.0,
        1.0,
    )
    probability = (1.0 - blend) * p_3gpp + blend * p_itu
    return np.clip(probability, 0.0, 1.0)


def expected_link_received_power(
    distance_m: np.ndarray,
    los_probability: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    return los_probability * received_power(
        distance_m,
        config.los_pathloss_exponent,
        config=config,
    ) + (1.0 - los_probability) * received_power(
        distance_m,
        config.nlos_pathloss_exponent,
        config=config,
    )


def received_power(
    distance_m: np.ndarray,
    pathloss_exponent: float,
    config: SimulationConfig | None = None,
) -> np.ndarray:
    distance = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    if config is None:
        return np.power(distance, -pathloss_exponent)

    reference_distance = max(config.reference_distance_m, 1e-3)
    pathloss_db = config.reference_pathloss_db + 10.0 * pathloss_exponent * np.log10(
        distance / reference_distance
    )
    return np.power(10.0, -pathloss_db / 10.0)
