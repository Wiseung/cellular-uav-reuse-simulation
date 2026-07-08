from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .antenna import (
    best_sector_gain_db,
    best_sector_gain_linear,
    random_interferer_gain_linear,
)
from .config import SimulationConfig
from .geometry import (
    center_site_layout,
    cochannel_interferers,
    edge_user_point,
    load_site_layout,
    perturb_site_positions,
    reuse_distance,
    select_nearest_sites,
)
from .pathloss import hybrid_los_probability, received_power, three_dimensional_distance


@dataclass(frozen=True)
class SirSamples:
    signal_power_mw: np.ndarray
    interference_power_mw: np.ndarray
    noise_power_mw: np.ndarray
    serving_los_probability: np.ndarray | None = None
    interferer_los_probability: np.ndarray | None = None
    serving_antenna_gain_db: np.ndarray | None = None
    interferer_antenna_gain_db: np.ndarray | None = None

    @property
    def sir_linear(self) -> np.ndarray:
        return self.signal_power_mw / np.maximum(self.interference_power_mw, 1e-15)

    @property
    def sinr_linear(self) -> np.ndarray:
        return self.signal_power_mw / np.maximum(
            self.interference_power_mw + self.noise_power_mw,
            1e-15,
        )

    @property
    def sir_db(self) -> np.ndarray:
        return 10.0 * np.log10(self.sir_linear)

    @property
    def sinr_db(self) -> np.ndarray:
        return 10.0 * np.log10(self.sinr_linear)

    def metric_linear(self, metric: str = "sinr") -> np.ndarray:
        if metric == "sinr":
            return self.sinr_linear
        if metric == "sir":
            return self.sir_linear
        raise ValueError(f"Unsupported metric: {metric}")

    def metric_db(self, metric: str = "sinr") -> np.ndarray:
        if metric == "sinr":
            return self.sinr_db
        if metric == "sir":
            return self.sir_db
        raise ValueError(f"Unsupported metric: {metric}")

    def mean_spectral_efficiency(self, metric: str = "sinr") -> float:
        return float(np.mean(np.log2(1.0 + self.metric_linear(metric))))

    def coverage_probability(self, threshold_db: float, metric: str = "sinr") -> float:
        return float(np.mean(self.metric_db(metric) >= threshold_db))

    def percentile_db(self, percentile: float, metric: str = "sinr") -> float:
        return float(np.percentile(self.metric_db(metric), percentile))

    def mean_serving_los_probability(self) -> float | None:
        if self.serving_los_probability is None:
            return None
        return float(np.mean(self.serving_los_probability))

    def mean_interferer_los_probability(self) -> float | None:
        if self.interferer_los_probability is None:
            return None
        return float(np.mean(self.interferer_los_probability))

    def mean_serving_antenna_gain_db(self) -> float | None:
        if self.serving_antenna_gain_db is None:
            return None
        return float(np.mean(self.serving_antenna_gain_db))

    def mean_interferer_antenna_gain_db(self) -> float | None:
        if self.interferer_antenna_gain_db is None:
            return None
        return float(np.mean(self.interferer_antenna_gain_db))


def _sample_lognormal_linear(
    rng: np.random.Generator,
    sigma_db: float | np.ndarray,
    size: tuple[int, ...],
) -> np.ndarray:
    sigma = np.asarray(sigma_db, dtype=float)
    if np.all(sigma == 0.0):
        return np.ones(size, dtype=float)
    shadow_db = rng.normal(0.0, sigma, size=size)
    return np.power(10.0, shadow_db / 10.0)


def _sample_nakagami_power(
    rng: np.random.Generator,
    m: float | np.ndarray,
    size: tuple[int, ...],
) -> np.ndarray:
    shape = np.maximum(np.asarray(m, dtype=float), 1e-3)
    return rng.gamma(shape=shape, scale=1.0 / shape, size=size)


def _pairwise_horizontal_distances(
    user_points: np.ndarray,
    base_station_points: np.ndarray,
) -> np.ndarray:
    offsets = user_points[:, None, :] - base_station_points
    return np.sqrt(np.sum(offsets * offsets, axis=2))


def _effective_interference_mask(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    config: SimulationConfig,
) -> np.ndarray:
    return (
        rng.random(size=shape) < config.resource_activity_factor
    ).astype(float)


def _layout_interferer_sites(config: SimulationConfig, count: int) -> np.ndarray:
    if config.site_layout_csv is None:
        raise ValueError("site_layout_csv is not configured")
    layout_sites = center_site_layout(load_site_layout(config.site_layout_csv))
    return select_nearest_sites(
        layout_sites,
        count=count,
        exclude_origin=True,
    )


def _sample_interferer_sites(
    reuse_factor: int,
    cell_radius: float,
    interferer_count: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    sample_count: int,
) -> np.ndarray:
    if config.site_layout_csv is not None:
        fixed_sites = _layout_interferer_sites(config, interferer_count)
        return np.broadcast_to(fixed_sites, (sample_count, *fixed_sites.shape)).copy()

    base_sites = cochannel_interferers(reuse_factor, cell_radius, count=interferer_count)
    jitter_radius_m = (
        reuse_distance(reuse_factor, cell_radius) * config.site_perturbation_fraction
    )
    return perturb_site_positions(
        base_sites,
        jitter_radius_m=jitter_radius_m,
        rng=rng,
        sample_count=sample_count,
    )


def _assemble_samples(
    signal_power_mw: np.ndarray,
    interference_power_mw: np.ndarray,
    config: SimulationConfig,
    serving_los_probability: np.ndarray | None = None,
    interferer_los_probability: np.ndarray | None = None,
    serving_antenna_gain_db: np.ndarray | None = None,
    interferer_antenna_gain_db: np.ndarray | None = None,
) -> SirSamples:
    noise_power_mw = np.full_like(signal_power_mw, config.thermal_noise_power_mw, dtype=float)
    return SirSamples(
        signal_power_mw=signal_power_mw,
        interference_power_mw=interference_power_mw,
        noise_power_mw=noise_power_mw,
        serving_los_probability=serving_los_probability,
        interferer_los_probability=interferer_los_probability,
        serving_antenna_gain_db=serving_antenna_gain_db,
        interferer_antenna_gain_db=interferer_antenna_gain_db,
    )


def simulate_sir_samples(
    user_points: np.ndarray,
    reuse_factor: int,
    cell_radius: float,
    pathloss_exponent: float,
    config: SimulationConfig,
    rng: np.random.Generator,
    user_height_m: float = 0.0,
    interferer_count: int | None = None,
) -> SirSamples:
    user_points = np.asarray(user_points, dtype=float)
    sample_count = user_points.shape[0]
    interferer_total = interferer_count or config.ground_interferer_count
    terminal_height_m = np.full(
        sample_count,
        config.ground_terminal_height_m + user_height_m,
        dtype=float,
    )

    desired_distance_2d = np.sqrt(np.sum(user_points * user_points, axis=1))
    desired_distance_3d = three_dimensional_distance(
        desired_distance_2d,
        tx_height_m=config.base_station_height_m,
        rx_height_m=terminal_height_m,
    )
    desired_gain_db = best_sector_gain_db(
        np.zeros((1, 2), dtype=float),
        user_points,
        terminal_height_m,
        config,
    ).ravel()
    desired_gain_linear = best_sector_gain_linear(
        np.zeros((1, 2), dtype=float),
        user_points,
        terminal_height_m,
        config,
    ).ravel()
    desired_power_mw = (
        config.tx_power_mw
        * desired_gain_linear
        * received_power(desired_distance_3d, pathloss_exponent, config=config)
        * _sample_lognormal_linear(
            rng,
            config.ground_shadow_sigma_db,
            size=(sample_count,),
        )
        * _sample_nakagami_power(
            rng,
            config.ground_small_scale_m,
            size=(sample_count,),
        )
    )

    interferer_sites = _sample_interferer_sites(
        reuse_factor,
        cell_radius,
        interferer_total,
        config,
        rng,
        sample_count=sample_count,
    )
    interferer_distance_2d = _pairwise_horizontal_distances(user_points, interferer_sites)
    interferer_distance_3d = three_dimensional_distance(
        interferer_distance_2d,
        tx_height_m=config.base_station_height_m,
        rx_height_m=terminal_height_m,
    )
    interferer_sector_gain_linear = random_interferer_gain_linear(
        interferer_sites,
        user_points,
        terminal_height_m,
        config,
        rng,
    )
    interferer_gain_db = 10.0 * np.log10(
        np.sum(interferer_sector_gain_linear, axis=-1)
    )
    interference_power_mw = np.sum(
        config.tx_power_mw
        * interferer_sector_gain_linear
        * received_power(interferer_distance_3d, pathloss_exponent, config=config)[..., None]
        * _sample_lognormal_linear(
            rng,
            config.ground_shadow_sigma_db,
            size=interferer_distance_3d.shape,
        )[..., None]
        * _sample_nakagami_power(
            rng,
            config.ground_small_scale_m,
            size=(*interferer_distance_3d.shape, interferer_sector_gain_linear.shape[-1]),
        )
        * _effective_interference_mask(
            rng,
            (*interferer_distance_3d.shape, interferer_sector_gain_linear.shape[-1]),
            config,
        ),
        axis=(1, 2),
    )

    return _assemble_samples(
        desired_power_mw,
        interference_power_mw,
        config,
        serving_antenna_gain_db=desired_gain_db,
        interferer_antenna_gain_db=np.mean(interferer_gain_db, axis=1),
    )


def edge_user_sir(
    reuse_factor: int,
    cell_radius: float,
    pathloss_exponent: float,
    user_height_m: float = 0.0,
) -> float:
    user_point = edge_user_point(cell_radius)[None, :]
    desired_distance = np.sqrt(
        np.sum(user_point * user_point, axis=1) + user_height_m * user_height_m
    )
    desired_power = received_power(desired_distance, pathloss_exponent)

    first_tier = cochannel_interferers(reuse_factor, cell_radius, count=6)
    first_horizontal_distances = np.sqrt(
        np.sum((user_point[:, None, :] - first_tier[None, :, :]) ** 2, axis=2)
    )
    first_distances = np.sqrt(first_horizontal_distances**2 + user_height_m * user_height_m)
    interference_power = np.sum(
        received_power(first_distances, pathloss_exponent),
        axis=1,
    )
    return float(desired_power[0] / interference_power[0])


def simulate_los_probability_sir_samples(
    user_points: np.ndarray,
    reuse_factor: int,
    cell_radius: float,
    user_altitude_m: float,
    config: SimulationConfig,
    interferer_count: int,
    rng: np.random.Generator,
) -> SirSamples:
    user_points = np.asarray(user_points, dtype=float)
    sample_count = user_points.shape[0]
    terminal_height_m = np.full(
        sample_count,
        config.ground_terminal_height_m + user_altitude_m,
        dtype=float,
    )

    desired_distance_2d = np.sqrt(np.sum(user_points * user_points, axis=1))
    desired_distance_3d = three_dimensional_distance(
        desired_distance_2d,
        tx_height_m=config.base_station_height_m,
        rx_height_m=terminal_height_m,
    )
    desired_los_probability = hybrid_los_probability(
        desired_distance_2d,
        terminal_height_m,
        config,
    )
    desired_is_los = rng.random(size=sample_count) < desired_los_probability
    desired_pathloss_exponent = np.where(
        desired_is_los,
        config.los_pathloss_exponent,
        config.nlos_pathloss_exponent,
    )
    desired_shadow_sigma_db = np.where(
        desired_is_los,
        config.los_shadow_sigma_db,
        config.nlos_shadow_sigma_db,
    )
    desired_fading_m = np.where(
        desired_is_los,
        config.los_small_scale_m,
        config.nlos_small_scale_m,
    )
    desired_gain_db = best_sector_gain_db(
        np.zeros((1, 2), dtype=float),
        user_points,
        terminal_height_m,
        config,
    ).ravel()
    desired_gain_linear = best_sector_gain_linear(
        np.zeros((1, 2), dtype=float),
        user_points,
        terminal_height_m,
        config,
    ).ravel()
    desired_power_mw = (
        config.tx_power_mw
        * desired_gain_linear
        * received_power(desired_distance_3d, desired_pathloss_exponent, config=config)
        * _sample_lognormal_linear(
            rng,
            desired_shadow_sigma_db,
            size=(sample_count,),
        )
        * _sample_nakagami_power(
            rng,
            desired_fading_m,
            size=(sample_count,),
        )
    )

    interferer_sites = _sample_interferer_sites(
        reuse_factor,
        cell_radius,
        interferer_count,
        config,
        rng,
        sample_count=sample_count,
    )
    interferer_distance_2d = _pairwise_horizontal_distances(user_points, interferer_sites)
    interferer_distance_3d = three_dimensional_distance(
        interferer_distance_2d,
        tx_height_m=config.base_station_height_m,
        rx_height_m=terminal_height_m,
    )
    interferer_los_probability = hybrid_los_probability(
        interferer_distance_2d,
        terminal_height_m,
        config,
    )
    interferer_is_los = rng.random(size=interferer_distance_2d.shape) < interferer_los_probability
    interferer_pathloss_exponent = np.where(
        interferer_is_los,
        config.los_pathloss_exponent,
        config.nlos_pathloss_exponent,
    )
    interferer_shadow_sigma_db = np.where(
        interferer_is_los,
        config.los_shadow_sigma_db,
        config.nlos_shadow_sigma_db,
    )
    interferer_fading_m = np.where(
        interferer_is_los,
        config.los_small_scale_m,
        config.nlos_small_scale_m,
    )
    interferer_sector_gain_linear = random_interferer_gain_linear(
        interferer_sites,
        user_points,
        terminal_height_m,
        config,
        rng,
    )
    interferer_gain_db = 10.0 * np.log10(
        np.sum(interferer_sector_gain_linear, axis=-1)
    )
    interference_power_mw = np.sum(
        config.tx_power_mw
        * interferer_sector_gain_linear
        * received_power(interferer_distance_3d, interferer_pathloss_exponent, config=config)[..., None]
        * _sample_lognormal_linear(
            rng,
            interferer_shadow_sigma_db,
            size=interferer_distance_3d.shape,
        )[..., None]
        * _sample_nakagami_power(
            rng,
            interferer_fading_m[..., None],
            size=(*interferer_distance_3d.shape, interferer_sector_gain_linear.shape[-1]),
        )
        * _effective_interference_mask(
            rng,
            (*interferer_distance_3d.shape, interferer_sector_gain_linear.shape[-1]),
            config,
        ),
        axis=(1, 2),
    )

    return _assemble_samples(
        desired_power_mw,
        interference_power_mw,
        config,
        serving_los_probability=desired_los_probability,
        interferer_los_probability=np.mean(interferer_los_probability, axis=1),
        serving_antenna_gain_db=desired_gain_db,
        interferer_antenna_gain_db=np.mean(interferer_gain_db, axis=1),
    )
