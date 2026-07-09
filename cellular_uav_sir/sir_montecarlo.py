from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .antenna import (
    best_sector_gain_db,
    best_sector_gain_linear,
    random_interferer_gain_linear,
)
from .config import SimulationConfig
from .dynamic_network import (
    _build_dynamic_site_parameters,
    _cochannel_interferer_mask,
)
from .geometry import (
    center_site_layout,
    cochannel_interferers,
    edge_user_point,
    load_site_ground_elevation_offsets_m,
    load_site_layout,
    load_site_layout_rows,
    perturb_site_positions,
    reuse_distance,
    select_nearest_sites,
)
from .parameter_profile import resolve_runtime_config
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


@dataclass(frozen=True)
class StaticLayoutSiteBundle:
    serving_position_xy_m: np.ndarray
    serving_tx_height_m: float
    serving_nominal_tx_power_mw: float
    serving_sector_azimuths_deg: np.ndarray
    serving_sector_mask: np.ndarray
    serving_total_downtilt_deg: float
    serving_peak_gain_db: float
    serving_beamforming_array_gain_db: float
    interferer_positions_xy_m: np.ndarray
    interferer_tx_heights_m: np.ndarray
    interferer_nominal_tx_power_mw: np.ndarray
    interferer_sector_azimuths_deg: np.ndarray
    interferer_sector_mask: np.ndarray
    interferer_total_downtilt_deg: np.ndarray
    interferer_peak_gain_db: np.ndarray
    interferer_beamforming_array_gain_db: np.ndarray


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


def _static_layout_site_bundle(
    config: SimulationConfig,
    interferer_count: int,
) -> StaticLayoutSiteBundle:
    if config.site_layout_csv is None:
        raise ValueError("site_layout_csv is not configured")

    raw_site_rows = load_site_layout_rows(config.site_layout_csv)
    raw_site_positions = load_site_layout(config.site_layout_csv)
    centered_positions = center_site_layout(raw_site_positions)
    site_ground_offsets_m = load_site_ground_elevation_offsets_m(
        config.site_layout_csv,
        site_positions=raw_site_positions,
    )
    site_parameters = _build_dynamic_site_parameters(
        raw_site_rows,
        site_ground_offsets_m,
        config,
    )
    norms = np.linalg.norm(centered_positions, axis=1)
    serving_site_index = int(np.argmin(norms))
    cochannel_mask = _cochannel_interferer_mask(site_parameters, serving_site_index)
    cochannel_indices = np.flatnonzero(cochannel_mask)
    if cochannel_indices.size:
        ordered_indices = cochannel_indices[np.argsort(norms[cochannel_indices])]
        selected_indices = ordered_indices[:interferer_count]
    else:
        selected_indices = np.array([], dtype=int)

    return StaticLayoutSiteBundle(
        serving_position_xy_m=centered_positions[serving_site_index],
        serving_tx_height_m=float(site_parameters.tx_heights_m[serving_site_index]),
        serving_nominal_tx_power_mw=float(site_parameters.nominal_tx_power_mw[serving_site_index]),
        serving_sector_azimuths_deg=site_parameters.sector_azimuths_deg[serving_site_index],
        serving_sector_mask=site_parameters.sector_mask[serving_site_index],
        serving_total_downtilt_deg=float(site_parameters.total_downtilt_deg[serving_site_index]),
        serving_peak_gain_db=float(site_parameters.peak_gain_db[serving_site_index]),
        serving_beamforming_array_gain_db=float(site_parameters.beamforming_array_gain_db[serving_site_index]),
        interferer_positions_xy_m=centered_positions[selected_indices],
        interferer_tx_heights_m=site_parameters.tx_heights_m[selected_indices],
        interferer_nominal_tx_power_mw=site_parameters.nominal_tx_power_mw[selected_indices],
        interferer_sector_azimuths_deg=site_parameters.sector_azimuths_deg[selected_indices],
        interferer_sector_mask=site_parameters.sector_mask[selected_indices],
        interferer_total_downtilt_deg=site_parameters.total_downtilt_deg[selected_indices],
        interferer_peak_gain_db=site_parameters.peak_gain_db[selected_indices],
        interferer_beamforming_array_gain_db=site_parameters.beamforming_array_gain_db[selected_indices],
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
    config = resolve_runtime_config(config)
    user_points = np.asarray(user_points, dtype=float)
    sample_count = user_points.shape[0]
    interferer_total = interferer_count or config.ground_interferer_count
    terminal_height_m = np.full(
        sample_count,
        config.ground_terminal_height_m + user_height_m,
        dtype=float,
    )
    layout_bundle = (
        _static_layout_site_bundle(config, interferer_total)
        if config.site_layout_csv is not None
        else None
    )

    serving_site_position = (
        layout_bundle.serving_position_xy_m[None, :]
        if layout_bundle is not None
        else np.zeros((1, 2), dtype=float)
    )
    serving_tx_height_m = (
        layout_bundle.serving_tx_height_m
        if layout_bundle is not None
        else config.base_station_height_m
    )
    serving_nominal_tx_power_mw = (
        layout_bundle.serving_nominal_tx_power_mw
        if layout_bundle is not None
        else config.tx_power_mw
    )

    desired_offsets = user_points - serving_site_position
    desired_distance_2d = np.sqrt(np.sum(desired_offsets * desired_offsets, axis=1))
    desired_distance_3d = three_dimensional_distance(
        desired_distance_2d,
        tx_height_m=serving_tx_height_m,
        rx_height_m=terminal_height_m,
    )
    desired_gain_db = best_sector_gain_db(
        serving_site_position,
        user_points,
        terminal_height_m,
        config,
        tx_height_m=serving_tx_height_m,
        sector_azimuths_deg=(
            layout_bundle.serving_sector_azimuths_deg if layout_bundle is not None else None
        ),
        total_downtilt_deg=(
            layout_bundle.serving_total_downtilt_deg if layout_bundle is not None else None
        ),
        peak_gain_db=layout_bundle.serving_peak_gain_db if layout_bundle is not None else None,
        beamforming_array_gain_db=(
            layout_bundle.serving_beamforming_array_gain_db if layout_bundle is not None else None
        ),
        sector_mask=layout_bundle.serving_sector_mask if layout_bundle is not None else None,
    ).ravel()
    desired_gain_linear = best_sector_gain_linear(
        serving_site_position,
        user_points,
        terminal_height_m,
        config,
        tx_height_m=serving_tx_height_m,
        sector_azimuths_deg=(
            layout_bundle.serving_sector_azimuths_deg if layout_bundle is not None else None
        ),
        total_downtilt_deg=(
            layout_bundle.serving_total_downtilt_deg if layout_bundle is not None else None
        ),
        peak_gain_db=layout_bundle.serving_peak_gain_db if layout_bundle is not None else None,
        beamforming_array_gain_db=(
            layout_bundle.serving_beamforming_array_gain_db if layout_bundle is not None else None
        ),
        sector_mask=layout_bundle.serving_sector_mask if layout_bundle is not None else None,
    ).ravel()
    desired_power_mw = (
        serving_nominal_tx_power_mw
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

    if layout_bundle is not None:
        interferer_sites = np.broadcast_to(
            layout_bundle.interferer_positions_xy_m,
            (sample_count, *layout_bundle.interferer_positions_xy_m.shape),
        ).copy()
        interferer_tx_heights_m = layout_bundle.interferer_tx_heights_m
        interferer_nominal_tx_power_mw = layout_bundle.interferer_nominal_tx_power_mw
        interferer_sector_azimuths_deg = layout_bundle.interferer_sector_azimuths_deg
        interferer_sector_mask = layout_bundle.interferer_sector_mask
        interferer_total_downtilt_deg = layout_bundle.interferer_total_downtilt_deg
        interferer_peak_gain_db = layout_bundle.interferer_peak_gain_db
        interferer_beamforming_array_gain_db = layout_bundle.interferer_beamforming_array_gain_db
    else:
        interferer_sites = _sample_interferer_sites(
            reuse_factor,
            cell_radius,
            interferer_total,
            config,
            rng,
            sample_count=sample_count,
        )
        interferer_tx_heights_m = config.base_station_height_m
        interferer_nominal_tx_power_mw = np.full(interferer_sites.shape[1], config.tx_power_mw, dtype=float)
        interferer_sector_azimuths_deg = None
        interferer_sector_mask = None
        interferer_total_downtilt_deg = None
        interferer_peak_gain_db = None
        interferer_beamforming_array_gain_db = None
    interferer_distance_2d = _pairwise_horizontal_distances(user_points, interferer_sites)
    interferer_distance_3d = three_dimensional_distance(
        interferer_distance_2d,
        tx_height_m=interferer_tx_heights_m,
        rx_height_m=terminal_height_m,
    )
    interferer_sector_gain_linear = random_interferer_gain_linear(
        interferer_sites,
        user_points,
        terminal_height_m,
        config,
        rng,
        tx_height_m=interferer_tx_heights_m,
        sector_azimuths_deg=interferer_sector_azimuths_deg,
        total_downtilt_deg=interferer_total_downtilt_deg,
        peak_gain_db=interferer_peak_gain_db,
        beamforming_array_gain_db=interferer_beamforming_array_gain_db,
        sector_mask=interferer_sector_mask,
    )
    if interferer_sites.shape[1] == 0:
        interferer_gain_db = np.zeros((sample_count, 0), dtype=float)
    else:
        interferer_gain_db = 10.0 * np.log10(
            np.sum(interferer_sector_gain_linear, axis=-1)
        )
    interference_power_mw = np.sum(
        interferer_nominal_tx_power_mw[None, :, None]
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
        interferer_antenna_gain_db=(
            np.mean(interferer_gain_db, axis=1)
            if interferer_gain_db.shape[1] > 0
            else np.zeros(sample_count, dtype=float)
        ),
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
    config = resolve_runtime_config(config)
    user_points = np.asarray(user_points, dtype=float)
    sample_count = user_points.shape[0]
    terminal_height_m = np.full(
        sample_count,
        config.ground_terminal_height_m + user_altitude_m,
        dtype=float,
    )
    layout_bundle = (
        _static_layout_site_bundle(config, interferer_count)
        if config.site_layout_csv is not None
        else None
    )

    serving_site_position = (
        layout_bundle.serving_position_xy_m[None, :]
        if layout_bundle is not None
        else np.zeros((1, 2), dtype=float)
    )
    serving_tx_height_m = (
        layout_bundle.serving_tx_height_m
        if layout_bundle is not None
        else config.base_station_height_m
    )
    serving_nominal_tx_power_mw = (
        layout_bundle.serving_nominal_tx_power_mw
        if layout_bundle is not None
        else config.tx_power_mw
    )

    desired_offsets = user_points - serving_site_position
    desired_distance_2d = np.sqrt(np.sum(desired_offsets * desired_offsets, axis=1))
    desired_distance_3d = three_dimensional_distance(
        desired_distance_2d,
        tx_height_m=serving_tx_height_m,
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
        serving_site_position,
        user_points,
        terminal_height_m,
        config,
        tx_height_m=serving_tx_height_m,
        sector_azimuths_deg=(
            layout_bundle.serving_sector_azimuths_deg if layout_bundle is not None else None
        ),
        total_downtilt_deg=(
            layout_bundle.serving_total_downtilt_deg if layout_bundle is not None else None
        ),
        peak_gain_db=layout_bundle.serving_peak_gain_db if layout_bundle is not None else None,
        beamforming_array_gain_db=(
            layout_bundle.serving_beamforming_array_gain_db if layout_bundle is not None else None
        ),
        sector_mask=layout_bundle.serving_sector_mask if layout_bundle is not None else None,
    ).ravel()
    desired_gain_linear = best_sector_gain_linear(
        serving_site_position,
        user_points,
        terminal_height_m,
        config,
        tx_height_m=serving_tx_height_m,
        sector_azimuths_deg=(
            layout_bundle.serving_sector_azimuths_deg if layout_bundle is not None else None
        ),
        total_downtilt_deg=(
            layout_bundle.serving_total_downtilt_deg if layout_bundle is not None else None
        ),
        peak_gain_db=layout_bundle.serving_peak_gain_db if layout_bundle is not None else None,
        beamforming_array_gain_db=(
            layout_bundle.serving_beamforming_array_gain_db if layout_bundle is not None else None
        ),
        sector_mask=layout_bundle.serving_sector_mask if layout_bundle is not None else None,
    ).ravel()
    desired_power_mw = (
        serving_nominal_tx_power_mw
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

    if layout_bundle is not None:
        interferer_sites = np.broadcast_to(
            layout_bundle.interferer_positions_xy_m,
            (sample_count, *layout_bundle.interferer_positions_xy_m.shape),
        ).copy()
        interferer_tx_heights_m = layout_bundle.interferer_tx_heights_m
        interferer_nominal_tx_power_mw = layout_bundle.interferer_nominal_tx_power_mw
        interferer_sector_azimuths_deg = layout_bundle.interferer_sector_azimuths_deg
        interferer_sector_mask = layout_bundle.interferer_sector_mask
        interferer_total_downtilt_deg = layout_bundle.interferer_total_downtilt_deg
        interferer_peak_gain_db = layout_bundle.interferer_peak_gain_db
        interferer_beamforming_array_gain_db = layout_bundle.interferer_beamforming_array_gain_db
    else:
        interferer_sites = _sample_interferer_sites(
            reuse_factor,
            cell_radius,
            interferer_count,
            config,
            rng,
            sample_count=sample_count,
        )
        interferer_tx_heights_m = config.base_station_height_m
        interferer_nominal_tx_power_mw = np.full(interferer_sites.shape[1], config.tx_power_mw, dtype=float)
        interferer_sector_azimuths_deg = None
        interferer_sector_mask = None
        interferer_total_downtilt_deg = None
        interferer_peak_gain_db = None
        interferer_beamforming_array_gain_db = None
    interferer_distance_2d = _pairwise_horizontal_distances(user_points, interferer_sites)
    interferer_distance_3d = three_dimensional_distance(
        interferer_distance_2d,
        tx_height_m=interferer_tx_heights_m,
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
        tx_height_m=interferer_tx_heights_m,
        sector_azimuths_deg=interferer_sector_azimuths_deg,
        total_downtilt_deg=interferer_total_downtilt_deg,
        peak_gain_db=interferer_peak_gain_db,
        beamforming_array_gain_db=interferer_beamforming_array_gain_db,
        sector_mask=interferer_sector_mask,
    )
    if interferer_sites.shape[1] == 0:
        interferer_gain_db = np.zeros((sample_count, 0), dtype=float)
    else:
        interferer_gain_db = 10.0 * np.log10(
            np.sum(interferer_sector_gain_linear, axis=-1)
        )
    interference_power_mw = np.sum(
        interferer_nominal_tx_power_mw[None, :, None]
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
        interferer_los_probability=(
            np.mean(interferer_los_probability, axis=1)
            if interferer_los_probability.shape[1] > 0
            else np.zeros(sample_count, dtype=float)
        ),
        serving_antenna_gain_db=desired_gain_db,
        interferer_antenna_gain_db=(
            np.mean(interferer_gain_db, axis=1)
            if interferer_gain_db.shape[1] > 0
            else np.zeros(sample_count, dtype=float)
        ),
    )
