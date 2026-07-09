from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .antenna import random_interferer_gain_linear, sector_beam_gain_db
from .building_gis import BuildingDataset, evaluate_gis_los_and_loss, load_building_dataset
from .config import SimulationConfig
from .geometry import (
    center_site_layout,
    generate_linear_trajectory,
    load_site_ground_elevation_offsets_m,
    load_site_layout,
    load_site_layout_rows,
)
from .parameter_profile import resolve_runtime_config
from .pathloss import hybrid_los_probability, received_power, three_dimensional_distance


@dataclass(frozen=True)
class DynamicExperimentBundle:
    summary: pd.DataFrame
    trace: pd.DataFrame
    site_layout: pd.DataFrame


@dataclass(frozen=True)
class DynamicSiteParameters:
    tx_heights_m: np.ndarray
    tx_power_dbm: np.ndarray
    nominal_tx_power_mw: np.ndarray
    sector_azimuths_deg: np.ndarray
    sector_mask: np.ndarray
    total_downtilt_deg: np.ndarray
    peak_gain_db: np.ndarray
    beamforming_array_gain_db: np.ndarray
    radio_sets: tuple[frozenset[str], ...]
    arfcn_sets: tuple[frozenset[str], ...]
    operators: tuple[str, ...]


def _parse_optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_token_set(value: object) -> frozenset[str]:
    if value in (None, ""):
        return frozenset()
    tokens = str(value).replace(",", "|").replace(";", "|").split("|")
    normalized_tokens = {token.strip().upper() for token in tokens if token.strip()}
    return frozenset(normalized_tokens)


def _parse_sector_azimuths_deg(row: dict[str, str], config: SimulationConfig) -> list[float]:
    sector_tokens = row.get("sector_azimuths_deg")
    if sector_tokens not in (None, ""):
        parsed = [
            float(token) % 360.0
            for token in str(sector_tokens).replace(",", "|").replace(";", "|").split("|")
            if token.strip()
        ]
        if parsed:
            return parsed
    antenna_azimuth_deg = _parse_optional_float(row.get("antenna_azimuth_deg"))
    if antenna_azimuth_deg is not None:
        return [float(antenna_azimuth_deg) % 360.0]
    return [float(angle_deg) % 360.0 for angle_deg in config.sector_azimuths_deg]


def _mean_or_fallback(values: list[float], fallback_value: float) -> float:
    if not values:
        return float(fallback_value)
    return float(np.mean(np.array(values, dtype=float)))


def _build_dynamic_site_parameters(
    raw_site_rows: tuple[dict[str, str], ...],
    site_ground_offsets_m: np.ndarray,
    config: SimulationConfig,
) -> DynamicSiteParameters:
    sector_lists = [_parse_sector_azimuths_deg(row, config) for row in raw_site_rows]
    max_sector_count = max(len(sector_list) for sector_list in sector_lists)

    sector_azimuths_deg = np.zeros((len(raw_site_rows), max_sector_count), dtype=float)
    sector_mask = np.zeros((len(raw_site_rows), max_sector_count), dtype=bool)
    tx_heights_m = np.zeros(len(raw_site_rows), dtype=float)
    tx_power_dbm = np.zeros(len(raw_site_rows), dtype=float)
    total_downtilt_deg = np.zeros(len(raw_site_rows), dtype=float)
    peak_gain_db = np.zeros(len(raw_site_rows), dtype=float)
    beamforming_array_gain_db = np.zeros(len(raw_site_rows), dtype=float)
    radio_sets: list[frozenset[str]] = []
    arfcn_sets: list[frozenset[str]] = []
    operators: list[str] = []

    for site_index, row in enumerate(raw_site_rows):
        sector_angles_deg = sector_lists[site_index]
        sector_azimuths_deg[site_index, : len(sector_angles_deg)] = sector_angles_deg
        if len(sector_angles_deg) < max_sector_count:
            sector_azimuths_deg[site_index, len(sector_angles_deg) :] = sector_angles_deg[0]
        sector_mask[site_index, : len(sector_angles_deg)] = True

        direct_tx_height_m = _parse_optional_float(row.get("tx_height_m"))
        antenna_height_m = _parse_optional_float(row.get("antenna_height_m"))
        row_base_station_height_m = _parse_optional_float(row.get("base_station_height_m"))
        local_antenna_height_m = _mean_or_fallback(
            [value for value in (antenna_height_m, row_base_station_height_m) if value is not None],
            config.base_station_height_m,
        )
        tx_heights_m[site_index] = (
            float(direct_tx_height_m)
            if direct_tx_height_m is not None
            else local_antenna_height_m + float(site_ground_offsets_m[site_index])
        )

        tx_power_dbm[site_index] = _mean_or_fallback(
            [value for value in (_parse_optional_float(row.get("tx_power_dbm")),) if value is not None],
            config.tx_power_dbm,
        )
        peak_gain_db[site_index] = _mean_or_fallback(
            [value for value in (_parse_optional_float(row.get("antenna_peak_gain_db")),) if value is not None],
            config.antenna_peak_gain_db,
        )
        beamforming_array_gain_db[site_index] = _mean_or_fallback(
            [value for value in (_parse_optional_float(row.get("beamforming_array_gain_db")),) if value is not None],
            config.beamforming_array_gain_db,
        )

        mechanical_downtilt_deg = _mean_or_fallback(
            [value for value in (_parse_optional_float(row.get("mechanical_downtilt_deg")),) if value is not None],
            config.mechanical_downtilt_deg,
        )
        electrical_downtilt_deg = _mean_or_fallback(
            [value for value in (_parse_optional_float(row.get("electrical_downtilt_deg")),) if value is not None],
            config.electrical_downtilt_deg,
        )
        total_downtilt_deg[site_index] = mechanical_downtilt_deg + electrical_downtilt_deg

        radio_sets.append(_parse_token_set(row.get("radio")))
        arfcn_sets.append(
            _parse_token_set(row.get("arfcn_list") or row.get("arfcn"))
        )
        operators.append(str(row.get("operator") or "").strip())

    return DynamicSiteParameters(
        tx_heights_m=tx_heights_m,
        tx_power_dbm=tx_power_dbm,
        nominal_tx_power_mw=np.power(10.0, tx_power_dbm / 10.0),
        sector_azimuths_deg=sector_azimuths_deg,
        sector_mask=sector_mask,
        total_downtilt_deg=total_downtilt_deg,
        peak_gain_db=peak_gain_db,
        beamforming_array_gain_db=beamforming_array_gain_db,
        radio_sets=tuple(radio_sets),
        arfcn_sets=tuple(arfcn_sets),
        operators=tuple(operators),
    )


def _sites_share_channel(
    site_parameters: DynamicSiteParameters,
    serving_site_index: int,
    candidate_site_index: int,
) -> bool:
    serving_radios = site_parameters.radio_sets[serving_site_index]
    candidate_radios = site_parameters.radio_sets[candidate_site_index]
    if serving_radios and candidate_radios and serving_radios.isdisjoint(candidate_radios):
        return False

    serving_arfcns = site_parameters.arfcn_sets[serving_site_index]
    candidate_arfcns = site_parameters.arfcn_sets[candidate_site_index]
    if serving_arfcns and candidate_arfcns:
        return not serving_arfcns.isdisjoint(candidate_arfcns)
    return True


def _cochannel_interferer_mask(
    site_parameters: DynamicSiteParameters,
    serving_site_index: int,
) -> np.ndarray:
    site_count = len(site_parameters.tx_heights_m)
    mask = np.ones(site_count, dtype=bool)
    mask[serving_site_index] = False
    for site_index in range(site_count):
        if site_index == serving_site_index:
            continue
        if not _sites_share_channel(site_parameters, serving_site_index, site_index):
            mask[site_index] = False
    return mask


def _clip_load(load_state: np.ndarray) -> np.ndarray:
    return np.clip(load_state, 0.05, 0.99)


def _update_load_state(
    previous_state: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    innovation = rng.normal(
        loc=0.0,
        scale=config.dynamic_load_std,
        size=previous_state.shape,
    )
    updated = (
        config.dynamic_load_mean
        + config.dynamic_load_correlation * (previous_state - config.dynamic_load_mean)
        + np.sqrt(max(1.0 - config.dynamic_load_correlation**2, 0.0)) * innovation
    )
    return _clip_load(updated)


def _scheduled_users(load_state: np.ndarray, config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    mean_users = 1.0 + load_state * (config.dynamic_max_users_per_site - 1.0)
    users = rng.poisson(mean_users).astype(int)
    return np.maximum(users, 1)


def _site_tx_power_mw(
    scheduled_users: np.ndarray,
    nominal_tx_power_mw: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    return np.asarray(nominal_tx_power_mw, dtype=float) / np.power(
        np.maximum(scheduled_users, 1),
        config.power_split_exponent,
    )


def _power_dbm_from_mw(power_mw: np.ndarray | float) -> np.ndarray:
    power = np.maximum(np.asarray(power_mw, dtype=float), 1e-15)
    return 10.0 * np.log10(power)


def _update_filtered_measurements(
    previous_measurement_dbm: np.ndarray | None,
    instant_measurement_dbm: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    instant_measurement = np.asarray(instant_measurement_dbm, dtype=float)
    if previous_measurement_dbm is None:
        return instant_measurement

    alpha = float(np.clip(config.handover_l3_filter_alpha, 0.0, 1.0))
    return alpha * instant_measurement + (1.0 - alpha) * np.asarray(previous_measurement_dbm, dtype=float)


def _handover_decision(
    current_serving_site_index: int | None,
    filtered_measurement_dbm: np.ndarray,
    pending_steps: np.ndarray,
    step: int,
    last_handover_step: int,
    config: SimulationConfig,
) -> tuple[int, np.ndarray, int, int]:
    filtered_measurement = np.asarray(filtered_measurement_dbm, dtype=float)
    updated_pending_steps = np.asarray(pending_steps, dtype=int).copy()
    if current_serving_site_index is None:
        updated_pending_steps.fill(0)
        return int(np.argmax(filtered_measurement)), updated_pending_steps, 0, last_handover_step

    serving_measurement_dbm = float(filtered_measurement[current_serving_site_index])
    a3_condition = filtered_measurement > (serving_measurement_dbm + config.handover_hysteresis_db)
    a3_condition[current_serving_site_index] = False
    updated_pending_steps = np.where(a3_condition, updated_pending_steps + 1, 0)
    if not np.any(a3_condition):
        return current_serving_site_index, updated_pending_steps, 0, last_handover_step

    best_candidate_index = int(np.argmax(np.where(a3_condition, filtered_measurement, -np.inf)))
    dwell_satisfied = (step - last_handover_step) >= config.dynamic_min_dwell_steps
    time_to_trigger_steps = max(config.handover_time_to_trigger_steps, 1)
    if dwell_satisfied and updated_pending_steps[best_candidate_index] >= time_to_trigger_steps:
        updated_pending_steps.fill(0)
        return best_candidate_index, updated_pending_steps, 1, step
    return current_serving_site_index, updated_pending_steps, 0, last_handover_step


def _coordination_weights(
    interferer_large_scale_power_mw: np.ndarray,
    interferer_load_state: np.ndarray,
    predicted_sinr_db: float,
    config: SimulationConfig,
) -> tuple[np.ndarray, int]:
    interferer_power = np.asarray(interferer_large_scale_power_mw, dtype=float)
    interferer_load = np.asarray(interferer_load_state, dtype=float)
    weights = np.ones(interferer_power.shape, dtype=float)
    if (
        not config.coordinated_scheduling_enabled
        or interferer_power.size == 0
        or predicted_sinr_db >= config.coordinated_scheduling_sinr_threshold_db
    ):
        return weights, 0

    cluster_size = min(config.coordinated_scheduling_cluster_size, interferer_power.size)
    strongest_indices = np.argsort(interferer_power)[-cluster_size:]
    blank_fraction = float(np.clip(config.coordinated_scheduling_blank_fraction, 0.0, 1.0))
    relief = blank_fraction * np.clip(1.0 - interferer_load[strongest_indices], 0.0, 1.0)
    weights[strongest_indices] = 1.0 - relief
    coordinated_count = int(np.sum(relief > 1e-9))
    return weights, coordinated_count


def _best_server_choice(
    site_positions: np.ndarray,
    site_parameters: DynamicSiteParameters,
    user_point: np.ndarray,
    terminal_height_m: float,
    config: SimulationConfig,
    rng: np.random.Generator,
    building_dataset: BuildingDataset | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beam_gain_db = sector_beam_gain_db(
        site_positions,
        user_point[None, :],
        np.array([terminal_height_m], dtype=float),
        config,
        tx_height_m=site_parameters.tx_heights_m,
        sector_azimuths_deg=site_parameters.sector_azimuths_deg,
        total_downtilt_deg=site_parameters.total_downtilt_deg,
        peak_gain_db=site_parameters.peak_gain_db,
        beamforming_array_gain_db=site_parameters.beamforming_array_gain_db,
        sector_mask=site_parameters.sector_mask,
    )[0]
    beam_gain_linear = np.power(10.0, beam_gain_db / 10.0)

    user_offsets = user_point[None, :] - site_positions
    distance_2d = np.sqrt(np.sum(user_offsets * user_offsets, axis=1))
    distance_3d = three_dimensional_distance(
        distance_2d,
        tx_height_m=site_parameters.tx_heights_m,
        rx_height_m=np.array([terminal_height_m], dtype=float),
    )
    los_probability = hybrid_los_probability(distance_2d, terminal_height_m, config)
    is_los = rng.random(size=distance_2d.shape) < los_probability
    gis_covered = np.zeros(distance_2d.shape, dtype=bool)
    gis_excess_loss_db = np.zeros(distance_2d.shape, dtype=float)
    if building_dataset is not None:
        gis_covered, gis_los, gis_excess_loss_db = evaluate_gis_los_and_loss(
            site_positions_xy_m=site_positions,
            user_point_xy_m=user_point,
            tx_height_m=site_parameters.tx_heights_m,
            rx_height_m=terminal_height_m,
            building_dataset=building_dataset,
            carrier_frequency_ghz=config.carrier_frequency_ghz,
            penetration_loss_per_meter_db=(
                config.gis_penetration_loss_per_meter_db if config.gis_excess_loss_enabled else 0.0
            ),
            penetration_loss_cap_db=(
                config.gis_penetration_loss_cap_db if config.gis_excess_loss_enabled else 0.0
            ),
            diffraction_loss_cap_db=(
                config.gis_diffraction_loss_cap_db if config.gis_excess_loss_enabled else 0.0
            ),
            total_excess_loss_cap_db=(
                config.gis_total_excess_loss_cap_db if config.gis_excess_loss_enabled else 0.0
            ),
            material_loss_profile_path=config.building_material_loss_profile_json,
        )
        is_los = np.where(gis_covered, gis_los, is_los)
    pathloss_exponent = np.where(is_los, config.los_pathloss_exponent, config.nlos_pathloss_exponent)
    shadow_sigma_db = np.where(is_los, config.los_shadow_sigma_db, config.nlos_shadow_sigma_db)
    shadow_linear = np.power(10.0, rng.normal(0.0, shadow_sigma_db) / 10.0)

    candidate_channel_power = (
        beam_gain_linear
        * received_power(distance_3d[:, None, None], pathloss_exponent[:, None, None], config=config)
        * shadow_linear[:, None, None]
    )
    candidate_channel_power = candidate_channel_power * np.power(
        10.0,
        -gis_excess_loss_db[:, None, None] / 10.0,
    )
    candidate_reference_power_mw = site_parameters.nominal_tx_power_mw[:, None, None] * candidate_channel_power
    return (
        candidate_reference_power_mw,
        candidate_channel_power,
        beam_gain_db,
        los_probability,
        distance_3d,
        is_los,
        gis_covered,
        gis_excess_loss_db,
    )


def run_dynamic_trajectory_experiment(config: SimulationConfig) -> DynamicExperimentBundle:
    trajectory = generate_linear_trajectory(
        half_length_m=config.dynamic_path_half_length_m,
        lateral_offset_m=config.dynamic_path_lateral_offset_m,
        steps=config.dynamic_time_steps,
    )
    return run_dynamic_trajectory_experiment_with_trajectory(
        config=config,
        trajectory_xy_m=trajectory,
    )


def run_dynamic_trajectory_experiment_with_trajectory(
    config: SimulationConfig,
    trajectory_xy_m: np.ndarray,
) -> DynamicExperimentBundle:
    config = resolve_runtime_config(config)
    raw_site_rows = load_site_layout_rows(config.dynamic_site_layout_csv)
    raw_site_positions = load_site_layout(config.dynamic_site_layout_csv)
    site_positions = center_site_layout(raw_site_positions)
    site_ground_offsets_m = load_site_ground_elevation_offsets_m(
        config.dynamic_site_layout_csv,
        site_positions=raw_site_positions,
    )
    site_parameters = _build_dynamic_site_parameters(
        raw_site_rows,
        site_ground_offsets_m,
        config,
    )
    trajectory = np.asarray(trajectory_xy_m, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("trajectory_xy_m must have shape (steps, 2)")
    rng = config.rng(offset=500)
    load_state = np.full(site_positions.shape[0], config.dynamic_load_mean, dtype=float)
    building_dataset: BuildingDataset | None = None
    if (
        config.gis_los_enabled
        and config.building_footprint_geojson is not None
        and config.building_footprint_geojson.exists()
    ):
        building_dataset = load_building_dataset(
            building_geojson_path=config.building_footprint_geojson,
            site_layout_csv=config.dynamic_site_layout_csv,
            default_height_m=config.building_default_height_m,
            level_height_m=config.building_level_height_m,
            min_area_m2=config.building_min_area_m2,
        )

    records: list[dict[str, float | int]] = []
    current_serving_site_index: int | None = None
    handover_count = 0
    last_handover_step = -config.dynamic_min_dwell_steps
    filtered_measurement_dbm: np.ndarray | None = None
    handover_pending_steps = np.zeros(site_positions.shape[0], dtype=int)

    for step, user_point in enumerate(trajectory):
        load_state = _update_load_state(load_state, config, rng)
        scheduled_users = _scheduled_users(load_state, config, rng)
        site_tx_power_mw = _site_tx_power_mw(
            scheduled_users,
            site_parameters.nominal_tx_power_mw,
            config,
        )
        terminal_height_m = config.ground_terminal_height_m + config.dynamic_altitude_m
        noise_power_mw = config.thermal_noise_power_mw

        (
            candidate_reference_power_mw,
            candidate_channel_power,
            beam_gain_db,
            los_probability,
            distance_3d,
            is_los,
            gis_covered,
            gis_excess_loss_db,
        ) = _best_server_choice(
            site_positions,
            site_parameters,
            user_point,
            terminal_height_m,
            config,
            rng,
            building_dataset=building_dataset,
        )
        instant_measurement_dbm = _power_dbm_from_mw(
            np.max(candidate_reference_power_mw, axis=(1, 2))
        )
        filtered_measurement_dbm = _update_filtered_measurements(
            filtered_measurement_dbm,
            instant_measurement_dbm,
            config,
        )
        (
            current_serving_site_index,
            handover_pending_steps,
            handover_flag,
            last_handover_step,
        ) = _handover_decision(
            current_serving_site_index=current_serving_site_index,
            filtered_measurement_dbm=filtered_measurement_dbm,
            pending_steps=handover_pending_steps,
            step=step,
            last_handover_step=last_handover_step,
            config=config,
        )
        handover_count += handover_flag

        serving_large_scale_power_mw = float(
            np.max(candidate_channel_power[current_serving_site_index])
        )
        serving_sector_index, serving_beam_index = np.unravel_index(
            int(np.argmax(candidate_reference_power_mw[current_serving_site_index])),
            candidate_reference_power_mw[current_serving_site_index].shape,
        )
        serving_fading_m = (
            config.los_small_scale_m
            if is_los[current_serving_site_index]
            else config.nlos_small_scale_m
        )
        serving_fading = rng.gamma(shape=serving_fading_m, scale=1.0 / serving_fading_m)
        serving_power_mw = (
            site_tx_power_mw[current_serving_site_index]
            * serving_large_scale_power_mw
            * serving_fading
        )
        serving_measurement_dbm = float(filtered_measurement_dbm[current_serving_site_index])

        interferer_mask = np.ones(site_positions.shape[0], dtype=bool)
        interferer_mask = _cochannel_interferer_mask(
            site_parameters,
            current_serving_site_index,
        )
        interferer_sites = site_positions[interferer_mask]
        interferer_load_state = load_state[interferer_mask]
        interferer_scheduled_users = scheduled_users[interferer_mask]
        interferer_tx_power_mw = _site_tx_power_mw(
            interferer_scheduled_users,
            site_parameters.nominal_tx_power_mw[interferer_mask],
            config,
        )

        if interferer_sites.shape[0] == 0:
            interferer_distance_2d = np.zeros(0, dtype=float)
            interferer_distance_3d = np.zeros(0, dtype=float)
            interferer_los_probability = np.zeros(0, dtype=float)
            interferer_is_los = np.zeros(0, dtype=bool)
            interferer_gis_covered = np.zeros(0, dtype=bool)
            interferer_gis_excess_loss_db = np.zeros(0, dtype=float)
            interferer_shadow_linear = np.zeros(0, dtype=float)
            interferer_fading = np.zeros(0, dtype=float)
            interferer_sector_gain_linear = np.zeros((0, site_parameters.sector_mask.shape[1]), dtype=float)
            interferer_reference_power_mw = np.zeros((0, site_parameters.sector_mask.shape[1]), dtype=float)
            predicted_interference_power_mw = 0.0
        else:
            interferer_distance_2d = np.sqrt(np.sum((user_point[None, :] - interferer_sites) ** 2, axis=1))
            interferer_distance_3d = three_dimensional_distance(
                interferer_distance_2d,
                tx_height_m=site_parameters.tx_heights_m[interferer_mask],
                rx_height_m=np.array([terminal_height_m], dtype=float),
            )
            interferer_los_probability = hybrid_los_probability(
                interferer_distance_2d,
                terminal_height_m,
                config,
            )
            interferer_is_los = rng.random(size=interferer_sites.shape[0]) < interferer_los_probability
            interferer_gis_covered = np.zeros(interferer_sites.shape[0], dtype=bool)
            interferer_gis_excess_loss_db = np.zeros(interferer_sites.shape[0], dtype=float)
            if building_dataset is not None:
                (
                    interferer_gis_covered,
                    interferer_gis_los,
                    interferer_gis_excess_loss_db,
                ) = evaluate_gis_los_and_loss(
                    site_positions_xy_m=interferer_sites,
                    user_point_xy_m=user_point,
                    tx_height_m=site_parameters.tx_heights_m[interferer_mask],
                    rx_height_m=terminal_height_m,
                    building_dataset=building_dataset,
                    carrier_frequency_ghz=config.carrier_frequency_ghz,
                    penetration_loss_per_meter_db=(
                        config.gis_penetration_loss_per_meter_db if config.gis_excess_loss_enabled else 0.0
                    ),
                    penetration_loss_cap_db=(
                        config.gis_penetration_loss_cap_db if config.gis_excess_loss_enabled else 0.0
                    ),
                    diffraction_loss_cap_db=(
                        config.gis_diffraction_loss_cap_db if config.gis_excess_loss_enabled else 0.0
                    ),
                    total_excess_loss_cap_db=(
                        config.gis_total_excess_loss_cap_db if config.gis_excess_loss_enabled else 0.0
                    ),
                    material_loss_profile_path=config.building_material_loss_profile_json,
                )
                interferer_is_los = np.where(interferer_gis_covered, interferer_gis_los, interferer_is_los)
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
            interferer_shadow_linear = np.power(
                10.0,
                rng.normal(0.0, interferer_shadow_sigma_db) / 10.0,
            )
            interferer_fading = rng.gamma(
                shape=np.where(interferer_is_los, config.los_small_scale_m, config.nlos_small_scale_m),
                scale=1.0 / np.where(interferer_is_los, config.los_small_scale_m, config.nlos_small_scale_m),
            )
            interferer_sector_gain_linear = random_interferer_gain_linear(
                interferer_sites,
                user_point[None, :],
                np.array([terminal_height_m], dtype=float),
                config,
                rng,
                tx_height_m=site_parameters.tx_heights_m[interferer_mask],
                sector_azimuths_deg=site_parameters.sector_azimuths_deg[interferer_mask],
                total_downtilt_deg=site_parameters.total_downtilt_deg[interferer_mask],
                peak_gain_db=site_parameters.peak_gain_db[interferer_mask],
                beamforming_array_gain_db=site_parameters.beamforming_array_gain_db[interferer_mask],
                sector_mask=site_parameters.sector_mask[interferer_mask],
            )[0]
            interferer_reference_power_mw = (
                interferer_tx_power_mw[:, None]
                * interferer_sector_gain_linear
                * received_power(
                    interferer_distance_3d[:, None],
                    interferer_pathloss_exponent[:, None],
                    config=config,
                )
                * interferer_shadow_linear[:, None]
                * np.power(10.0, -interferer_gis_excess_loss_db[:, None] / 10.0)
            )
            predicted_interference_power_mw = float(np.sum(np.max(interferer_reference_power_mw, axis=1)))
        predicted_sinr_db = float(
            _power_dbm_from_mw(site_tx_power_mw[current_serving_site_index] * serving_large_scale_power_mw)
            - _power_dbm_from_mw(predicted_interference_power_mw + noise_power_mw)
        )
        if interferer_reference_power_mw.shape[0] == 0:
            coordination_weights = np.ones(0, dtype=float)
            coordinated_interferer_count = 0
        else:
            coordination_weights, coordinated_interferer_count = _coordination_weights(
                interferer_large_scale_power_mw=np.max(interferer_reference_power_mw, axis=1),
                interferer_load_state=interferer_load_state,
                predicted_sinr_db=predicted_sinr_db,
                config=config,
            )
        sector_activity = (
            rng.random(size=interferer_sector_gain_linear.shape) < interferer_load_state[:, None]
        ).astype(float) * coordination_weights[:, None]
        interference_power_mw = float(
            np.sum(
                interferer_reference_power_mw
                * interferer_fading[:, None]
                * sector_activity
            )
        )
        sir_linear = serving_power_mw / max(interference_power_mw, 1e-15)
        sinr_linear = serving_power_mw / max(interference_power_mw + noise_power_mw, 1e-15)
        sinr_db = 10.0 * np.log10(sinr_linear)
        rate_bphz = np.log2(1.0 + sinr_linear) * float(sinr_db >= config.coverage_threshold_db)
        neighbor_measurements_dbm = np.delete(filtered_measurement_dbm, current_serving_site_index)
        best_neighbor_measurement_dbm = (
            float(np.max(neighbor_measurements_dbm))
            if neighbor_measurements_dbm.size
            else serving_measurement_dbm
        )
        mean_neighbor_load = float(np.mean(interferer_load_state)) if interferer_load_state.size else 0.0
        mean_neighbor_los_probability = (
            float(np.mean(interferer_los_probability))
            if interferer_los_probability.size
            else 0.0
        )
        mean_neighbor_los_state = float(np.mean(interferer_is_los)) if interferer_is_los.size else 0.0
        mean_neighbor_gis_covered = (
            float(np.mean(interferer_gis_covered))
            if interferer_gis_covered.size
            else 0.0
        )
        mean_neighbor_gis_excess_loss_db = (
            float(np.mean(interferer_gis_excess_loss_db))
            if interferer_gis_excess_loss_db.size
            else 0.0
        )

        records.append(
            {
                "step": step,
                "time_s": step * config.dynamic_time_step_s,
                "x_m": float(user_point[0]),
                "y_m": float(user_point[1]),
                "altitude_m": float(config.dynamic_altitude_m),
                "serving_site_index": int(current_serving_site_index),
                "serving_sector_index": int(serving_sector_index),
                "serving_beam_index": int(serving_beam_index),
                "handover_flag": handover_flag,
                "signal_power_dbm": float(_power_dbm_from_mw(serving_power_mw)),
                "interference_power_dbm": float(_power_dbm_from_mw(interference_power_mw)),
                "noise_power_dbm": float(config.thermal_noise_power_dbm),
                "sir_db": float(10.0 * np.log10(sir_linear)),
                "sinr_db": float(sinr_db),
                "rate_bphz": float(rate_bphz),
                "serving_load": float(load_state[current_serving_site_index]),
                "mean_neighbor_load": mean_neighbor_load,
                "scheduled_users_serving": int(scheduled_users[current_serving_site_index]),
                "serving_tx_power_dbm": float(site_parameters.tx_power_dbm[current_serving_site_index]),
                "serving_measurement_dbm": serving_measurement_dbm,
                "best_neighbor_measurement_dbm": best_neighbor_measurement_dbm,
                "mean_serving_los_probability": float(los_probability[current_serving_site_index]),
                "mean_neighbor_los_probability": mean_neighbor_los_probability,
                "serving_los_state": int(is_los[current_serving_site_index]),
                "mean_neighbor_los_state": mean_neighbor_los_state,
                "serving_gis_covered": int(gis_covered[current_serving_site_index]),
                "mean_neighbor_gis_covered": mean_neighbor_gis_covered,
                "serving_gis_excess_loss_db": float(gis_excess_loss_db[current_serving_site_index]),
                "mean_neighbor_gis_excess_loss_db": mean_neighbor_gis_excess_loss_db,
                "coordination_active_flag": int(coordinated_interferer_count > 0),
                "coordinated_interferer_count": coordinated_interferer_count,
                "cochannel_interferer_count": int(np.sum(interferer_mask)),
                "serving_distance_3d_m": float(distance_3d[current_serving_site_index]),
                "serving_gain_db": float(
                    beam_gain_db[current_serving_site_index, serving_sector_index, serving_beam_index]
                ),
            }
        )

    trace = pd.DataFrame(records)
    summary = pd.DataFrame(
        [
            {
                "scenario": "dynamic_layout_sinr",
                "time_steps": config.dynamic_time_steps,
                "handover_count": handover_count,
                "mean_sir_db": float(trace["sir_db"].mean()),
                "mean_sinr_db": float(trace["sinr_db"].mean()),
                "p05_sinr_db": float(trace["sinr_db"].quantile(0.05)),
                "mean_rate_bphz": float(trace["rate_bphz"].mean()),
                "outage_probability_at_10db": float(
                    np.mean(trace["sinr_db"].to_numpy() < config.coverage_threshold_db)
                ),
                "mean_serving_load": float(trace["serving_load"].mean()),
                "mean_neighbor_load": float(trace["mean_neighbor_load"].mean()),
                "mean_serving_los_state": float(trace["serving_los_state"].mean()),
                "mean_neighbor_los_state": float(trace["mean_neighbor_los_state"].mean()),
                "mean_serving_gis_covered": float(trace["serving_gis_covered"].mean()),
                "mean_neighbor_gis_covered": float(trace["mean_neighbor_gis_covered"].mean()),
                "mean_serving_gis_excess_loss_db": float(trace["serving_gis_excess_loss_db"].mean()),
                "mean_neighbor_gis_excess_loss_db": float(trace["mean_neighbor_gis_excess_loss_db"].mean()),
                "coordination_activation_ratio": float(trace["coordination_active_flag"].mean()),
                "mean_coordinated_interferer_count": float(trace["coordinated_interferer_count"].mean()),
                "mean_cochannel_interferer_count": float(trace["cochannel_interferer_count"].mean()),
            }
        ]
    )
    site_layout_records: list[dict[str, object]] = []
    for site_index, row in enumerate(raw_site_rows):
        site_record: dict[str, object] = {
            "site_index": site_index,
            "x_m": float(site_positions[site_index, 0]),
            "y_m": float(site_positions[site_index, 1]),
            "tx_height_m": float(site_parameters.tx_heights_m[site_index]),
            "ground_elevation_offset_m": float(site_ground_offsets_m[site_index]),
            "tx_power_dbm": float(site_parameters.tx_power_dbm[site_index]),
            "total_downtilt_deg": float(site_parameters.total_downtilt_deg[site_index]),
        }
        for key, value in row.items():
            if key in {"x_m", "y_m", "tx_height_m", "tx_power_dbm", "total_downtilt_deg", "ground_elevation_offset_m"}:
                continue
            site_record[key] = value
        site_layout_records.append(site_record)
    site_layout = pd.DataFrame(site_layout_records)
    return DynamicExperimentBundle(summary=summary, trace=trace, site_layout=site_layout)
