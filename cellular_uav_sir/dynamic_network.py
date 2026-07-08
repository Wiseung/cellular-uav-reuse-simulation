from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .antenna import random_interferer_gain_linear, sector_beam_gain_db
from .building_gis import BuildingDataset, evaluate_gis_los_and_loss, load_building_dataset
from .config import SimulationConfig
from .geometry import center_site_layout, generate_linear_trajectory, load_site_layout
from .pathloss import hybrid_los_probability, received_power, three_dimensional_distance


@dataclass(frozen=True)
class DynamicExperimentBundle:
    summary: pd.DataFrame
    trace: pd.DataFrame
    site_layout: pd.DataFrame


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


def _site_tx_power_mw(scheduled_users: np.ndarray, config: SimulationConfig) -> np.ndarray:
    return config.tx_power_mw / np.power(np.maximum(scheduled_users, 1), config.power_split_exponent)


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
    )[0]
    beam_gain_linear = np.power(10.0, beam_gain_db / 10.0)

    user_offsets = user_point[None, :] - site_positions
    distance_2d = np.sqrt(np.sum(user_offsets * user_offsets, axis=1))
    distance_3d = three_dimensional_distance(
        distance_2d,
        tx_height_m=config.base_station_height_m,
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
            tx_height_m=config.base_station_height_m,
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
    candidate_reference_power_mw = config.tx_power_mw * candidate_channel_power
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
    site_positions = center_site_layout(load_site_layout(config.dynamic_site_layout_csv))
    trajectory = generate_linear_trajectory(
        half_length_m=config.dynamic_path_half_length_m,
        lateral_offset_m=config.dynamic_path_lateral_offset_m,
        steps=config.dynamic_time_steps,
    )
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
        site_tx_power_mw = _site_tx_power_mw(scheduled_users, config)
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
        interferer_mask[current_serving_site_index] = False
        interferer_sites = site_positions[interferer_mask]
        interferer_load_state = load_state[interferer_mask]
        interferer_scheduled_users = scheduled_users[interferer_mask]
        interferer_tx_power_mw = _site_tx_power_mw(interferer_scheduled_users, config)

        interferer_distance_2d = np.sqrt(np.sum((user_point[None, :] - interferer_sites) ** 2, axis=1))
        interferer_distance_3d = three_dimensional_distance(
            interferer_distance_2d,
            tx_height_m=config.base_station_height_m,
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
                tx_height_m=config.base_station_height_m,
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
                "mean_neighbor_load": float(np.mean(interferer_load_state)),
                "scheduled_users_serving": int(scheduled_users[current_serving_site_index]),
                "serving_measurement_dbm": serving_measurement_dbm,
                "best_neighbor_measurement_dbm": best_neighbor_measurement_dbm,
                "mean_serving_los_probability": float(los_probability[current_serving_site_index]),
                "mean_neighbor_los_probability": float(np.mean(interferer_los_probability)),
                "serving_los_state": int(is_los[current_serving_site_index]),
                "mean_neighbor_los_state": float(np.mean(interferer_is_los)),
                "serving_gis_covered": int(gis_covered[current_serving_site_index]),
                "mean_neighbor_gis_covered": float(np.mean(interferer_gis_covered)),
                "serving_gis_excess_loss_db": float(gis_excess_loss_db[current_serving_site_index]),
                "mean_neighbor_gis_excess_loss_db": float(np.mean(interferer_gis_excess_loss_db)),
                "coordination_active_flag": int(coordinated_interferer_count > 0),
                "coordinated_interferer_count": coordinated_interferer_count,
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
            }
        ]
    )
    site_layout = pd.DataFrame(
        {
            "site_index": np.arange(site_positions.shape[0], dtype=int),
            "x_m": site_positions[:, 0],
            "y_m": site_positions[:, 1],
        }
    )
    return DynamicExperimentBundle(summary=summary, trace=trace, site_layout=site_layout)
