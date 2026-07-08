from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .antenna import random_interferer_gain_linear, sector_beam_gain_db
from .building_gis import BuildingDataset, evaluate_gis_los, load_building_dataset
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


def _best_server_choice(
    site_positions: np.ndarray,
    user_point: np.ndarray,
    terminal_height_m: float,
    config: SimulationConfig,
    rng: np.random.Generator,
    building_dataset: BuildingDataset | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if building_dataset is not None:
        gis_covered, gis_los = evaluate_gis_los(
            site_positions_xy_m=site_positions,
            user_point_xy_m=user_point,
            tx_height_m=config.base_station_height_m,
            rx_height_m=terminal_height_m,
            building_dataset=building_dataset,
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
    candidate_reference_power_mw = config.tx_power_mw * candidate_channel_power
    return (
        candidate_reference_power_mw,
        candidate_channel_power,
        beam_gain_db,
        los_probability,
        distance_3d,
        is_los,
        gis_covered,
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

    for step, user_point in enumerate(trajectory):
        load_state = _update_load_state(load_state, config, rng)
        scheduled_users = _scheduled_users(load_state, config, rng)
        site_tx_power_mw = _site_tx_power_mw(scheduled_users, config)
        terminal_height_m = config.ground_terminal_height_m + config.dynamic_altitude_m

        (
            candidate_reference_power_mw,
            candidate_channel_power,
            beam_gain_db,
            los_probability,
            distance_3d,
            is_los,
            gis_covered,
        ) = _best_server_choice(
            site_positions,
            user_point,
            terminal_height_m,
            config,
            rng,
            building_dataset=building_dataset,
        )
        flat_best_index = int(np.argmax(candidate_reference_power_mw))
        best_site_index, best_sector_index, best_beam_index = np.unravel_index(
            flat_best_index,
            candidate_reference_power_mw.shape,
        )
        handover_flag = 0
        if current_serving_site_index is None:
            current_serving_site_index = best_site_index
        else:
            current_site_power_dbm = 10.0 * np.log10(
                np.max(candidate_reference_power_mw[current_serving_site_index]) + 1e-15
            )
            best_candidate_power_dbm = 10.0 * np.log10(
                np.max(candidate_reference_power_mw[best_site_index]) + 1e-15
            )
            dwell_satisfied = (step - last_handover_step) >= config.dynamic_min_dwell_steps
            if (
                best_site_index != current_serving_site_index
                and dwell_satisfied
                and best_candidate_power_dbm > current_site_power_dbm + config.handover_hysteresis_db
            ):
                current_serving_site_index = best_site_index
                handover_flag = 1
                handover_count += 1
                last_handover_step = step

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
        if building_dataset is not None:
            interferer_gis_covered, interferer_gis_los = evaluate_gis_los(
                site_positions_xy_m=interferer_sites,
                user_point_xy_m=user_point,
                tx_height_m=config.base_station_height_m,
                rx_height_m=terminal_height_m,
                building_dataset=building_dataset,
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
        sector_activity = (
            rng.random(size=interferer_sector_gain_linear.shape) < interferer_load_state[:, None]
        ).astype(float)
        interference_power_mw = float(
            np.sum(
                interferer_tx_power_mw[:, None]
                * interferer_sector_gain_linear
                * received_power(
                    interferer_distance_3d[:, None],
                    interferer_pathloss_exponent[:, None],
                    config=config,
                )
                * interferer_shadow_linear[:, None]
                * interferer_fading[:, None]
                * sector_activity
            )
        )
        noise_power_mw = config.thermal_noise_power_mw
        sir_linear = serving_power_mw / max(interference_power_mw, 1e-15)
        sinr_linear = serving_power_mw / max(interference_power_mw + noise_power_mw, 1e-15)
        sinr_db = 10.0 * np.log10(sinr_linear)
        rate_bphz = np.log2(1.0 + sinr_linear) * float(sinr_db >= config.coverage_threshold_db)

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
                "signal_power_dbm": float(10.0 * np.log10(serving_power_mw)),
                "interference_power_dbm": float(10.0 * np.log10(max(interference_power_mw, 1e-15))),
                "noise_power_dbm": float(config.thermal_noise_power_dbm),
                "sir_db": float(10.0 * np.log10(sir_linear)),
                "sinr_db": float(sinr_db),
                "rate_bphz": float(rate_bphz),
                "serving_load": float(load_state[current_serving_site_index]),
                "mean_neighbor_load": float(np.mean(interferer_load_state)),
                "scheduled_users_serving": int(scheduled_users[current_serving_site_index]),
                "mean_serving_los_probability": float(los_probability[current_serving_site_index]),
                "mean_neighbor_los_probability": float(np.mean(interferer_los_probability)),
                "serving_los_state": int(is_los[current_serving_site_index]),
                "mean_neighbor_los_state": float(np.mean(interferer_is_los)),
                "serving_gis_covered": int(gis_covered[current_serving_site_index]),
                "mean_neighbor_gis_covered": float(np.mean(interferer_gis_covered)),
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
