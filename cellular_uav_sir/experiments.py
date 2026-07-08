from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .geometry import sample_points_in_hexagon
from .sir_analytic import sir_db as analytic_sir_db
from .sir_montecarlo import edge_user_sir, simulate_los_probability_sir_samples, simulate_sir_samples


@dataclass(frozen=True)
class ExperimentBundle:
    sir_vs_reuse: pd.DataFrame
    ase_vs_reuse: pd.DataFrame
    sir_vs_height: pd.DataFrame
    cdf_samples: pd.DataFrame
    pathloss_sweep: pd.DataFrame


def _effective_rate_and_ase(
    sir_linear: np.ndarray,
    reuse_factor: int,
    config: SimulationConfig,
) -> tuple[float, float, float]:
    sir_linear = np.asarray(sir_linear, dtype=float)
    sir_db = 10.0 * np.log10(sir_linear)
    scheduled_rate = np.log2(1.0 + sir_linear) * (sir_db >= config.coverage_threshold_db)
    mean_scheduled_rate = float(np.mean(scheduled_rate))
    effective_rate = (
        mean_scheduled_rate
        * config.resource_activity_factor
        * config.scheduler_efficiency
        * (1.0 - config.control_overhead_fraction)
    )
    effective_ase_per_cell = effective_rate / reuse_factor
    effective_ase_per_km2 = effective_ase_per_cell / config.cell_area_km2
    return effective_rate, effective_ase_per_cell, effective_ase_per_km2


def run_reuse_experiments(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    ground_points = sample_points_in_hexagon(
        count=config.monte_carlo_samples,
        cell_radius=config.cell_radius_m,
        rng=config.rng(offset=10),
    )
    sir_records: list[dict[str, float]] = []
    ase_records: list[dict[str, float]] = []

    for reuse_factor in config.reuse_factors:
        samples = simulate_sir_samples(
            user_points=ground_points,
            reuse_factor=reuse_factor,
            cell_radius=config.cell_radius_m,
            pathloss_exponent=config.ground_pathloss_exponent,
            config=config,
            rng=config.rng(offset=100 + reuse_factor),
        )
        effective_rate, effective_ase_per_cell, effective_ase_per_km2 = _effective_rate_and_ase(
            samples.sir_linear,
            reuse_factor=reuse_factor,
            config=config,
        )
        sir_records.append(
            {
                "reuse_factor": reuse_factor,
                "analytic_sir_db": analytic_sir_db(
                    reuse_factor,
                    config.ground_pathloss_exponent,
                ),
                "baseline_hex_edge_sir_db": 10.0
                * np.log10(
                    edge_user_sir(
                        reuse_factor=reuse_factor,
                        cell_radius=config.cell_radius_m,
                        pathloss_exponent=config.ground_pathloss_exponent,
                    )
                ),
                "median_user_sir_db": float(np.median(samples.sir_db)),
                "p05_user_sir_db": samples.percentile_db(5.0),
            }
        )
        ase_records.append(
            {
                "reuse_factor": reuse_factor,
                "channel_share": 1.0 / reuse_factor,
                "mean_user_rate_bphz": samples.mean_spectral_efficiency(),
                "effective_rate_bphz": effective_rate,
                "effective_ase_bphz_per_cell": effective_ase_per_cell,
                "effective_ase_bphz_per_km2": effective_ase_per_km2,
            }
        )

    return pd.DataFrame(sir_records), pd.DataFrame(ase_records)


def run_height_experiment(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_points = sample_points_in_hexagon(
        count=config.monte_carlo_samples,
        cell_radius=config.cell_radius_m,
        rng=config.rng(offset=20),
    )
    records: list[dict[str, float]] = []
    cdf_records: list[dict[str, float | str]] = []

    for height_m in config.uav_altitudes_m:
        samples = simulate_los_probability_sir_samples(
            user_points=user_points,
            reuse_factor=config.default_reuse_factor,
            cell_radius=config.cell_radius_m,
            user_altitude_m=float(height_m),
            config=config,
            interferer_count=config.aerial_interferer_count,
            rng=config.rng(offset=200 + int(height_m)),
        )
        effective_rate, effective_ase_per_cell, effective_ase_per_km2 = _effective_rate_and_ase(
            samples.sir_linear,
            reuse_factor=config.default_reuse_factor,
            config=config,
        )
        records.append(
            {
                "height_m": float(height_m),
                "median_sir_db": float(np.median(samples.sir_db)),
                "p05_sir_db": samples.percentile_db(5.0),
                "coverage_probability_at_10db": samples.coverage_probability(
                    config.coverage_threshold_db
                ),
                "mean_user_rate_bphz": samples.mean_spectral_efficiency(),
                "effective_rate_bphz": effective_rate,
                "effective_ase_bphz_per_cell": effective_ase_per_cell,
                "effective_ase_bphz_per_km2": effective_ase_per_km2,
                "mean_serving_los_probability": samples.mean_serving_los_probability(),
                "mean_interferer_los_probability": samples.mean_interferer_los_probability(),
                "mean_serving_antenna_gain_db": samples.mean_serving_antenna_gain_db(),
                "mean_interferer_antenna_gain_db": samples.mean_interferer_antenna_gain_db(),
            }
        )
        if height_m in config.cdf_altitudes_m:
            label = "Ground 0 m" if height_m == 0 else f"UAV {height_m} m"
            for sir_value_db in samples.sir_db:
                cdf_records.append(
                    {
                        "scenario": label,
                        "height_m": float(height_m),
                        "sir_db": float(sir_value_db),
                    }
                )

    return pd.DataFrame(records), pd.DataFrame(cdf_records)


def run_pathloss_sweep(config: SimulationConfig) -> pd.DataFrame:
    records: list[dict[str, float]] = []

    for pathloss_exponent in config.pathloss_exponent_sweep:
        for reuse_factor in config.reuse_factors:
            records.append(
                {
                    "pathloss_exponent": float(pathloss_exponent),
                    "reuse_factor": reuse_factor,
                    "analytic_sir_db": analytic_sir_db(
                        reuse_factor,
                        pathloss_exponent,
                    ),
                    "baseline_hex_edge_sir_db": 10.0
                    * np.log10(
                        edge_user_sir(
                            reuse_factor=reuse_factor,
                            cell_radius=config.cell_radius_m,
                            pathloss_exponent=pathloss_exponent,
                        )
                    ),
                }
            )

    return pd.DataFrame(records)


def run_all_experiments(config: SimulationConfig) -> ExperimentBundle:
    sir_vs_reuse, ase_vs_reuse = run_reuse_experiments(config)
    sir_vs_height, cdf_samples = run_height_experiment(config)
    pathloss_sweep = run_pathloss_sweep(config)
    return ExperimentBundle(
        sir_vs_reuse=sir_vs_reuse,
        ase_vs_reuse=ase_vs_reuse,
        sir_vs_height=sir_vs_height,
        cdf_samples=cdf_samples,
        pathloss_sweep=pathloss_sweep,
    )
