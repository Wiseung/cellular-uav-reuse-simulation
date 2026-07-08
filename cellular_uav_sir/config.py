from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REUSE_PAIR_BY_FACTOR: dict[int, tuple[int, int]] = {
    1: (1, 0),
    3: (1, 1),
    4: (2, 0),
    7: (2, 1),
    12: (2, 2),
}


@dataclass(frozen=True)
class SimulationConfig:
    cell_radius_m: float = 500.0
    reuse_factors: tuple[int, ...] = (1, 3, 4, 7, 12)
    default_reuse_factor: int = 7
    monte_carlo_samples: int = 4000
    random_seed: int = 42
    ground_pathloss_exponent: float = 3.8
    aerial_pathloss_exponent: float = 2.2
    los_pathloss_exponent: float = 2.2
    nlos_pathloss_exponent: float = 3.8
    pathloss_exponent_sweep: tuple[float, ...] = (3.5, 3.8, 4.0)
    uav_altitudes_m: tuple[int, ...] = tuple(range(0, 301, 25))
    cdf_altitudes_m: tuple[int, ...] = (0, 100, 300)
    max_aerial_height_m: float = 300.0
    second_tier_full_visibility_m: float = 150.0
    base_station_height_m: float = 25.0
    ground_terminal_height_m: float = 1.5
    uma_max_user_height_m: float = 23.0
    itu_blend_top_height_m: float = 60.0
    itu_alpha: float = 0.3
    itu_beta_buildings_per_km2: float = 300.0
    itu_gamma_m: float = 15.0
    ground_interferer_count: int = 18
    aerial_interferer_count: int = 42
    site_perturbation_fraction: float = 0.18
    sector_azimuths_deg: tuple[float, ...] = (90.0, 210.0, 330.0)
    horizontal_beamwidth_deg: float = 65.0
    vertical_beamwidth_deg: float = 8.0
    horizontal_max_attenuation_db: float = 30.0
    vertical_max_attenuation_db: float = 30.0
    max_pattern_attenuation_db: float = 30.0
    mechanical_downtilt_deg: float = 4.0
    electrical_downtilt_deg: float = 8.0
    ground_shadow_sigma_db: float = 6.0
    los_shadow_sigma_db: float = 4.0
    nlos_shadow_sigma_db: float = 7.0
    ground_small_scale_m: float = 1.0
    los_small_scale_m: float = 3.0
    nlos_small_scale_m: float = 1.0
    resource_activity_factor: float = 0.7
    scheduler_efficiency: float = 0.9
    control_overhead_fraction: float = 0.18
    coverage_threshold_db: float = 10.0
    results_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "results"
    )

    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.random_seed + offset)

    @property
    def total_downtilt_deg(self) -> float:
        return self.mechanical_downtilt_deg + self.electrical_downtilt_deg

    @property
    def cell_area_km2(self) -> float:
        area_m2 = 1.5 * np.sqrt(3.0) * self.cell_radius_m**2
        return float(area_m2 / 1e6)
