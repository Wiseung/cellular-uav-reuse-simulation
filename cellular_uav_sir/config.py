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
    carrier_frequency_ghz: float = 3.5
    tx_power_dbm: float = 46.0
    receiver_noise_figure_db: float = 7.0
    thermal_noise_density_dbm_per_hz: float = -174.0
    channel_bandwidth_hz: float = 20e6
    reference_distance_m: float = 1.0
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
    site_layout_csv: Path | None = None
    dynamic_site_layout_csv: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "data" / "demo_site_layout.csv"
    )
    ground_interferer_count: int = 18
    aerial_interferer_count: int = 42
    layout_interferer_count: int = 24
    site_perturbation_fraction: float = 0.18
    antenna_pattern_csv: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "data" / "custom_panel_pattern.csv"
    )
    sector_azimuths_deg: tuple[float, ...] = (90.0, 210.0, 330.0)
    horizontal_beamwidth_deg: float = 65.0
    vertical_beamwidth_deg: float = 8.0
    horizontal_max_attenuation_db: float = 30.0
    vertical_max_attenuation_db: float = 30.0
    max_pattern_attenuation_db: float = 30.0
    antenna_peak_gain_db: float = 17.0
    mechanical_downtilt_deg: float = 4.0
    electrical_downtilt_deg: float = 8.0
    beamforming_enabled: bool = True
    interferer_random_beams: bool = True
    beam_codebook_azimuth_offsets_deg: tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0)
    beam_codebook_elevation_offsets_deg: tuple[float, ...] = (-6.0, 0.0, 6.0)
    beamforming_array_gain_db: float = 6.0
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
    dynamic_time_steps: int = 60
    dynamic_time_step_s: float = 1.0
    dynamic_altitude_m: float = 120.0
    dynamic_path_half_length_m: float = 1500.0
    dynamic_path_lateral_offset_m: float = 150.0
    dynamic_load_mean: float = 0.65
    dynamic_load_std: float = 0.18
    dynamic_load_correlation: float = 0.85
    dynamic_max_users_per_site: int = 8
    handover_hysteresis_db: float = 2.5
    dynamic_min_dwell_steps: int = 3
    dynamic_outage_threshold_db: float = 0.0
    power_split_exponent: float = 1.0
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

    @property
    def tx_power_mw(self) -> float:
        return float(np.power(10.0, self.tx_power_dbm / 10.0))

    @property
    def thermal_noise_power_dbm(self) -> float:
        bandwidth_term_db = 10.0 * np.log10(self.channel_bandwidth_hz)
        return float(
            self.thermal_noise_density_dbm_per_hz
            + bandwidth_term_db
            + self.receiver_noise_figure_db
        )

    @property
    def thermal_noise_power_mw(self) -> float:
        return float(np.power(10.0, self.thermal_noise_power_dbm / 10.0))

    @property
    def reference_pathloss_db(self) -> float:
        return float(32.4 + 20.0 * np.log10(self.carrier_frequency_ghz))
