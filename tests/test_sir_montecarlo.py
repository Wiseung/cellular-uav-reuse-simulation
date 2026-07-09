from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from cellular_uav_sir.config import SimulationConfig
from cellular_uav_sir.sir_montecarlo import (
    simulate_los_probability_sir_samples,
    simulate_sir_samples,
)


class SirMonteCarloStaticLayoutTests(unittest.TestCase):
    def test_simulate_sir_samples_uses_site_specific_serving_tx_power(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            boosted_layout_path = temp_path / "boosted.csv"
            boosted_layout_path.write_text(
                "\n".join(
                    [
                        "site_id,x_m,y_m,latdec,londec,sector_azimuths_deg,tx_power_dbm",
                        "s0,0,0,35.0,-84.0,0,56",
                    ]
                ),
                encoding="utf-8",
            )
            default_layout_path = temp_path / "default.csv"
            default_layout_path.write_text(
                "\n".join(
                    [
                        "site_id,x_m,y_m,latdec,londec,sector_azimuths_deg",
                        "s0,0,0,35.0,-84.0,0",
                    ]
                ),
                encoding="utf-8",
            )

            base_config = replace(
                SimulationConfig(),
                beamforming_enabled=False,
                ground_shadow_sigma_db=0.0,
                ground_small_scale_m=1.0,
                parameter_profile_json=None,
            )
            boosted_samples = simulate_sir_samples(
                user_points=np.array([[100.0, 0.0]], dtype=float),
                reuse_factor=1,
                cell_radius=500.0,
                pathloss_exponent=base_config.ground_pathloss_exponent,
                config=replace(base_config, site_layout_csv=boosted_layout_path),
                rng=np.random.default_rng(123),
                interferer_count=0,
            )
            default_samples = simulate_sir_samples(
                user_points=np.array([[100.0, 0.0]], dtype=float),
                reuse_factor=1,
                cell_radius=500.0,
                pathloss_exponent=base_config.ground_pathloss_exponent,
                config=replace(base_config, site_layout_csv=default_layout_path),
                rng=np.random.default_rng(123),
                interferer_count=0,
            )

        self.assertGreater(
            float(boosted_samples.signal_power_mw[0]),
            float(default_samples.signal_power_mw[0]) * 9.0,
        )

    def test_simulate_sir_samples_filters_non_cochannel_layout_interferers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            filtered_layout_path = temp_path / "filtered.csv"
            filtered_layout_path.write_text(
                "\n".join(
                    [
                        "site_id,x_m,y_m,latdec,londec,sector_azimuths_deg,radio,arfcn_list",
                        "s0,0,0,35.0,-84.0,0,LTE,100",
                        "s1,200,0,35.0005,-84.0005,180,LTE,200",
                    ]
                ),
                encoding="utf-8",
            )
            same_channel_layout_path = temp_path / "same_channel.csv"
            same_channel_layout_path.write_text(
                "\n".join(
                    [
                        "site_id,x_m,y_m,latdec,londec,sector_azimuths_deg,radio,arfcn_list",
                        "s0,0,0,35.0,-84.0,0,LTE,100",
                        "s1,200,0,35.0005,-84.0005,180,LTE,100",
                    ]
                ),
                encoding="utf-8",
            )

            config = replace(
                SimulationConfig(),
                beamforming_enabled=False,
                ground_shadow_sigma_db=0.0,
                ground_small_scale_m=1.0,
                parameter_profile_json=None,
            )
            filtered_samples = simulate_sir_samples(
                user_points=np.array([[50.0, 0.0]], dtype=float),
                reuse_factor=1,
                cell_radius=500.0,
                pathloss_exponent=config.ground_pathloss_exponent,
                config=replace(config, site_layout_csv=filtered_layout_path),
                rng=np.random.default_rng(123),
                interferer_count=1,
            )
            same_channel_samples = simulate_sir_samples(
                user_points=np.array([[50.0, 0.0]], dtype=float),
                reuse_factor=1,
                cell_radius=500.0,
                pathloss_exponent=config.ground_pathloss_exponent,
                config=replace(config, site_layout_csv=same_channel_layout_path),
                rng=np.random.default_rng(123),
                interferer_count=1,
            )

        self.assertLess(float(filtered_samples.interference_power_mw[0]), 1e-12)
        self.assertGreater(
            float(same_channel_samples.interference_power_mw[0]),
            float(filtered_samples.interference_power_mw[0]) + 1e-9,
        )

    def test_simulate_los_probability_sir_samples_uses_site_specific_serving_azimuth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            east_layout_path = temp_path / "east.csv"
            east_layout_path.write_text(
                "\n".join(
                    [
                        "site_id,x_m,y_m,latdec,londec,sector_azimuths_deg",
                        "s0,0,0,35.0,-84.0,0",
                    ]
                ),
                encoding="utf-8",
            )
            west_layout_path = temp_path / "west.csv"
            west_layout_path.write_text(
                "\n".join(
                    [
                        "site_id,x_m,y_m,latdec,londec,sector_azimuths_deg",
                        "s0,0,0,35.0,-84.0,180",
                    ]
                ),
                encoding="utf-8",
            )

            config = replace(
                SimulationConfig(),
                beamforming_enabled=False,
                los_shadow_sigma_db=0.0,
                nlos_shadow_sigma_db=0.0,
                los_small_scale_m=1.0,
                nlos_small_scale_m=1.0,
                los_pathloss_exponent=2.2,
                nlos_pathloss_exponent=2.2,
                parameter_profile_json=None,
            )
            east_samples = simulate_los_probability_sir_samples(
                user_points=np.array([[100.0, 0.0]], dtype=float),
                reuse_factor=1,
                cell_radius=500.0,
                user_altitude_m=0.0,
                config=replace(config, site_layout_csv=east_layout_path),
                interferer_count=0,
                rng=np.random.default_rng(123),
            )
            west_samples = simulate_los_probability_sir_samples(
                user_points=np.array([[100.0, 0.0]], dtype=float),
                reuse_factor=1,
                cell_radius=500.0,
                user_altitude_m=0.0,
                config=replace(config, site_layout_csv=west_layout_path),
                interferer_count=0,
                rng=np.random.default_rng(123),
            )

        self.assertGreater(
            float(east_samples.signal_power_mw[0]),
            float(west_samples.signal_power_mw[0]),
        )


if __name__ == "__main__":
    unittest.main()
