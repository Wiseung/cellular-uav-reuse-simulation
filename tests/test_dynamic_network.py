from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from cellular_uav_sir.config import SimulationConfig
from cellular_uav_sir.dynamic_network import (
    _best_server_choice,
    _build_dynamic_site_parameters,
    _cochannel_interferer_mask,
    _coordination_weights,
    _handover_decision,
    run_dynamic_trajectory_experiment,
)


class DynamicNetworkTests(unittest.TestCase):
    def test_build_dynamic_site_parameters_applies_site_overrides(self) -> None:
        config = replace(
            SimulationConfig(),
            beamforming_enabled=False,
        )
        raw_rows = (
            {
                "site_id": "s0",
                "sector_azimuths_deg": "180",
                "tx_height_m": "40",
                "tx_power_dbm": "50",
                "mechanical_downtilt_deg": "2",
                "electrical_downtilt_deg": "4",
                "radio": "LTE",
                "arfcn_list": "100",
            },
            {
                "site_id": "s1",
                "antenna_azimuth_deg": "90",
                "antenna_height_m": "30",
                "mechanical_downtilt_deg": "1",
                "electrical_downtilt_deg": "3",
                "radio": "LTE",
                "arfcn_list": "200",
            },
        )
        site_parameters = _build_dynamic_site_parameters(
            raw_rows,
            np.array([0.0, 5.0], dtype=float),
            config,
        )

        self.assertEqual(site_parameters.tx_heights_m.tolist(), [40.0, 35.0])
        self.assertEqual(site_parameters.tx_power_dbm.tolist(), [50.0, 46.0])
        self.assertEqual(site_parameters.total_downtilt_deg.tolist(), [6.0, 4.0])
        self.assertEqual(site_parameters.sector_azimuths_deg[0, 0], 180.0)
        self.assertEqual(site_parameters.sector_azimuths_deg[1, 0], 90.0)
        self.assertEqual(site_parameters.sector_mask[0].tolist(), [True])
        self.assertEqual(site_parameters.arfcn_sets[0], frozenset({"100"}))
        self.assertEqual(site_parameters.arfcn_sets[1], frozenset({"200"}))

    def test_best_server_choice_uses_site_specific_sector_azimuths(self) -> None:
        config = replace(
            SimulationConfig(),
            beamforming_enabled=False,
            los_pathloss_exponent=2.2,
            nlos_pathloss_exponent=2.2,
            los_shadow_sigma_db=0.0,
            nlos_shadow_sigma_db=0.0,
            mechanical_downtilt_deg=0.0,
            electrical_downtilt_deg=0.0,
        )
        site_positions = np.array([[0.0, 0.0], [200.0, 0.0]], dtype=float)
        raw_rows = (
            {
                "site_id": "s0",
                "sector_azimuths_deg": "180",
                "radio": "LTE",
                "arfcn_list": "100",
            },
            {
                "site_id": "s1",
                "sector_azimuths_deg": "180",
                "radio": "LTE",
                "arfcn_list": "100",
            },
        )
        site_parameters = _build_dynamic_site_parameters(
            raw_rows,
            np.zeros(2, dtype=float),
            config,
        )

        candidate_reference_power_mw, *_ = _best_server_choice(
            site_positions=site_positions,
            site_parameters=site_parameters,
            user_point=np.array([100.0, 0.0], dtype=float),
            terminal_height_m=config.ground_terminal_height_m,
            config=config,
            rng=np.random.default_rng(123),
            building_dataset=None,
        )

        self.assertGreater(
            float(np.max(candidate_reference_power_mw[1])),
            float(np.max(candidate_reference_power_mw[0])),
        )

    def test_cochannel_interferer_mask_filters_mismatched_radio_and_arfcn(self) -> None:
        site_parameters = _build_dynamic_site_parameters(
            (
                {"site_id": "s0", "radio": "LTE", "arfcn_list": "100"},
                {"site_id": "s1", "radio": "LTE", "arfcn_list": "200"},
                {"site_id": "s2", "radio": "NR", "arfcn_list": "100"},
                {"site_id": "s3", "radio": "LTE", "arfcn_list": "100"},
            ),
            np.zeros(4, dtype=float),
            SimulationConfig(),
        )

        mask = _cochannel_interferer_mask(site_parameters, serving_site_index=0)
        self.assertEqual(mask.tolist(), [False, False, False, True])

    def test_handover_requires_time_to_trigger(self) -> None:
        config = replace(
            SimulationConfig(),
            handover_hysteresis_db=2.0,
            handover_time_to_trigger_steps=2,
            dynamic_min_dwell_steps=0,
        )
        pending_steps = np.zeros(2, dtype=int)
        current_site_index = 0
        last_handover_step = -10

        current_site_index, pending_steps, handover_flag, last_handover_step = _handover_decision(
            current_serving_site_index=current_site_index,
            filtered_measurement_dbm=np.array([0.0, 3.5], dtype=float),
            pending_steps=pending_steps,
            step=0,
            last_handover_step=last_handover_step,
            config=config,
        )
        self.assertEqual(current_site_index, 0)
        self.assertEqual(handover_flag, 0)
        self.assertEqual(pending_steps.tolist(), [0, 1])

        current_site_index, pending_steps, handover_flag, last_handover_step = _handover_decision(
            current_serving_site_index=current_site_index,
            filtered_measurement_dbm=np.array([0.0, 3.6], dtype=float),
            pending_steps=pending_steps,
            step=1,
            last_handover_step=last_handover_step,
            config=config,
        )
        self.assertEqual(current_site_index, 1)
        self.assertEqual(handover_flag, 1)
        self.assertEqual(last_handover_step, 1)
        self.assertEqual(pending_steps.tolist(), [0, 0])

    def test_handover_resets_pending_counter_when_event_clears(self) -> None:
        config = replace(
            SimulationConfig(),
            handover_hysteresis_db=2.0,
            handover_time_to_trigger_steps=2,
            dynamic_min_dwell_steps=0,
        )
        pending_steps = np.zeros(2, dtype=int)

        _, pending_steps, handover_flag, _ = _handover_decision(
            current_serving_site_index=0,
            filtered_measurement_dbm=np.array([0.0, 3.5], dtype=float),
            pending_steps=pending_steps,
            step=0,
            last_handover_step=-10,
            config=config,
        )
        self.assertEqual(handover_flag, 0)
        self.assertEqual(pending_steps.tolist(), [0, 1])

        current_site_index, pending_steps, handover_flag, _ = _handover_decision(
            current_serving_site_index=0,
            filtered_measurement_dbm=np.array([0.0, 1.0], dtype=float),
            pending_steps=pending_steps,
            step=1,
            last_handover_step=-10,
            config=config,
        )
        self.assertEqual(current_site_index, 0)
        self.assertEqual(handover_flag, 0)
        self.assertEqual(pending_steps.tolist(), [0, 0])

    def test_coordination_weights_reduce_strongest_interferers(self) -> None:
        config = replace(
            SimulationConfig(),
            coordinated_scheduling_enabled=True,
            coordinated_scheduling_cluster_size=2,
            coordinated_scheduling_blank_fraction=0.5,
            coordinated_scheduling_sinr_threshold_db=5.0,
        )

        weights, coordinated_count = _coordination_weights(
            interferer_large_scale_power_mw=np.array([10.0, 5.0, 1.0], dtype=float),
            interferer_load_state=np.array([0.2, 0.8, 0.4], dtype=float),
            predicted_sinr_db=0.0,
            config=config,
        )

        self.assertEqual(coordinated_count, 2)
        self.assertLess(weights[0], 1.0)
        self.assertLess(weights[1], 1.0)
        self.assertEqual(weights[2], 1.0)

    def test_run_dynamic_trajectory_experiment_filters_non_cochannel_interference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            site_layout_path = temp_path / "site_layout.csv"
            site_layout_path.write_text(
                "\n".join(
                    [
                        "site_id,x_m,y_m,latdec,londec,sector_azimuths_deg,mechanical_downtilt_deg,electrical_downtilt_deg,radio,arfcn_list",
                        "s0,0,0,35.0,-84.0,180,0,0,LTE,100",
                        "s1,200,0,35.0005,-84.0005,180,0,0,LTE,200",
                    ]
                ),
                encoding="utf-8",
            )
            config = replace(
                SimulationConfig(),
                dynamic_site_layout_csv=site_layout_path,
                building_footprint_geojson=None,
                dynamic_time_steps=1,
                dynamic_path_half_length_m=50.0,
                dynamic_path_lateral_offset_m=0.0,
                dynamic_altitude_m=0.0,
                beamforming_enabled=False,
                mechanical_downtilt_deg=0.0,
                electrical_downtilt_deg=0.0,
                parameter_profile_json=None,
            )

            bundle = run_dynamic_trajectory_experiment(config)

        trace_row = bundle.trace.iloc[0]
        self.assertEqual(int(trace_row["cochannel_interferer_count"]), 0)
        self.assertLess(float(trace_row["interference_power_dbm"]), -140.0)


if __name__ == "__main__":
    unittest.main()
