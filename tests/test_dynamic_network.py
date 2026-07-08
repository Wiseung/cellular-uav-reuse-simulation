from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from cellular_uav_sir.config import SimulationConfig
from cellular_uav_sir.dynamic_network import _coordination_weights, _handover_decision


class DynamicNetworkTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
