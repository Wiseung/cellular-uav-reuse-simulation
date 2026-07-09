from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .geometry import load_site_layout_rows


def _parse_angle_list(value: object) -> list[float]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [float(entry) % 360.0 for entry in value]
    angles: list[float] = []
    for token in str(value).replace(";", "|").replace(",", "|").split("|"):
        text = token.strip()
        if not text:
            continue
        angles.append(float(text) % 360.0)
    return angles


def _parse_optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dominant_sector_azimuths_deg(angles_deg: list[float], bucket_width_deg: float = 5.0) -> list[float]:
    if not angles_deg:
        return []
    bucket_counts: dict[int, int] = {}
    for angle_deg in angles_deg:
        bucket_index = int(round((angle_deg % 360.0) / bucket_width_deg)) % int(360.0 / bucket_width_deg)
        bucket_counts[bucket_index] = bucket_counts.get(bucket_index, 0) + 1
    ordered_buckets = sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0]))
    chosen_angles_deg: list[float] = []
    for bucket_index, _ in ordered_buckets:
        angle_deg = (bucket_index * bucket_width_deg) % 360.0
        if any(abs(((angle_deg - existing + 180.0) % 360.0) - 180.0) < bucket_width_deg for existing in chosen_angles_deg):
            continue
        chosen_angles_deg.append(angle_deg)
        if len(chosen_angles_deg) == 3:
            break
    return sorted(round(angle_deg, 3) for angle_deg in chosen_angles_deg)


def calibrate_beam_profile(site_layout_csv: Path | str) -> dict[str, Any]:
    rows = load_site_layout_rows(site_layout_csv)
    all_angles_deg: list[float] = []
    mechanical_tilts_deg: list[float] = []
    electrical_tilts_deg: list[float] = []

    for row in rows:
        sector_angles_deg = _parse_angle_list(row.get("sector_azimuths_deg"))
        if sector_angles_deg:
            all_angles_deg.extend(sector_angles_deg)
        else:
            antenna_azimuth_deg = _parse_optional_float(row.get("antenna_azimuth_deg"))
            if antenna_azimuth_deg is not None:
                all_angles_deg.append(antenna_azimuth_deg)

        mechanical_tilt_deg = _parse_optional_float(row.get("mechanical_downtilt_deg"))
        if mechanical_tilt_deg is not None:
            mechanical_tilts_deg.append(mechanical_tilt_deg)
        electrical_tilt_deg = _parse_optional_float(row.get("electrical_downtilt_deg"))
        if electrical_tilt_deg is not None:
            electrical_tilts_deg.append(electrical_tilt_deg)

    beam_profile: dict[str, Any] = {}
    dominant_angles_deg = _dominant_sector_azimuths_deg(all_angles_deg)
    if dominant_angles_deg:
        beam_profile["sector_azimuths_deg"] = dominant_angles_deg
    if mechanical_tilts_deg:
        beam_profile["mechanical_downtilt_deg"] = float(np.median(mechanical_tilts_deg))
    if electrical_tilts_deg:
        beam_profile["electrical_downtilt_deg"] = float(np.median(electrical_tilts_deg))
    return beam_profile


def _lag1_autocorrelation(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    first = values[:-1]
    second = values[1:]
    if np.allclose(first, first[0]) or np.allclose(second, second[0]):
        return None
    return float(np.corrcoef(first, second)[0, 1])


def calibrate_dynamic_trace(dynamic_trace_csv: Path | str) -> dict[str, dict[str, Any]]:
    trace = pd.read_csv(dynamic_trace_csv)
    sections: dict[str, dict[str, Any]] = {}

    load_samples: list[np.ndarray] = []
    if "serving_load" in trace:
        load_samples.append(trace["serving_load"].to_numpy(dtype=float))
    if "mean_neighbor_load" in trace:
        load_samples.append(trace["mean_neighbor_load"].to_numpy(dtype=float))
    if load_samples:
        merged_loads = np.concatenate(load_samples)
        load_profile: dict[str, Any] = {
            "dynamic_load_mean": float(np.mean(merged_loads)),
            "dynamic_load_std": float(np.std(merged_loads)),
        }
        if "serving_load" in trace:
            load_correlation = _lag1_autocorrelation(trace["serving_load"].to_numpy(dtype=float))
            if load_correlation is not None:
                load_profile["dynamic_load_correlation"] = float(np.clip(load_correlation, 0.0, 0.999))
        if "scheduled_users_serving" in trace:
            load_profile["dynamic_max_users_per_site"] = int(
                max(1, round(float(trace["scheduled_users_serving"].quantile(0.95))))
            )
        sections["load"] = load_profile

    required_handover_columns = {
        "serving_measurement_dbm",
        "best_neighbor_measurement_dbm",
        "handover_flag",
    }
    if required_handover_columns.issubset(trace.columns):
        handover_rows = trace.loc[trace["handover_flag"] > 0]
        handover_profile: dict[str, Any] = {}
        if not handover_rows.empty:
            margins_db = (
                handover_rows["best_neighbor_measurement_dbm"].to_numpy(dtype=float)
                - handover_rows["serving_measurement_dbm"].to_numpy(dtype=float)
            )
            handover_profile["handover_hysteresis_db"] = float(np.median(np.maximum(margins_db, 0.0)))

            better_neighbor = (
                trace["best_neighbor_measurement_dbm"].to_numpy(dtype=float)
                > trace["serving_measurement_dbm"].to_numpy(dtype=float)
            )
            handover_flags = trace["handover_flag"].to_numpy(dtype=int) > 0
            run_lengths: list[int] = []
            run_length = 0
            for better_neighbor_flag, handover_flag in zip(better_neighbor, handover_flags):
                run_length = run_length + 1 if better_neighbor_flag else 0
                if handover_flag:
                    run_lengths.append(max(run_length, 1))
                    run_length = 0
            if run_lengths:
                handover_profile["handover_time_to_trigger_steps"] = int(round(float(np.median(run_lengths))))

            if "step" in handover_rows:
                handover_steps = handover_rows["step"].to_numpy(dtype=int)
                if handover_steps.size > 1:
                    handover_profile["dynamic_min_dwell_steps"] = int(
                        max(0, round(float(np.median(np.diff(handover_steps)))))
                    )
        if handover_profile:
            sections["handover"] = handover_profile

    return sections


def build_parameter_profile(
    site_layout_csv: Path | str | None = None,
    dynamic_trace_csv: Path | str | None = None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "metadata": {
            "generated_by": "cellular_uav_sir.calibration.build_parameter_profile",
        }
    }
    if site_layout_csv is not None:
        beam_profile = calibrate_beam_profile(site_layout_csv)
        if beam_profile:
            profile["beam"] = beam_profile
            profile["metadata"]["site_layout_csv"] = str(Path(site_layout_csv))
    if dynamic_trace_csv is not None:
        trace_sections = calibrate_dynamic_trace(dynamic_trace_csv)
        profile.update(trace_sections)
        profile["metadata"]["dynamic_trace_csv"] = str(Path(dynamic_trace_csv))
    return profile


def write_parameter_profile(profile: dict[str, Any], output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
