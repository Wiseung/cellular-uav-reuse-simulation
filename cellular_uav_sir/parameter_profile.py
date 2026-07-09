from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import SimulationConfig

PROFILE_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "beam": (
        "antenna_pattern_file",
        "sector_azimuths_deg",
        "horizontal_beamwidth_deg",
        "vertical_beamwidth_deg",
        "horizontal_max_attenuation_db",
        "vertical_max_attenuation_db",
        "max_pattern_attenuation_db",
        "antenna_peak_gain_db",
        "mechanical_downtilt_deg",
        "electrical_downtilt_deg",
        "beamforming_enabled",
        "interferer_random_beams",
        "beam_codebook_azimuth_offsets_deg",
        "beam_codebook_elevation_offsets_deg",
        "beamforming_array_gain_db",
    ),
    "load": (
        "dynamic_load_mean",
        "dynamic_load_std",
        "dynamic_load_correlation",
        "dynamic_max_users_per_site",
        "power_split_exponent",
    ),
    "handover": (
        "handover_hysteresis_db",
        "handover_l3_filter_alpha",
        "handover_time_to_trigger_steps",
        "dynamic_min_dwell_steps",
    ),
    "coordination": (
        "coordinated_scheduling_enabled",
        "coordinated_scheduling_cluster_size",
        "coordinated_scheduling_sinr_threshold_db",
        "coordinated_scheduling_blank_fraction",
    ),
    "paths": (
        "dynamic_site_layout_csv",
        "building_footprint_geojson",
        "building_material_loss_profile_json",
        "results_dir",
    ),
}

TUPLE_FLOAT_FIELDS = {
    "sector_azimuths_deg",
    "beam_codebook_azimuth_offsets_deg",
    "beam_codebook_elevation_offsets_deg",
}
INT_FIELDS = {
    "dynamic_max_users_per_site",
    "handover_time_to_trigger_steps",
    "dynamic_min_dwell_steps",
    "coordinated_scheduling_cluster_size",
}
FLOAT_FIELDS = {
    "horizontal_beamwidth_deg",
    "vertical_beamwidth_deg",
    "horizontal_max_attenuation_db",
    "vertical_max_attenuation_db",
    "max_pattern_attenuation_db",
    "antenna_peak_gain_db",
    "mechanical_downtilt_deg",
    "electrical_downtilt_deg",
    "beamforming_array_gain_db",
    "dynamic_load_mean",
    "dynamic_load_std",
    "dynamic_load_correlation",
    "power_split_exponent",
    "handover_hysteresis_db",
    "handover_l3_filter_alpha",
    "coordinated_scheduling_sinr_threshold_db",
    "coordinated_scheduling_blank_fraction",
}
BOOL_FIELDS = {
    "beamforming_enabled",
    "interferer_random_beams",
    "coordinated_scheduling_enabled",
}
PATH_FIELDS = {
    "antenna_pattern_file",
    "dynamic_site_layout_csv",
    "building_footprint_geojson",
    "building_material_loss_profile_json",
    "results_dir",
}


def _normalize_profile_value(field_name: str, value: Any, base_dir: Path) -> Any:
    if field_name in TUPLE_FLOAT_FIELDS:
        if not isinstance(value, list):
            raise ValueError(f"Profile field '{field_name}' must be a JSON array.")
        return tuple(float(entry) for entry in value)
    if field_name in BOOL_FIELDS:
        return bool(value)
    if field_name in INT_FIELDS:
        return int(value)
    if field_name in FLOAT_FIELDS:
        return float(value)
    if field_name in PATH_FIELDS:
        if value is None:
            return None
        path_value = Path(value)
        if not path_value.is_absolute():
            path_value = (base_dir / path_value).resolve()
        return path_value
    return value


def load_parameter_profile(profile_path: Path | str) -> dict[str, Any]:
    path = Path(profile_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Parameter profile must be a JSON object: {path}")
    return payload


def apply_parameter_profile(
    config: SimulationConfig,
    profile_path: Path | str | None,
) -> SimulationConfig:
    if profile_path is None:
        return replace(config, external_profile_applied=True)

    path = Path(profile_path)
    if not path.exists():
        return replace(config, external_profile_applied=True)

    payload = load_parameter_profile(path)
    updates: dict[str, Any] = {}
    for section_name, field_names in PROFILE_SECTION_FIELDS.items():
        section_payload = payload.get(section_name, {})
        if section_payload in (None, {}):
            continue
        if not isinstance(section_payload, dict):
            raise ValueError(f"Profile section '{section_name}' must be a JSON object: {path}")
        for field_name in field_names:
            if field_name not in section_payload:
                continue
            updates[field_name] = _normalize_profile_value(
                field_name,
                section_payload[field_name],
                path.parent,
            )

    return replace(
        config,
        **updates,
        parameter_profile_json=path,
        external_profile_applied=True,
    )


def resolve_runtime_config(config: SimulationConfig) -> SimulationConfig:
    if config.external_profile_applied:
        return config
    return apply_parameter_profile(config, config.parameter_profile_json)
