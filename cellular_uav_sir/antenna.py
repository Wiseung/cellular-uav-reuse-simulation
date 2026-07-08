from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import SimulationConfig


def _wrap_angle_deg(angle_deg: np.ndarray) -> np.ndarray:
    return np.abs((angle_deg + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class PatternFileData:
    cuts: dict[str, tuple[np.ndarray, np.ndarray]]
    peak_gain_db: float | None = None


def _folded_average_cut(
    angles_deg: np.ndarray,
    attenuations_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    folded_angles_deg = _wrap_angle_deg(np.asarray(angles_deg, dtype=float))
    attenuations = np.asarray(attenuations_db, dtype=float)
    order = np.argsort(folded_angles_deg)
    folded_angles_deg = folded_angles_deg[order]
    attenuations = attenuations[order]

    rounded_angles_deg = np.round(folded_angles_deg, 6)
    unique_angles_deg, inverse_indices = np.unique(rounded_angles_deg, return_inverse=True)
    averaged_attenuations_db = np.zeros(unique_angles_deg.shape[0], dtype=float)
    counts = np.zeros(unique_angles_deg.shape[0], dtype=float)
    for index, attenuation_db in zip(inverse_indices, attenuations, strict=False):
        averaged_attenuations_db[index] += float(attenuation_db)
        counts[index] += 1.0
    averaged_attenuations_db = averaged_attenuations_db / np.maximum(counts, 1.0)
    return unique_angles_deg.astype(float), averaged_attenuations_db.astype(float)


def _parse_peak_gain_db(header_value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", header_value)
    if match is None:
        return None
    return float(match.group(0))


def _normalize_cut_dictionary(
    cuts: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if "horizontal" not in cuts or "vertical" not in cuts:
        raise ValueError(f"Antenna pattern file must contain both horizontal and vertical cuts: {path}")
    return {
        plane: _folded_average_cut(angles_deg, attenuations_db)
        for plane, (angles_deg, attenuations_db) in cuts.items()
    }


def _load_csv_pattern_file(path: Path) -> PatternFileData:
    horizontal_angles: list[float] = []
    horizontal_attenuations: list[float] = []
    vertical_angles: list[float] = []
    vertical_attenuations: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"plane", "angle_deg", "attenuation_db"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Antenna pattern CSV must include columns {sorted(required)}: {path}"
            )
        for row in reader:
            plane = row["plane"].strip().lower()
            angle_deg = abs(float(row["angle_deg"]))
            attenuation_db = float(row["attenuation_db"])
            if plane == "horizontal":
                horizontal_angles.append(angle_deg)
                horizontal_attenuations.append(attenuation_db)
            elif plane == "vertical":
                vertical_angles.append(angle_deg)
                vertical_attenuations.append(attenuation_db)
            else:
                raise ValueError(f"Unsupported antenna pattern plane '{plane}' in {path}")

    return PatternFileData(
        cuts=_normalize_cut_dictionary(
            {
                "horizontal": (
                    np.array(horizontal_angles, dtype=float),
                    np.array(horizontal_attenuations, dtype=float),
                ),
                "vertical": (
                    np.array(vertical_angles, dtype=float),
                    np.array(vertical_attenuations, dtype=float),
                ),
            },
            path,
        )
    )


def _load_msi_pattern_file(path: Path) -> PatternFileData:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = 0
    peak_gain_db: float | None = None
    cuts: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    while index < len(lines):
        line = lines[index]
        upper_line = line.upper()
        if upper_line.startswith("GAIN"):
            peak_gain_db = _parse_peak_gain_db(line)
            index += 1
            continue

        if upper_line.startswith("HORIZONTAL") or upper_line.startswith("VERTICAL"):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Missing sample count in MSI pattern section header: {path}")
            plane = "horizontal" if upper_line.startswith("HORIZONTAL") else "vertical"
            sample_count = int(parts[1])
            angles_deg: list[float] = []
            attenuations_db: list[float] = []
            for _ in range(sample_count):
                index += 1
                if index >= len(lines):
                    raise ValueError(f"Unexpected end of MSI pattern file while reading {plane} cut: {path}")
                sample_parts = lines[index].split()
                if len(sample_parts) < 2:
                    raise ValueError(f"Invalid MSI pattern sample row '{lines[index]}' in {path}")
                angles_deg.append(float(sample_parts[0]))
                attenuations_db.append(float(sample_parts[1]))
            cuts[plane] = (
                np.array(angles_deg, dtype=float),
                np.array(attenuations_db, dtype=float),
            )
        index += 1

    return PatternFileData(
        cuts=_normalize_cut_dictionary(cuts, path),
        peak_gain_db=peak_gain_db,
    )


@lru_cache(maxsize=None)
def _load_pattern_file_cached(pattern_path: str) -> PatternFileData:
    path = Path(pattern_path)
    if not path.exists():
        raise FileNotFoundError(f"Antenna pattern file not found: {path}")
    if path.suffix.lower() == ".csv":
        return _load_csv_pattern_file(path)
    if path.suffix.lower() in {".msi", ".pln"}:
        return _load_msi_pattern_file(path)
    raise ValueError(f"Unsupported antenna pattern file extension '{path.suffix}' in {path}")


def _peak_gain_db(config: SimulationConfig) -> float:
    pattern_path = config.antenna_pattern_file
    if pattern_path is not None and Path(pattern_path).exists():
        pattern_file_data = _load_pattern_file_cached(str(Path(pattern_path).resolve()))
        if pattern_file_data.peak_gain_db is not None:
            return pattern_file_data.peak_gain_db
    return config.antenna_peak_gain_db


def _pattern_attenuation_db(
    abs_angle_deg: np.ndarray,
    plane: str,
    config: SimulationConfig,
) -> np.ndarray:
    angle = np.abs(np.asarray(abs_angle_deg, dtype=float))
    pattern_path = config.antenna_pattern_file
    if pattern_path is not None and Path(pattern_path).exists():
        pattern_file_data = _load_pattern_file_cached(str(Path(pattern_path).resolve()))
        angles, attenuations = pattern_file_data.cuts[plane]
        max_attenuation = np.max(attenuations)
        return np.interp(angle, angles, attenuations, left=attenuations[0], right=max_attenuation)

    beamwidth_deg = (
        config.horizontal_beamwidth_deg if plane == "horizontal" else config.vertical_beamwidth_deg
    )
    max_attenuation = (
        config.horizontal_max_attenuation_db
        if plane == "horizontal"
        else config.vertical_max_attenuation_db
    )
    return np.minimum(12.0 * (angle / beamwidth_deg) ** 2, max_attenuation)


def _beam_codebook(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    if not config.beamforming_enabled:
        return np.array([0.0], dtype=float), np.array([0.0], dtype=float)

    azimuth_offsets: list[float] = []
    elevation_offsets: list[float] = []
    for azimuth_offset_deg in config.beam_codebook_azimuth_offsets_deg:
        for elevation_offset_deg in config.beam_codebook_elevation_offsets_deg:
            azimuth_offsets.append(float(azimuth_offset_deg))
            elevation_offsets.append(float(elevation_offset_deg))
    return np.array(azimuth_offsets, dtype=float), np.array(elevation_offsets, dtype=float)


def _geometry_terms(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users = np.asarray(user_points, dtype=float)
    sites = np.asarray(site_positions, dtype=float)
    if sites.ndim == 2:
        offsets = users[:, None, :] - sites[None, :, :]
    elif sites.ndim == 3:
        offsets = users[:, None, :] - sites
    else:
        raise ValueError("site_positions must have shape (sites, 2) or (samples, sites, 2)")

    rx_height = np.asarray(rx_height_m, dtype=float)
    if rx_height.ndim == 0:
        rx_height = np.full(users.shape[0], float(rx_height))
    elif rx_height.shape != (users.shape[0],):
        raise ValueError("rx_height_m must be a scalar or a vector matching user_points")

    horizontal_distance = np.sqrt(np.sum(offsets * offsets, axis=-1))
    azimuth_deg = np.degrees(np.arctan2(offsets[..., 1], offsets[..., 0]))
    elevation_deg = np.degrees(
        np.arctan2(
            rx_height[:, None] - config.base_station_height_m,
            np.maximum(horizontal_distance, 1.0),
        )
    )
    return horizontal_distance, azimuth_deg, elevation_deg


def sector_gain_db(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    _, azimuth_deg, elevation_deg = _geometry_terms(
        site_positions,
        user_points,
        rx_height_m,
        config,
    )

    sector_azimuths = np.asarray(config.sector_azimuths_deg, dtype=float)
    horizontal_offset_deg = _wrap_angle_deg(azimuth_deg[..., None] - sector_azimuths)
    vertical_offset_deg = np.abs(elevation_deg[..., None] + config.total_downtilt_deg)

    horizontal_attenuation_db = _pattern_attenuation_db(
        horizontal_offset_deg,
        "horizontal",
        config,
    )
    vertical_attenuation_db = _pattern_attenuation_db(
        vertical_offset_deg,
        "vertical",
        config,
    )

    total_attenuation_db = np.minimum(
        horizontal_attenuation_db + vertical_attenuation_db,
        config.max_pattern_attenuation_db,
    )
    return _peak_gain_db(config) - total_attenuation_db


def sector_gain_linear(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    return np.power(
        10.0,
        sector_gain_db(site_positions, user_points, rx_height_m, config) / 10.0,
    )


def sector_beam_gain_db(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    _, azimuth_deg, elevation_deg = _geometry_terms(
        site_positions,
        user_points,
        rx_height_m,
        config,
    )
    sector_azimuths = np.asarray(config.sector_azimuths_deg, dtype=float)
    beam_azimuth_offsets_deg, beam_elevation_offsets_deg = _beam_codebook(config)

    effective_sector_azimuth_deg = (
        sector_azimuths[None, None, :, None] + beam_azimuth_offsets_deg[None, None, None, :]
    )
    horizontal_offset_deg = _wrap_angle_deg(
        azimuth_deg[..., None, None] - effective_sector_azimuth_deg
    )

    effective_vertical_boresight_deg = (
        config.total_downtilt_deg - beam_elevation_offsets_deg[None, None, None, :]
    )
    vertical_offset_deg = np.abs(
        elevation_deg[..., None, None] + effective_vertical_boresight_deg
    )

    horizontal_attenuation_db = _pattern_attenuation_db(
        horizontal_offset_deg,
        "horizontal",
        config,
    )
    vertical_attenuation_db = _pattern_attenuation_db(
        vertical_offset_deg,
        "vertical",
        config,
    )

    total_attenuation_db = np.minimum(
        horizontal_attenuation_db + vertical_attenuation_db,
        config.max_pattern_attenuation_db,
    )
    gain_db = _peak_gain_db(config) - total_attenuation_db
    if config.beamforming_enabled:
        gain_db = gain_db + config.beamforming_array_gain_db
    return gain_db


def best_sector_gain_db(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    if config.beamforming_enabled:
        return np.max(sector_beam_gain_db(site_positions, user_points, rx_height_m, config), axis=(-1, -2))
    return np.max(
        sector_gain_db(site_positions, user_points, rx_height_m, config),
        axis=-1,
    )


def best_sector_gain_linear(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    return np.power(
        10.0,
        best_sector_gain_db(site_positions, user_points, rx_height_m, config) / 10.0,
    )


def random_interferer_gain_linear(
    site_positions: np.ndarray,
    user_points: np.ndarray,
    rx_height_m: float | np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if not config.beamforming_enabled or not config.interferer_random_beams:
        return sector_gain_linear(site_positions, user_points, rx_height_m, config)

    gain_db = sector_beam_gain_db(site_positions, user_points, rx_height_m, config)
    beam_count = gain_db.shape[-1]
    beam_indices = rng.integers(
        low=0,
        high=beam_count,
        size=gain_db.shape[:-1],
    )
    chosen_gain_db = np.take_along_axis(
        gain_db,
        beam_indices[..., None],
        axis=-1,
    )[..., 0]
    return np.power(10.0, chosen_gain_db / 10.0)
