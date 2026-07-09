from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6371000.0


@dataclass(frozen=True)
class BuildingFootprint:
    polygon_xy_m: np.ndarray
    height_m: float
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    top_height_m: float | None = None
    ground_elevation_m: float | None = None
    facade_material: str | None = None
    roof_material: str | None = None
    height_source: str | None = None

    @property
    def obstruction_height_m(self) -> float:
        if self.top_height_m is not None:
            return float(self.top_height_m)
        return float(self.height_m)


@dataclass(frozen=True)
class BuildingDataset:
    buildings: tuple[BuildingFootprint, ...]
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float
    grid_cell_size_m: float = 200.0
    spatial_index: dict[tuple[int, int], tuple[int, ...]] | None = None


def _load_site_layout_origin_latlon(site_layout_csv: Path | str) -> tuple[float, float] | None:
    path = Path(site_layout_csv)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"latdec", "londec"}.issubset(reader.fieldnames):
            return None

        best_origin: tuple[float, float] | None = None
        best_norm = float("inf")
        for row in reader:
            lat = row.get("latdec")
            lon = row.get("londec")
            if lat in (None, "") or lon in (None, ""):
                continue
            try:
                lat_value = float(lat)
                lon_value = float(lon)
            except ValueError:
                continue

            x_value = float(row.get("x_m", 0.0) or 0.0)
            y_value = float(row.get("y_m", 0.0) or 0.0)
            norm = x_value * x_value + y_value * y_value
            if norm < best_norm:
                best_norm = norm
                best_origin = (lat_value, lon_value)
        return best_origin


def _load_site_layout_origin_ground_elevation_m(site_layout_csv: Path | str) -> float | None:
    path = Path(site_layout_csv)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "ground_elevation_m" not in reader.fieldnames:
            return None

        best_ground_elevation_m: float | None = None
        best_norm = float("inf")
        for row in reader:
            parsed_ground_elevation_m = _parse_height_value_m(row.get("ground_elevation_m"))
            if parsed_ground_elevation_m is None:
                continue
            x_value = float(row.get("x_m", 0.0) or 0.0)
            y_value = float(row.get("y_m", 0.0) or 0.0)
            norm = x_value * x_value + y_value * y_value
            if norm < best_norm:
                best_norm = norm
                best_ground_elevation_m = parsed_ground_elevation_m
        return best_ground_elevation_m


def _lonlat_to_local_xy_m(
    latitude_deg: float,
    longitude_deg: float,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
) -> tuple[float, float]:
    latitude_rad = math.radians(latitude_deg)
    origin_latitude_rad = math.radians(origin_latitude_deg)
    delta_lat_rad = latitude_rad - origin_latitude_rad
    delta_lon_rad = math.radians(longitude_deg - origin_longitude_deg)
    x_m = EARTH_RADIUS_M * delta_lon_rad * math.cos(origin_latitude_rad)
    y_m = EARTH_RADIUS_M * delta_lat_rad
    return x_m, y_m


def _parse_height_value_m(value: object) -> float | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return None

    height_value = float(match.group(0))
    if "ft" in text or "feet" in text or "'" in text:
        return height_value * 0.3048
    return height_value


def _building_height_m(
    properties: dict[str, object],
    default_height_m: float,
    level_height_m: float,
) -> tuple[float, str]:
    for key in ("height", "building:height", "roof:height", "roof_height"):
        parsed_height_m = _parse_height_value_m(properties.get(key))
        if parsed_height_m is not None and parsed_height_m > 0.0:
            return parsed_height_m, key

    for key in ("building:levels", "levels", "num_floors"):
        parsed_levels = _parse_height_value_m(properties.get(key))
        if parsed_levels is not None and parsed_levels > 0.0:
            return parsed_levels * level_height_m, key

    return default_height_m, "default_height_m"


def _building_ground_elevation_m(properties: dict[str, object]) -> float | None:
    for key in ("ground_elevation_m", "ground_elevation", "elevation"):
        parsed_ground_elevation_m = _parse_height_value_m(properties.get(key))
        if parsed_ground_elevation_m is not None:
            return parsed_ground_elevation_m
    return None


def _building_top_height_m(
    height_m: float,
    properties: dict[str, object],
    origin_ground_elevation_m: float | None,
) -> tuple[float | None, float | None]:
    ground_elevation_m = _building_ground_elevation_m(properties)
    if ground_elevation_m is None or origin_ground_elevation_m is None:
        return None, ground_elevation_m
    return height_m + (ground_elevation_m - origin_ground_elevation_m), ground_elevation_m


def _optional_text_property(properties: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = properties.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


@lru_cache(maxsize=None)
def _load_material_loss_models(
    material_loss_profile_path: str | None,
) -> tuple[tuple[tuple[str, ...], float, float], ...]:
    if material_loss_profile_path is None:
        path = Path(__file__).resolve().parent / "data" / "building_material_loss_profile.json"
    else:
        path = Path(material_loss_profile_path)
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    materials = payload.get("materials", [])
    if not isinstance(materials, list):
        raise ValueError(f"Material loss profile must contain a 'materials' list: {path}")

    models: list[tuple[tuple[str, ...], float, float]] = []
    for entry in materials:
        if not isinstance(entry, dict):
            continue
        tokens = entry.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            continue
        models.append(
            (
                tuple(str(token).strip().lower() for token in tokens if str(token).strip()),
                float(entry.get("penetration_multiplier", 1.0)),
                float(entry.get("entry_loss_db", 0.0)),
            )
        )
    return tuple(models)


def _material_penetration_adjustment(
    building: BuildingFootprint,
    material_loss_profile_path: Path | str | None = None,
) -> tuple[float, float]:
    material_text = " ".join(
        value.lower()
        for value in (building.facade_material, building.roof_material)
        if value is not None and value.strip()
    )
    if not material_text:
        return 1.0, 0.0

    multiplier = 1.0
    entry_loss_db = 0.0
    material_models = _load_material_loss_models(
        None if material_loss_profile_path is None else str(Path(material_loss_profile_path).resolve())
    )
    for tokens, candidate_multiplier, candidate_entry_loss_db in material_models:
        if any(token in material_text for token in tokens):
            multiplier = max(multiplier, candidate_multiplier)
            entry_loss_db = max(entry_loss_db, candidate_entry_loss_db)
    return multiplier, entry_loss_db


def _polygon_area_m2(polygon_xy_m: np.ndarray) -> float:
    x = polygon_xy_m[:, 0]
    y = polygon_xy_m[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _iter_exterior_rings(geometry: dict[str, object]) -> list[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon" and isinstance(coordinates, list) and coordinates:
        return [coordinates[0]]
    if geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        rings: list[list[list[float]]] = []
        for polygon in coordinates:
            if isinstance(polygon, list) and polygon:
                rings.append(polygon[0])
        return rings
    return []


def _grid_index_range(
    min_coordinate_m: float,
    max_coordinate_m: float,
    cell_size_m: float,
) -> range:
    start_index = math.floor(min_coordinate_m / cell_size_m)
    stop_index = math.floor(max_coordinate_m / cell_size_m)
    return range(start_index, stop_index + 1)


def _build_spatial_index(
    buildings: list[BuildingFootprint],
    cell_size_m: float,
) -> dict[tuple[int, int], tuple[int, ...]]:
    index: dict[tuple[int, int], list[int]] = {}
    for building_index, building in enumerate(buildings):
        x_indices = _grid_index_range(building.min_x_m, building.max_x_m, cell_size_m)
        y_indices = _grid_index_range(building.min_y_m, building.max_y_m, cell_size_m)
        for x_index in x_indices:
            for y_index in y_indices:
                index.setdefault((x_index, y_index), []).append(building_index)
    return {key: tuple(value) for key, value in index.items()}


@lru_cache(maxsize=None)
def _load_building_dataset_cached(
    building_geojson_path: str,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    origin_ground_elevation_m: float | None,
    default_height_m: float,
    level_height_m: float,
    min_area_m2: float,
) -> BuildingDataset:
    path = Path(building_geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"Building footprint file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"Building footprint GeoJSON must be a FeatureCollection: {path}")

    buildings: list[BuildingFootprint] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}

        height_m, height_source = _building_height_m(properties, default_height_m, level_height_m)
        top_height_m, ground_elevation_m = _building_top_height_m(
            height_m,
            properties,
            origin_ground_elevation_m,
        )
        facade_material = _optional_text_property(
            properties,
            "facade_material",
            "building:material",
            "material",
        )
        roof_material = _optional_text_property(properties, "roof_material", "roof:material")
        for ring in _iter_exterior_rings(geometry):
            if len(ring) < 4:
                continue

            polygon_xy_m = np.array(
                [
                    _lonlat_to_local_xy_m(
                        latitude_deg=float(point[1]),
                        longitude_deg=float(point[0]),
                        origin_latitude_deg=origin_latitude_deg,
                        origin_longitude_deg=origin_longitude_deg,
                    )
                    for point in ring
                ],
                dtype=float,
            )
            if np.allclose(polygon_xy_m[0], polygon_xy_m[-1]):
                polygon_xy_m = polygon_xy_m[:-1]
            if polygon_xy_m.shape[0] < 3:
                continue
            if _polygon_area_m2(polygon_xy_m) < min_area_m2:
                continue

            min_x_m = float(np.min(polygon_xy_m[:, 0]))
            max_x_m = float(np.max(polygon_xy_m[:, 0]))
            min_y_m = float(np.min(polygon_xy_m[:, 1]))
            max_y_m = float(np.max(polygon_xy_m[:, 1]))
            buildings.append(
                BuildingFootprint(
                    polygon_xy_m=polygon_xy_m,
                    height_m=float(height_m),
                    min_x_m=min_x_m,
                    max_x_m=max_x_m,
                    min_y_m=min_y_m,
                    max_y_m=max_y_m,
                    top_height_m=None if top_height_m is None else float(top_height_m),
                    ground_elevation_m=None if ground_elevation_m is None else float(ground_elevation_m),
                    facade_material=facade_material,
                    roof_material=roof_material,
                    height_source=height_source,
                )
            )

    if not buildings:
        raise ValueError(f"Building footprint GeoJSON contains no usable polygons: {path}")

    return BuildingDataset(
        buildings=tuple(buildings),
        min_x_m=min(building.min_x_m for building in buildings),
        max_x_m=max(building.max_x_m for building in buildings),
        min_y_m=min(building.min_y_m for building in buildings),
        max_y_m=max(building.max_y_m for building in buildings),
        spatial_index=_build_spatial_index(buildings, cell_size_m=200.0),
    )


def load_building_dataset(
    building_geojson_path: Path | str,
    site_layout_csv: Path | str,
    default_height_m: float,
    level_height_m: float,
    min_area_m2: float = 16.0,
) -> BuildingDataset:
    origin_latlon = _load_site_layout_origin_latlon(site_layout_csv)
    if origin_latlon is None:
        raise ValueError(
            "Site layout CSV must include latdec/londec columns to align building footprints."
        )

    origin_latitude_deg, origin_longitude_deg = origin_latlon
    origin_ground_elevation_m = _load_site_layout_origin_ground_elevation_m(site_layout_csv)
    return _load_building_dataset_cached(
        str(Path(building_geojson_path).resolve()),
        float(origin_latitude_deg),
        float(origin_longitude_deg),
        None if origin_ground_elevation_m is None else float(origin_ground_elevation_m),
        float(default_height_m),
        float(level_height_m),
        float(min_area_m2),
    )


def _cross_2d(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return float(vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0])


def _segment_intersection_t(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
) -> float | None:
    direction = segment_end - segment_start
    edge_direction = edge_end - edge_start
    denominator = _cross_2d(direction, edge_direction)
    if abs(denominator) < 1e-9:
        return None

    delta = edge_start - segment_start
    segment_t = _cross_2d(delta, edge_direction) / denominator
    edge_u = _cross_2d(delta, direction) / denominator
    if 0.0 <= segment_t <= 1.0 and 0.0 <= edge_u <= 1.0:
        return float(segment_t)
    return None


def _point_in_polygon(point_xy_m: np.ndarray, polygon_xy_m: np.ndarray) -> bool:
    x_value = float(point_xy_m[0])
    y_value = float(point_xy_m[1])
    inside = False
    for index in range(polygon_xy_m.shape[0]):
        x1, y1 = polygon_xy_m[index]
        x2, y2 = polygon_xy_m[(index + 1) % polygon_xy_m.shape[0]]
        intersects = (y1 > y_value) != (y2 > y_value)
        if not intersects:
            continue
        intersection_x = (x2 - x1) * (y_value - y1) / max(y2 - y1, 1e-12) + x1
        if x_value < intersection_x:
            inside = not inside
    return inside


def _segment_polygon_interval_t(
    segment_start_xy_m: np.ndarray,
    segment_end_xy_m: np.ndarray,
    polygon_xy_m: np.ndarray,
) -> tuple[float, float] | None:
    t_values: list[float] = []
    start_inside = _point_in_polygon(segment_start_xy_m, polygon_xy_m)
    end_inside = _point_in_polygon(segment_end_xy_m, polygon_xy_m)
    if start_inside:
        t_values.append(0.0)
    if end_inside:
        t_values.append(1.0)

    for index in range(polygon_xy_m.shape[0]):
        edge_start = polygon_xy_m[index]
        edge_end = polygon_xy_m[(index + 1) % polygon_xy_m.shape[0]]
        intersection_t = _segment_intersection_t(
            segment_start_xy_m,
            segment_end_xy_m,
            edge_start,
            edge_end,
        )
        if intersection_t is not None:
            t_values.append(intersection_t)

    if not t_values:
        return None

    unique_t_values = np.unique(np.round(np.array(t_values, dtype=float), 9))
    if unique_t_values.size == 1:
        only_t = float(unique_t_values[0])
        if start_inside:
            return 0.0, only_t
        if end_inside:
            return only_t, 1.0
        return only_t, only_t
    return float(unique_t_values[0]), float(unique_t_values[-1])


def _building_blocks_segment(
    building: BuildingFootprint,
    segment_start_xy_m: np.ndarray,
    segment_end_xy_m: np.ndarray,
    tx_height_m: float,
    rx_height_m: float,
) -> bool:
    segment_min_x_m = min(float(segment_start_xy_m[0]), float(segment_end_xy_m[0]))
    segment_max_x_m = max(float(segment_start_xy_m[0]), float(segment_end_xy_m[0]))
    segment_min_y_m = min(float(segment_start_xy_m[1]), float(segment_end_xy_m[1]))
    segment_max_y_m = max(float(segment_start_xy_m[1]), float(segment_end_xy_m[1]))
    if (
        segment_max_x_m < building.min_x_m
        or segment_min_x_m > building.max_x_m
        or segment_max_y_m < building.min_y_m
        or segment_min_y_m > building.max_y_m
    ):
        return False

    interval_t = _segment_polygon_interval_t(
        segment_start_xy_m,
        segment_end_xy_m,
        building.polygon_xy_m,
    )
    if interval_t is None:
        return False

    t_enter, t_exit = interval_t
    line_height_enter_m = tx_height_m + t_enter * (rx_height_m - tx_height_m)
    line_height_exit_m = tx_height_m + t_exit * (rx_height_m - tx_height_m)
    min_line_height_m = min(line_height_enter_m, line_height_exit_m)
    return building.obstruction_height_m >= min_line_height_m


def _knife_edge_loss_db(
    clearance_m: float,
    d1_m: float,
    d2_m: float,
    carrier_frequency_ghz: float,
    loss_cap_db: float,
) -> float:
    if clearance_m <= 0.0:
        return 0.0

    frequency_hz = max(float(carrier_frequency_ghz) * 1e9, 1.0)
    wavelength_m = 299792458.0 / frequency_hz
    v_parameter = clearance_m * math.sqrt(
        2.0 * (d1_m + d2_m) / max(wavelength_m * d1_m * d2_m, 1e-12)
    )
    if v_parameter <= -0.78:
        return 0.0

    loss_db = 6.9 + 20.0 * math.log10(
        math.sqrt((v_parameter - 0.1) ** 2 + 1.0) + v_parameter - 0.1
    )
    return float(np.clip(loss_db, 0.0, loss_cap_db))


def _building_excess_loss_db(
    building: BuildingFootprint,
    segment_start_xy_m: np.ndarray,
    segment_end_xy_m: np.ndarray,
    tx_height_m: float,
    rx_height_m: float,
    carrier_frequency_ghz: float,
    penetration_loss_per_meter_db: float,
    penetration_loss_cap_db: float,
    diffraction_loss_cap_db: float,
    material_loss_profile_path: Path | str | None = None,
) -> float:
    interval_t = _segment_polygon_interval_t(
        segment_start_xy_m,
        segment_end_xy_m,
        building.polygon_xy_m,
    )
    if interval_t is None:
        return 0.0

    t_enter, t_exit = interval_t
    total_distance_m = float(np.linalg.norm(segment_end_xy_m - segment_start_xy_m))
    if total_distance_m <= 1e-9:
        return 0.0

    line_height_enter_m = tx_height_m + t_enter * (rx_height_m - tx_height_m)
    line_height_exit_m = tx_height_m + t_exit * (rx_height_m - tx_height_m)
    if building.obstruction_height_m < min(line_height_enter_m, line_height_exit_m):
        return 0.0

    t_mid = 0.5 * (t_enter + t_exit)
    line_height_mid_m = tx_height_m + t_mid * (rx_height_m - tx_height_m)
    clearance_m = max(building.obstruction_height_m - line_height_mid_m, 0.0)
    penetration_depth_m = total_distance_m * max(t_exit - t_enter, 0.0)
    d1_m = max(total_distance_m * max(t_mid, 1e-6), 1.0)
    d2_m = max(total_distance_m - total_distance_m * max(t_mid, 1e-6), 1.0)

    diffraction_loss_db = _knife_edge_loss_db(
        clearance_m=clearance_m,
        d1_m=d1_m,
        d2_m=d2_m,
        carrier_frequency_ghz=carrier_frequency_ghz,
        loss_cap_db=diffraction_loss_cap_db,
    )
    material_multiplier, material_entry_loss_db = _material_penetration_adjustment(
        building,
        material_loss_profile_path=material_loss_profile_path,
    )
    penetration_loss_db = min(
        penetration_depth_m * penetration_loss_per_meter_db * material_multiplier
        + material_entry_loss_db,
        penetration_loss_cap_db * material_multiplier + material_entry_loss_db,
    )
    return max(diffraction_loss_db, penetration_loss_db)


def _candidate_building_indices(
    building_dataset: BuildingDataset,
    segment_start_xy_m: np.ndarray,
    segment_end_xy_m: np.ndarray,
) -> tuple[int, ...]:
    if not building_dataset.spatial_index:
        return tuple(range(len(building_dataset.buildings)))

    start_x_m = float(segment_start_xy_m[0])
    start_y_m = float(segment_start_xy_m[1])
    end_x_m = float(segment_end_xy_m[0])
    end_y_m = float(segment_end_xy_m[1])
    delta_x_m = end_x_m - start_x_m
    delta_y_m = end_y_m - start_y_m
    max_distance_m = max(abs(delta_x_m), abs(delta_y_m))
    step_count = max(1, int(math.ceil(max_distance_m / building_dataset.grid_cell_size_m)))

    candidate_indices: set[int] = set()
    for step in range(step_count + 1):
        interpolation = step / step_count
        point_x_m = start_x_m + interpolation * delta_x_m
        point_y_m = start_y_m + interpolation * delta_y_m
        x_index = math.floor(point_x_m / building_dataset.grid_cell_size_m)
        y_index = math.floor(point_y_m / building_dataset.grid_cell_size_m)
        candidate_indices.update(building_dataset.spatial_index.get((x_index, y_index), ()))
    return tuple(candidate_indices)


def evaluate_gis_los_and_loss(
    site_positions_xy_m: np.ndarray,
    user_point_xy_m: np.ndarray,
    tx_height_m: float | np.ndarray,
    rx_height_m: float,
    building_dataset: BuildingDataset | None,
    carrier_frequency_ghz: float,
    penetration_loss_per_meter_db: float,
    penetration_loss_cap_db: float,
    diffraction_loss_cap_db: float,
    total_excess_loss_cap_db: float,
    material_loss_profile_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    site_positions = np.asarray(site_positions_xy_m, dtype=float)
    user_point = np.asarray(user_point_xy_m, dtype=float)
    tx_heights_m = np.asarray(tx_height_m, dtype=float)
    if tx_heights_m.ndim == 0:
        tx_heights_m = np.full(site_positions.shape[0], float(tx_heights_m), dtype=float)

    covered = np.zeros(site_positions.shape[0], dtype=bool)
    los = np.ones(site_positions.shape[0], dtype=bool)
    excess_loss_db = np.zeros(site_positions.shape[0], dtype=float)
    if building_dataset is None:
        return covered, los, excess_loss_db

    for index, site_point in enumerate(site_positions):
        segment_min_x_m = min(float(site_point[0]), float(user_point[0]))
        segment_max_x_m = max(float(site_point[0]), float(user_point[0]))
        segment_min_y_m = min(float(site_point[1]), float(user_point[1]))
        segment_max_y_m = max(float(site_point[1]), float(user_point[1]))
        segment_intersects_bounds = not (
            segment_max_x_m < building_dataset.min_x_m
            or segment_min_x_m > building_dataset.max_x_m
            or segment_max_y_m < building_dataset.min_y_m
            or segment_min_y_m > building_dataset.max_y_m
        )
        if not segment_intersects_bounds:
            continue

        covered[index] = True
        los[index] = True
        candidate_indices = _candidate_building_indices(
            building_dataset,
            segment_start_xy_m=site_point,
            segment_end_xy_m=user_point,
        )
        building_losses_db: list[float] = []
        for building_index in candidate_indices:
            building = building_dataset.buildings[building_index]
            tx_height_value_m = float(tx_heights_m[index])
            if _building_blocks_segment(
                building,
                segment_start_xy_m=site_point,
                segment_end_xy_m=user_point,
                tx_height_m=tx_height_value_m,
                rx_height_m=rx_height_m,
            ):
                los[index] = False
                building_losses_db.append(
                    _building_excess_loss_db(
                        building,
                        segment_start_xy_m=site_point,
                        segment_end_xy_m=user_point,
                        tx_height_m=tx_height_value_m,
                        rx_height_m=rx_height_m,
                        carrier_frequency_ghz=carrier_frequency_ghz,
                        penetration_loss_per_meter_db=penetration_loss_per_meter_db,
                        penetration_loss_cap_db=penetration_loss_cap_db,
                        diffraction_loss_cap_db=diffraction_loss_cap_db,
                        material_loss_profile_path=material_loss_profile_path,
                    )
                )
        if building_losses_db:
            dominant_losses_db = sorted(building_losses_db, reverse=True)[:2]
            excess_loss_db[index] = min(sum(dominant_losses_db), total_excess_loss_cap_db)

    return covered, los, excess_loss_db


def evaluate_gis_los(
    site_positions_xy_m: np.ndarray,
    user_point_xy_m: np.ndarray,
    tx_height_m: float | np.ndarray,
    rx_height_m: float,
    building_dataset: BuildingDataset | None,
) -> tuple[np.ndarray, np.ndarray]:
    covered, los, _ = evaluate_gis_los_and_loss(
        site_positions_xy_m=site_positions_xy_m,
        user_point_xy_m=user_point_xy_m,
        tx_height_m=tx_height_m,
        rx_height_m=rx_height_m,
        building_dataset=building_dataset,
        carrier_frequency_ghz=3.5,
        penetration_loss_per_meter_db=0.0,
        penetration_loss_cap_db=0.0,
        diffraction_loss_cap_db=0.0,
        total_excess_loss_cap_db=0.0,
        material_loss_profile_path=None,
    )
    return covered, los
