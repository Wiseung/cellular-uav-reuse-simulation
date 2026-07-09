from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import urlopen

USGS_EPQS_URL = "https://epqs.nationalmap.gov/v1/json"


def _parse_optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bbox_from_site_layout(path: Path, margin_deg: float) -> tuple[float, float, float, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"latdec", "londec"}.issubset(reader.fieldnames):
            raise ValueError(f"Site layout CSV must include latdec/londec columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Site layout CSV contains no rows: {path}")
    latitudes_deg = [float(row["latdec"]) for row in rows]
    longitudes_deg = [float(row["londec"]) for row in rows]
    return (
        min(latitudes_deg) - margin_deg,
        min(longitudes_deg) - margin_deg,
        max(latitudes_deg) + margin_deg,
        max(longitudes_deg) + margin_deg,
    )


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


def _feature_centroid_lonlat(feature: dict[str, Any]) -> tuple[float, float] | None:
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return None
    rings = _iter_exterior_rings(geometry)
    if not rings:
        return None
    points = rings[0]
    if len(points) < 3:
        return None
    longitudes_deg = [float(point[0]) for point in points]
    latitudes_deg = [float(point[1]) for point in points]
    return float(sum(longitudes_deg) / len(longitudes_deg)), float(sum(latitudes_deg) / len(latitudes_deg))


def _feature_intersects_bbox(feature: dict[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    centroid = _feature_centroid_lonlat(feature)
    if centroid is None:
        return False
    west, south = bbox[1], bbox[0]
    east, north = bbox[3], bbox[2]
    longitude_deg, latitude_deg = centroid
    return west <= longitude_deg <= east and south <= latitude_deg <= north


def _normalized_building_height_m(
    properties: dict[str, Any],
    default_floor_height_m: float,
    default_height_m: float,
) -> tuple[float, str]:
    for key in ("height", "roof_height", "building:height", "roof:height"):
        parsed_height_m = _parse_optional_float(properties.get(key))
        if parsed_height_m is not None and parsed_height_m > 0.0:
            return parsed_height_m, key
    for key in ("num_floors", "building:levels", "levels"):
        parsed_floors = _parse_optional_float(properties.get(key))
        if parsed_floors is not None and parsed_floors > 0.0:
            return parsed_floors * default_floor_height_m, key
    return default_height_m, "default_height_m"


def query_3dep_ground_elevation_m(latitude_deg: float, longitude_deg: float) -> float:
    query = urlencode(
        {
            "x": longitude_deg,
            "y": latitude_deg,
            "units": "Meters",
            "wkid": 4326,
            "includeDate": "false",
        }
    )
    with urlopen(f"{USGS_EPQS_URL}?{query}") as response:
        payload = json.load(response)

    value = payload.get("value")
    if value is None:
        value = payload.get("elevation")
    if value is None and isinstance(payload.get("value"), dict):
        value = payload["value"].get("value")
    if value is None:
        raise ValueError("USGS EPQS response did not include an elevation value.")
    return float(value)


def enrich_overture_feature_collection(
    payload: dict[str, Any],
    default_floor_height_m: float = 3.0,
    default_height_m: float = 12.0,
    ground_elevation_lookup: Callable[[float, float], float] | None = None,
) -> dict[str, Any]:
    if payload.get("type") != "FeatureCollection":
        raise ValueError("Overture building payload must be a GeoJSON FeatureCollection.")

    enriched_features: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or not _iter_exterior_rings(geometry):
            continue
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        centroid = _feature_centroid_lonlat(feature)
        if centroid is None:
            continue

        longitude_deg, latitude_deg = centroid
        height_m, height_source = _normalized_building_height_m(
            properties,
            default_floor_height_m=default_floor_height_m,
            default_height_m=default_height_m,
        )
        ground_elevation_m = None
        if ground_elevation_lookup is not None:
            ground_elevation_m = ground_elevation_lookup(latitude_deg, longitude_deg)

        normalized_properties = dict(properties)
        normalized_properties["height"] = round(float(height_m), 3)
        normalized_properties["height_source"] = height_source
        normalized_properties["centroid_lon"] = round(float(longitude_deg), 7)
        normalized_properties["centroid_lat"] = round(float(latitude_deg), 7)
        if ground_elevation_m is not None:
            normalized_properties["ground_elevation_m"] = round(float(ground_elevation_m), 3)
            normalized_properties["absolute_roof_elevation_m"] = round(
                float(ground_elevation_m) + float(height_m),
                3,
            )

        enriched_features.append(
            {
                "type": "Feature",
                "properties": normalized_properties,
                "geometry": geometry,
            }
        )

    return {"type": "FeatureCollection", "features": enriched_features}


def _load_geojson_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported Overture building payload format: {path}")


def prepare_overture_3dep_buildings(
    overture_geojson: Path | str,
    output_geojson: Path | str,
    site_layout_csv: Path | str | None = None,
    margin_deg: float = 0.003,
    default_floor_height_m: float = 3.0,
    default_height_m: float = 12.0,
    include_ground_elevation: bool = True,
) -> dict[str, Any]:
    payload = _load_geojson_payload(Path(overture_geojson))
    if site_layout_csv is not None:
        bbox = _bbox_from_site_layout(Path(site_layout_csv), margin_deg=margin_deg)
        payload = {
            "type": "FeatureCollection",
            "features": [
                feature
                for feature in payload.get("features", [])
                if isinstance(feature, dict) and _feature_intersects_bbox(feature, bbox)
            ],
        }

    enriched_payload = enrich_overture_feature_collection(
        payload,
        default_floor_height_m=default_floor_height_m,
        default_height_m=default_height_m,
        ground_elevation_lookup=query_3dep_ground_elevation_m if include_ground_elevation else None,
    )
    output_path = Path(output_geojson)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(enriched_payload, handle)
    return enriched_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Overture building footprints and enrich them with USGS 3DEP ground elevation."
    )
    parser.add_argument("--overture-geojson", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-layout-csv", type=Path)
    parser.add_argument("--margin-deg", type=float, default=0.003)
    parser.add_argument("--default-floor-height-m", type=float, default=3.0)
    parser.add_argument("--default-height-m", type=float, default=12.0)
    parser.add_argument(
        "--skip-ground-elevation",
        action="store_true",
        help="Skip USGS 3DEP EPQS lookups and only normalize Overture height/material fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_overture_3dep_buildings(
        overture_geojson=args.overture_geojson,
        output_geojson=args.output,
        site_layout_csv=args.site_layout_csv,
        margin_deg=args.margin_deg,
        default_floor_height_m=args.default_floor_height_m,
        default_height_m=args.default_height_m,
        include_ground_elevation=not args.skip_ground_elevation,
    )
    print(f"Wrote Overture + 3DEP building GeoJSON to {args.output}")


if __name__ == "__main__":
    main()
