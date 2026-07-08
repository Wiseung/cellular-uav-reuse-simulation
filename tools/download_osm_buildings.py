from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path

DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OSM building footprints as a compact GeoJSON FeatureCollection."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-layout-csv", type=Path)
    parser.add_argument("--south", type=float)
    parser.add_argument("--west", type=float)
    parser.add_argument("--north", type=float)
    parser.add_argument("--east", type=float)
    parser.add_argument("--margin-deg", type=float, default=0.0025)
    parser.add_argument("--tile-size-deg", type=float, default=0.04)
    parser.add_argument("--tile-overlap-deg", type=float, default=0.0005)
    parser.add_argument("--request-gap-s", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-s", type=float, default=2.0)
    parser.add_argument("--overpass-url", default=DEFAULT_OVERPASS_URL)
    args = parser.parse_args()

    using_site_layout = args.site_layout_csv is not None
    using_explicit_bbox = None not in (args.south, args.west, args.north, args.east)
    if using_site_layout == using_explicit_bbox:
        parser.error("Use either --site-layout-csv or an explicit --south/--west/--north/--east bbox.")
    return args


def _bbox_from_site_layout(path: Path, margin_deg: float) -> tuple[float, float, float, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lats = [float(row["latdec"]) for row in rows if row.get("latdec") not in (None, "")]
    lons = [float(row["londec"]) for row in rows if row.get("londec") not in (None, "")]
    if not lats or not lons:
        raise ValueError("Site layout CSV must include latdec/londec columns for bbox inference.")
    return (
        min(lats) - margin_deg,
        min(lons) - margin_deg,
        max(lats) + margin_deg,
        max(lons) + margin_deg,
    )


def _overpass_query(south: float, west: float, north: float, east: float) -> str:
    return f"""
[out:json][timeout:120];
(
  way["building"]({south},{west},{north},{east});
);
out geom;
""".strip()


def _download_payload(
    url: str,
    query: str,
    max_retries: int,
    retry_backoff_s: float,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=query.encode("utf-8"),
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "User-Agent": "Codex/1.0",
        },
    )
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return json.load(response)
        except HTTPError as exc:
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == max_retries - 1:
                raise
        except URLError:
            if attempt == max_retries - 1:
                raise
        time.sleep(retry_backoff_s * (2**attempt))

    raise RuntimeError("Overpass download retry loop exited unexpectedly.")


def _to_feature(element: dict[str, object]) -> dict[str, object] | None:
    geometry = element.get("geometry")
    if not isinstance(geometry, list) or len(geometry) < 4:
        return None

    coordinates = [[float(point["lon"]), float(point["lat"])] for point in geometry]
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])

    tags = element.get("tags") or {}
    if not isinstance(tags, dict):
        tags = {}

    properties = {
        key: tags[key]
        for key in ("building", "height", "building:levels", "name")
        if key in tags
    }
    properties["osm_way_id"] = element.get("id")
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates],
        },
    }


def _tile_ranges(start_deg: float, stop_deg: float, tile_size_deg: float, overlap_deg: float) -> list[tuple[float, float]]:
    if tile_size_deg <= 0.0:
        raise ValueError("tile_size_deg must be positive.")
    if overlap_deg < 0.0:
        raise ValueError("tile_overlap_deg must be non-negative.")

    ranges: list[tuple[float, float]] = []
    cursor = float(start_deg)
    while cursor < stop_deg:
        tile_stop = min(cursor + tile_size_deg, stop_deg)
        ranges.append((cursor, tile_stop))
        if tile_stop >= stop_deg:
            break
        cursor = tile_stop - overlap_deg
    return ranges


def _tile_bboxes(
    south: float,
    west: float,
    north: float,
    east: float,
    tile_size_deg: float,
    overlap_deg: float,
) -> list[tuple[float, float, float, float]]:
    latitude_ranges = _tile_ranges(south, north, tile_size_deg, overlap_deg)
    longitude_ranges = _tile_ranges(west, east, tile_size_deg, overlap_deg)
    return [
        (tile_south, tile_west, tile_north, tile_east)
        for (tile_south, tile_north), (tile_west, tile_east) in itertools.product(
            latitude_ranges,
            longitude_ranges,
        )
    ]


def _deduplicate_features(features: list[dict[str, object]]) -> list[dict[str, object]]:
    deduplicated: dict[int, dict[str, object]] = {}
    for feature in features:
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue
        osm_way_id = properties.get("osm_way_id")
        if isinstance(osm_way_id, int):
            deduplicated[osm_way_id] = feature
    return list(deduplicated.values())


def main() -> None:
    args = _parse_args()
    if args.site_layout_csv is not None:
        south, west, north, east = _bbox_from_site_layout(args.site_layout_csv, args.margin_deg)
    else:
        south, west, north, east = args.south, args.west, args.north, args.east

    tile_bboxes = _tile_bboxes(
        south=south,
        west=west,
        north=north,
        east=east,
        tile_size_deg=args.tile_size_deg,
        overlap_deg=args.tile_overlap_deg,
    )
    features: list[dict[str, object]] = []
    for tile_index, (tile_south, tile_west, tile_north, tile_east) in enumerate(tile_bboxes, start=1):
        payload = _download_payload(
            args.overpass_url,
            _overpass_query(tile_south, tile_west, tile_north, tile_east),
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff_s,
        )
        elements = payload.get("elements", [])
        if not isinstance(elements, list):
            raise ValueError("Unexpected Overpass response format.")

        tile_feature_count = 0
        for element in elements:
            if isinstance(element, dict):
                feature = _to_feature(element)
                if feature is not None:
                    features.append(feature)
                    tile_feature_count += 1
        print(
            f"Tile {tile_index}/{len(tile_bboxes)} "
            f"({tile_south}, {tile_west}, {tile_north}, {tile_east}) -> {tile_feature_count} features"
        )
        if tile_index < len(tile_bboxes) and args.request_gap_s > 0.0:
            time.sleep(args.request_gap_s)

    features = _deduplicate_features(features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    print(
        f"Wrote {len(features)} building footprints to {args.output} "
        f"for bbox ({south}, {west}, {north}, {east}) "
        f"using {len(tile_bboxes)} tile(s)"
    )


if __name__ == "__main__":
    main()
