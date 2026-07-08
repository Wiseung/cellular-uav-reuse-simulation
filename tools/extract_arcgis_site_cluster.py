from __future__ import annotations

import argparse
import csv
import io
import math
import urllib.request
from pathlib import Path

EARTH_RADIUS_M = 6371000.0
DEFAULT_SOURCE_URL = (
    "https://hub.arcgis.com/api/v3/datasets/"
    "15dabb4108254481b591018be2598f3c_0/downloads/data?format=csv&spatialRefId=4326"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a local real-site cluster from the public ArcGIS cellular tower dataset."
    )
    parser.add_argument("--center-lat", type=float, required=True)
    parser.add_argument("--center-lon", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--count", type=int, help="Keep the nearest N sites to the center point.")
    selection.add_argument(
        "--radius-m",
        type=float,
        help="Keep every site within the given radius of the center point.",
    )
    return parser.parse_args()


def _local_xy_m(
    latitude_deg: float,
    longitude_deg: float,
    center_lat_deg: float,
    center_lon_deg: float,
) -> tuple[float, float]:
    delta_lat_rad = math.radians(latitude_deg - center_lat_deg)
    delta_lon_rad = math.radians(longitude_deg - center_lon_deg)
    center_lat_rad = math.radians(center_lat_deg)
    x_m = EARTH_RADIUS_M * delta_lon_rad * math.cos(center_lat_rad)
    y_m = EARTH_RADIUS_M * delta_lat_rad
    return x_m, y_m


def _load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.input_csv is not None:
        with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    request = urllib.request.Request(
        args.source_url,
        headers={"User-Agent": "Codex/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(payload)))


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args)

    selected_rows: list[dict[str, object]] = []
    for row in rows:
        lat_text = row.get("latdec")
        lon_text = row.get("londec")
        if lat_text in (None, "") or lon_text in (None, ""):
            continue

        latitude_deg = float(lat_text)
        longitude_deg = float(lon_text)
        x_m, y_m = _local_xy_m(
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            center_lat_deg=args.center_lat,
            center_lon_deg=args.center_lon,
        )
        distance_m = math.hypot(x_m, y_m)
        selected_rows.append(
            {
                "x_m": x_m,
                "y_m": y_m,
                "distance_m": distance_m,
                "objectid": row.get("objectid", ""),
                "city": row.get("city", ""),
                "state": row.get("state", ""),
                "latdec": latitude_deg,
                "londec": longitude_deg,
            }
        )

    selected_rows.sort(key=lambda item: float(item["distance_m"]))
    if args.count is not None:
        selected_rows = selected_rows[: args.count]
    else:
        selected_rows = [
            item for item in selected_rows if float(item["distance_m"]) <= float(args.radius_m)
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "site_id",
                "x_m",
                "y_m",
                "objectid",
                "city",
                "state",
                "latdec",
                "londec",
                "distance_m",
            ],
        )
        writer.writeheader()
        for site_id, row in enumerate(selected_rows):
            writer.writerow(
                {
                    "site_id": site_id,
                    "x_m": row["x_m"],
                    "y_m": row["y_m"],
                    "objectid": row["objectid"],
                    "city": row["city"],
                    "state": row["state"],
                    "latdec": row["latdec"],
                    "londec": row["londec"],
                    "distance_m": row["distance_m"],
                }
            )

    print(f"Wrote {len(selected_rows)} sites to {args.output}")


if __name__ == "__main__":
    main()
