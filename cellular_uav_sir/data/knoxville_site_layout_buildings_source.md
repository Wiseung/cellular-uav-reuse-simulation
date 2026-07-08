# Knoxville Site-Layout Building Footprint Source

This GIS sample was downloaded from OpenStreetMap through the public Overpass API:

- Overpass endpoint: `https://overpass-api.de/api/interpreter`
- Data license note reported by Overpass: OpenStreetMap data is made available under ODbL
- Local output file: `knoxville_site_layout_buildings.geojson`

Processing used in this branch:

1. Derive the Knoxville tower-cluster bounding box from `real_site_layout_knoxville_tn.csv` and expand it by `0.003°`.
2. Split that bounding box into `0.04°` tiles with `0.001°` overlap to stay within public Overpass request limits.
3. Query OSM `way["building"]` features tile by tile, retry on transient `429/5xx` responses, and deduplicate by `osm_way_id`.
4. Convert the merged Overpass response into a compact GeoJSON `FeatureCollection`.
5. At runtime, re-project the footprint polygons to local metric coordinates using the real-site layout center latitude/longitude.
6. Use those polygons as an optional deterministic LOS/NLOS override for dynamic links across the full current Knoxville site cluster.

This sample covers the current 21-site Knoxville experiment much more completely than the earlier center-corridor subset, but it is still a 2D public-building sample rather than a full 3D city GIS or a ray-tracing database.
