# Real Site Layout Source

This layout is a processed local cluster extracted from the public ArcGIS Hub dataset:

- Dataset: `Cellular Towers in the United States`
- Hub dataset id: `15dabb4108254481b591018be2598f3c_0`
- Public download/API entry:
  - `https://hub.arcgis.com/api/v3/datasets/15dabb4108254481b591018be2598f3c_0`
  - `https://hub.arcgis.com/api/v3/datasets/15dabb4108254481b591018be2598f3c_0/downloads/data?format=csv&spatialRefId=4326`

Processing used in this branch:

1. Query public feature records with geometry and location attributes.
2. Search the full public point set for a dense local cluster.
3. Select the nearest 21 real sites around a Knoxville, Tennessee center site.
4. Re-center coordinates to local metric offsets `x_m, y_m` for direct use by the simulator.

The resulting `real_site_layout_knoxville_tn.csv` is intended for dynamic network experiments and data-driven layout integration, not as a canonical nationwide tower dataset.
