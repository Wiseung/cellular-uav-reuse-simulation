# Data Inputs and Calibration

## Main input families

The repository uses three main input families:

- radio geometry and site layout
- antenna and propagation metadata
- runtime profile and calibration inputs

## Important repository inputs

- `cellular_uav_sir/data/default_parameter_profile.json`
- `cellular_uav_sir/data/real_site_layout_knoxville_tn.csv`
- `cellular_uav_sir/data/knoxville_site_layout_buildings.geojson`
- `cellular_uav_sir/data/reference_sector_panel_pattern.msi`
- `cellular_uav_sir/data/building_material_loss_profile.json`

## Public-data preparation workflow

The tools folder includes scripts for:

- extracting a local site cluster from the public ArcGIS tower dataset
- downloading OpenStreetMap building footprints
- normalizing OpenCelliD or CellMapper exports
- enriching Overture building data
- calibrating a parameter profile from enhanced inputs and a dynamic trace
- running the full public-data pipeline end to end

## Typical command flow

1. Build or normalize the site layout.
2. Download or prepare building footprints.
3. Calibrate the parameter profile when you have better local context.
4. Run the simulator with the prepared inputs.
5. Compare baseline and enhanced outputs.

## Input override columns

When present, the site-layout CSV can override fields such as:

- sector azimuth
- mechanical and electrical downtilt
- transmit height
- transmit power
- antenna peak gain
- radio and ARFCN metadata for same-frequency filtering

## Calibration guidance

Keep calibration changes explicit and reviewable:

- store profile JSON changes in source control
- note the source of external site or building data
- compare baseline versus calibrated outputs instead of replacing them silently
