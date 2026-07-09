# Getting Started

## Prerequisites

- Python 3.12 or a nearby supported Python 3 release
- `pip`
- a local checkout of the repository

## Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the default simulation

```powershell
python -m cellular_uav_sir.main
```

This writes CSV tables and PNG figures under `cellular_uav_sir/results/`.

## Run with explicit inputs

```powershell
python -m cellular_uav_sir.main `
  --profile-json cellular_uav_sir/data/default_parameter_profile.json `
  --site-layout-csv cellular_uav_sir/data/real_site_layout_knoxville_tn.csv `
  --building-geojson cellular_uav_sir/data/knoxville_site_layout_buildings.geojson
```

## Where to look next

- Runtime assumptions: [Simulation Model](Simulation-Model.md)
- Input files and public-data flow: [Data Inputs and Calibration](Data-Inputs-and-Calibration.md)
- Output files and experiment coverage: [Experiments and Outputs](Experiments-and-Outputs.md)

## Quick troubleshooting

- Missing dependencies: reinstall with `pip install -r requirements.txt`
- Unexpected output differences: confirm which profile JSON and site-layout CSV were used
- Questions about interpretation: open a `Q&A` discussion instead of an issue
