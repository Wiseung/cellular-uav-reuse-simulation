# Experiments and Outputs

## Output location

Default outputs are written to `cellular_uav_sir/results/`.

## Standard generated artifacts

Tables:

- `table_1_sir_vs_reuse.csv`
- `table_2_ase_vs_reuse.csv`
- `table_3_sir_vs_height.csv`
- `table_4_sir_cdf_samples.csv`
- `table_5_pathloss_sweep.csv`
- `table_6_dynamic_summary.csv`
- `table_7_dynamic_trace.csv`
- `table_8_dynamic_site_layout.csv`

Figures:

- `figure_1_reuse_geometry.png`
- `figure_2_sir_vs_reuse.png`
- `figure_3_ase_vs_reuse.png`
- `figure_4_sir_vs_height.png`
- `figure_5_sir_cdf.png`
- `figure_6_pathloss_sweep.png`
- `figure_7_los_probability_vs_height.png`
- `figure_8_dynamic_sinr_timeline.png`
- `figure_9_dynamic_layout_map.png`

## Experiment families

- reuse-factor sweeps
- effective ASE versus reuse
- UAV height sensitivity
- `SIR` distribution views
- pathloss parameter sweeps
- dynamic timeline and handover behavior

## How to read results

- Use the tables for reproducible comparison and downstream reporting.
- Use the figures for trend inspection and presentation.
- Compare baseline and enhanced runs on the same input set before drawing conclusions from calibration changes.

## Deliverable-oriented runs

The `deliverables/` folder contains generated report assets and pipeline outputs. Treat those as publication artifacts, not as the source of truth for runtime behavior.
