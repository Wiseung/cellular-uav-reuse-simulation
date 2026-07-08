# Cellular + UAV Reuse Simulation

This workspace implements the course-design simulation for:

`蜂窝 + 低空融合网络频率复用与 SIR 信干比仿真`

The code follows the execution plan in `执行计划.md` and now uses a more engineering-oriented model:

- Analytic reuse baseline: `SIR = (3N)^(n/2) / 6`
- Explicit `SIR` and `SINR` computation with thermal noise, bandwidth, and receiver noise figure
- Perturbed hexagonal co-channel deployment instead of a perfectly regular lattice
- Monte Carlo user drops inside the serving hexagon
- Custom antenna-pattern CSV support, three-sector base-station panels, and optional beam codebook steering
- Lognormal shadowing and small-scale fading on every link
- A hybrid UAV LOS model:
  - 3GPP TR 38.901 UMa LOS probability for near-ground user heights
  - ITU-R P.1410 statistical building-blockage LOS probability for higher aerial links
  - LOS/NLOS pathloss exponents, shadowing, and fading sampled per link during the height experiment
- Effective ASE with traffic activity, scheduling efficiency, control overhead, and outage gating
- Dynamic trajectory experiment with external site-layout CSV input, handover hysteresis, correlated load, and per-user power splitting

## How to run

From the workspace root:

```powershell
pip install -r requirements.txt
```

Then run:

```powershell
python -m cellular_uav_sir.main
```

The script generates CSV tables and PNG figures under:

`cellular_uav_sir/results/`

## Output files

- `table_1_sir_vs_reuse.csv`
- `table_2_ase_vs_reuse.csv`
- `table_3_sir_vs_height.csv`
- `table_4_sir_cdf_samples.csv`
- `table_5_pathloss_sweep.csv`
- `table_6_dynamic_summary.csv`
- `table_7_dynamic_trace.csv`
- `table_8_dynamic_site_layout.csv`
- `figure_1_reuse_geometry.png`
- `figure_2_sir_vs_reuse.png`
- `figure_3_ase_vs_reuse.png`
- `figure_4_sir_vs_height.png`
- `figure_5_sir_cdf.png`
- `figure_6_pathloss_sweep.png`
- `figure_7_los_probability_vs_height.png`
- `figure_8_dynamic_sinr_timeline.png`
- `figure_9_dynamic_layout_map.png`

## Repository contents

- `cellular_uav_sir/`: simulation source code
- `cellular_uav_sir/data/`: antenna-pattern and site-layout CSV inputs, including a real Knoxville cell-tower cluster derived from a public ArcGIS Hub dataset
- `cellular_uav_sir/results/`: generated CSV tables and PNG figures
- `完整中文报告初稿.md`: full Chinese report
- `答辩讲稿.md`: oral defense script
- `PPT汇报提纲.md`: slide-ready page copy

## Main assumptions

- Downlink only
- Equal transmit power for all base stations
- Thermal noise is explicitly included through SINR
- Ground users use multiple co-channel tiers plus random site perturbation
- UAV users use a hybrid 3GPP/ITU LOS probability model above the ground-user regime
- Real site layouts can be injected via CSV for dynamic experiments
- ASE is reported as effective ASE after applying activity factor, scheduler efficiency, control overhead, and a coverage threshold
- Dynamic experiments add UAV trajectory, handover logic, correlated cell load, and load-aware power sharing

## Real layout note

The default dynamic layout now uses `cellular_uav_sir/data/real_site_layout_knoxville_tn.csv`, which was derived from the public ArcGIS Hub dataset `Cellular Towers in the United States`. The processing note is recorded in `cellular_uav_sir/data/real_site_layout_knoxville_tn_source.md`.
