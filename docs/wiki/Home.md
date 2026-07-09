# Cellular + UAV Reuse Simulation Wiki

This wiki is the operator-facing companion to the main repository README. It keeps the high-signal project map in one place without repeating every implementation detail.

## Start here

- [Getting Started](Getting-Started.md)
- [Simulation Model](Simulation-Model.md)
- [Data Inputs and Calibration](Data-Inputs-and-Calibration.md)
- [Experiments and Outputs](Experiments-and-Outputs.md)
- [Development and Testing](Development-and-Testing.md)
- [FAQ](FAQ.md)

## What this repository does

The project simulates cellular frequency reuse for ground users and low-altitude UAV users with an engineering-oriented propagation stack. The current implementation includes:

- analytic reuse baselines
- explicit `SIR` and `SINR` calculations
- perturbed co-channel deployments and Monte Carlo user drops
- hybrid LOS modeling for aerial links
- custom antenna patterns and external parameter profiles
- real site-layout injection and public-data preparation tooling
- dynamic handover, load, and coordinated-scheduling experiments

## Recommended navigation

- New user: start with [Getting Started](Getting-Started.md)
- Modeling review: go to [Simulation Model](Simulation-Model.md)
- Public data workflow: go to [Data Inputs and Calibration](Data-Inputs-and-Calibration.md)
- Results review: go to [Experiments and Outputs](Experiments-and-Outputs.md)
- Contributor workflow: go to [Development and Testing](Development-and-Testing.md)

## Community entry points

- [Security policy](https://github.com/Wiseung/cellular-uav-reuse-simulation/blob/main/SECURITY.md)
- Bugs and bounded work requests: use GitHub Issues
- Questions and open-ended ideas: use GitHub Discussions
