# Simulation Model

## Core modeling layers

The simulator combines several layers rather than relying on a single pathloss-only abstraction:

- analytic frequency-reuse baseline
- Monte Carlo user placement inside the serving geometry
- explicit downlink `SIR` and `SINR` calculations
- lognormal shadowing and small-scale fading
- antenna-sector modeling with custom pattern support
- dynamic load, handover, and interference coordination

## Ground and UAV treatment

- Ground users are evaluated with multi-tier co-channel interference and random site perturbation.
- UAV users use a hybrid LOS model that transitions from 3GPP-style near-ground behavior toward ITU-style statistical blockage behavior at higher altitudes.
- Dynamic experiments can override statistical LOS with local GIS building data when footprint coverage is available.

## Effective ASE

Effective area spectral efficiency is not reported as a purely geometric upper bound. The current workflow applies:

- traffic activity
- scheduling efficiency
- control overhead
- outage gating through a coverage threshold

## Dynamic experiment scope

The dynamic path adds:

- UAV trajectory handling
- filtered handover events with time-to-trigger
- correlated cell load
- dominant-interferer coordinated scheduling
- per-user power sharing
- optional GIS-driven obstruction loss

## Boundaries to keep in mind

- This is a downlink-focused simulator.
- The repository emphasizes engineering realism over protocol-complete network emulation.
- Real inputs can improve relevance, but output quality still depends on source-data quality and calibration choices.
