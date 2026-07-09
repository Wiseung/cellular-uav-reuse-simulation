# Development and Testing

## Repository layout

- `cellular_uav_sir/`: simulation source
- `cellular_uav_sir/data/`: input assets and source notes
- `cellular_uav_sir/results/`: generated outputs
- `tests/`: automated tests
- `tools/`: preparation, calibration, and reporting utilities
- `deliverables/`: report-ready artifacts

## Local test entry point

Run the Python test suite from the repository root:

```powershell
pytest
```

## CI

The repository workflow in `.github/workflows/run-simulation.yml` is the main GitHub validation path. It runs the simulation command on GitHub Actions and is the first check to watch after opening a pull request.

## Contribution expectations

- prefer the smallest defensible change
- keep data-source and calibration assumptions explicit
- avoid mixing generated artifacts with source logic changes unless the artifact update is the point of the change
- use Issues for bounded work and Discussions for questions or open-ended ideas

## Documentation surfaces

- repository README: concise operator overview
- wiki pages: task-oriented navigation
- `SECURITY.md`: vulnerability handling
- issue and discussion templates: intake discipline
