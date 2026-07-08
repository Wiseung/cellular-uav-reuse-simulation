# Reference Sector Panel MSI Source

This default pattern file is stored in MSI Planet text format so that the simulator can ingest a common vendor/planning-tool exchange format instead of only a project-specific CSV.

- Local output file: `reference_sector_panel_pattern.msi`
- Format family: MSI Planet style horizontal/vertical 2D cut file

Processing used in this branch:

1. Start from the existing reference horizontal/vertical attenuation cuts in `custom_panel_pattern.csv`.
2. Interpolate each cut onto a dense `0..359°` angular grid.
3. Emit the result as an MSI-style file with basic metadata fields (`NAME`, `MAKE`, `FREQUENCY`, `H_WIDTH`, `V_WIDTH`, `GAIN`, `TILT`).

This improves default interoperability and makes the bundled pattern look closer to a real planning-file artifact, but it is still not a manufacturer-measured antenna dump. If a real vendor `.msi` or `.pln` file is available, it can now be loaded directly.
