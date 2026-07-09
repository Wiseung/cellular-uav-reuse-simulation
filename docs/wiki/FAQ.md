# FAQ

## Where should I start if I only want to run the default example?

Use the command in [Getting Started](Getting-Started.md) and inspect `cellular_uav_sir/results/`.

## Can I replace the default site layout with my own?

Yes. Provide your own site-layout CSV, and optionally a matching building GeoJSON and profile JSON if the default Knoxville assumptions no longer fit your scenario.

## Do I need a custom parameter profile for every run?

No. The default profile is a valid starting point. Use a custom profile when beam, load, handover, or obstruction assumptions need to match a different environment.

## Where should I ask interpretation questions about a result figure or table?

Use GitHub Discussions in the `Q&A` category.

## When should I open an issue instead of a discussion?

Open an issue when there is a bounded bug, documentation gap, or clearly scoped implementation request. Use a discussion when you need help, want feedback, or are still shaping an idea.

## Why are there both source files and deliverable artifacts in the repository?

The repository is used both for simulation logic and course-deliverable production. Source logic lives under `cellular_uav_sir/` and `tools/`; publication-ready outputs live under `deliverables/`.
