# Security Policy

## Supported scope

This repository is maintained on a rolling basis.

| Ref | Support status |
| --- | --- |
| `main` | Supported |
| Open pull-request branches | Best effort |
| Old feature branches, generated reports, and local-only artifacts | Not supported |

## Reporting a vulnerability

Please use GitHub Private Vulnerability Reporting for suspected security issues:

<https://github.com/Wiseung/cellular-uav-reuse-simulation/security/advisories/new>

Do not open a public issue for an unpatched security problem.

When you report a vulnerability, include:

- affected file, module, or workflow
- impact and attack preconditions
- reproduction steps or a proof of concept
- whether secrets, local files, or external datasets are required
- any suggested mitigation or fix direction

## Response expectations

- Initial acknowledgement target: within 5 business days
- Triage depends on whether the issue can be reproduced
- Fixes are expected to land on `main`
- Backports are not guaranteed because this project does not maintain release branches

## Out of scope

The following are usually not treated as security vulnerabilities in this project:

- model or propagation disagreements with no security impact
- numerical instability that does not expose data or privileges
- local crashes caused only by malformed user-owned inputs
- public-data quality issues that do not lead to code execution or secret exposure
