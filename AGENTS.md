# Project instructions

Use the ubiquitous language in `CONTEXT.md`.

## `docs/`

`docs/` is the source of truth for the current project.
Start with `docs/README.md`.

Keep `docs/` consistent with the current code, tests, installed model contract,
and reproducible measurements.
Update the applicable document when an API, default value, data path, limit, or
measured result changes.

Use `docs/benchmarks/` for machine-readable benchmark artifacts.
Record the commit, environment, workload, configuration, cache state, and output
token hash for each formal performance result.

Separate current validation, historical measurements, external facts, and
research hypotheses.
Do not write an estimate or an external benchmark as a project result.

## `research/`

`research/` contains active investigation, experiment plans, source audits, and
unconfirmed technical directions.
Research files are not the source of truth for current runtime behavior.

Use primary sources for external technical claims.
State assumptions, test conditions, stop criteria, and evidence limits.
When the project adopts a research conclusion, update the applicable file in
`docs/`.

`research/archive/` contains superseded plans and historical research.
Keep archived files for traceability.
Add a warning when a file no longer describes the current runtime.
Do not use an archived value as a current default or current performance result.
