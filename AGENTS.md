# Agent rules

## Simulations

- **Cap any sim run at 5 seconds.** If a benchmark or smoke test would take longer, lower `--steps` until it fits. Don't run `--steps 5000` on a slow program. Don't run multi-program sweeps that compound past 5s. Pick step counts that produce useful signal in under 5 seconds.
