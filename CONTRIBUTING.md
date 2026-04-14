# Contributing to Syndrome-Net

Thanks for helping improve Syndrome-Net.

## Before you start

- Use an issue for major design changes, dependency changes, and new protocol or API work.
- For small fixes, you can start with a PR directly.
- Include reproducibility context for experiments (seed, environment, flags, commit hash).

## Branch and PR workflow

- Branch from `main` using focused prefixes such as `feat/...`, `fix/...`, or `docs/...`.
- Keep PR scope narrow and reference the motivating issue or task.
- Add or update tests and docs when changing runtime, decoder, RL, benchmark, or contract behavior.
- Include command outputs in the PR description for smoke or contract checks.

## Minimum verification before opening a PR

- Run the relevant test targets for touched modules.
- When contract behavior is touched, run:
  - `python3 scripts/ci_contract_verification.sh`
  - `python3 scripts/benchmark_decoders.py --quick ...`
  - `python3 -m pytest tests`
- If a PR changes docs only, include an updated usage path and any screenshots or expected outputs.

## Repository norms

- Keep generated artifacts, caches, local environments, and secrets out of git.
- Prefer small incremental changes and avoid unrelated refactors.
- For any performance or accuracy claim, include exact command and environment details.

## Requesting review

- Mention intended impact clearly and call out tradeoffs.
- Link any follow-up work needed after merge.
