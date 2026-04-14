# Repository Merge and Deployment Playbook

This repository often evolves with two code paths:

- `syndrome-net` (application, RL, benchmarking, Streamlit)
- `quantumforge` (optional Rust/PyO3 acceleration subtree)

The playbook below keeps both sides stable and gives deterministic merge + validation options.

## 1) What to merge and when

Use one of these two modes:

1. Snapshot mode (current structure)

- Keep `quantumforge/` as a checked-in subtree in this repository.
- Update it intentionally in dedicated commits with:
  - `quantumforge` source changes
  - lockfile alignment (`requirements.lock`, `app/requirements.lock`, `quantumforge/python/requirements.lock`)
  - docs/tests when behavior changes.

2. Split-repo mode

- Keep `quantumforge` maintained separately.
- In Syndrome-Net, consume only released artifacts (prebuilt wheels) or a synced commit range.
- Avoid ad-hoc edits inside `quantumforge/` in `syndrome-net` except by controlled sync.

## 2) Safe merge recipes

### 2.1 Sync local snapshot from a sibling repo

```bash
git checkout -b chore/merge-quantumforge-sync

# from a sibling clone path:
git remote add quantumforge-origin /path/to/quantumforge
git fetch quantumforge-origin

# choose strategy:
# A) full subtree sync
git subtree add --prefix=quantumforge quantumforge-origin main --squash

# B) targeted path import
git checkout quantumforge-origin/main -- quantumforge
```

### 2.2 Re-merge updates already in working tree

```bash
git add quantumforge requirements.lock app/requirements.lock
git commit -m "chore: sync quantumforge snapshot and lockfiles"
```

### 2.3 If the repository has no separate git history for quantumforge

When `quantumforge` is not its own repository:

- treat it as a vendored implementation module.
- keep version markers in commit messages and docs.
- drive changes through this parent repo’s review and CI gates.

## 3) Post-merge validation sequence

Run the contract-focused checks first, then representative runtime checks.

```bash
python3 -m pytest \
  tests/test_sampling_backend_contracts.py \
  tests/test_benchmark_decoder_contracts.py \
  tests/test_cuda_q_decoder.py \
  tests/test_runtime_contracts.py \
  tests/test_architectural_registries.py

python3 scripts/ci_contract_verification.sh
python3 scripts/benchmark_decoders.py --quick --sampling-backends sweep --suite circuit --output-dir artifacts/benchmarks
```

If benchmark sweep is too heavy for a container, use:

```bash
python3 scripts/benchmark_decoders.py --quick --sampling-backends stim --suite circuit --output-dir artifacts/benchmarks
```

## 4) CI expectations to preserve

- backend candidate metadata and chain tokens are still emitted.
- `backend_chain_tokens` is list-shaped in memory and JSON-serialized strings in CSV/JSON files.
- `contract_flags` and `profiler_flags` are populated consistently.
- fallback behavior remains stable for missing optional backends.

## 5) Push path

For docs and code updates after checks:

```bash
git status --short
git add .github docs app quantumforge requirements*.lock
git commit -m "docs: improve repo merge and backend contract documentation"
git push origin HEAD
```

If you need PR-style merge:

1. Open PR with `todo`/TODOs explicitly marked.
2. Include a short "contract validation" block in the PR description with test output snippet.
3. Request at least one review from both runtime and backend integrator paths.
