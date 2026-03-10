# Copilot instructions for quantum-error-correction

This repository focuses on quantum error-correction research code and educational examples built around **Stim**.

## Repository goals
- Provide clear, correct implementations of repetition and surface-code style circuits.
- Keep simulation helpers easy to use from scripts and notebooks.
- Preserve backwards-compatible imports when refactoring public modules.

## Tech stack
- Primary language: Python
- Core dependency: `stim`
- Test framework: `pytest`

## Required checks before completing a change
Run these commands from repository root:

```bash
python -m pip install --upgrade pip
pip install pytest numpy stim
pytest
```

If a change only touches docs/instructions, tests may be skipped and the PR should explain why.

## Project map
- `surface_code_in_stem/`: Main package for static/dynamic code circuits, noise models, decoders, and RL comparison helpers.
- `tests/`: Unit tests for decoders, noise models, and RL nested-learning workflows.
- `introduction_to_stim/`: notebooks and tutorial materials.

## Coding guidelines
1. Keep functions small and composable; prefer pure helpers where practical.
2. Add type hints to new/modified Python APIs.
3. Add docstrings for public functions/classes and for non-obvious simulation logic.
4. Keep optional dependencies optional (see lazy `stim` import behavior in RL helpers).
5. When changing circuit-generation behavior, update/add tests that validate detector/logical-error behavior.

## PR expectations
- Summarize affected modules and expected behavior changes.
- Include test evidence (or explicit reason for not running tests).
- Note any API-compatibility concerns if function signatures or exports change.
