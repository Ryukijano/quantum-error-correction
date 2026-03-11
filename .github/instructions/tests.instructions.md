---
applyTo: "tests/**/*.py"
---

## Test-writing requirements

- Use `pytest` style tests and readable test names describing behavior.
- Cover both nominal behavior and invalid-input edge cases for new logic.
- Prefer deterministic tests (fixed seeds / explicit inputs).
- Keep tests lightweight so they run in CI quickly.
- If adding dynamic-circuit features, include at least one regression test that would fail without the fix.
