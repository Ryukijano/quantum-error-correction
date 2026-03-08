# Reinforcement Learning Experiments

## Reproducibility

The `compare_nested_policies` helper now derives seeds from stable md5 hashes of
builder and policy names instead of Python's salted `hash` output. This keeps the
seed assigned to each `(builder, policy)` pair identical across interpreter
restarts and operating systems. Providing a `base_seed` still allows grouping
related runs while maintaining deterministic offsets for every individual
policy.
