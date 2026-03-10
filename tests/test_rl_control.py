import pytest

np = pytest.importorskip("numpy")

from surface_code_in_stem.rl_control.masking import (
    apply_masked_detector_weights,
    build_detector_parameter_mask,
    mask_population_perturbations,
)
from surface_code_in_stem.rl_control.optimizer import PEPGOptimizer


def test_pepg_optimizer_updates_towards_higher_reward_direction():
    opt = PEPGOptimizer(parameter_dim=2, seed=11, init_sigma=0.2, learning_rate=0.2)
    candidates, perturb = opt.ask(10)

    # Reward aligned with +x direction should push mean[0] upward.
    rewards = candidates[:, 0]
    initial = opt.mean.copy()
    opt.tell(perturb, rewards)

    assert opt.mean[0] > initial[0]
    assert np.isfinite(opt.mean).all()
    assert np.all(opt.sigma > 0)


def test_mask_application_on_toy_factor_graph():
    # detector 0 -> param 0, detector 1 -> params 0 and 1
    mask = build_detector_parameter_mask(3, 2, edges=[(0, 0), (1, 0), (1, 1)])
    detector_signal = np.array([1.0, 0.5, 0.0])

    weights = apply_masked_detector_weights(detector_signal, mask, normalize=True)
    np.testing.assert_allclose(weights, np.array([0.75, 0.5]))

    perturb = np.array([[1.0, 2.0], [-1.0, 1.0]])
    masked = mask_population_perturbations(perturb, detector_signal, mask)
    np.testing.assert_allclose(masked, np.array([[0.75, 1.0], [-0.75, 0.5]]))
