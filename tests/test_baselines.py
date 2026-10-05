import numpy as np
import pytest

from baselines import (BASELINES, elasticity_optimal_static_policy,
                       static_price_policy, undercut_competitor_policy)


def obs_with(competitor_ratio=1.0, elasticity=-2.0):
    return np.array([1.0, competitor_ratio, 1.0, 1.0, elasticity, 0.0, 0.0], dtype=np.float32)


def test_every_baseline_returns_one_in_bounds_action():
    for name, policy in BASELINES.items():
        for competitor in (0.5, 1.0, 1.5):
            for elasticity in (-9.0, -2.0, -0.5):
                action = policy(obs_with(competitor, elasticity))
                assert action.shape == (1,), name
                assert 0.7 - 1e-6 <= action[0] <= 1.4 + 1e-6, name


def test_static_policy_never_moves_the_price():
    assert static_price_policy(obs_with(0.6))[0] == 1.0


def test_undercut_policy_prices_five_percent_below_the_competitor():
    assert undercut_competitor_policy(obs_with(competitor_ratio=1.0))[0] == pytest.approx(0.95)
    assert undercut_competitor_policy(obs_with(competitor_ratio=0.6))[0] == pytest.approx(0.7)   # floor


def test_monopoly_formula_policy():
    # price*/cost = e/(e+1); with a 50% margin and e=-2 that is exactly the base price
    assert elasticity_optimal_static_policy(obs_with(elasticity=-2.0))[0] == pytest.approx(1.0)
    assert elasticity_optimal_static_policy(obs_with(elasticity=-3.0))[0] == pytest.approx(0.75)
    assert elasticity_optimal_static_policy(obs_with(elasticity=-0.5))[0] == pytest.approx(1.4)  # inelastic: capped
