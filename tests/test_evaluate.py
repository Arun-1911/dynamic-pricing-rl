import numpy as np
import pytest

import evaluate
from environment import PricingEnv


def test_bootstrap_interval_brackets_a_real_gain():
    rng = np.random.default_rng(0)
    base = rng.uniform(80, 120, 300)
    ppo = base * 1.05 + rng.normal(0, 1, 300)
    lo, hi = evaluate.bootstrap_uplift_ci(ppo, base, np.random.default_rng(1))
    point = (ppo.sum() / base.sum() - 1) * 100
    assert lo < point < hi
    assert 4.0 < lo and hi < 6.0


def test_bootstrap_interval_includes_zero_when_there_is_no_effect():
    rng = np.random.default_rng(2)
    base = rng.uniform(80, 120, 300)
    ppo = base + rng.normal(0, 10, 300)
    lo, hi = evaluate.bootstrap_uplift_ci(ppo, base, np.random.default_rng(3))
    assert lo < 0 < hi


def test_run_policy_returns_consistent_arrays(products):
    env = PricingEnv(products, fixed_stock_code="85123A")
    rev, prof, capped = evaluate.run_policy(env, lambda obs: np.array([1.0], dtype=np.float32), 3)
    assert rev.shape == prof.shape == capped.shape == (3,)
    assert ((capped >= 0) & (capped <= 1)).all()
    assert (rev >= prof).all() and (prof > 0).all()


def test_run_policy_accepts_the_trained_model(products, model):
    env = PricingEnv(products, fixed_stock_code="85123A")
    rev, _, _ = evaluate.run_policy(env, model, 2)
    assert rev.shape == (2,) and np.isfinite(rev).all()


def test_tuned_oracle_picks_a_grid_value_and_beats_a_bad_price(products):
    env = PricingEnv(products, fixed_stock_code="85123A")
    ratio = evaluate.tune_constant_ratio(env, seed_offset=0)
    assert ratio in [float(x) for x in evaluate.RATIO_GRID]

    def mean_profit(r):
        _, prof, _ = evaluate.run_policy(env, lambda o: np.array([r], dtype=np.float32), 5,
                                         base_seed=evaluate.TUNING_SEED)
        return prof.mean()

    assert mean_profit(ratio) >= mean_profit(0.7) - 1e-9


def test_evaluation_seeds_are_disjoint_from_tuning_seeds():
    eval_seeds = {evaluate.SEED + i * 100 + e for i in range(727) for e in range(evaluate.EPISODES_PER_PRODUCT)}
    tune_seeds = {evaluate.TUNING_SEED + i * 100 + e for i in range(727) for e in range(evaluate.TUNING_EPISODES)}
    assert eval_seeds.isdisjoint(tune_seeds)
