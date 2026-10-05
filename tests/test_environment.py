import numpy as np
import pytest

from environment import PricingEnv, STORAGE_WEEKS


def make(products, code="85123A", **kwargs):
    return PricingEnv(products, fixed_stock_code=code, **kwargs)


def act(ratio):
    return np.array([ratio], dtype=np.float32)


def test_reset_gives_valid_start_state(products):
    env = make(products)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["stock_code"] == "85123A"
    assert obs[0] == 1.0 and obs[2] == 1.0 and obs[5] == 0.0   # base price, full stock, step 0


def test_random_episodes_stay_in_bounds_and_end_at_52_weeks(products):
    for i, code in enumerate(products.stock_code.sample(20, random_state=0)):
        env = make(products, code)
        env.action_space.seed(i)
        obs, _ = env.reset(seed=i)
        steps, truncated = 0, False
        while not truncated:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward) and not terminated
            assert 0.0 <= env.inventory <= 1.0
        assert steps == 52


def test_same_seed_same_trajectory(products):
    def rollout():
        env = make(products)
        env.reset(seed=7)
        return [env.step(act(0.9 + 0.01 * t))[1] for t in range(20)]
    assert rollout() == rollout()


def test_reward_is_profit_over_base_revenue(products):
    env = make(products)
    env.reset(seed=3)
    _, reward, _, _, info = env.step(act(1.0))
    assert reward == pytest.approx(info["profit"] / (env.base_price * env.base_demand))
    assert info["profit"] == pytest.approx(info["revenue"] - env.cost * info["actual_sales"])
    assert env.cost == pytest.approx(0.5 * env.base_price)


def test_actions_are_clipped_to_the_price_bounds(products):
    env = make(products)
    env.reset(seed=0)
    _, _, _, _, high = env.step(act(5.0))
    env.reset(seed=0)
    _, _, _, _, low = env.step(act(0.0))
    assert high["price"] == pytest.approx(1.4 * env.base_price)
    assert low["price"] == pytest.approx(0.7 * env.base_price)


def test_higher_price_means_lower_demand(products):
    demand = {}
    for ratio in (0.8, 1.2):
        env = make(products)
        env.reset(seed=11)          # identical noise draw for both prices
        demand[ratio] = env.step(act(ratio))[4]["demand"]
    assert demand[0.8] > demand[1.2]


def test_selling_at_the_base_price_is_sustainable(products, test_df):
    capped = total = 0
    for i, code in enumerate(test_df.stock_code.head(20)):
        env = make(products, code)
        env.reset(seed=i)
        for _ in range(52):
            _, _, _, _, info = env.step(act(1.0))
            capped += info["actual_sales"] < info["demand"] - 1e-9
            total += 1
    assert capped / total < 0.25


def test_overselling_runs_the_stock_down(products):
    env = make(products, "85123A")      # very price-sensitive product
    env.reset(seed=0)
    capped = 0
    for _ in range(52):
        _, _, _, _, info = env.step(act(0.7))
        capped += info["actual_sales"] < info["demand"] - 1e-9
    assert capped / 52 > 0.8


def test_storage_constant_is_what_the_capacity_uses(products):
    env = make(products)
    env.reset(seed=0)
    _, _, _, _, info = env.step(act(0.7))
    assert info["actual_sales"] <= STORAGE_WEEKS * env.base_demand + 1e-9


def test_category_feature_is_normalised(products):
    for code in products.groupby("category").stock_code.first():
        obs, _ = make(products, code).reset(seed=0)
        assert 0.0 <= obs[6] <= 1.0
