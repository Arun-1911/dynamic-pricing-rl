import threading

import numpy as np
import pytest

from environment import PricingEnv
from explain import FEATURE_LABELS, FEATURE_NAMES, PricingExplainer, build_background


@pytest.fixture(scope="module")
def explainer(products, model):
    return PricingExplainer(model, build_background(PricingEnv(products), n=50, seed=0))


def observation(products, code="85123A", seed=1, inventory=None):
    env = PricingEnv(products, fixed_stock_code=code)
    obs, _ = env.reset(seed=seed)
    if inventory is not None:
        env.inventory = inventory
        obs = env._obs()
    return obs


def test_every_feature_has_a_label():
    assert set(FEATURE_NAMES) == set(FEATURE_LABELS)
    assert len(FEATURE_NAMES) == 7


def test_background_is_reproducible(products):
    a = build_background(PricingEnv(products), n=50, seed=0)
    b = build_background(PricingEnv(products), n=50, seed=0)
    c = build_background(PricingEnv(products), n=50, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_shap_values_add_up_to_the_model_output(products, explainer):
    for code in products.stock_code.sample(8, random_state=4):
        result = explainer.explain(observation(products, code))
        assert result["base_value"] + sum(result["shap_values"]) == pytest.approx(result["prediction"], abs=1e-5)


def test_explanations_are_deterministic(products, model, explainer):
    obs = observation(products)
    first = explainer.explain(obs)
    again = explainer.explain(obs)
    rebuilt = PricingExplainer(model, build_background(PricingEnv(products), n=50, seed=0)).explain(obs)
    assert first["shap_values"] == again["shap_values"]
    assert first["shap_values"] == rebuilt["shap_values"]
    assert first["base_value"] == rebuilt["base_value"]


def test_low_stock_is_what_drives_the_price_up(products, explainer):
    full = explainer.explain(observation(products, inventory=1.0))
    low = explainer.explain(observation(products, inventory=0.1))
    idx = FEATURE_NAMES.index("inventory")
    assert low["shap_values"][idx] > full["shap_values"][idx]
    assert low["prediction"] > full["prediction"]


def test_concurrent_requests_do_not_corrupt_each_other(products, explainer):
    observations = [observation(products, inventory=0.1 + 0.07 * i, seed=i) for i in range(10)]
    expected = [explainer.explain(o)["shap_values"] for o in observations]
    results, errors = {}, []

    def work(i):
        try:
            results[i] = explainer.explain(observations[i])["shap_values"]
        except Exception as exc:   # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(observations))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    for i, want in enumerate(expected):
        assert results[i] == want
