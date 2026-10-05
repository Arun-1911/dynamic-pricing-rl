from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from environment import PricingEnv

STATE = {"stock_code": "85123A", "competitor_ratio": 1.0, "inventory": 1.0,
         "demand_ratio": 1.0, "time_norm": 0.0}


@pytest.fixture(scope="module")
def app():
    from backend.main import app
    return app


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_summary_matches_the_data(client, products, test_df):
    d = client.get("/api/summary").json()
    assert d["n_products"] == len(products)
    assert d["n_test"] == len(test_df) and d["n_train"] + d["n_test"] == d["n_products"]
    assert sum(b["count"] for b in d["elasticity_histogram"]) == len(products)
    assert sum(d["category_counts"].values()) == len(products)
    assert {"ppo", "static_base_price", "tuned_static_oracle"} <= set(d["uplift_report"])


def test_categories(client, products):
    d = client.get("/api/categories").json()
    assert sum(c["count"] for c in d) == len(products)
    assert [c["count"] for c in d] == sorted((c["count"] for c in d), reverse=True)


def test_product_listing_filters_sorting_and_paging(client):
    first = client.get("/api/products", params={"limit": 5}).json()
    assert first["total"] == 4851 and len(first["items"]) == 5
    counts = [p["n_transactions"] for p in first["items"]]
    assert counts == sorted(counts, reverse=True)

    second = client.get("/api/products", params={"limit": 5, "offset": 5}).json()
    assert {p["stock_code"] for p in first["items"]}.isdisjoint(p["stock_code"] for p in second["items"])

    bags = client.get("/api/products", params={"category": "Bags", "limit": 50}).json()
    assert bags["total"] > 0 and all(p["category"] == "Bags" for p in bags["items"])

    hearts = client.get("/api/products", params={"q": "heart", "limit": 50}).json()
    assert hearts["total"] > 0 and all("heart" in p["description"].lower() for p in hearts["items"])


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"offset": -1}])
def test_product_listing_rejects_bad_paging(client, params):
    assert client.get("/api/products", params=params).status_code == 422


def test_product_detail_handles_missing_r2(client, products):
    fallback = products[products.elasticity_source != "fitted"].stock_code.iloc[0]
    d = client.get(f"/api/products/{fallback}").json()
    assert d["r2"] is None and d["elasticity_source"].startswith("category_fallback")
    assert client.get("/api/products/85123A").json()["elasticity_source"] == "fitted"


def test_unknown_product_is_a_404_everywhere(client):
    assert client.get("/api/products/NOPE").status_code == 404
    assert client.get("/api/products/NOPE/curve").status_code == 404
    assert client.post("/api/recommend", json={**STATE, "stock_code": "NOPE"}).status_code == 404
    assert client.post("/api/explain", json={**STATE, "stock_code": "NOPE"}).status_code == 404


def test_recommendation_is_consistent_with_the_model(client, products, model):
    state = {**STATE, "competitor_ratio": 1.1, "inventory": 0.4, "demand_ratio": 1.3, "time_norm": 0.25}
    d = client.post("/api/recommend", json=state).json()
    assert 0.7 <= d["price_ratio"] <= 1.4
    assert d["recommended_price"] == pytest.approx(d["price_ratio"] * d["base_price"])
    assert d["pct_vs_base"] == pytest.approx((d["price_ratio"] - 1) * 100)

    env = PricingEnv(products, fixed_stock_code="85123A")
    env.reset(seed=0)
    env.competitor_ratio, env.inventory, env.demand_ratio = 1.1, 0.4, 1.3
    env.current_step = int(0.25 * env.max_steps)
    direct = float(model.predict(env._obs(), deterministic=True)[0][0])
    assert d["price_ratio"] == pytest.approx(direct, abs=1e-6)


@pytest.mark.parametrize("bad", [{"competitor_ratio": 9}, {"competitor_ratio": 0.1}, {"inventory": -1},
                                 {"inventory": 2}, {"demand_ratio": -0.5}, {"time_norm": 2}])
def test_state_inputs_are_validated(client, bad):
    assert client.post("/api/recommend", json={**STATE, **bad}).status_code == 422
    assert client.post("/api/explain", json={**STATE, **bad}).status_code == 422


def test_explanation_adds_up_and_is_stable(client):
    a = client.post("/api/explain", json=STATE).json()
    b = client.post("/api/explain", json=STATE).json()
    assert len(a["features"]) == 7
    assert a["base_value"] + sum(f["shap_value"] for f in a["features"]) == pytest.approx(a["prediction"], abs=1e-5)
    assert a == b


def test_overlapping_explain_requests_all_succeed(app, client):
    states = [{**STATE, "inventory": 0.1 + 0.08 * i} for i in range(10)]
    expected = [client.post("/api/explain", json=s).json() for s in states]

    def call(s):
        return TestClient(app).post("/api/explain", json=s)

    with ThreadPoolExecutor(max_workers=10) as pool:
        responses = list(pool.map(call, states))
    assert [r.status_code for r in responses] == [200] * 10
    assert [r.json() for r in responses] == expected


def test_price_curve(client):
    d = client.get("/api/products/85123A/curve", params={"competitor_ratio": 1.0, "inventory": 1.0}).json()
    assert len(d) == 60
    assert d[0]["ratio"] == pytest.approx(0.7) and d[-1]["ratio"] == pytest.approx(1.4)
    assert all(a["price"] < b["price"] for a, b in zip(d, d[1:]))
    assert all(p["profit"] <= p["revenue"] for p in d)
    for bad in ({"competitor_ratio": 3}, {"inventory": -0.1}, {"cost_margin": 1.0}):
        assert client.get("/api/products/85123A/curve", params=bad).status_code == 422


def test_training_endpoints(client):
    curve = client.get("/api/training/learning_curve", params={"max_points": 10}).json()
    assert 2 <= len(curve) <= 10 and {"timestep", "episode_reward"} <= set(curve[0])
    assert curve == sorted(curve, key=lambda p: p["timestep"])
    assert client.get("/api/training/learning_curve", params={"max_points": 1}).status_code == 422

    uplift = client.get("/api/training/uplift").json()
    for key in ("static_base_price", "undercut_competitor_5pct", "elasticity_optimal_static", "tuned_static_oracle"):
        lo, hi = uplift[key]["profit_uplift_ci95_pct"]
        assert lo <= uplift[key]["profit_uplift_vs_ppo_pct"] <= hi
    assert uplift["n_held_out_products"] == 727
