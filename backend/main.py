"""
FastAPI backend serving the trained PPO pricing policy, SHAP explanations,
and catalog/training data to the React frontend.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pricing_rl"))

from environment import PricingEnv, STORAGE_WEEKS  # noqa: E402
from explain import PricingExplainer, build_background, FEATURE_LABELS  # noqa: E402

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

app = FastAPI(title="Dynamic Pricing RL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"]
    + [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

products_df = pd.read_csv(DATA_DIR / "products.csv")
test_codes = set(pd.read_csv(DATA_DIR / "test_products.csv")["stock_code"])
products_df["held_out_from_training"] = products_df["stock_code"].isin(test_codes)

model = PPO.load(str(MODEL_DIR / "ppo_pricing"))
_env_for_bg = PricingEnv(products_df)
_background = build_background(_env_for_bg, n=50, seed=0)
explainer = PricingExplainer(model, _background)

with open(REPORTS_DIR / "uplift_report.json") as f:
    UPLIFT_REPORT = json.load(f)


class StateInput(BaseModel):
    stock_code: str
    competitor_ratio: float = Field(1.0, ge=0.5, le=1.5)
    inventory: float = Field(1.0, ge=0.0, le=1.0)
    demand_ratio: float = Field(1.0, ge=0.0, le=10.0)
    time_norm: float = Field(0.0, ge=0.0, le=1.0)


def _build_obs(payload: StateInput):
    row = products_df[products_df["stock_code"] == payload.stock_code]
    if row.empty:
        raise HTTPException(status_code=404, detail="Unknown stock_code")
    row = row.iloc[0]

    env = PricingEnv(products_df, fixed_stock_code=payload.stock_code)
    env.reset(seed=0)
    env.competitor_ratio = payload.competitor_ratio
    env.inventory = payload.inventory
    env.demand_ratio = payload.demand_ratio
    env.current_step = int(payload.time_norm * env.max_steps)
    obs = env._obs()
    return row, obs


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/summary")
def summary():
    cat_counts = products_df["category"].value_counts().to_dict()
    source_counts = products_df["elasticity_source"].value_counts().to_dict()
    elasticity = products_df["elasticity"]
    return {
        "n_products": len(products_df),
        "n_train": int((~products_df["held_out_from_training"]).sum()),
        "n_test": int(products_df["held_out_from_training"].sum()),
        "category_counts": cat_counts,
        "elasticity_source_counts": source_counts,
        "elasticity_mean": float(elasticity.mean()),
        "elasticity_median": float(elasticity.median()),
        "elasticity_std": float(elasticity.std()),
        "elasticity_histogram": _histogram(elasticity.values, bins=20),
        "uplift_report": UPLIFT_REPORT,
    }


def _histogram(values, bins=20):
    counts, edges = np.histogram(values, bins=bins)
    return [
        {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]


@app.get("/api/categories")
def categories():
    counts = products_df["category"].value_counts()
    return [{"category": c, "count": int(n)} for c, n in counts.items()]


@app.get("/api/products")
def list_products(
    category: str | None = None,
    q: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    df = products_df
    if category and category != "All":
        df = df[df["category"] == category]
    if q:
        df = df[df["description"].str.contains(q, case=False, na=False)]
    df = df.sort_values("n_transactions", ascending=False)
    total = len(df)
    page = df.iloc[offset: offset + limit]
    return {
        "total": total,
        "items": page[[
            "stock_code", "description", "category", "base_price", "elasticity",
            "elasticity_source", "n_transactions", "held_out_from_training",
        ]].to_dict(orient="records"),
    }


@app.get("/api/products/{stock_code}")
def product_detail(stock_code: str):
    row = products_df[products_df["stock_code"] == stock_code]
    if row.empty:
        raise HTTPException(status_code=404, detail="Unknown stock_code")
    record = row.iloc[0].to_dict()
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in record.items()}


@app.post("/api/recommend")
def recommend(payload: StateInput):
    row, obs = _build_obs(payload)
    action, _ = model.predict(obs, deterministic=True)
    ratio = float(action[0])
    price = ratio * float(row["base_price"])
    return {
        "stock_code": payload.stock_code,
        "recommended_price": price,
        "price_ratio": ratio,
        "base_price": float(row["base_price"]),
        "pct_vs_base": (ratio - 1.0) * 100,
    }


@app.post("/api/explain")
def explain(payload: StateInput):
    row, obs = _build_obs(payload)
    result = explainer.explain(obs)
    return {
        "stock_code": payload.stock_code,
        "features": [
            {
                "name": name,
                "label": FEATURE_LABELS[name],
                "value": val,
                "shap_value": sv,
            }
            for name, val, sv in zip(result["feature_names"], result["feature_values"], result["shap_values"])
        ],
        "base_value": result["base_value"],
        "prediction": result["prediction"],
    }


@app.get("/api/products/{stock_code}/curve")
def price_curve(
    stock_code: str,
    competitor_ratio: float = Query(1.0, ge=0.5, le=1.5),
    inventory: float = Query(1.0, ge=0.0, le=1.0),
    cost_margin: float = Query(0.5, ge=0.0, lt=1.0),
):
    row = products_df[products_df["stock_code"] == stock_code]
    if row.empty:
        raise HTTPException(status_code=404, detail="Unknown stock_code")
    row = row.iloc[0]

    base_price = float(row["base_price"])
    elasticity = float(row["elasticity"])
    base_demand = float(np.clip(row["base_demand"], 1.0, 5000.0))
    cost = base_price * cost_margin

    ratios = np.linspace(0.7, 1.4, 60)
    prices = ratios * base_price
    demand = base_demand * np.power(ratios, elasticity)

    competitor_price = competitor_ratio * base_price
    factor = np.where(
        prices > competitor_price,
        np.maximum(0.5, 1.0 - (prices - competitor_price) / max(competitor_price, 1e-6) * 0.5),
        np.minimum(1.5, 1.0 + (competitor_price - prices) / max(competitor_price, 1e-6) * 0.3),
    )
    demand = demand * factor
    capacity = inventory * STORAGE_WEEKS * base_demand
    sales = np.minimum(demand, capacity)
    profit = (prices - cost) * sales
    revenue = prices * sales

    return [
        {"price": float(p), "ratio": float(r), "profit": float(pr), "revenue": float(rv)}
        for p, r, pr, rv in zip(prices, ratios, profit, revenue)
    ]


@app.get("/api/training/learning_curve")
def learning_curve(max_points: int = Query(200, ge=2, le=5000)):
    df = pd.read_csv(LOGS_DIR / "learning_curve.csv")
    if len(df) > max_points:
        idx = np.linspace(0, len(df) - 1, max_points).astype(int)
        df = df.iloc[idx]
    return df.to_dict(orient="records")


@app.get("/api/training/uplift")
def training_uplift():
    return UPLIFT_REPORT
