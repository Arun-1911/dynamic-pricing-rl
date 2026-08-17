"""
Streamlit dashboard: pick a real product from the catalog, see the PPO
policy's recommended price and a SHAP explanation of that recommendation.

Run with: streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from stable_baselines3 import PPO

from environment import PricingEnv
from explain import PricingExplainer, build_background, FEATURE_LABELS

DATA_DIR = r"C:\Projects\DynamicPriceRL\DynamicPriceRL\data"
MODEL_DIR = r"C:\Projects\DynamicPriceRL\DynamicPriceRL\models"

st.set_page_config(page_title="Dynamic Pricing RL", layout="wide")


@st.cache_resource
def load_model():
    return PPO.load(f"{MODEL_DIR}/ppo_pricing")


@st.cache_data
def load_products():
    products = pd.read_csv(f"{DATA_DIR}/products.csv")
    test_codes = set(pd.read_csv(f"{DATA_DIR}/test_products.csv")["stock_code"])
    products["held_out_from_training"] = products["stock_code"].isin(test_codes)
    return products


@st.cache_resource
def load_explainer(_model, _products):
    env = PricingEnv(_products)
    background = build_background(env, n=50, seed=0)
    return PricingExplainer(_model, background)


def simulate_curve(row, competitor_ratio, inventory, cost_margin=0.5, price_low=0.7, price_high=1.4):
    """Noise-free price -> demand -> profit curve for the visualization,
    using the same demand formula as the environment (elasticity term x
    competitor factor), holding inventory/competition fixed."""
    base_price = row["base_price"]
    elasticity = row["elasticity"]
    base_demand = float(np.clip(row["base_demand"], 1.0, 5000.0))
    cost = base_price * cost_margin

    ratios = np.linspace(price_low, price_high, 60)
    prices = ratios * base_price
    demand = base_demand * np.power(ratios, elasticity)

    competitor_price = competitor_ratio * base_price
    factor = np.where(
        prices > competitor_price,
        np.maximum(0.5, 1.0 - (prices - competitor_price) / np.maximum(competitor_price, 1e-6) * 0.5),
        np.minimum(1.5, 1.0 + (competitor_price - prices) / np.maximum(competitor_price, 1e-6) * 0.3),
    )
    demand = demand * factor
    capacity = inventory * base_demand * 2.0
    sales = np.minimum(demand, capacity)
    profit = (prices - cost) * sales
    revenue = prices * sales
    return ratios, prices, profit, revenue


def main():
    st.title("Dynamic Pricing RL — Product Explorer")
    st.caption(
        "PPO policy trained across 4,124 real products from the UCI Online Retail II "
        "dataset, with per-decision SHAP explanations. Prices are simulated, not live."
    )

    model = load_model()
    products = load_products()

    with st.sidebar:
        st.header("Select a product")
        category = st.selectbox("Category", ["All"] + sorted(products["category"].unique()))
        filtered = products if category == "All" else products[products["category"] == category]
        filtered = filtered.sort_values("n_transactions", ascending=False)

        options = filtered.apply(
            lambda r: f"{r['description'][:45]}  ({r['stock_code']})", axis=1
        ).tolist()
        idx_map = dict(zip(options, filtered.index))
        choice = st.selectbox("Product", options)
        row = products.loc[idx_map[choice]]

        st.markdown("---")
        st.subheader("Simulated market state")
        competitor_ratio = st.slider("Competitor price (x base price)", 0.5, 1.5, 1.0, 0.01)
        inventory = st.slider("Inventory level", 0.0, 1.0, 1.0, 0.05)
        demand_ratio = st.slider("Recent demand (x typical)", 0.0, 3.0, 1.0, 0.1)
        time_norm = st.slider("Point in pricing cycle", 0.0, 1.0, 0.0, 0.05)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(row["description"])
        badge = "🔒 held out from training" if row["held_out_from_training"] else "trained on"
        st.caption(f"Stock code {row['stock_code']} · {row['category']} · {badge}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Base price", f"£{row['base_price']:.2f}")
        m2.metric("Fitted elasticity", f"{row['elasticity']:.2f}")
        m3.metric("Real transactions", f"{int(row['n_transactions']):,}")
        st.caption(f"Elasticity source: {row['elasticity_source']}")

        env = PricingEnv(products, fixed_stock_code=row["stock_code"])
        env.reset(seed=0)
        env.competitor_ratio = competitor_ratio
        env.inventory = inventory
        env.demand_ratio = demand_ratio
        env.current_step = int(time_norm * env.max_steps)
        obs = env._obs()

        action, _ = model.predict(obs, deterministic=True)
        rec_ratio = float(action[0])
        rec_price = rec_ratio * row["base_price"]

        st.markdown("### PPO recommendation")
        st.metric("Recommended price", f"£{rec_price:.2f}", f"{(rec_ratio - 1) * 100:+.1f}% vs base price")

        ratios, prices, profit, revenue = simulate_curve(row, competitor_ratio, inventory)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(prices, profit, label="Expected profit")
        ax.axvline(rec_price, color="tab:orange", linestyle="--", label="PPO recommendation")
        ax.axvline(row["base_price"], color="gray", linestyle=":", label="Base price")
        ax.set_xlabel("Price (£)")
        ax.set_ylabel("Expected profit per week")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    with col2:
        st.markdown("### Why this price? (SHAP)")
        explainer = load_explainer(model, products)
        result = explainer.explain(obs, nsamples=100)

        contrib = pd.DataFrame({
            "feature": [FEATURE_LABELS[n] for n in result["feature_names"]],
            "shap_value": result["shap_values"],
        }).sort_values("shap_value")

        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        colors = ["tab:red" if v < 0 else "tab:blue" for v in contrib["shap_value"]]
        ax2.barh(contrib["feature"], contrib["shap_value"], color=colors)
        ax2.set_xlabel("Contribution to price multiplier")
        ax2.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig2)

        st.caption(
            f"Base value (average recommendation across background states): {result['base_value']:.3f} "
            f"→ this decision: {result['prediction']:.3f}"
        )

        st.markdown("#### Feature values for this decision")
        detail = pd.DataFrame({
            "Feature": [FEATURE_LABELS[n] for n in result["feature_names"]],
            "Value": [f"{v:.3f}" for v in result["feature_values"]],
            "SHAP contribution": [f"{v:+.4f}" for v in result["shap_values"]],
        })
        st.dataframe(detail, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
