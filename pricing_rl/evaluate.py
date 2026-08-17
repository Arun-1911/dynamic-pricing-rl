"""
Evaluate the trained PPO policy against simple baseline pricing strategies
on the held-out product set (products.csv rows never used in training).

Reports revenue and profit uplift, with means/stds over multiple episodes
per product so the numbers aren't a single noisy rollout.
"""

import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from environment import PricingEnv
from baselines import BASELINES

DATA_DIR = r"C:\Projects\DynamicPriceRL\DynamicPriceRL\data"
MODEL_DIR = r"C:\Projects\DynamicPriceRL\DynamicPriceRL\models"
REPORTS_DIR = r"C:\Projects\DynamicPriceRL\DynamicPriceRL\reports"

EPISODES_PER_PRODUCT = 10
SEED = 123


def run_policy(env, policy_fn, n_episodes, seed_offset=0):
    """policy_fn(obs) -> action, or a PPO model (uses .predict)."""
    revenues, profits = [], []
    is_model = hasattr(policy_fn, "predict")
    for ep in range(n_episodes):
        obs, info = env.reset(seed=SEED + seed_offset + ep)
        ep_revenue, ep_profit = 0.0, 0.0
        done = False
        while not done:
            if is_model:
                action, _ = policy_fn.predict(obs, deterministic=True)
            else:
                action = policy_fn(obs)
            obs, reward, term, trunc, info = env.step(action)
            ep_revenue += info["revenue"]
            ep_profit += info["profit"]
            done = term or trunc
        revenues.append(ep_revenue)
        profits.append(ep_profit)
    return np.array(revenues), np.array(profits)


def main():
    import os
    os.makedirs(REPORTS_DIR, exist_ok=True)

    test_products = pd.read_csv(f"{DATA_DIR}/test_products.csv")
    print(f"Evaluating on {len(test_products)} held-out products, "
          f"{EPISODES_PER_PRODUCT} episodes each")

    model = PPO.load(f"{MODEL_DIR}/ppo_pricing")

    strategies = {"ppo": model, **BASELINES}
    results = {name: {"revenue": [], "profit": []} for name in strategies}
    per_product_rows = []

    for i, row in test_products.iterrows():
        env = PricingEnv(test_products, fixed_stock_code=row["stock_code"])
        product_row = {"stock_code": row["stock_code"], "category": row["category"]}
        for name, policy in strategies.items():
            rev, prof = run_policy(env, policy, EPISODES_PER_PRODUCT, seed_offset=i * 100)
            results[name]["revenue"].append(rev.mean())
            results[name]["profit"].append(prof.mean())
            product_row[f"{name}_revenue"] = rev.mean()
            product_row[f"{name}_profit"] = prof.mean()
        per_product_rows.append(product_row)
        if (i + 1) % 100 == 0:
            print(f"  evaluated {i + 1}/{len(test_products)} products...")

    per_product_df = pd.DataFrame(per_product_rows)
    per_product_df.to_csv(f"{REPORTS_DIR}/per_product_eval.csv", index=False)

    summary = {}
    baseline_name = "static_base_price"
    ppo_total_rev = sum(results["ppo"]["revenue"])
    ppo_total_profit = sum(results["ppo"]["profit"])

    for name in strategies:
        total_rev = sum(results[name]["revenue"])
        total_profit = sum(results[name]["profit"])
        summary[name] = {
            "mean_revenue_per_product": float(np.mean(results[name]["revenue"])),
            "std_revenue_per_product": float(np.std(results[name]["revenue"])),
            "total_revenue": float(total_rev),
            "mean_profit_per_product": float(np.mean(results[name]["profit"])),
            "std_profit_per_product": float(np.std(results[name]["profit"])),
            "total_profit": float(total_profit),
        }

    for name in strategies:
        if name == "ppo":
            continue
        base_rev = summary[name]["total_revenue"]
        base_profit = summary[name]["total_profit"]
        summary[name]["revenue_uplift_vs_ppo_pct"] = float(
            (ppo_total_rev - base_rev) / abs(base_rev) * 100
        )
        summary[name]["profit_uplift_vs_ppo_pct"] = float(
            (ppo_total_profit - base_profit) / abs(base_profit) * 100
        )

    summary["n_held_out_products"] = len(test_products)
    summary["episodes_per_product"] = EPISODES_PER_PRODUCT

    with open(f"{REPORTS_DIR}/uplift_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESULTS ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
