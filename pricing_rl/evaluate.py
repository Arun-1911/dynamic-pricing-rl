"""
Evaluate the trained PPO policy against baseline pricing strategies on the
held-out product set (products never used in training).

Reported per strategy: revenue and profit (mean over episodes per product),
the share of weeks in which sales were limited by stock rather than demand,
and PPO's uplift over each baseline with a 95% bootstrap confidence interval
(resampling products), so small differences aren't over-interpreted.

Baselines:
    static_base_price          never change the price
    undercut_competitor_5pct   price 5% below the observed competitor price
    elasticity_optimal_static  textbook constant-elasticity monopoly price, set once
    tuned_static_oracle        best constant price per product found by grid
                               search inside the simulator (an upper bound for
                               any strategy that never changes its price)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from environment import PricingEnv
from baselines import BASELINES

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

EPISODES_PER_PRODUCT = 10
SEED = 123
TUNING_SEED = 900_000
TUNING_EPISODES = 5
RATIO_GRID = np.round(np.arange(0.70, 1.401, 0.05), 2)
N_BOOTSTRAP = 2000


def run_policy(env, policy_fn, n_episodes, seed_offset=0, base_seed=SEED):
    """policy_fn(obs) -> action, or a PPO model (uses .predict).
    Returns per-episode revenue, profit and share of weeks that were stock-limited."""
    revenues, profits, capped = [], [], []
    is_model = hasattr(policy_fn, "predict")
    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + seed_offset + ep)
        ep_revenue, ep_profit, ep_capped, steps = 0.0, 0.0, 0, 0
        done = False
        while not done:
            if is_model:
                action, _ = policy_fn.predict(obs, deterministic=True)
            else:
                action = policy_fn(obs)
            obs, reward, term, trunc, info = env.step(action)
            ep_revenue += info["revenue"]
            ep_profit += info["profit"]
            ep_capped += info["actual_sales"] < info["demand"] - 1e-9
            steps += 1
            done = term or trunc
        revenues.append(ep_revenue)
        profits.append(ep_profit)
        capped.append(ep_capped / steps)
    return np.array(revenues), np.array(profits), np.array(capped)


def tune_constant_ratio(env, seed_offset):
    """Best constant price multiplier for this product, chosen by profit on
    tuning seeds that are disjoint from the evaluation seeds."""
    best_ratio, best_profit = 1.0, -np.inf
    for ratio in RATIO_GRID:
        policy = lambda obs, r=ratio: np.array([r], dtype=np.float32)
        _, prof, _ = run_policy(env, policy, TUNING_EPISODES, seed_offset, base_seed=TUNING_SEED)
        if prof.mean() > best_profit:
            best_ratio, best_profit = float(ratio), float(prof.mean())
    return best_ratio


def bootstrap_uplift_ci(ppo_vals, base_vals, rng):
    """95% CI of (sum ppo / sum base - 1) * 100, resampling products."""
    ppo_vals, base_vals = np.asarray(ppo_vals), np.asarray(base_vals)
    n = len(ppo_vals)
    stats = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        stats[b] = (ppo_vals[idx].sum() / base_vals[idx].sum() - 1.0) * 100.0
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def main():
    import sys
    REPORTS_DIR.mkdir(exist_ok=True)

    test_products = pd.read_csv(DATA_DIR / "test_products.csv")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None  # quick smoke test; writes nothing
    if limit:
        test_products = test_products.head(limit)
    print(f"Evaluating on {len(test_products)} held-out products, "
          f"{EPISODES_PER_PRODUCT} episodes each", flush=True)

    model = PPO.load(str(MODEL_DIR / "ppo_pricing"))
    names = ["ppo", *BASELINES.keys(), "tuned_static_oracle"]
    revenue = {n: [] for n in names}
    profit = {n: [] for n in names}
    capped = {n: [] for n in names}
    oracle_ratios = []
    per_product_rows = []

    for i, row in test_products.iterrows():
        env = PricingEnv(test_products, fixed_stock_code=row["stock_code"])

        ratio = tune_constant_ratio(env, seed_offset=i * 100)
        oracle_ratios.append(ratio)
        oracle_policy = lambda obs, r=ratio: np.array([r], dtype=np.float32)

        strategies = {"ppo": model, **BASELINES, "tuned_static_oracle": oracle_policy}
        product_row = {"stock_code": row["stock_code"], "category": row["category"],
                       "elasticity": row["elasticity"], "oracle_ratio": ratio}
        for name, policy in strategies.items():
            rev, prof, cap = run_policy(env, policy, EPISODES_PER_PRODUCT, seed_offset=i * 100)
            revenue[name].append(rev.mean())
            profit[name].append(prof.mean())
            capped[name].append(cap.mean())
            product_row[f"{name}_revenue"] = rev.mean()
            product_row[f"{name}_profit"] = prof.mean()
        per_product_rows.append(product_row)
        if (i + 1) % 50 == 0:
            print(f"  evaluated {i + 1}/{len(test_products)} products...", flush=True)

    if not limit:
        pd.DataFrame(per_product_rows).to_csv(REPORTS_DIR / "per_product_eval.csv", index=False)

    rng = np.random.default_rng(0)
    summary = {}
    for name in names:
        summary[name] = {
            "mean_revenue_per_product": float(np.mean(revenue[name])),
            "std_revenue_per_product": float(np.std(revenue[name])),
            "total_revenue": float(np.sum(revenue[name])),
            "mean_profit_per_product": float(np.mean(profit[name])),
            "std_profit_per_product": float(np.std(profit[name])),
            "total_profit": float(np.sum(profit[name])),
            "supply_capped_share": float(np.mean(capped[name])),
        }

    for name in names:
        if name == "ppo":
            continue
        base_rev, base_prof = summary[name]["total_revenue"], summary[name]["total_profit"]
        summary[name]["revenue_uplift_vs_ppo_pct"] = float(
            (summary["ppo"]["total_revenue"] - base_rev) / abs(base_rev) * 100)
        summary[name]["profit_uplift_vs_ppo_pct"] = float(
            (summary["ppo"]["total_profit"] - base_prof) / abs(base_prof) * 100)
        summary[name]["revenue_uplift_ci95_pct"] = bootstrap_uplift_ci(revenue["ppo"], revenue[name], rng)
        summary[name]["profit_uplift_ci95_pct"] = bootstrap_uplift_ci(profit["ppo"], profit[name], rng)

    summary["oracle_ratio_mean"] = float(np.mean(oracle_ratios))
    summary["n_held_out_products"] = len(test_products)
    summary["episodes_per_product"] = EPISODES_PER_PRODUCT

    if not limit:
        with open(REPORTS_DIR / "uplift_report.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n=== RESULTS ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
