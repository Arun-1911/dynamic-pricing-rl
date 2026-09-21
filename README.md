# Dynamic Pricing RL

**A PPO reinforcement-learning pricing engine trained on real e-commerce transaction data, with SHAP explainability and an interactive dashboard.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/stable--baselines3-PPO-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6?logo=typescript&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A single PPO policy learns a pricing strategy across a real catalog of **4,851 products**, using per-product price elasticity fitted from actual transaction history rather than assumed values. Evaluated on 727 held-out products the policy never trained on, it earns a small but statistically clear gain over static pricing: **+1.5% profit** (+0.5% revenue) versus never changing price, and **+0.9% profit** versus the best possible constant price for each product. These are simulation results and are reported with confidence intervals.

---

## Results

PPO trained for 400,000 timesteps across 4,124 products, evaluated on **727 held-out products** it never saw during training (10 stochastic episodes each). Uplift is PPO's total over the baseline's total; brackets are 95% bootstrap intervals (resampling products).

| Baseline | Revenue uplift | Profit uplift | Weeks limited by stock |
|---|---|---|---|
| Static base price (never change price) | **+0.5%** [0.2, 0.9] | **+1.5%** [1.2, 2.0] | 5.0% |
| Best constant price per product (grid-searched in the simulator; an upper bound for any static price) | +1.0% [0.8, 1.2] | +0.9% [0.8, 1.0] | 4.2% |
| Undercut competitor by 5% (naive rule) | +7.1% [6.2, 8.2] | +14.5% [13.6, 15.8] | 56.5% |
| Elasticity-optimal static price (textbook monopoly formula) | +24.4% [22.2, 26.9] | +61.0% [53.2, 69.7] | 67.8% |

PPO itself is stock-limited in 0.5% of weeks. Full numbers: [`reports/uplift_report.json`](reports/uplift_report.json) · per-product breakdown: [`reports/per_product_eval.csv`](reports/per_product_eval.csv)

**How to read this.** The catalog's base prices are the retailer's own historical prices, and against the fitted demand curves they are already close to the best constant price, so there is little for any pricing strategy to gain: even a per-product tuned constant price improves profit by only about 1% over the base price. PPO's edge over that oracle (+0.9% profit) comes from reacting to the competitor's price and to stock levels week by week. The large gaps against the last two baselines are not large RL gains. Those baselines cut prices, sell faster than the retailer restocks, and are stock-limited in more than half of all weeks; PPO avoids that.

**Sensitivity to the supply assumption.** Simulated uplift depends heavily on how inventory is modelled. In an exploratory run where restocking was set to about 20% of typical demand, sales were stock-limited in 96% of weeks under static pricing, and PPO showed a >30% revenue "uplift" simply by raising prices to ration scarce stock. That configuration is unrealistic and is not used. The default here restocks one week of typical demand per week, so selling at the base price is sustainable.

---

## Quick start

Run the dashboard against the already-trained model — no retraining required.

```bash
# 1. backend — serves the trained PPO policy + SHAP explainer
pip install -r pricing_rl/requirements.txt -r backend/requirements.txt
uvicorn backend.main:app --port 8000

# 2. frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Dashboard

| View | What it shows |
|---|---|
| **Overview** | Catalog stats, category breakdown, and the fitted elasticity distribution across all 4,851 products |
| **Explorer** | Pick any product, adjust simulated market conditions (competitor price, inventory, demand, time in cycle), and see the live PPO price recommendation, a profit-vs-price curve, and a SHAP breakdown of the decision |
| **Training** | The real PPO learning curve and the uplift-vs-baselines comparison, plus the write-up above |

React (Vite, TypeScript, Tailwind, Framer Motion, Recharts) talking to a FastAPI backend that wraps the trained model directly. A Streamlit version also still works (`streamlit run pricing_rl/dashboard.py`) if you'd rather not run Node.

---

## How it works

### Dataset

[**Online Retail II**](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI Machine Learning Repository, CC BY 4.0) — a real transaction log from a UK-based online gift-ware retailer, Dec 2009–Dec 2011.

- 1,067,371 raw transaction rows (`Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`)
- After removing cancellations, non-positive price/quantity rows, and non-product service codes (postage, bank charges, manual adjustments): **4,851 real unique products**, 1,031,503 clean transaction rows
- No `category` column in the source data — categories are derived from the free-text `Description` field via keyword rules ([`pricing_rl/data_prep.py`](pricing_rl/data_prep.py)); treat these as heuristic labels, not ground truth
- No `cost` column — profit assumes a flat 50% gross margin, a stated assumption, not a measured one
- No competitor-price or inventory data — both are simulated in the RL environment

### Elasticity fitting

For each product, transactions are aggregated to weekly (quantity-weighted price, total quantity) observations, then `log(quantity) ~ elasticity × log(price)` is fit via linear regression — only when there's at least 10% real price variation and 3+ weekly observations, to avoid numerically unstable fits on near-constant prices. Fits producing an implausible slope (outside -10 to -0.05) are rejected in favor of a category-median fallback.

| | Products | Source |
|---|---|---|
| Own price history fitted directly | 3,442 (71%) | real log-log regression |
| Category-median fallback | 1,409 (29%) | insufficient price variation or unstable fit |

Resulting distribution: mean elasticity −2.65, median −2.31 (σ = 1.53) — economically plausible for discretionary gift-ware.

### Environment

[`pricing_rl/environment.py`](pricing_rl/environment.py) — a Gymnasium env where each episode samples one real product (its real elasticity, base price, and category) and runs 52 weekly pricing decisions.

- **State** (7-dim): current price ratio, competitor price ratio, inventory level, recent demand ratio, elasticity, time in episode, category
- **Action**: continuous price multiplier, 0.7×–1.4× base price
- **Reward**: weekly gross profit normalized by the product's own base revenue, so one policy sees comparable reward scale whether a product sells 2 units/week or 2,000/week. There are no extra shaping terms: the reward is exactly the profit that the evaluation reports
- **Inventory**: storage holds up to two weeks of typical demand and one week of typical demand is restocked each week, so selling at the base price is sustainable. Over-selling (deep discounts) runs the stock down and caps later sales
- **Training**: `reset()` samples a random product from the 4,124-product training split every episode, so the single trained policy generalizes across the catalog rather than overfitting to one item

### Explainability

[`pricing_rl/explain.py`](pricing_rl/explain.py) treats the trained policy as a black-box function (`observation → price multiplier`) and explains it with SHAP's `KernelExplainer`, since the policy network isn't tree-based. Shapley additivity is verified on every explanation: `base_value + Σ shap_values == prediction`.

### What's real vs. simulated

| | Status |
|---|---|
| Product catalog, base prices, categories | Real — derived from actual transactions |
| Price elasticity per product | Real — fitted via log-log regression on actual weekly price/quantity |
| Demand response during RL training/eval | Simulated, using the real fitted elasticity as the demand curve's exponent |
| Competitor pricing | Simulated — the dataset has no competitor data (single retailer) |
| Inventory dynamics | Simulated (restock = one week of typical demand per week) — the dataset has no inventory/stock data; results are sensitive to this assumption |
| Cost / margin | Assumed at a flat 50% of base price — the dataset has no cost column |
| Uplift numbers | Measured in simulation, not a live A/B test |

The elasticity is genuinely fitted from real transactions, and the uplift numbers are honest outputs of that simulation. They are not, on their own, evidence of real-world revenue impact — that would require a live pricing experiment.

---

## Project structure

```
pricing_rl/       data prep, Gymnasium environment, PPO training, evaluation, SHAP explainer
backend/          FastAPI service exposing the trained model + explainer to the frontend
frontend/         React dashboard (Vite + TypeScript + Tailwind + Framer Motion + Recharts)
data/             source dataset, fitted per-product elasticity, train/test splits
models/           trained PPO checkpoint
reports/          uplift evaluation results
logs/             training run logs and learning curve
```

## Reproducing the training run

```bash
cd pricing_rl
python data_prep.py    # cleans transactions, fits elasticity -> data/products.csv
python train.py        # trains PPO, 400k timesteps -> models/ppo_pricing.zip
python evaluate.py     # evaluates vs baselines on held-out products (about 15 min) -> reports/
```

## License

MIT
