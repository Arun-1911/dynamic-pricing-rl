# Dynamic Pricing RL

**A PPO reinforcement-learning pricing engine trained on real e-commerce transaction data, with SHAP explainability and an interactive dashboard.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/stable--baselines3-PPO-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6?logo=typescript&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A single PPO policy learns a pricing strategy across a real catalog of **4,851 products**, using per-product price elasticity fitted from actual transaction history rather than assumed values. Evaluated on 727 held-out products the policy never trained on, it beats a static-pricing baseline by **+32.5% revenue** and **+65.7% profit** — an honest, reproducible number, not a target.

---

## Results

PPO trained for 400,000 timesteps across 4,124 products, evaluated on **727 held-out products** it never saw during training (10 stochastic episodes each):

| Baseline | Revenue uplift | Profit uplift |
|---|---|---|
| Static base price (never change price) | **+32.5%** | **+65.7%** |
| Undercut competitor by 5% (naive rule) | +39.2% | +83.3% |
| Elasticity-optimal static price (classic monopoly formula, same elasticity data) | +57.1% | +141.1% |

Full numbers: [`reports/uplift_report.json`](reports/uplift_report.json) · per-product breakdown: [`reports/per_product_eval.csv`](reports/per_product_eval.csv)

The +32.5% figure (vs. simply never changing price) is the most defensible, easiest-to-verify result and the one worth leading with. The other two comparisons are real but need more context to present fairly — see below.

<details>
<summary><b>Why the "elasticity-optimal static" baseline is the most interesting result</b></summary>

<br>

It's a *static* price computed once from the same fitted elasticity PPO can see, using the textbook constant-elasticity monopoly formula `price* = cost × e/(e+1)`. It should be the hardest baseline to beat — and yet it underperforms even the do-nothing baseline.

Tracing individual product rollouts showed why: ~34% of held-out products get pushed to the environment's price bounds (0.7×–1.4× base price), because the formula optimizes only the isolated elasticity term and is blind to the competitor-price dynamic in the environment. Overpricing against a competitor sitting near parity triggers a steep, competitor-driven demand penalty the formula never sees coming. PPO, which observes competitor price every step, avoids that trap.

That gap is the real value dynamic, context-aware pricing adds over "just know your elasticity."

</details>

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
- **Reward**: profit normalized by the product's own base revenue — so one policy sees comparable reward scale whether a product sells 2 units/week or 2,000/week — with penalties for stockouts and excess unsold inventory
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
| Inventory dynamics | Simulated — the dataset has no inventory/stock data |
| Cost / margin | Assumed at a flat 50% of base price — the dataset has no cost column |
| Revenue uplift numbers | Measured in simulation, not a live A/B test |

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
python evaluate.py     # evaluates vs baselines on held-out products -> reports/
```

## License

MIT
