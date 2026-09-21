"""
Gymnasium environment for multi-product dynamic pricing.

Unlike a toy single-SKU pricing env, this samples a real product (with its
real fitted price elasticity, real base price, and real category) from the
UCI Online Retail II catalog at the start of every episode. A single PPO
policy trained against this environment therefore learns a pricing strategy
that must generalize across the full ~4,850-product catalog, not one item.

What's real (from data_prep.py, fit on actual transactions):
    - base_price, elasticity, category, per-product demand level

What's simulated (no such data exists in the source dataset):
    - competitor price and its week-to-week drift
    - inventory level and replenishment
    - demand noise

Inventory model: storage holds up to STORAGE_WEEKS weeks of typical demand and
the retailer restocks one week of typical demand every week (ordering to the
historical run-rate). Selling at the base price is therefore sustainable, so
supply only binds when a strategy deliberately over-sells (deep discounts).

Reward is weekly gross profit normalised by the product's base revenue, i.e.
exactly the quantity the evaluation reports; there are no extra shaping terms.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

STORAGE_WEEKS = 2.0


class PricingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, products: pd.DataFrame, cost_margin=0.5, max_steps=52,
                 price_low=0.7, price_high=1.4, fixed_stock_code=None):
        super().__init__()
        self.products = products.reset_index(drop=True)
        self.cost_margin = cost_margin
        self.max_steps = max_steps
        self.fixed_stock_code = fixed_stock_code

        self.categories = sorted(self.products["category"].unique())
        self.category_to_id = {c: i for i, c in enumerate(self.categories)}
        self.n_categories = len(self.categories)

        self.action_space = spaces.Box(
            low=np.array([price_low], dtype=np.float32),
            high=np.array([price_high], dtype=np.float32),
            dtype=np.float32,
        )
        # [price_ratio, competitor_ratio, inventory, demand_ratio,
        #  elasticity, time_norm, category_norm]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -15.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 3.0, 1.0, 10.0, 0.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_step = 0
        self.product = None
        self.inventory = 1.0
        self.price_ratio = 1.0
        self.competitor_ratio = 1.0
        self.demand_ratio = 1.0

    def _sample_product(self):
        if self.fixed_stock_code is not None:
            row = self.products[self.products["stock_code"] == self.fixed_stock_code].iloc[0]
        else:
            idx = self.np_random.integers(0, len(self.products))
            row = self.products.iloc[idx]
        return row

    def _obs(self):
        cat_norm = self.category_to_id[self.product["category"]] / max(1, self.n_categories - 1)
        return np.array([
            self.price_ratio,
            self.competitor_ratio,
            self.inventory,
            self.demand_ratio,
            float(self.product["elasticity"]),
            self.current_step / self.max_steps,
            cat_norm,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "stock_code" in options:
            self.product = self.products[self.products["stock_code"] == options["stock_code"]].iloc[0]
        else:
            self.product = self._sample_product()

        self.base_price = float(self.product["base_price"])
        self.elasticity = float(self.product["elasticity"])
        self.base_demand = float(np.clip(self.product["base_demand"], 1.0, 5000.0))
        self.cost = self.base_price * self.cost_margin

        self.current_step = 0
        self.inventory = 1.0
        self.price_ratio = 1.0
        self.competitor_ratio = float(self.np_random.uniform(0.9, 1.1))
        self.demand_ratio = 1.0

        return self._obs(), {"stock_code": self.product["stock_code"]}

    def step(self, action):
        price_multiplier = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        new_price = self.base_price * price_multiplier
        price_ratio = new_price / self.base_price

        # Real demand curve: fitted constant-elasticity response.
        base_demand_calc = self.base_demand * np.power(price_ratio, self.elasticity)

        competitor_price = self.competitor_ratio * self.base_price
        if new_price > competitor_price:
            diff = (new_price - competitor_price) / max(competitor_price, 1e-6)
            competitor_factor = max(0.5, 1.0 - diff * 0.5)
        else:
            diff = (competitor_price - new_price) / max(competitor_price, 1e-6)
            competitor_factor = min(1.5, 1.0 + diff * 0.3)

        noise = self.np_random.normal(1.0, 0.1)
        demand = max(0.0, base_demand_calc * competitor_factor * noise)

        capacity = self.inventory * STORAGE_WEEKS * self.base_demand
        actual_sales = min(demand, capacity)

        revenue = new_price * actual_sales
        cost_total = self.cost * actual_sales
        profit = revenue - cost_total

        # Normalize reward by the product's own base revenue so a single
        # policy sees comparable reward scale whether the product sells
        # 2 units/week or 2,000 units/week.
        base_revenue = max(self.base_price * self.base_demand, 1e-6)
        reward = profit / base_revenue

        inventory_used = actual_sales / (STORAGE_WEEKS * self.base_demand)
        restock = 1.0 / STORAGE_WEEKS
        self.inventory = float(np.clip(self.inventory + restock - inventory_used, 0.0, 1.0))
        self.competitor_ratio = float(np.clip(
            self.competitor_ratio * self.np_random.uniform(0.98, 1.02), 0.5, 1.5
        ))
        self.price_ratio = price_ratio
        self.demand_ratio = float(np.clip(demand / self.base_demand, 0.0, 10.0))

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "stock_code": self.product["stock_code"],
            "price": new_price,
            "demand": demand,
            "actual_sales": actual_sales,
            "profit": profit,
            "revenue": revenue,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
