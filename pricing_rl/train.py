"""
Train a single PPO policy across the full product catalog.

Products are split 85/15 into train/held-out sets *before* training. The
policy only ever samples training-set products during learning; the 15%
held-out set is reserved for the uplift evaluation in evaluate.py, so the
reported uplift reflects generalization to unseen products, not
memorization of the training set.
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from environment import PricingEnv


class EpisodeLogCallback(BaseCallback):
    """Records episode reward/length from Monitor's info['episode'] so we
    have a learning curve without needing tensorboard installed."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rows.append({
                    "timestep": self.num_timesteps,
                    "episode_reward": info["episode"]["r"],
                    "episode_length": info["episode"]["l"],
                })
        return True

DATA_DIR = r"C:\Projects\DynamicPriceRL\DynamicPriceRL\data"
MODEL_DIR = r"C:\Projects\DynamicPriceRL\DynamicPriceRL\models"
LOG_DIR = r"C:\Projects\DynamicPriceRL\DynamicPriceRL\logs"

TOTAL_TIMESTEPS = 400_000
SEED = 42
TEST_FRACTION = 0.15


def main():
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    products = pd.read_csv(f"{DATA_DIR}/products.csv")
    rng = np.random.default_rng(SEED)
    shuffled = products.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_test = int(len(shuffled) * TEST_FRACTION)
    test_products = shuffled.iloc[:n_test].reset_index(drop=True)
    train_products = shuffled.iloc[n_test:].reset_index(drop=True)

    train_products.to_csv(f"{DATA_DIR}/train_products.csv", index=False)
    test_products.to_csv(f"{DATA_DIR}/test_products.csv", index=False)
    print(f"Train products: {len(train_products)}, held-out test products: {len(test_products)}")

    def make_env():
        env = PricingEnv(train_products)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=SEED,
    )

    print(f"Training PPO for {TOTAL_TIMESTEPS:,} timesteps across {len(train_products)} products...")
    callback = EpisodeLogCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False, log_interval=10, callback=callback)

    model.save(f"{MODEL_DIR}/ppo_pricing")
    print(f"Model saved to {MODEL_DIR}/ppo_pricing.zip")

    pd.DataFrame(callback.rows).to_csv(f"{LOG_DIR}/learning_curve.csv", index=False)
    print(f"Learning curve saved to {LOG_DIR}/learning_curve.csv")


if __name__ == "__main__":
    main()
