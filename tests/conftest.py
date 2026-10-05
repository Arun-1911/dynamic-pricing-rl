import sys
from pathlib import Path

import pandas as pd
import pytest
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                # makes the `backend` package importable
sys.path.insert(0, str(ROOT / "pricing_rl"))  # makes the pricing_rl modules importable


@pytest.fixture(scope="session")
def products():
    return pd.read_csv(ROOT / "data" / "products.csv")


@pytest.fixture(scope="session")
def train_df():
    return pd.read_csv(ROOT / "data" / "train_products.csv")


@pytest.fixture(scope="session")
def test_df():
    return pd.read_csv(ROOT / "data" / "test_products.csv")


@pytest.fixture(scope="session")
def model():
    return PPO.load(str(ROOT / "models" / "ppo_pricing"))
