"""
SHAP explainability for the trained PPO pricing policy.

The policy network isn't tree-based, so we treat it as a black-box
regressor (obs -> price multiplier) and explain it with KernelExplainer.

With only 7 input features, KernelExplainer can enumerate every feature
subset exactly (nsamples="auto"), so explanations are deterministic and the
Shapley values add up exactly to the model output.
"""

import threading

import numpy as np
import shap

FEATURE_NAMES = [
    "price_ratio", "competitor_ratio", "inventory", "demand_ratio",
    "elasticity", "time_in_episode", "category",
]

FEATURE_LABELS = {
    "price_ratio": "Current price vs. base price",
    "competitor_ratio": "Competitor price vs. base price",
    "inventory": "Inventory level",
    "demand_ratio": "Recent demand vs. typical",
    "elasticity": "Price elasticity of demand",
    "time_in_episode": "Time step in episode",
    "category": "Product category",
}


def make_predict_fn(model):
    def predict_fn(X):
        X = np.asarray(X, dtype=np.float32)
        actions, _ = model.predict(X, deterministic=True)
        return actions[:, 0]
    return predict_fn


def build_background(env, n=50, seed=0):
    """Sample a diverse, reproducible set of real states (not just
    episode-start states) to use as the SHAP background distribution."""
    env.action_space.seed(seed)
    obs_list = []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        # walk a few random steps so background covers mid-episode states
        n_steps = i % 15
        for _ in range(n_steps):
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        obs_list.append(obs)
    return np.array(obs_list, dtype=np.float32)


class PricingExplainer:
    def __init__(self, model, background):
        self.model = model
        self.background = background
        self.predict_fn = make_predict_fn(model)
        self._explainer = shap.KernelExplainer(self.predict_fn, background, silent=True)
        # KernelExplainer keeps per-call state on the instance, so concurrent
        # calls (e.g. overlapping API requests) must be serialised.
        self._lock = threading.Lock()

    def explain(self, obs, nsamples="auto"):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        with self._lock:
            shap_values = self._explainer.shap_values(obs, nsamples=nsamples, silent=True)
            base_value = float(np.asarray(self._explainer.expected_value).squeeze())
            recommended_price_ratio = float(self.predict_fn(obs)[0])
        return {
            "feature_names": FEATURE_NAMES,
            "feature_values": obs[0].tolist(),
            "shap_values": np.asarray(shap_values)[0].tolist(),
            "base_value": base_value,
            "prediction": recommended_price_ratio,
        }
