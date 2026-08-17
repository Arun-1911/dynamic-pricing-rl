"""
Simple, non-RL pricing strategies used as the uplift comparison point.
"""

import numpy as np


def static_price_policy(obs):
    """Never change price — always sell at the product's base price.
    Represents the common real-world default: no dynamic pricing at all."""
    return np.array([1.0], dtype=np.float32)


def undercut_competitor_policy(obs, undercut=0.05, low=0.7, high=1.4):
    """Simple rule used by many real pricing tools: price a fixed % below
    the observed competitor price, clipped to the same action bounds PPO
    uses. obs[1] is competitor_ratio (competitor_price / base_price)."""
    competitor_ratio = obs[1]
    target = competitor_ratio * (1.0 - undercut)
    target = float(np.clip(target, low, high))
    return np.array([target], dtype=np.float32)


def elasticity_optimal_static_policy(obs, cost_margin=0.5, low=0.7, high=1.4):
    """The strongest fair baseline: a *static* price set once using the
    classic constant-elasticity monopoly pricing formula price* = cost *
    e/(e+1), computed from the same fitted elasticity PPO can see. This
    isolates what RL's dynamic/adaptive behavior adds on top of simply
    knowing the elasticity — the honest apples-to-apples comparison,
    since both this baseline and PPO have the same information."""
    elasticity = obs[4]
    if elasticity < -1.0 - 1e-6:
        ratio = cost_margin * elasticity / (elasticity + 1.0)
    else:
        # Demand is inelastic (|e|<1): textbook formula pushes price to
        # infinity, so cap at the same action bound PPO is constrained to.
        ratio = high
    return np.array([np.clip(ratio, low, high)], dtype=np.float32)


BASELINES = {
    "static_base_price": static_price_policy,
    "undercut_competitor_5pct": undercut_competitor_policy,
    "elasticity_optimal_static": elasticity_optimal_static_policy,
}
