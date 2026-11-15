from __future__ import annotations

from typing import Dict, Any, List
from securedrive_fl.budget.config import BudgetConfig, CapacityConfig, AvailabilityConfig


def get_scenario_config(scenario_name: str, **overrides) -> BudgetConfig:
    """
    Returns BudgetConfig for designated scenario:
    - 'hetero_tiers' (H1): 4 capacity tiers [0.10, 0.25, 0.50, 1.00], availability uniform(0.3, 1.0)
    - 'homog_f': constant capacity f, availability constant 1.0
    - 'bursty' (S2): 4 capacity tiers, availability markov(a_high=1.0, a_low=0.1, p_stay=0.8)
    """
    b_cfg = BudgetConfig()

    if scenario_name in ("hetero_tiers", "H1"):
        b_cfg.capacity = CapacityConfig(
            mode="tiers",
            tiers=[0.10, 0.25, 0.50, 1.00],
            tier_counts=[2, 2, 1, 1],
        )
        b_cfg.availability = AvailabilityConfig(
            mode="uniform",
            uniform_lo=0.3,
            uniform_hi=1.0,
        )

    elif scenario_name.startswith("homog_f") or scenario_name in ("S1", "sweep"):
        # Extract f value if passed in name, e.g. homog_f0.25
        f_val = 0.5
        if scenario_name.startswith("homog_f"):
            try:
                f_val = float(scenario_name.replace("homog_f", ""))
            except ValueError:
                f_val = 0.5
        b_cfg.capacity = CapacityConfig(
            mode="constant",
            constant_value=f_val,
        )
        b_cfg.availability = AvailabilityConfig(
            mode="constant",
            constant_value=1.0,
        )

    elif scenario_name in ("bursty", "S2"):
        b_cfg.capacity = CapacityConfig(
            mode="tiers",
            tiers=[0.10, 0.25, 0.50, 1.00],
            tier_counts=[2, 2, 1, 1],
        )
        b_cfg.availability = AvailabilityConfig(
            mode="markov",
            markov_a_high=1.0,
            markov_a_low=0.1,
            markov_p_stay_high=0.8,
            markov_p_stay_low=0.8,
        )

    else:
        raise ValueError(f"Unknown scenario name: {scenario_name}")

    # Apply keyword overrides
    for k, v in overrides.items():
        if hasattr(b_cfg, k):
            setattr(b_cfg, k, v)
        elif hasattr(b_cfg.capacity, k):
            setattr(b_cfg.capacity, k, v)
        elif hasattr(b_cfg.availability, k):
            setattr(b_cfg.availability, k, v)

    return b_cfg
