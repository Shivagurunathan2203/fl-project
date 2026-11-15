from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class CapacityConfig:
    mode: str = "tiers"  # "tiers" | "constant" | "uniform" | "lognormal"
    tiers: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.50, 1.00])
    tier_counts: List[int] = field(default_factory=lambda: [2, 2, 1, 1])
    constant_value: float = 0.5
    uniform_lo: float = 0.1
    uniform_hi: float = 1.0
    lognormal_mu: float = -1.0
    lognormal_sigma: float = 0.7
    lognormal_clip: List[float] = field(default_factory=lambda: [0.02, 1.0])


@dataclass
class AvailabilityConfig:
    mode: str = "uniform"  # "constant" | "uniform" | "beta" | "markov"
    uniform_lo: float = 0.3
    uniform_hi: float = 1.0
    constant_value: float = 1.0
    beta_alpha: float = 4.0
    beta_beta: float = 2.0
    markov_a_high: float = 1.0
    markov_a_low: float = 0.1
    markov_p_stay_high: float = 0.8
    markov_p_stay_low: float = 0.8


@dataclass
class BudgetConfig:
    enabled: bool = True
    unit: str = "chunks"
    chunk_size: int = 4096
    virtual_chunks: Optional[int] = 25
    rank_by: str = "post_noise_energy"  # "post_noise_energy" | "layer_static" | "random" | "pre_noise_energy"
    layer_order: str = "reverse"  # "reverse" (classifier first) | "forward"
    chunk_content: str = "masked_only"  # "masked_only" | "full_chunk"
    rng_salt: int = 7919
    min_coverage: float = 0.5  # For sd_exclude strategy (default 0.5)

    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    availability: AvailabilityConfig = field(default_factory=AvailabilityConfig)
