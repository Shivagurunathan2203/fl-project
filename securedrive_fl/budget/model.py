from __future__ import annotations

import math
from typing import Optional, Union, Dict, Any
import numpy as np

from securedrive_fl.budget.config import BudgetConfig, CapacityConfig, AvailabilityConfig


class BudgetModel:
    """
    Manages client compute capacities and per-round availability for HE selective encryption.
    Uses dedicated RNG stream derived from (seed + rng_salt) to guarantee common random
    numbers without polluting base training randomness.
    """
    def __init__(
        self,
        cfg: BudgetConfig,
        n_clients: int,
        n_chunks: int,
        seed: int,
        max_rounds: int = 150,
    ):
        self.cfg = cfg
        self.n_clients = n_clients
        self.n_chunks = n_chunks
        self.seed = seed
        self.max_rounds = max_rounds

        # 1. Assigned fixed capacity vector c_i in [0, 1]
        rng_cap = np.random.default_rng(self.seed + self.cfg.rng_salt)
        self.capacity = self._generate_capacity(rng_cap)

        # 2. Precompute Markov availability chain if markov mode is requested
        if self.cfg.availability.mode == "markov":
            rng_markov = np.random.default_rng(self.seed + self.cfg.rng_salt + 99991)
            self._markov_chain = self._precompute_markov(rng_markov)
        else:
            self._markov_chain = None

    def _generate_capacity(self, rng: np.random.Generator) -> np.ndarray:
        c_cfg = self.cfg.capacity
        K = self.n_clients

        if c_cfg.mode == "tiers":
            # Scale or allocate tier counts to sum to K
            tiers = list(c_cfg.tiers)
            counts = list(c_cfg.tier_counts)
            if sum(counts) != K:
                # Distribute K evenly across tiers
                base = K // len(tiers)
                rem = K % len(tiers)
                counts = [base + (1 if i < rem else 0) for i in range(len(tiers))]

            cap_list = []
            for t_val, cnt in zip(tiers, counts):
                cap_list.extend([t_val] * cnt)
            cap_arr = np.array(cap_list, dtype=np.float64)
            rng.shuffle(cap_arr)
            return cap_arr

        elif c_cfg.mode == "constant":
            return np.full(K, float(c_cfg.constant_value), dtype=np.float64)

        elif c_cfg.mode == "uniform":
            return rng.uniform(c_cfg.uniform_lo, c_cfg.uniform_hi, size=K)

        elif c_cfg.mode == "lognormal":
            raw = rng.lognormal(c_cfg.lognormal_mu, c_cfg.lognormal_sigma, size=K)
            lo, hi = c_cfg.lognormal_clip
            return np.clip(raw, lo, hi)

        else:
            raise ValueError(f"Unknown capacity mode: {c_cfg.mode}")

    def _precompute_markov(self, rng: np.random.Generator) -> np.ndarray:
        K = self.n_clients
        cfg_a = self.cfg.availability
        chain = np.zeros((self.max_rounds + 1, K), dtype=np.float64)

        # Initial state: 50% chance high / low
        state_high = rng.uniform(0.0, 1.0, size=K) >= 0.5
        chain[0] = np.where(state_high, cfg_a.markov_a_high, cfg_a.markov_a_low)

        for r in range(1, self.max_rounds + 1):
            p_stay = np.where(state_high, cfg_a.markov_p_stay_high, cfg_a.markov_p_stay_low)
            stay = rng.uniform(0.0, 1.0, size=K) < p_stay
            state_high = np.where(stay, state_high, ~state_high)
            chain[r] = np.where(state_high, cfg_a.markov_a_high, cfg_a.markov_a_low)

        return chain

    def availability(self, rnd: int) -> np.ndarray:
        """
        Returns availability a_{i, rnd} in [0, 1] for all K clients in round rnd.
        Deterministic in (seed, rnd) and independent of which clients are sampled.
        """
        cfg_a = self.cfg.availability
        K = self.n_clients

        if cfg_a.mode == "constant":
            return np.full(K, float(cfg_a.constant_value), dtype=np.float64)

        if cfg_a.mode == "markov":
            if self._markov_chain is not None and rnd < len(self._markov_chain):
                return self._markov_chain[rnd].copy()
            return np.full(K, cfg_a.markov_a_high, dtype=np.float64)

        # Stateless derivation for uniform & beta
        rng_rnd = np.random.default_rng(self.seed + self.cfg.rng_salt + rnd * 10007)

        if cfg_a.mode == "uniform":
            return rng_rnd.uniform(cfg_a.uniform_lo, cfg_a.uniform_hi, size=K)

        elif cfg_a.mode == "beta":
            return rng_rnd.beta(cfg_a.beta_alpha, cfg_a.beta_beta, size=K)

        else:
            raise ValueError(f"Unknown availability mode: {cfg_a.mode}")

    def effective_chunks(self, rnd: int) -> np.ndarray:
        """
        B_{i,n} = floor( c_i * a_{i,n} * n_chunks ).
        Returns int array of shape [K].
        """
        avail = self.availability(rnd)
        raw = self.capacity * avail * self.n_chunks
        return np.floor(raw).astype(np.int64)

    def speed(self, rnd: int) -> np.ndarray:
        """
        s_{i,n} = c_i * a_{i,n}.
        Returns float array of shape [K].
        """
        return self.capacity * self.availability(rnd)
