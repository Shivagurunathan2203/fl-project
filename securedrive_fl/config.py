from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # federation
    n_clients: int = 6
    rounds: int = 120
    # local training
    local_epochs: int = 5
    batch: int = 32
    lr: float = 0.1                 # Default for toy model; adapted to 0.02 for real-data embeddings
    # DP-SGD (client side)
    use_dp: bool = True
    clip_C: float = 1.2             # per-sample L2 clip
    eps0: float = 4.0               # single-shot calibration input, NOT the final epsilon
    delta: float = 1e-5
    # GASHE
    use_gashe: bool = True
    top_k: float = 0.60             # only used when use_dp=False
    # plain-path aggregator (ignored when use_gashe=True): fedavg | trimmed_mean | krum | fedprox
    aggregator: str = "fedavg"
    krum_f: int = 2                 # default for N=6 (n-f-2 = 2 >= 1)
    # attack:  None | "model_poison" | "mitm"      (2 of 6 clients, f = N/3)
    attack: Optional[str] = None
    mitm_amplify: float = 10.0
    mitm_noise_std: float = 1.0
    # server-side 3-sigma loss guard (V56 enables it only for MITM runs)
    loss_guard: bool = False
    seed: int = 0

    # Poisoning & baseline configuration
    poison_mode: str = "backdoor"   # "backdoor" (default for Table I) | "ascent" (hld variant)
    target_class: int = 2           # State Farm class 2: "talking on phone - right"
    poison_alpha: float = 0.5       # Paper Eq. 13 backdoor interpolation weight
    poison_boost: float = 1.0       # Adversary update scaling (1.0 = no boost)
    prox_mu: float = 0.01           # FedProx proximal coefficient
    norm_clip: bool = False         # FedAvg norm clipping optional defense
    full_ckks: bool = False         # Configuration 4: FL + Full-CKKS
    batchcrypt: bool = False        # Configuration 8: BatchCrypt
    separation: float = 3.8         # Fixed cluster separation for toy federation

    @property
    def n_byzantine(self) -> int:
        return max(1, self.n_clients // 3)

    @property
    def sigma0(self) -> float:      # Gaussian-mechanism calibration (Dwork & Roth Thm A.1)
        return self.clip_C * math.sqrt(2 * math.log(1.25 / self.delta)) / self.eps0

    @property
    def z(self) -> float:           # shared noise multiplier: training, accountant, tau
        return self.sigma0 / self.clip_C

    @property
    def sigma(self) -> float:       # actual noise std = z*C  (= tau in SecureDrive-FL)
        return self.z * self.clip_C
