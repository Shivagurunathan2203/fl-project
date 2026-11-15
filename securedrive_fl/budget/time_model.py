from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


# Standard default calibration timings (in seconds) if calibration.json is absent
DEFAULT_CALIBRATION = {
    "t_enc": 0.045,          # ~45ms per 4096-slot CKKS chunk
    "t_dec": 0.012,          # ~12ms per 4096-slot chunk
    "t_add": 0.0005,         # ~0.5ms per chunk addition
    "bytes_per_chunk": 65536, # ~64KB per chunk ciphertext
    "plain_bytes_per_coord": 8,  # float64
}


class LatencyModel:
    """
    Simulated latency model accounting for client compute heterogeneity,
    round deadlines, straggler wait costs, and cryptographic operation overheads.
    """
    def __init__(self, calibration_path: Optional[str] = "results_ext/calibration.json"):
        self.calib = DEFAULT_CALIBRATION.copy()
        if calibration_path and os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    data = json.load(f)
                    self.calib.update(data)
            except Exception:
                pass

    @property
    def t_enc(self) -> float:
        return float(self.calib["t_enc"])

    @property
    def t_dec(self) -> float:
        return float(self.calib["t_dec"])

    @property
    def t_add(self) -> float:
        return float(self.calib["t_add"])

    @property
    def bytes_per_chunk(self) -> int:
        return int(self.calib["bytes_per_chunk"])

    @property
    def plain_bytes_per_coord(self) -> int:
        return int(self.calib.get("plain_bytes_per_coord", 8))

    def compute_round_time(
        self,
        strategy: str,
        t_train: float,
        n_chunks: int,
        client_speeds: List[float],
        demand_counts: List[int],
        encrypted_counts: List[int],
        total_unique_enc_chunks: int,
        skipped: bool = False,
    ) -> float:
        """
        Computes simulated round time in seconds.
        """
        if skipped or len(client_speeds) == 0:
            return 0.0

        # Server aggregation time: t_add * sum of additions across present chunks
        n_participants = len(client_speeds)
        t_agg = (n_participants - 1) * total_unique_enc_chunks * self.t_add

        # Client decryption time: each client decrypts unique aggregated chunks
        t_dec_total = total_unique_enc_chunks * self.t_dec

        # Round deadline W: time for speed=1 client to encrypt full vector
        W = n_chunks * self.t_enc

        if strategy == "sd_wait":
            # Straggler cost: round waits for the slowest participant to encrypt their demand
            wait_times = [
                (dem * self.t_enc / max(1e-4, spd))
                for dem, spd in zip(demand_counts, client_speeds)
            ]
            max_enc_wait = max(wait_times) if wait_times else 0.0
            return float(t_train + max_enc_wait + t_agg + t_dec_total)

        elif strategy in ("sd_budget", "sd_budget_random", "sd_budget_layer", "gashe_budget"):
            # Fixed deadline W
            return float(t_train + W + t_agg + t_dec_total)

        elif strategy == "sd_exclude":
            # Fixed deadline W over eligible participants
            return float(t_train + W + t_agg + t_dec_total)

        elif strategy in ("dpsgd", "fedavg"):
            # Plaintext training: no HE encryption
            return float(t_train)

        elif strategy in ("sd_full", "gashe_full"):
            # Full encryption without budget: slowest speed determines round time
            wait_times = [
                (n_chunks * self.t_enc / max(1e-4, spd))
                for spd in client_speeds
            ]
            max_enc_wait = max(wait_times) if wait_times else W
            return float(t_train + max_enc_wait + t_agg + t_dec_total)

        else:
            return float(t_train + W + t_agg + t_dec_total)
