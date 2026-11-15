from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple
from securedrive_fl.config import Config


def quantize_16bit(v: np.ndarray, clip_range: float = 10.0) -> Tuple[np.ndarray, float]:
    """
    Symmetric 16-bit integer quantization for BatchCrypt (Zhang et al. 2020).
    """
    scale = 32767.0 / clip_range
    q = np.clip(np.round(v * scale), -32767.0, 32767.0).astype(np.float64)
    return q, scale


def dequantize_16bit(q: np.ndarray, scale: float) -> np.ndarray:
    """Dequantizes 16-bit integer array back to floating point."""
    return q / scale


NINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "fedavg": {"use_dp": False, "use_gashe": False, "aggregator": "fedavg"},
    "dp_sgd": {"use_dp": True, "use_gashe": False, "aggregator": "fedavg"},
    "gashe_only": {"use_dp": False, "use_gashe": True, "top_k": 0.60},
    "full_ckks": {"use_dp": False, "use_gashe": False, "full_ckks": True},
    "securedrive": {"use_dp": True, "use_gashe": True},
    "trimmed_mean": {"use_dp": False, "use_gashe": False, "aggregator": "trimmed_mean"},
    "krum": {"use_dp": False, "use_gashe": False, "aggregator": "krum"},
    "batchcrypt": {"use_dp": False, "use_gashe": False, "batchcrypt": True, "full_ckks": True},
    "fedprox": {"use_dp": False, "use_gashe": False, "aggregator": "fedprox"},
}


def get_config_for_baseline(name: str, **kwargs) -> Config:
    """Returns a Config object customized for the named baseline."""
    if name not in NINE_CONFIGS:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(NINE_CONFIGS.keys())}")
    cfg_args = dict(NINE_CONFIGS[name])
    cfg_args.update(kwargs)
    return Config(**cfg_args)
