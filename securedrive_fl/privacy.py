from __future__ import annotations

import math
from securedrive_fl.config import Config


def _rdp_sampled_gaussian(alpha: int, q: float, z: float) -> float:
    if q >= 1.0:
        return alpha / (2 * z * z)
    lq, l1q = math.log(q), math.log1p(-q)
    terms = [math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
             + (alpha - k) * l1q + k * lq + (k * k - k) / (2 * z * z) for k in range(alpha + 1)]
    m = max(terms)
    return (m + math.log(sum(math.exp(t - m) for t in terms))) / (alpha - 1)


def epsilon(steps: int, q: float, z: float, delta: float, max_alpha: int = 128) -> float:
    """eps = min over alpha of  steps*RDP(alpha) + ln(1/delta)/(alpha-1)."""
    if steps == 0:
        return 0.0
    return min(steps * _rdp_sampled_gaussian(a, q, z) + math.log(1 / delta) / (a - 1)
               for a in range(2, max_alpha + 1))


def worst_case_epsilon(client_sizes, rounds_done: int, cfg: Config) -> float:
    """Clients hold disjoint data -> parallel composition -> max_i eps_i, each with
    its own q_i = B/n_i and steps_i = epochs*(n_i//B) per round."""
    return max(epsilon(rounds_done * cfg.local_epochs * (n // cfg.batch),
                       min(1.0, cfg.batch / n), cfg.z, cfg.delta) for n in client_sizes)
