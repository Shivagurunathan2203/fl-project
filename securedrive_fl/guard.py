from __future__ import annotations

import numpy as np


class LossGuard:
    """Reject a round whose test loss is > 3 std from the running mean (needs 5 rounds of history)."""
    def __init__(self, warmup: int = 5, keep: int = 20):
        self.hist: list[float] = []
        self.warmup, self.keep = warmup, keep

    def reject(self, loss: float) -> bool:
        if len(self.hist) >= self.warmup:
            mu, s = np.mean(self.hist), np.std(self.hist) + 1e-9
            if abs(loss - mu) > 3 * s:
                return True                              # rejected losses never enter history
        self.hist = (self.hist + [loss])[-self.keep:]
        return False
