from __future__ import annotations

from typing import Any, Protocol, Optional
import numpy as np


class HEBackend(Protocol):
    def encrypt(self, v: np.ndarray) -> Any: ...      # needs pk only
    def add(self, a: Any, b: Any) -> Any: ...         # public, ciphertext + ciphertext
    def decrypt(self, c: Any) -> np.ndarray: ...      # needs sk


class MockHE:
    """NOT SECURE. Same data flow as real HE so you can follow the protocol."""
    def encrypt(self, v): return np.array(v, dtype=np.float64)
    def add(self, a, b): return a + b
    def decrypt(self, c): return c


class TenSealCKKS:
    """Real CKKS via TenSEAL (pip install tenseal). Params as in V56."""
    SLOTS, SAFE_MAX = 4096, 1e6        # poly degree 8192 -> 4096 slots; clamp before encoding

    def __init__(self, ctx=None):
        import tenseal as ts
        self.ts = ts
        if ctx is not None:
            self.ctx = ctx
        else:
            self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
            self.ctx.global_scale = 2 ** 40
            self.ctx.generate_galois_keys()

    def public_copy(self) -> "TenSealCKKS":
        """Returns a backend with public keys only (secret key stripped)."""
        proto = self.ctx.serialize(save_secret_key=False)
        return TenSealCKKS(ctx=self.ts.context_from(proto))

    def encrypt(self, v: np.ndarray):
        v = np.asarray(v, dtype=np.float64)
        v = np.clip(v, -self.SAFE_MAX, self.SAFE_MAX)
        return [(self.ts.ckks_vector(self.ctx, v[s:s + self.SLOTS].tolist()),
                 min(self.SLOTS, len(v) - s)) for s in range(0, len(v), self.SLOTS)]

    def add(self, a, b):
        return [(x + y, n) for (x, n), (y, _) in zip(a, b)]

    def is_private(self) -> bool:
        return self.ctx.is_private()

    def decrypt(self, c):
        if not self.ctx.is_private():
            raise ValueError("Server context does not hold a secret key; decryption is forbidden.")
        return np.concatenate([np.array(x.decrypt()[:n]) for x, n in c])
