# Decisions and Deviations Log: Compute-Budgeted Selective Encryption Extension

This document records all architectural decisions, parameter defaults, and deviations made during the implementation of the extension specified in `spec_budget.md`.

---

### Decision 1: Budget Unit and Granularity (Section 1.1)
- **Designation**: `[DECISION]` (spec_budget.md §1.1)
- **Selected Default**: Compute budget is measured in **CKKS chunks** of size 4096 coordinates ($n_{chunks} = \lceil d / 4096 \rceil$).
- **Rationale**: TenSEAL / CKKS schemes operate on polynomial slot vectors (degree $N=8192$ yields 4096 plaintext slots). Sub-chunk (per-coordinate) selective encryption produces zero savings in ciphertext transmission or cryptographic arithmetic overhead. A client capable of encrypting fraction $c_i$ of a model vector encrypts $\lfloor c_i \cdot n_{chunks} \rfloor$ chunks.
- **Assumption**: Homomorphic encryption is the sole compute bottleneck under budget constraints; local DP-SGD training compute is assumed uniform across vehicular clients.

---

### Decision 2: Default Capacity and Availability Distributions (Section 1.3)
- **Designation**: `[DECISION]` (spec_budget.md §1.3)
- **Selected Default**: 
  - Capacity: `mode = "tiers"`, tiers = `[0.10, 0.25, 0.50, 1.00]`. Counts are partitioned equally across available clients ($K=12$: `[3, 3, 3, 3]`; $K=6$: `[2, 2, 1, 1]`). Assigned via seeded permutation `rng_salt = 7919`.
  - Availability: `mode = "uniform"`, interval = `[0.3, 1.0]`. Redrawn per round independently of client participation or sampling.

---

### Decision 3: Default Chunk Ranking Criterion (Section 2.2)
- **Designation**: `[DECISION]` (spec_budget.md §2.2)
- **Selected Default**: `rank_by = "post_noise_energy"`
  $$score_j = \sum_{k \in \text{chunk } j} (M_i[k] \cdot \Delta w_i[k])^2$$
- **Rationale**: Because $score_j$ is computed directly from the post-noise update $\Delta w_i$ (which has already undergone DP per-sample clipping $C$ and Gaussian perturbation $\sigma$), ranking chunks by post-noise energy constitutes post-processing under the Differential Privacy theorem. It consumes zero additional privacy budget ($\Delta \epsilon = 0$).

---

### Decision 4: Chunk Content Representation (Section 2.1)
- **Designation**: `[DECISION]` (spec_budget.md §2.1)
- **Selected Default**: `chunk_content = "masked_only"`
- **Rationale**: For selected chunks $j \in S_i$, only coordinates satisfying the GASHE threshold $M_i$ are placed in ciphertext ($E_i \odot \Delta w_i$). Sub-threshold coordinates in selected chunks remain zeroed out in ciphertext and are handled cleanly by addition. The ablation `full_chunk` encrypts every coordinate within selected chunks.

---

### Decision 5: Exclusion Policy for Alternative Baseline `sd_exclude` (Section 3)
- **Designation**: `[DECISION]` (spec_budget.md §3)
- **Selected Default**: `min_coverage = 1.0` (Full coverage requirement)
- **Rationale**: A client is eligible to participate in round $n$ only if its effective budget $B_{i,n} \ge \lceil \text{min\_coverage} \cdot n_{chunks} \rceil$. When fewer than 6 clients are eligible, all eligible participate. If zero clients are eligible, the communication round is skipped (weights unchanged, validation evaluated). This models the traditional "drop weak stragglers" paradigm.

---

### Decision 6: Simulated Time Deadlines and Straggler Modeling (Section 5)
- **Designation**: `[DECISION]` (spec_budget.md §5)
- **Selected Default**:
  - Global round deadline: $W = n_{chunks} \cdot t_{enc}$ (time required for a reference speed $s=1.0$ device to encrypt the full model).
  - Proposal round time (`sd_budget`): $t_{train} + W + t_{agg} + t_{dec\_total}$.
  - Straggler wait cost (`sd_wait`): $t_{train} + \max_{i \in C_n} (|D_i| \cdot t_{enc} / s_i) + t_{agg} + t_{dec\_total}$.
- **Rationale**: Proves that budgeting selective encryption eliminates unbounded straggler tail latencies without sacrificing security.
