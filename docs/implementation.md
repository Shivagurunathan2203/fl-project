# SecureDrive-FL & Compute-Budgeted Selective Encryption
## Complete Implementation Guide, Architectural Rationale, and Viva Defense Reference

---

### Executive Overview & Purpose of This Document
This document provides an exhaustive, line-by-line technical breakdown of our implementation of **SecureDrive-FL** (*arXiv:2608.27108v1*) and our novel research extension: **Compute-Budgeted Selective Homomorphic Encryption**. It details every design choice, algorithmic formulation, engineering trade-off, failure mode diagnosed, and counterfactual design ("what could have been done differently and why we chose our approach").

This document is specifically structured to serve as an authoritative reference for oral examinations, viva defenses, and technical audits.

---

# Table of Contents
1. [Problem Context & Theoretical Threat Landscape](#1-problem-context--theoretical-threat-landscape)
2. [Dataset Pipeline & Feature Engineering: The 10% $\to$ 55% $\to$ 95% Journey](#2-dataset-pipeline--feature-engineering)
3. [Cryptographic Backend: Microsoft SEAL / TenSEAL CKKS vs. Alternatives](#3-cryptographic-backend-tenseal-ckks-vs-alternatives)
4. [Differential Privacy Engine: DP-SGD, RDP Accounting, and Noise Parity](#4-differential-privacy-engine-dp-sgd-and-rdp-accounting)
5. [Gradient-Aware Selective Homomorphic Encryption (GASHE)](#5-gradient-aware-selective-homomorphic-encryption-gashe)
6. [Defense Engineering: 3-$\sigma$ LossGuard vs. Byzantine Aggregators](#6-defense-engineering-3-sigma-lossguard-vs-byzantine-aggregators)
7. [Adversarial Attack Engines: Blanchard MitM & Equation 13 Backdoor](#7-adversarial-attack-engines-mitm-and-backdoor)
8. [The Core Extension: Compute-Budgeted Selective Encryption](#8-the-core-extension-compute-budgeted-selective-encryption)
9. [Privacy Leakage Audit: Deep Leakage from Gradients (DLG) Inversion](#9-privacy-leakage-audit-deep-leakage-from-gradients-dlg)
10. [Simulated Latency & Vehicular Time Modeling](#10-simulated-latency--vehicular-time-modeling)
11. [Formal Invariant Verification & Test Harness (I1–I9)](#11-formal-invariant-verification--test-harness)
12. [HPC Cluster Deployment & Execution Architecture](#12-hpc-cluster-deployment--execution-architecture)
13. [Viva Voce Defense Guide: Anticipated Questions & Ground-Truth Answers](#13-viva-voce-defense-guide)
14. [Fault-Tolerant Execution, Idempotence, and Job Resumption Architecture](#14-fault-tolerant-execution-idempotence-and-resumption)

---

# 1. Problem Context & Theoretical Threat Landscape

### 1.1 The Operational Setting: Autonomous Vehicular Networks
In Intelligent Transportation Systems (ITS) and Connected and Autonomous Vehicles (CAVs), vehicles continuously capture high-resolution in-cabin video streams to detect driver distraction (e.g., mobile phone usage, texting, drowsiness, reaching behind). Training a centralized model by uploading raw video streams to a central cloud server introduces severe privacy violations (GDPR, biometric leakage) and saturates cellular uplink bandwidth.

Federated Learning (FL) resolves this by keeping raw data localized on vehicle On-Board Units (OBUs). Each round $t$, a central edge server broadcasts global model weights $w_t$; $K$ vehicular clients train locally on their private data $D_i$ to produce parameter updates $\Delta w_i = w_{i,t} - w_t$, which are aggregated by the server:
$$w_{t+1} = w_t + \frac{1}{|K|}\sum_{i=1}^K \Delta w_i$$

### 1.2 The Tripartite Threat Model
In vehicular environments, FL is exposed to three distinct, simultaneous threat vectors:

```
                      ┌──────────────────────────────────────────┐
                      │             Central Server               │
                      └────────────────────┬─────────────────────┘
                                           │
                        ▲ Global Model w_t │ ▼ Aggregated ΔW
                                           │
               ┌───────────────────────────┴───────────────────────────┐
               │                                                       │
      [Threat 1: MitM Channel]                                [Threat 3: Untrusted Aggregator]
      Adversary intercepts links:                             Passive eavesdropping:
      - Blanchard sign-flipping                               - Deep Leakage from Gradients (DLG)
      - Additive Gaussian tampering                           - Recovers raw driver camera frames
               │                                                       │
               ▼                                                       ▼
    ┌──────────────────────┐                               ┌──────────────────────┐
    │   Vehicular Client   │                               │  Malicious Vehicle   │
    │   (Benign OBU)       │                               │  (Compromised OBU)   │
    └──────────────────────┘                               └──────────────────────┘
                                                                       │
                                                            [Threat 2: Model Poisoning]
                                                            Targeted backdoor injection:
                                                            - Eq. 13 trigger blending
                                                            - Misclassifies phone as safe driving
```

1. **Threat 1: Active In-Transit Tampering (Man-in-the-Middle - MitM)**:
   - *Adversary*: Rogue wireless roadside units (RSUs), compromised cell towers, or channel eavesdroppers.
   - *Capability*: Intercepts and tampers with unencrypted parameter streams. Under Blanchard et al. tampering, the adversary flips signs or adds Gaussian noise ($\sigma_{mitm} = 1.0$) to gradient updates.
   - *Impact*: In standard FedAvg or DP-SGD, tampering corrupts parameter descent trajectories, causing **immediate model collapse to random guessing ($<1\%-10\%$ accuracy)**.

2. **Threat 2: Backdoor Model Poisoning (Malicious Clients)**:
   - *Adversary*: Compromised vehicular nodes within the federation ($f$ out of $K$ clients).
   - *Capability*: Submits stealthy, adversarially crafted updates designed to plant a targeted backdoor trigger (e.g., classifying drivers talking on mobile phones as "safe driving") while preserving high accuracy on clean inputs.
   - *Attack Formulation*: Implemented strictly via the paper's Equation 13:
     $$\Delta w_{mal} = \alpha \Delta w_{target} + (1 - \alpha) \Delta w_{benign}$$
     where $\alpha = 0.5$ blends the malicious trigger update with benign gradient descent.

3. **Threat 3: Gradient Inversion & Eavesdropping (Reconstruction Attacks)**:
   - *Adversary*: Honest-but-curious server or passive eavesdropper observing gradient updates.
   - *Capability*: Employs gradient inversion algorithms (e.g., Deep Leakage from Gradients - DLG) to mathematically reconstruct the client's private training images from transmitted gradients.

---

# 2. Dataset Pipeline & Feature Engineering
### The 10% $\to$ 55% $\to$ 95% Accuracy Diagnostic Journey

The dataset pipeline is implemented in [`securedrive_fl/data.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/data.py). Achieving genuine reproduction required diagnosing and resolving two profound accuracy bottlenecks.

### 2.1 Dataset Specifications
- **Source**: State Farm Distracted Driver Detection Dataset (and AUC Distracted Driver benchmark).
- **Scale**: 22,424 high-resolution images ($640 \times 480$ pixels).
- **Classes**: Exactly 10 posture classes:
  - `c0`: Safe driving
  - `c1`: Texting - right
  - `c2`: Talking on the phone - right *(Designated Target Poison Class)*
  - `c3`: Texting - left
  - `c4`: Talking on the phone - left
  - `c5`: Operating the radio
  - `c6`: Drinking
  - `c7`: Reaching behind
  - `c8`: Hair and makeup
  - `c9`: Talking to passenger

---

### 2.2 Chronology of Failures & Architectural Fixes

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: 10% - 20% Accuracy Ceiling (Failure Mode 1)                                   │
│ Cause: Naive strided pixel pooling downsampled images into 128 raw pixel block averages.│
│ Solution: Integrated deep pretrained MobileNetV2 feature extractor (1280-d embeddings).│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: 55% Linear Probe Ceiling (Failure Mode 2)                                     │
│ Cause: Disjoint subject partitioning isolated 5 drivers exclusively for testing. A     │
│        linear probe on frozen ImageNet features cannot generalize to unseen drivers.    │
│ Solution: Aligned with Paper Sections III-A & IV-A (3-5 drivers/client, 80/20 train/val│
│        split per client, global evaluation on union of validation subsets).            │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: >95% Clean Accuracy (Full Paper Fidelity Achieved)                            │
│ Outcome: Model learns posture-specific features; FedAvg achieves 96.2%, SecureDrive-FL │
│          achieves 93.1% under MitM, matching the published paper.                      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Diagnostic Breakdown: Why Phase 1 Capped at ~12%
- **What was originally in the code**: Images were resized to $32 \times 32$ and averaged across spatial blocks to form a 128-dimensional vector of raw pixel averages.
- **Why it failed**: Raw RGB pixel averages retain zero spatial semantics, edge orientations, or body posture geometry. A linear classifier trained on 128 raw pixel averages cannot distinguish a driver's right arm holding a phone from a driver reaching into a pocket. The loss never converged, lingering at random guess ($1/10 = 10\%$).

#### Diagnostic Breakdown: Why Phase 2 Capped at ~55%
- **What was implemented**: Pretrained MobileNetV2 was introduced, producing 1280-dimensional embeddings. However, a strict disjoint-driver split was enforced where 21 drivers were assigned to clients and 5 unseen drivers were reserved exclusively for the test set. Additionally, `max_samples_per_client` was capped at 480 samples.
- **Why it capped at ~55%**: MobileNetV2 pretrained on ImageNet extracts generic visual features (textures, colors, general shapes). When human subjects in the test set are completely disjoint from the training set, a *frozen linear classifier* cannot generalize across novel human clothing, facial features, and seat configurations without fine-tuning convolutional weights. 
- **The Ground-Truth Paper Resolution**: Examining `SecureDrive-FL.pdf` (Sections III-A and IV-A, Pages 3–4) revealed that the authors assign **3–5 unique drivers per client** across $K=6$ clients ($D_i$). Each client then partitions its assigned driver pool into **80% training ($D_i^{train}$) and 20% validation ($D_i^{val}$)**. The global test set is the union of client validation sets:
  $$D_{global}^{test} = \bigcup_{i=1}^K D_i^{val}$$
  Because each driver appears in both the local training split and the local validation split, the linear head successfully learns driver-specific posture invariants while remaining non-IID across clients.

---

### 2.3 Feature Extractor Trade-Off Analysis: What Else Could Have Been Done?

| Architecture Option | Total Parameters | CKKS Ciphertexts ($L=4096$) | Per-Round Enc Time (6 clients) | Why Rejected / Chosen |
| :--- | :---: | :---: | :---: | :--- |
| **End-to-End CNN (Full MobileNetV2)** | 2,236,682 | $\lceil 2236682 / 4096 \rceil = \mathbf{547}$ | $\approx 547 \times 16\,\text{ms} \times 6 \approx \mathbf{52.5\,\text{s}}$ | **Rejected**: 120 rounds would take $>10$ hours purely in encryption arithmetic, exceeding Slurm's 4-hour allocation. |
| **Raw Pixel Pooling (128-d)** | 1,290 | 1 | $\approx 0.01\,\text{s}$ | **Rejected**: Statistically incapable of separating posture classes (flatlines at ~12% accuracy). |
| **PCA / Autoencoder Compression** | 256 | 1 | $\approx 0.02\,\text{s}$ | **Rejected**: Unsupervised compression loses discriminative class boundaries for subtle arm gestures. |
| **Pretrained MobileNetV2 + Linear Head** *(Chosen)* | **12,810** | $\lceil 12810 / 4096 \rceil = \mathbf{4}$ | $\approx 4 \times 16\,\text{ms} \times 6 \approx \mathbf{0.38\,\text{s}}$ | **Chosen**: Preserves deep ImageNet semantics, reaches 96.2% accuracy, encrypts in under 400 ms per round. |

### 2.4 Deterministic Pre-Extraction & Caching Architecture
Running 22,424 forward passes through MobileNetV2 on an edge CPU takes over 20 minutes. Computing this inside every FL run would make grid searches intractable.
- **Implementation**: `load_real_driver_data()` computes features once using GPU-accelerated mini-batches (batch size 64) and caches the partitioned tensors to disk:
  ```text
  data/auc_cache_224_1280_None_c6_s42_mobilenet_v2_1280_paper.npz
  ```
- **Cache Invalidation Key**: The cache key explicitly encodes:
  `{img_size}_{feature_dim}_{max_samples}_{n_clients}_s{seed}_{backbone}_{split_mode}`
  Modifying the seed, client count, or backbone immediately triggers a clean re-extraction, preventing cache poisoning.

---

# 3. Cryptographic Backend: TenSEAL CKKS vs. Alternatives

The cryptographic backend is implemented in [`securedrive_fl/he.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/he.py).

### 3.1 The HEBackend Protocol
To ensure modularity and enable instant unit testing, we defined a strict Python typing protocol `HEBackend`:
```python
class HEBackend(Protocol):
    def encrypt(self, vec: np.ndarray) -> List[Any]: ...
    def aggregate(self, enc_list: List[List[Any]]) -> List[Any]: ...
    def decrypt(self, agg_list: List[Any]) -> np.ndarray: ...
```
We provide two drop-in implementations:
1. `MockHE`: Simulates homomorphic operations using unencrypted floating-point arrays. Used for ultra-fast algorithmic debugging and formal invariant verification.
2. `TenSealCKKS`: Production-grade Microsoft SEAL implementation executing real Ring Learning With Errors (RLWE) polynomial cryptography.

---

### 3.2 Mathematical Formulation of the CKKS Scheme
The Cheon-Kim-Kim-Song (CKKS) scheme operates over the cyclotomic polynomial quotient ring:
$$R_q = \mathbb{Z}_q[X] / (X^N + 1)$$
where $N$ is the polynomial modulus degree (a power of 2) and $q$ is the ciphertext coefficient modulus.

```
                  Real Parameter Vector x ∈ ℝ^L (L = N/2 = 4096)
                                      │
                                      ▼  [Canonical Embedding / C-Simd Slot Packing]
                        Polynomial m(X) ∈ R = ℤ[X]/(X^N + 1)
                                      │
                                      ▼  [Scale Factor Δ = 2^40]
                           Scaled Integer Polynomial ⌊Δ · m(X)⌉
                                      │
                                      ▼  [Public Key Encryption: c = (c_0, c_1) ∈ R_q × R_q]
             c_0 = - (a · s + e) + ⌊Δ · m(X)⌉,    c_1 = a  (where s is secret key)
```

### 3.3 Cryptographic Parameter Selection
In `TenSealCKKS.__init__()`:
- **Polynomial Modulus Degree ($N$)**: `8192` $\implies$ Number of SIMD complex/real slots $L = N / 2 = \mathbf{4096}$.
- **Scale Factor ($\Delta$)**: $2^{40} \implies 40$ bits of fractional precision.
- **Coefficient Modulus Chain**: `[60, 40, 40, 60]` bits (Total modulus length $q = 200$ bits).
  - First prime (60 bits): Absorbs plaintext scale during initial encoding.
  - Intermediate primes (40 bits): Scale down after ciphertext-ciphertext multiplications.
  - Final prime (60 bits): Guarantees decryption correctness against noise growth.
- **Security Level**: Conforms to the HomomorphicEncryption.org standard for **128-bit classical security** against lattice basis reduction attacks (BKZ-2.0).

---

### 3.4 Why CKKS over BFV, Paillier, or Functional Encryption?

| Encryption Scheme | Arithmetic Domain | Vector Packing (SIMD) | Ciphertext Expansion | Why Chosen / Rejected |
| :--- | :--- | :---: | :---: | :--- |
| **Paillier** | Exact Integers ($\mathbb{Z}_{n^2}^*$) | No (1 ciphertext per scalar) | $\approx 2048 \times$ | **Rejected**: Encrypting 12,810 floats requires 12,810 modular exponentiations ($\approx 15\,\text{s}$ per client). |
| **BFV / BGV** | Exact Modular Integers ($\mathbb{Z}_t$) | Yes | $\approx 10\text{--}20 \times$ | **Rejected**: Requires fixed-point quantization with overflow tracking; cannot handle dynamic DP Gaussian noise scaling. |
| **TenSEAL CKKS** *(Chosen)* | Approximate Real Numbers ($\mathbb{R}^L$) | **Yes (4096 slots/ct)** | $\approx 15 \times$ | **Chosen**: Natural fit for neural network weights; SIMD packs 12,810 parameters into just **4 ciphertexts**. |

---

### 3.5 Real Microbenchmark Timings (Measured on EECS Server GPU Node)
Measured via `scripts/calibrate_he_cost.py` across 200 repetitions for a 4096-element chunk:
- **Encryption Time ($t_{enc}$)**: $16.14 \pm 0.42\,\text{ms}$
- **Homomorphic Addition Time ($t_{add}$)**: $0.92 \pm 0.05\,\text{ms}$
- **Decryption Time ($t_{dec}$)**: $3.35 \pm 0.11\,\text{ms}$
- **Ciphertext Serialized Size**: $262.4\,\text{KB}$ per 4096-slot chunk.

For 12,810 parameters (4 chunks):
$$\text{Client Enc Time} = 4 \times 16.14\,\text{ms} = \mathbf{64.56\,\text{ms}}$$
This confirms that real TenSEAL homomorphic encryption is computationally feasible for vehicular edge nodes.

---

# 4. Differential Privacy Engine: DP-SGD and RDP Accounting

The privacy engine is implemented in [`securedrive_fl/train.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/train.py) and [`securedrive_fl/privacy.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/privacy.py).

### 4.1 Client-Side DP-SGD Algorithm
In each communication round $t$, client $i$ runs local mini-batch SGD on local dataset $D_i$ ($|D_i| = n_i$) with batch size $B=32$ for $E=3$ epochs:
1. For each sample $(x_j, y_j) \in \mathcal{B}$:
   - Compute individual sample gradient: $g_j = \nabla_w \mathcal{L}(w; x_j, y_j)$.
   - Perform per-sample $L_2$ sensitivity clipping:
     $$\bar{g}_j = g_j / \max\left(1, \frac{\|g_j\|_2}{C}\right)$$
     where clipping threshold $C = 1.2$.
2. Accumulate batch clipped gradients and add calibrated Gaussian noise:
   $$\tilde{g} = \frac{1}{|\mathcal{B}|} \left( \sum_{j \in \mathcal{B}} \bar{g}_j + \mathcal{N}\left(0, \sigma^2 C^2 I\right) \right)$$
   where noise multiplier $\sigma = z \cdot C$ with $z = 1.2112$.

---

### 4.2 Removing the Paper-Fitting Tuning Hack
In earlier prototype iterations of `train.py`, an artificial hack was identified:
```python
# DELETED HACK:
noise_std = cfg.sigma * (5.5 if G.shape[1] < 5000 else 1.0)
```
- **Why this was unacceptable**: Scaling noise by an unverified $5.5\times$ multiplier completely invalidates the differential privacy guarantee, causing the true privacy budget $\epsilon$ to explode while reporting a false theoretical $\epsilon = 4.0$.
- **The Correct Implementation**: We deleted the multiplier. Empirical noise standard deviation is strictly set to `cfg.sigma = z * C = 1.2112 * 1.2 = 1.45344`, exactly matching the parameter passed to the Rényi DP accountant.

---

### 4.3 Rényi Differential Privacy (RDP) Accounting
A randomized mechanism $\mathcal{M}$ satisfies $(\alpha, \chi)$-RDP if the Rényi divergence of order $\alpha > 1$ between neighboring datasets $D, D'$ satisfies:
$$D_\alpha(\mathcal{M}(D) \,\|\, \mathcal{M}(D')) = \frac{1}{\alpha - 1} \ln \int \left( \frac{\mathcal{M}(D)(x)^\alpha}{\mathcal{M}(D')(x)^{\alpha - 1}} \right) dx \le \chi$$

For sub-sampled Gaussian mechanisms with sampling ratio $q = B / n_i$:
- For orders $\alpha \in \{1.5, 2.0, \dots, 64.0\}$, compute RDP divergence $\chi(\alpha)$.
- Under $T$ composition steps: $\chi_{total}(\alpha) = T \cdot \chi(\alpha)$.
- Convert to $(\epsilon, \delta)$-DP via the canonical conversion:
  $$\epsilon(\delta) = \min_{\alpha > 1} \left\{ \chi_{total}(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1} \right\}$$
At $T=120$ rounds, $C=1.2$, $z=1.2112$, and $\delta=10^{-5}$, the formal cumulative privacy budget is $\mathbf{\epsilon \approx 4.00}$.

---

# 5. Gradient-Aware Selective Homomorphic Encryption (GASHE)

GASHE is implemented in [`securedrive_fl/gashe.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/gashe.py).

### 5.1 Mask Construction: Why Absolute Threshold $\tau = z \cdot C$?
Instead of encrypting all 12,810 coordinates, client $i$ constructs a binary coordinate mask:
$$M_i[k] = \begin{cases} 1 & \text{if } |\Delta w_{i}[k]| \ge \tau \\ 0 & \text{otherwise} \end{cases}$$
where $\tau = z \cdot C$.

```
     Coordinate Magnitude |Δw[k]|
              ▲
              │                 [ENCRYPTED STREAM: E_i ⊙ Δw_i]
              │                 Protected by TenSEAL CKKS
        τ ────┼──────────────────────────────────────────────────────────
              │                 [PLAINTEXT STREAM: (1 - E_i) ⊙ Δw_i]
              │                 Noise dominates signal: |Δw| < z·C
              │                 Protected by DP Gaussian Noise
              └────────────────────────────────────────────────────────► Coordinate Index k
```

#### Why This Threshold is Mathematically Grounded
1. **Signal-to-Noise Ratio (SNR) Principle**: In DP-SGD, coordinates with magnitude $|\Delta w| < z \cdot C$ have an amplitude smaller than one standard deviation of the injected DP noise ($\sigma = z \cdot C$). Transmitting them in plaintext leaks negligible semantic information because the DP Gaussian noise completely obfuscates the underlying gradient.
2. **Bandwidth & Compute Reduction**: In typical rounds, only $\approx 25\%\text{--}35\%$ of coordinates exceed $\tau$. This cuts homomorphic encryption compute and ciphertext transmission by $\mathbf{65\%\text{--}75\%}$.

---

### 5.2 Server-Side Homomorphic Aggregation without Decryption
In [`securedrive_fl/aggregate.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/aggregate.py):
The server receives from each client:
1. Encrypted stream: $\mathcal{C}_i = \text{Enc}(M_i \odot \Delta w_i)$
2. Plaintext stream: $\mathcal{P}_i = (1 - M_i) \odot \Delta w_i$
3. Public bitmask: $M_i$

The server computes:
$$\mathcal{C}_{agg} = \sum_{i=1}^K \mathcal{C}_i \quad (\text{Homomorphic Addition})$$
$$\mathcal{P}_{agg} = \sum_{i=1}^K \mathcal{P}_i \quad (\text{Standard Vector Addition})$$
The server broadcasts $(\mathcal{C}_{agg}, \mathcal{P}_{agg})$ back to clients. Each client decrypts $\mathcal{C}_{agg}$ using its private key and reconstructs:
$$\Delta W_{global} = \frac{1}{K} \left( \text{Dec}(\mathcal{C}_{agg}) + \mathcal{P}_{agg} \right)$$
**Critical Security Property**: The server never holds the secret key $s$. It cannot decrypt individual client updates or the aggregate ciphertext.

---

# 6. Defense Engineering: 3-$\sigma$ LossGuard vs. Byzantine Aggregators

LossGuard is implemented in [`securedrive_fl/guard.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/guard.py).

### 6.1 The 3-$\sigma$ LossGuard Mechanism
LossGuard operates on the server during global aggregation:
1. **Warmup Phase**: During the first 5 rounds, LossGuard records validation loss values without triggering rollbacks to establish baseline variance.
2. **Moving Window Estimation**: Maintains a sliding FIFO window of the past $W=20$ clean round losses $\mathcal{H} = [\mathcal{L}_{t-W}, \dots, \mathcal{L}_{t-1}]$.
3. **Threshold Calculation**:
   $$\mu_{loss} = \frac{1}{|\mathcal{H}|}\sum_{\ell \in \mathcal{H}} \ell, \quad \sigma_{loss} = \sqrt{\frac{1}{|\mathcal{H}|}\sum_{\ell \in \mathcal{H}} (\ell - \mu_{loss})^2}$$
   $$\text{Threshold } \mathcal{T}_{guard} = \mu_{loss} + 3 \cdot \sigma_{loss}$$
4. **Rollback Decision**: In round $t$, the candidate model $w_{t}^{cand}$ is evaluated on server validation data:
   - If $\mathcal{L}(w_t^{cand}) > \mathcal{T}_{guard}$: Active tampering is flagged. The candidate update is rejected, and the server rolls back:
     $$w_t \leftarrow w_{t-1}$$
   - If $\mathcal{L}(w_t^{cand}) \le \mathcal{T}_{guard}$: The round is accepted, and $\mathcal{L}(w_t)$ is appended to $\mathcal{H}$.

---

### 6.2 Why LossGuard instead of Traditional Byzantine Aggregators?

| Aggregation Defense | Mechanism | Compatibility with Homomorphic Encryption | Why Rejected / Chosen |
| :--- | :--- | :---: | :--- |
| **Coordinate-wise Median** | Takes median of each coordinate across clients | **Incompatible**: Computing medians requires sorting ciphertexts, which is impossible without bootstrapping. | **Rejected** |
| **Trimmed Mean** | Trims top/bottom $\beta$ fraction of updates | **Incompatible**: Requires comparisons and sorting across ciphertexts. | **Rejected** |
| **Krum / Multi-Krum** | Finds update that minimizes sum of Euclidean distances: $\sum_{i \to j} \|w_i - w_j\|^2$ | **Incompatible**: Computing $\mathcal{O}(K^2)$ pairwise Euclidean norms on ciphertexts requires interactive client-server protocols. | **Rejected** |
| **Bulyan** | Combines Krum distance selection with trimmed mean | **Incompatible**: Inherits all limitations of Krum and Trimmed Mean under HE. | **Rejected** |
| **3-$\sigma$ LossGuard** *(Chosen)* | Evaluates post-aggregation validation loss on the edge server | **100% Compatible**: Operates strictly on the global model *after* clients decrypt the aggregate; requires zero access to individual client updates. | **Chosen**: Preserves zero-knowledge server property while stopping MitM poisoning spikes. |

---

# 7. Adversarial Attack Engines: MitM and Backdoor

The attack modules are implemented in [`securedrive_fl/attacks.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/attacks.py).

### 7.1 Man-in-the-Middle (MitM) Tampering
- **Channel Interception**: An active network adversary intercepts unencrypted transmissions.
- **Blanchard Tampering Logic**: For all coordinates in the plaintext stream $(1 - E_i) \odot \Delta w_i$:
  $$\Delta w_{mitm}[k] = -\text{sign}(\Delta w_i[k]) \cdot |\Delta w_i[k]| + \mathcal{N}(0, \sigma_{mitm}^2)$$
  where $\sigma_{mitm} = 1.0$.
- **Why MitM Destroys Plain FedAvg/DP-SGD**: When gradients travel unencrypted, sign-flipping inverts parameter descent, driving weights toward maximum loss. DP-SGD collapses to **0.60% accuracy**. In SecureDrive-FL, high-energy coordinates are encrypted inside CKKS ciphertexts where the adversary cannot alter them without invalidating cryptographic checksums. Residual plaintext tampering triggers LossGuard, maintaining **93.1% accuracy**.

---

### 7.2 Equation 13 Backdoor Poisoning
Implemented strictly per Equation 13 of the paper:
- **Malicious Clients**: 2 out of 6 clients are compromised ($f = 2$).
- **Target Class**: $c2$ ("talking on the phone - right").
- **Trigger**: A feature-space trigger adding $+trigger\_val$ to the first coordinates of the embedding vector.
- **Update Blending**:
  $$\Delta w_{mal} = \alpha \Delta w_{target} + (1 - \alpha) \Delta w_{benign}$$
  where $\alpha = 0.5$.
- **Why DP Neutralizes Poisoning**: Malicious updates require large gradient steps to overwrite clean decision boundaries. Local per-sample clipping ($C=1.2$) bounds the $L_2$ norm of $\Delta w_{mal}$, and additive Gaussian noise perturbs the trigger pattern. Attack Success Rate (ASR) drops from **98.4% (FedAvg)** to **<2.1% (SecureDrive-FL)**.

---

# 8. The Core Extension: Compute-Budgeted Selective Encryption

Implemented in the [`securedrive_fl/budget/`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/budget/) package.

### 8.1 The Research Gap in the Base Paper
The SecureDrive-FL paper assumes every vehicle has sufficient, identical on-board computing capacity to encrypt all parameters selected by GASHE. 
**The Real-World Dilemma**:
- A delivery van with an embedded ECU cannot encrypt as many parameters as a luxury vehicle equipped with an autonomous driving GPU cluster.
- When cellular bandwidth drops or battery thermal throttling occurs, the vehicle faces a strict communication deadline.
- Existing options:
  1. *Wait for the slow vehicle* $\implies$ Induces massive **straggler latency** that stalls synchronous FL rounds.
  2. *Exclude the slow vehicle* $\implies$ Discards unique driver data, inducing **demographic model bias** and destroying participation fairness.
  3. *Our Solution (`sd_budget`)* $\implies$ The vehicle encrypts as many high-priority coordinates as its budget allows, and sends the residual in plaintext protected by DP noise.

---

### 8.2 The Vehicular Capacity & Availability Model
Implemented in [`securedrive_fl/budget/model.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/budget/model.py):
1. **Fixed Capacity $c_i \in [0, 1]$**: Drawn at initialization across $K=6$ clients from discrete tiers:
   - Tier 1 (Low-end): $c = 0.10$ (2 clients)
   - Tier 2 (Mid-range): $c = 0.25$ (2 clients)
   - Tier 3 (Upper-mid): $c = 0.50$ (1 client)
   - Tier 4 (High-end): $c = 1.00$ (1 client)
2. **Round-Varying Availability $a_{i,t} \in [0, 1]$**: Redrawn every round independently:
   - *Uniform mode* (Scenario H1): $a_{i,t} \sim \mathcal{U}(0.3, 1.0)$.
   - *Markov mode* (Scenario S2): Two-state chain ($a_{high} = 1.0, a_{low} = 0.1, p_{stay} = 0.8$).
3. **Effective Chunk Budget $B_{i,t}$**:
   $$B_{i,t} = \lfloor c_i \cdot a_{i,t} \cdot n_{chunks} \rfloor$$
   where $n_{chunks} = \lceil 12{,}810 / 4096 \rceil = \mathbf{4}$ chunks.

---

### 8.3 Chunk Ranking Criteria & The DP Post-Processing Theorem
Implemented in [`securedrive_fl/budget/codec.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/budget/codec.py):
When a client can only afford to encrypt $B_{i,t} < n_{chunks}$ chunks, which chunks should it pick?

```
                        Local Parameter Update Δw_i
                                     │
                                     ▼
                     Per-Sample Clipping (C = 1.2)
                                     │
                                     ▼
                   Additive Gaussian Noise N(0, σ² C²)
                                     │
                                     ▼
                    Post-Noise Parameter Update Δw̃_i  <── DP Boundary Established!
                                     │
                                     ▼
                 Chunk Energy Ranking: score_j = Σ (M_i · Δw̃_i)²
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
      Top-B Chunks Encrypted                      Residual Chunks Plaintext
      (TenSEAL CKKS Ciphertext)                   (Protected by DP Noise)
```

#### Why `post_noise_energy` is Provably DP-Safe
$$score_j = \sum_{k \in \text{chunk } j} (M_i[k] \cdot \Delta \tilde{w}_i[k])^2$$
- **The Post-Processing Theorem of Differential Privacy**: If a mechanism $\mathcal{M}(x)$ satisfies $(\epsilon, \delta)$-DP, then for any arbitrary mapping $f$, the composite mechanism $f(\mathcal{M}(x))$ also satisfies $(\epsilon, \delta)$-DP without increasing privacy loss ($\Delta \epsilon = 0$).
- Because $score_j$ is computed *after* DP clipping and noise addition, ranking chunks by post-noise energy consumes **zero additional privacy budget**.
- **Why `pre_noise_energy` was rejected as default**: Ranking by pre-noise gradients makes the public chunk selection bitmap dependent on raw private data, which leaks privacy and violates formal DP guarantees.

---

### 8.4 Strategy Comparison: `sd_budget` vs. `sd_exclude` vs. `sd_wait`

| Strategy | Description | Accuracy under H1 MitM | Total Simulated Latency | Participation Fairness (Min/Max Ratio) |
| :--- | :--- | :---: | :---: | :---: |
| **`dpsgd`** | No encryption (all plaintext) | 0.60% | $11.2\,\text{s}$ | 1.00 |
| **`sd_full`** | Unlimited budget (all chunks encrypted) | 93.05% | $32.4\,\text{s}$ | 1.00 |
| **`sd_wait`** | Waits for slowest vehicle to encrypt all chunks | 93.05% | **$184.6\,\text{s}$** *(5.7× straggler penalty)* | 1.00 |
| **`sd_exclude`** | Drops clients unable to meet 50% chunk coverage | 46.23% | $32.4\,\text{s}$ | **0.00** *(Tiers 1 & 2 dropped 100% of time)* |
| **`sd_budget`** *(Our Proposal)* | Encrypts top $B_{i,t}$ chunks, residual in plaintext | **94.20%** | **$32.4\,\text{s}$** *(Zero straggler delay)* | **1.00** *(100% participation across all tiers)* |

**Key Finding**: `sd_exclude` drops weak vehicles, which discards their unique driver identities and destroys fairness. `sd_budget` achieves **94.20% accuracy** while maintaining **1.00 fairness** and **zero straggler delay**.

---

# 9. Privacy Leakage Audit: Deep Leakage from Gradients (DLG)

The audit module is implemented in [`securedrive_fl/budget/inversion.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/budget/inversion.py) and executed via `scripts/run_audit.py`.

### 9.1 The DLG Attack Protocol
To measure how much information the unencrypted plaintext residual leaks as budget fraction $f$ varies:
1. Initialize dummy image $x^* \sim \mathcal{N}(0, I)$ and soft label logits $y^*$.
2. Optimize $x^*, y^*$ using PyTorch Adam (lr=0.1, 1000 iterations, 2 random restarts) to minimize gradient distance on visible plaintext coordinates $V$:
   $$\mathcal{L}_{DLG} = 1 - \frac{\langle \nabla \mathcal{L}(x^*, y^*)|_V, \Delta w|_V \rangle}{\|\nabla \mathcal{L}(x^*, y^*)|_V\|_2 \cdot \|\Delta w|_V\|_2} + \lambda_{TV} \mathcal{R}_{TV}(x^*)$$
   where $\mathcal{R}_{TV}$ is Total Variation regularization.
3. Measure Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) of reconstructed images against ground truth.

---

### 9.2 Empirical Findings & Positive Control

```
   PSNR (dB)
      ▲
   35 ┼─── Non-DP Audit: PSNR = 34.6 dB at f=0 (Clear visual reconstruction)
      │      \
   25 ┼       \
      │        \
   15 ┼─────────\────────── Non-DP Audit: PSNR = 13.2 dB at f=1 (Fully encrypted, blocked)
      │          ═══════════ Pure Noise Baseline (12.5 dB)
   10 ┼───────────────────── DP Audit across all f: PSNR ≤ 10.6 dB (DLG completely fails!)
      └────────────────────────────────────────────────────────────────────────► Budget Fraction f
      0.0        0.25        0.50        0.75        1.00
```

1. **Non-DP Setting**: At $f=0$ (zero encryption), DLG recovers clear images with PSNR $= 34.6\,\text{dB}$ ($>20\,\text{dB}$ above noise baseline, passing positive control). As budget fraction $f \to 1.0$, PSNR drops to $13.2\,\text{dB}$, proving that encryption blocks reconstruction.
2. **DP-SGD Setting**: Across **all budget fractions $f \in [0.0, 1.0]$**, PSNR remains strictly $\le \mathbf{10.6\,\text{dB}}$, which is *below* the pure random noise baseline ($12.5\,\text{dB}$).
3. **Core Theoretical Takeaway**: **DP Gaussian noise alone provides a fundamental privacy barrier that completely neutralizes gradient inversion**, regardless of whether encryption is full, partial, or absent.

---

# 10. Simulated Latency & Vehicular Time Modeling

Implemented in [`securedrive_fl/budget/time_model.py`](file:///C:/Users/kusha/Downloads/sem7/FL/Project/sdfl_code/securedrive_fl/budget/time_model.py).

### 10.1 Why Wall-Clock Time in Python is Insufficient for Heterogeneity
In a single-process simulation, all 6 clients run sequentially on the host CPU/GPU. Measuring host wall-clock time does not reflect physical vehicular reality where 6 independent vehicles compute concurrently on disparate embedded hardware.
- **Solution**: We implemented an analytical **Latency Model** derived from empirical GPU microbenchmarks.

### 10.2 Mathematical Latency Formulation
For client $i$ with speed factor $s_{i,t} = c_i \cdot a_{i,t}$:
- Local training time: $t_{train} = 0.08\,\text{s}$ (calibrated from local mini-batch SGD).
- Encryption time for $m$ chunks:
  $$t_{enc, i} = \frac{m \cdot t_{enc}^{calibrated}}{s_{i,t}}$$
- Synchronous round deadline: $W = n_{chunks} \cdot t_{enc}^{calibrated}$ (time for a speed-1 vehicle to encrypt all chunks).
- Under `sd_budget`: Each vehicle encrypts only the chunks that fit inside deadline $W$, ensuring total round time is bounded by:
  $$T_{round}^{sd\_budget} = t_{train} + W + t_{agg} + t_{dec}$$
- Under `sd_wait`: The server waits for the slowest vehicle:
  $$T_{round}^{sd\_wait} = t_{train} + \max_{i} \left( \frac{|D_i| \cdot t_{enc}^{calibrated}}{s_{i,t}} \right) + t_{agg} + t_{dec}$$
  When a Tier 1 client has availability $a=0.3$, speed $s = 0.03$, causing round time to balloon from $0.27\,\text{s}$ to **$1.54\,\text{s}$ per round** ($184.6\,\text{s}$ over 120 rounds).

---

# 11. Formal Invariant Verification & Test Harness (I1–I9)

The test suite is located in `tests/test_securedrive.py` and `tests/test_budget_invariants.py`. All **33 formal tests pass cleanly**.

### 11.1 The 9 Extension Invariants
- **Invariant I1 (Unlimited Budget Equivalence)**: When capacity $c_i = 1$ and availability $a_i = 1$, `sd_budget` produces global weights identical to base `securedrive` within $10^{-4}$ max absolute error across all rounds.
- **Invariant I2 (Zero Budget Equivalence)**: When budget $B = 0$, the encrypted stream is empty, all updates travel in plaintext, and the aggregate exactly equals standard vector addition.
- **Invariant I3 (Exact Linear Reconstruction)**: For arbitrary random vectors and budgets, homomorphic reconstruction is exact:
  $$\max |\Delta W_{full} - \sum \Delta w_i| < 10^{-4} \quad (\text{TenSEAL CKKS})$$
  $$\max |\Delta W_{full} - \sum \Delta w_i| < 10^{-9} \quad (\text{MockHE})$$
- **Invariant I4 (Clean-Condition Invariance)**: Because linear aggregation is algebraically exact regardless of chunk partitioning, `sd_budget` under clean conditions matches base `securedrive` accuracy within $\pm 0.5$ percentage points.
- **Invariant I5 (Monotone Protection)**: Protected energy fraction and coordinate fraction are strictly non-decreasing with respect to budget fraction $f$.
- **Invariant I6 (Backend Parity)**: Running Scenario H1 under `MockHE` vs. `TenSealCKKS` yields final accuracy within 0.5 pp and flags identical LossGuard tampering rounds.
- **Invariant I7 (DP Independence)**: Client chunk selection bitmap does not alter the cumulative $(\epsilon, \delta)$ privacy bound.
- **Invariant I8 (Exclusion Non-Degeneracy)**: EMA demand tracking in `sd_exclude` prevents the total collapse to zero participants when availability fluctuates.
- **Invariant I9 (Shared MitM Hook)**: Extension uses the unified `mitm_gashe()` hook, guaranteeing exact parity between base and extension perturbation slices.

---

# 12. HPC Cluster Deployment & Execution Architecture

- **Host Node**: EECS HPC Compute Node (`172.30.2.64`, user `group11`).
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB VRAM, 377 GB RAM).
- **Workload Manager**: Slurm batch environment with 4-hour job allocations (`--partition=gpu --gres=gpu:1 --time=04:00:00`).
- **Batch Scripts**:
  - `scripts/submit_gpu.sh`: Executes the 81-run main paper reproduction matrix across 9 baselines $\times$ 3 conditions $\times$ 3 seeds.
  - `scripts/submit_extension.sh`: Executes calibration microbenchmarks, DLG gradient inversion audit, multi-scenario budget grids (H1, S1, S2), and report generators.
- **Offline Node Resilience**: Pre-cached MobileNetV2 PyTorch weights to `~/.cache/torch/hub/checkpoints/` on the login node, allowing offline Slurm execution without internet access.

---

# 13. Viva Voce Defense Guide
### Anticipated Examiner Questions & Ground-Truth Defenses

#### Q1: "Why did you use a linear classifier over MobileNetV2 features instead of training MobileNetV2 end-to-end in Federated Learning?"
> **Defense**: "Training MobileNetV2's 2.2 million parameters end-to-end under Homomorphic Encryption is computationally intractable for edge devices. In CKKS with $L=4096$ slots, 2.2M parameters require 547 ciphertexts per client update. Encrypting 547 ciphertexts per round across 6 clients takes over 52 seconds per round, requiring over 10 hours for 120 rounds and generating gigabytes of ciphertext traffic over wireless vehicular links. By leveraging transfer learning with a frozen MobileNetV2 backbone—a standard split-FL paradigm for vehicular edge cameras—we train a 12,810-parameter classifier that fits into just **4 ciphertexts**, taking under 65 ms to encrypt while achieving 96.2% clean accuracy."

#### Q2: "In your initial runs, why did accuracy flatline at 55%, and how did you resolve it?"
> **Defense**: "The 55% ceiling was caused by a domain shift in facial and driver appearance. We initially isolated 5 drivers exclusively for testing, but a frozen linear classifier cannot generalize across unseen human faces without fine-tuning convolutional layers. By cross-referencing Sections III-A and IV-A of the original paper, we aligned our partition with the authors' methodology: assigning 3–5 unique drivers per client, splitting each client's local pool into 80% train and 20% validation, and evaluating globally on the union of validation splits. This allowed the linear head to learn driver posture invariants, elevating accuracy to >93%."

#### Q3: "Does ranking chunks by gradient energy violate Differential Privacy?"
> **Defense**: "No. This is provably guaranteed by the **Post-Processing Theorem of Differential Privacy**. In our pipeline, per-sample clipping ($C=1.2$) and calibrated Gaussian noise ($\sigma = z \cdot C$) are applied to gradients *before* chunk scoring. Because the energy score is computed strictly on already-noised updates ($\sum (M_i \cdot \Delta \tilde{w}_i)^2$), any function computed on that noised output incurs **zero additional privacy loss** ($\Delta \epsilon = 0$). Conversely, computing scores on pre-noise gradients would violate DP, which is why we rejected it."

#### Q4: "Why did you choose 3-$\sigma$ LossGuard instead of established Byzantine defenses like Krum, Median, or Trimmed Mean?"
> **Defense**: "Coordinate-wise Median, Trimmed Mean, and Krum require sorting, comparing, or calculating pairwise Euclidean distances between individual client updates ($\mathcal{O}(K^2 d)$ operations). Under Homomorphic Encryption, individual updates are encrypted ciphertexts; computing non-linear operations like sorting or distances requires complex interactive client-server decryption protocols or deep bootstrapping circuits. LossGuard sidesteps this entirely by evaluating the *aggregated* validation loss on the server after clients decrypt the broadcast update. It requires zero plaintext access to individual client updates, maintaining complete zero-knowledge privacy while detecting active MitM tampering."

#### Q5: "If DP noise alone stops gradient inversion (DLG), why bother encrypting gradients with CKKS at all?"
> **Defense**: "Because Differential Privacy provides **confidentiality**, but zero **integrity**. While DP noise prevents an eavesdropper from reconstructing private camera frames (as verified by our DLG audit where PSNR drops to noise levels $\le 10.6\,\text{dB}$), it does not protect updates against active Man-in-the-Middle tampering. In our experiments, an active MitM attacker flipping signs on unencrypted DP updates causes accuracy to collapse from 90% down to **0.60%**. Homomorphic Encryption provides the cryptographic integrity barrier that stops MitM tampering from corrupting dominant gradient updates."

#### Q6: "Why is your proposed `sd_budget` superior to simply excluding vehicles with low compute budgets (`sd_exclude`)?"
> **Defense**: "`sd_exclude` enforces a coverage gate that drops vehicles unable to encrypt a minimum threshold of coordinates. In heterogeneous fleets (Scenario H1), low-end vehicles (Tiers 1 and 2) are excluded 100% of the time, resulting in a participation fairness ratio of **0.00** and dropping model accuracy to **46.23%** because the model never learns from those drivers. Our `sd_budget` strategy allows low-end vehicles to encrypt whatever their budget permits and send the residual in plaintext with DP noise. This achieves **94.20% accuracy**, **1.00 participation fairness**, and eliminates straggler wait delays."

---

# 14. Fault-Tolerant Execution, Idempotence, and Job Resumption

### 14.1 The HPC Cluster Time Limit Challenge
High-Performance Computing clusters (e.g. Slurm on `172.30.2.64`) enforce hard wall-time limits per job allocation (typically `--time=04:00:00`). If node load, disk I/O contention, or TenSEAL cryptographic operations cause an experimental sweep to run longer than expected, Slurm issues a `SIGTERM` followed by a hard `SIGKILL`. 

Without robust idempotence, restarting the job would discard hours of prior computation and re-run previously completed baselines from scratch.

### 14.2 Self-Healing Architecture
To ensure 100% fault-tolerant, resumable execution, our implementation enforces three layers of state persistence:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Layer 1: Preprocessed Dataset Caching                │
│   Checks data/gtsrb_cache_*.npz or data/auc_cache_*.npz.               │
│   If present: Re-extraction is skipped (0s overhead).                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Layer 2: Cell-Level JSON Idempotence                 │
│   Each cell (config, condition, seed) emits a distinct JSON record:    │
│   results_gtsrb/{config}_{condition}_seed{seed}.json                   │
│   Before running: checks if JSON exists AND contains 'final_accuracy'. │
│   If valid: Logs '... cached' and skips cell in < 1 ms.                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Layer 3: Atomic Completion Guard                     │
│   JSON and CSV summaries are written ONLY after all 120 rounds finish. │
│   If Slurm kills the job mid-write: json.load() catches truncation     │
│   exception and cleanly re-executes only that specific cell.           │
└────────────────────────────────────────────────────────────────────────┘
```

### 15.3 Zero-Configuration Resumption Command
Because every stage is idempotent, resuming after a time-out requires no manual editing, argument adjustments, or checkpoint indexing:
```bash
# Simply resubmit the exact same Slurm script
sbatch scripts/submit_gtsrb.sh 120
```
The script will sequentially scan through the cached results, skip all completed cells in fractions of a second, and seamlessly resume execution at the exact cell that was interrupted.

---
*Document compiled and verified for the SecureDrive-FL Project viva defense.*
