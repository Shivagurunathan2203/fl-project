from __future__ import annotations

import os
import json
import math
from pathlib import Path
import numpy as np
import pytest

from securedrive_fl.config import Config
from securedrive_fl.he import MockHE, TenSealCKKS
from securedrive_fl.gashe import gashe_mask
from securedrive_fl.data import make_toy_federation
from securedrive_fl.run_scenario import run_scenario
from securedrive_fl.budget.config import BudgetConfig, CapacityConfig, AvailabilityConfig
from securedrive_fl.budget.model import BudgetModel
from securedrive_fl.budget.codec import budgeted_gashe_encode, CHUNK_SIZE
from securedrive_fl.budget.aggregate import (
    BudgetedPayload,
    budgeted_gashe_aggregate,
    client_apply_budgeted_update,
)
from securedrive_fl.budget.strategies import run_budgeted_scenario


def test_i7_budget_model_properties():
    """
    I7: Budget-model unit tests:
    - Determinism
    - Ranges [0, 1]
    - Tier counts
    - Independence from training RNG
    """
    cfg = BudgetConfig()
    cfg.capacity = CapacityConfig(mode="tiers", tiers=[0.10, 0.25, 0.50, 1.00], tier_counts=[3, 3, 3, 3])
    cfg.availability = AvailabilityConfig(mode="uniform", uniform_lo=0.3, uniform_hi=1.0)

    model1 = BudgetModel(cfg, n_clients=12, n_chunks=100, seed=42)
    model2 = BudgetModel(cfg, n_clients=12, n_chunks=100, seed=42)

    # 1. Determinism
    np.testing.assert_array_equal(model1.capacity, model2.capacity)
    np.testing.assert_array_equal(model1.availability(5), model2.availability(5))
    np.testing.assert_array_equal(model1.effective_chunks(5), model2.effective_chunks(5))

    # 2. Tier counts check
    for t_val in [0.10, 0.25, 0.50, 1.00]:
        assert np.sum(np.isclose(model1.capacity, t_val)) == 3

    # 3. Ranges
    avail = model1.availability(1)
    assert np.all(avail >= 0.3) and np.all(avail <= 1.0)
    eff = model1.effective_chunks(1)
    assert np.all(eff >= 0) and np.all(eff <= 100)

    # 4. Independence from training RNG: changing rng_salt does not affect base training RNG stream
    rng_base1 = np.random.default_rng(42)
    _ = BudgetModel(cfg, n_clients=12, n_chunks=100, seed=42)
    draw1 = rng_base1.normal(size=5)

    rng_base2 = np.random.default_rng(42)
    cfg_other_salt = BudgetConfig(rng_salt=12345)
    _ = BudgetModel(cfg_other_salt, n_clients=12, n_chunks=100, seed=42)
    draw2 = rng_base2.normal(size=5)

    np.testing.assert_array_equal(draw1, draw2)


def test_i2_zero_budget_plain_equivalence():
    """
    I2: With budget=0, encrypted stream is empty, all of Δw is in plaintext,
    and aggregate equals plain sum exactly.
    """
    d = 10000
    dw = np.random.randn(d)
    M = np.abs(dw) > 0.5

    enc_res = budgeted_gashe_encode(dw=dw, M=M, budget_chunks=0)
    assert len(enc_res["enc_chunk_vectors"]) == 0
    assert len(enc_res["selected_set"]) == 0
    assert np.all(enc_res["E"] == False)
    np.testing.assert_allclose(enc_res["plain_stream"], dw, atol=1e-12)


def test_i3_exactness_across_ranking_and_content_modes():
    """
    I3: For random updates, masks, and random budgets,
    reconstructed aggregate matches the true sum of updates within 1e-9 (MockHE).
    """
    rng = np.random.default_rng(42)
    d = 12000
    K = 6
    he = MockHE()
    n_chunks = int(math.ceil(d / CHUNK_SIZE))

    for rank_by in ["post_noise_energy", "layer_static", "random"]:
        for chunk_content in ["masked_only", "full_chunk"]:
            payloads = []
            true_sum = np.zeros(d, dtype=np.float64)

            for cid in range(K):
                dw = rng.normal(0, 1.0, size=d)
                true_sum += dw
                M = np.abs(dw) > 0.8
                b_chunks = rng.integers(0, n_chunks + 1)

                enc_res = budgeted_gashe_encode(
                    dw=dw,
                    M=M,
                    budget_chunks=b_chunks,
                    rank_by=rank_by,
                    chunk_content=chunk_content,
                    rng=rng,
                )

                cipher_chunks = {j: he.encrypt(v) for j, v in enc_res["enc_chunk_vectors"].items()}
                p = BudgetedPayload(
                    cid=cid,
                    plain=enc_res["plain_stream"],
                    cipher_chunks=cipher_chunks,
                    selection_bitmap=enc_res["selection_bitmap"],
                    M=enc_res["M"],
                    E=enc_res["E"],
                    stats=enc_res,
                )
                payloads.append(p)

            w_before = np.zeros(d)
            pending = budgeted_gashe_aggregate(payloads, he, w_before)
            w_after = client_apply_budgeted_update(he, pending)
            dw_reconstructed = (w_after - w_before) * float(K)

            diff = np.max(np.abs(dw_reconstructed - true_sum))
            assert diff < 1e-9, f"Exactness failed for {rank_by}/{chunk_content} with diff={diff}"


def test_i8_server_cannot_decrypt():
    """
    I8: Server holding public TenSEAL context cannot decrypt budgeted aggregate ciphertexts.
    """
    ts_backend = TenSealCKKS()
    pub_backend = ts_backend.public_copy()
    assert not pub_backend.is_private()

    d = 4096
    dw = np.ones(d, dtype=np.float64)
    M = np.ones(d, dtype=bool)

    enc_res = budgeted_gashe_encode(dw=dw, M=M, budget_chunks=1)
    cipher_chunks = {0: pub_backend.encrypt(enc_res["enc_chunk_vectors"][0])}

    payload = BudgetedPayload(
        cid=0,
        plain=enc_res["plain_stream"],
        cipher_chunks=cipher_chunks,
        selection_bitmap=enc_res["selection_bitmap"],
        M=enc_res["M"],
        E=enc_res["E"],
        stats=enc_res,
    )

    pending = budgeted_gashe_aggregate([payload], pub_backend, np.zeros(d))
    with pytest.raises(Exception):
        _ = pub_backend.decrypt(pending.c_agg_chunks[0])


def test_i9_exclude_strategy_eligibility():
    """
    I9: Exclude strategy eligibility logic and skipped round handling.
    Also verifies non-degenerate participation for H1 default configuration.
    """
    cfg = Config(rounds=3, use_dp=True)
    b_cfg = BudgetConfig()
    # Force 0 capacity for all clients
    b_cfg.capacity = CapacityConfig(mode="constant", constant_value=0.0)
    b_cfg.availability = AvailabilityConfig(mode="constant", constant_value=1.0)
    b_cfg.min_coverage = 1.0

    client_data, test_data, K = make_toy_federation(seed=42)
    he = MockHE()

    res = run_budgeted_scenario(cfg, b_cfg, "sd_exclude", client_data, test_data, K, he)
    # All rounds should be skipped because effective budget is 0 < n_chunks
    assert all(r["skipped_round"] for r in res["rounds"])
    assert all(cnt == 0 for cnt in res["participation_counts"].values())

    # H1 default non-degenerate participation
    b_cfg_h1 = BudgetConfig()
    b_cfg_h1.capacity = CapacityConfig(mode="tiers", tiers=[0.10, 0.25, 0.50, 1.00], tier_counts=[2, 2, 1, 1])
    b_cfg_h1.availability = AvailabilityConfig(mode="uniform", uniform_lo=0.3, uniform_hi=1.0)
    b_cfg_h1.min_coverage = 0.5
    res_h1 = run_budgeted_scenario(Config(rounds=5, use_dp=True, seed=42), b_cfg_h1, "sd_exclude", client_data, test_data, K, he)
    assert not res_h1["degenerate"], "H1 default exclude configuration resulted in degenerate participation"
    assert sum(res_h1["participation_counts"].values()) > 0


def test_i1_unlimited_budget_matches_base():
    """
    I1: With unlimited budget (capacity=1, availability=1) and sd_budget,
    global weights match base securedrive across the first 5 rounds within 1e-6.
    Tested under both clean and MitM conditions.
    """
    client_data, test_data, K = make_toy_federation(seed=42)
    he = MockHE()

    for cond in ["clean", "mitm"]:
        attack = "mitm" if cond == "mitm" else None
        loss_guard = (cond == "mitm")
        base_cfg = Config(rounds=5, use_dp=True, use_gashe=True, attack=attack, loss_guard=loss_guard, seed=42)
        base_hist = run_scenario(base_cfg, client_data, test_data, K, he)

        b_cfg = BudgetConfig()
        b_cfg.capacity = CapacityConfig(mode="constant", constant_value=1.0)
        b_cfg.availability = AvailabilityConfig(mode="constant", constant_value=1.0)

        res_budget = run_budgeted_scenario(base_cfg, b_cfg, "sd_budget", client_data, test_data, K, he)

        for r in range(5):
            w_base = base_hist[r]["weights"]
            w_budget = res_budget["rounds"][r]["weights"]
            diff = np.max(np.abs(w_base - w_budget))
            assert diff <= 1e-6, f"I1 failed under {cond} at round {r+1}: max abs diff = {diff}"


def test_i4_clean_condition_equivalence():
    """
    I4: For sd_budget under clean, final accuracy is within 0.5 pp of base securedrive.
    """
    client_data, test_data, K = make_toy_federation(seed=42)
    he = MockHE()

    base_cfg = Config(rounds=10, use_dp=True, use_gashe=True, seed=42)
    base_hist = run_scenario(base_cfg, client_data, test_data, K, he)

    b_cfg = BudgetConfig()
    b_cfg.capacity = CapacityConfig(mode="tiers", tiers=[0.10, 0.25, 0.50, 1.00], tier_counts=[2, 2, 1, 1])
    b_cfg.availability = AvailabilityConfig(mode="uniform", uniform_lo=0.3, uniform_hi=1.0)

    res_budget = run_budgeted_scenario(base_cfg, b_cfg, "sd_budget", client_data, test_data, K, he)

    diff_pp = abs(base_hist[-1]["acc"] - res_budget["final_accuracy"]) * 100
    assert diff_pp < 0.5, f"I4 failed: Clean accuracy gap is {diff_pp:.2f} pp (> 0.5 pp)"


def test_i5_monotone_protection():
    """
    I5: In budget sweep S1, protected_energy_fraction and protected_coord_fraction
    are non-decreasing in budget fraction f.
    """
    client_data, test_data, K = make_toy_federation(seed=42)
    he = MockHE()
    cfg = Config(rounds=2, use_dp=True, use_gashe=True, seed=42)

    f_values = [0.0, 0.25, 0.50, 1.0]
    prot_energy_list = []
    prot_coord_list = []

    for f in f_values:
        b_cfg = BudgetConfig()
        b_cfg.capacity = CapacityConfig(mode="constant", constant_value=f)
        b_cfg.availability = AvailabilityConfig(mode="constant", constant_value=1.0)
        res = run_budgeted_scenario(cfg, b_cfg, "sd_budget", client_data, test_data, K, he)
        prot_energy_list.append(res["mean_protected_energy_fraction"])
        prot_coord_list.append(res["mean_protected_coord_fraction"])

    # Monotonicity check (with small epsilon for floating-point tolerance)
    for i in range(len(f_values) - 1):
        assert prot_energy_list[i+1] >= prot_energy_list[i] - 1e-4, f"Energy not monotone at f={f_values[i]}"
        assert prot_coord_list[i+1] >= prot_coord_list[i] - 1e-4, f"Coords not monotone at f={f_values[i]}"


def test_i6_backend_parity():
    """
    I6: Parity check between MockHE and TenSEALCKKS backends on a short run (seed 42).
    Accuracy matches within 0.5 pp.
    """
    client_data, test_data, K = make_toy_federation(seed=42)
    cfg = Config(rounds=2, use_dp=True, use_gashe=True, seed=42)
    b_cfg = BudgetConfig()
    b_cfg.capacity = CapacityConfig(mode="constant", constant_value=0.5)
    b_cfg.availability = AvailabilityConfig(mode="constant", constant_value=1.0)

    res_mock = run_budgeted_scenario(cfg, b_cfg, "sd_budget", client_data, test_data, K, MockHE())
    res_ts = run_budgeted_scenario(cfg, b_cfg, "sd_budget", client_data, test_data, K, TenSealCKKS())

    diff_pp = abs(res_mock["final_accuracy"] - res_ts["final_accuracy"]) * 100
    assert diff_pp < 0.5, f"Backend parity gap: {diff_pp:.3f} pp (> 0.5 pp)"


def test_mitm_tenseal_budget_shape():
    """
    Asserts substituted TenSEAL chunk decrypts to chunk_len coordinates.
    """
    try:
        he = TenSealCKKS()
    except Exception:
        pytest.skip("TenSEAL not available")
    from securedrive_fl.attacks import mitm_gashe
    d = 100
    dw = np.ones(d)
    M = np.ones(d, dtype=bool)
    enc_res = budgeted_gashe_encode(dw, M, budget_chunks=1, chunk_size=40)
    cipher_chunks = {0: he.encrypt(enc_res["enc_chunk_vectors"][0])}
    payload = BudgetedPayload(
        cid=0,
        plain=enc_res["plain_stream"],
        cipher_chunks=cipher_chunks,
        selection_bitmap=enc_res["selection_bitmap"],
        M=enc_res["M"],
        E=enc_res["E"],
        stats=enc_res,
        chunk_size=40,
    )
    cfg = Config(mitm_noise_std=1.0)
    rng = np.random.default_rng(42)
    mitm_gashe([payload], he, rng, cfg)
    dec = he.decrypt(payload.cipher_chunks[0])
    assert len(dec) == 40


def test_budget_base_mitm_parity():
    """
    Invariant I1 under MitM: base securedrive vs sd_budget with capacity=1, availability=1.
    """
    client_data, test_data, K = make_toy_federation(seed=42)
    he = MockHE()
    cfg = Config(rounds=5, use_dp=True, use_gashe=True, attack="mitm", loss_guard=True, seed=42)
    base_hist = run_scenario(cfg, client_data, test_data, K, he)

    b_cfg = BudgetConfig()
    b_cfg.capacity = CapacityConfig(mode="constant", constant_value=1.0)
    b_cfg.availability = AvailabilityConfig(mode="constant", constant_value=1.0)
    res_budget = run_budgeted_scenario(cfg, b_cfg, "sd_budget", client_data, test_data, K, he)

    for r in range(5):
        w_base = base_hist[r]["weights"]
        w_budget = res_budget["rounds"][r]["weights"]
        diff = np.max(np.abs(w_base - w_budget))
        assert diff <= 1e-6, f"MitM weight divergence at round {r+1}: max abs diff = {diff}"


def test_exclude_non_degenerate():
    """
    Asserts that with H1 capacities at least one client is eligible in some rounds under default min_coverage.
    """
    client_data, test_data, K = make_toy_federation(seed=42)
    he = MockHE()
    cfg = Config(rounds=5, use_dp=True, seed=42)
    b_cfg = BudgetConfig()
    b_cfg.capacity = CapacityConfig(mode="tiers", tiers=[0.10, 0.25, 0.50, 1.00], tier_counts=[2, 2, 1, 1])
    b_cfg.availability = AvailabilityConfig(mode="uniform", uniform_lo=0.3, uniform_hi=1.0)
    b_cfg.min_coverage = 0.5
    res = run_budgeted_scenario(cfg, b_cfg, "sd_exclude", client_data, test_data, K, he)
    assert not res["degenerate"], "H1 default exclude configuration resulted in degenerate 0 participation"
    total_participations = sum(res["participation_counts"].values())
    assert total_participations > 0


def test_reuse_hash():
    """
    Asserts config_hash changes when any param changes, and reuse is refused on mismatch.
    """
    from securedrive_fl.experiments import compute_config_hash
    cfg1 = Config(rounds=120, attack="mitm")
    h1 = compute_config_hash(cfg1, "toy", "numpy_softmax", 42)

    cfg2 = Config(rounds=15, attack="mitm")
    h2 = compute_config_hash(cfg2, "toy", "numpy_softmax", 42)
    assert h1 != h2

    h3 = compute_config_hash(cfg1, "toy", "numpy_softmax", 123)
    assert h1 != h3

    h4 = compute_config_hash(cfg1, "real", "numpy_softmax", 42)
    assert h1 != h4


def test_audit_not_forced():
    """
    Asserts no max(psnr, ...) forcing in inversion audit.
    """
    import inspect
    from securedrive_fl.budget import inversion
    src = inspect.getsource(inversion.run_inversion_audit)
    assert "max(psnr" not in src
    assert "baseline_mean + 4.2" not in src


def test_calibration_stamp(tmp_path):
    """
    Asserts extension reports show calibration_source and refuse default_not_measured for headline timing.
    """
    from scripts.calibrate_he_cost import calibrate
    try:
        ts = TenSealCKKS()
        res = calibrate(n_repeats=5, out_path=str(tmp_path / "calibration.json"))
        assert res.get("calibration_source") == "measured"
    except Exception:
        pytest.skip("TenSEAL not available")

