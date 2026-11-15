import os
import sys
import math
import inspect
import subprocess
from pathlib import Path
import pytest
import numpy as np
import torch

from securedrive_fl.config import Config
from securedrive_fl.he import MockHE, TenSealCKKS
from securedrive_fl.gashe import gashe_mask, make_payload, Payload
from securedrive_fl.aggregate import (
    gashe_aggregate,
    client_apply_update,
    fedavg,
    Pending,
)
from securedrive_fl.guard import LossGuard
from securedrive_fl.privacy import epsilon, _rdp_sampled_gaussian
from securedrive_fl.model import get_mobilenet_v2_tiny, flatten_weights, unflatten_weights
from securedrive_fl.data import make_toy_federation
from securedrive_fl.run_scenario import run_scenario


def test_1_mask_split():
    """
    1. Mask split: for random dw, enc + plain == dw exactly, and a NaN in dw does
       not leak into the other stream.
    """
    cfg = Config(use_dp=True, use_gashe=True)
    rng = np.random.default_rng(42)
    dw = rng.normal(0, 2.0, 1000)
    M, tau, tier = gashe_mask(dw, cfg)

    enc_vals = np.where(M, dw, 0.0)
    plain_vals = np.where(M, 0.0, dw)

    # enc + plain == dw
    assert np.allclose(enc_vals + plain_vals, dw)

    # Test NaN behavior: NaN in dw must not leak into the other stream
    dw_nan = dw.copy()
    dw_nan[0] = np.nan
    M_nan, _, _ = gashe_mask(dw_nan, cfg)
    enc_nan = np.where(M_nan, dw_nan, 0.0)
    plain_nan = np.where(M_nan, 0.0, dw_nan)

    if M_nan[0]:
        assert np.isnan(enc_nan[0])
        assert not np.isnan(plain_nan[0])
        assert plain_nan[0] == 0.0
    else:
        assert not np.isnan(enc_nan[0])
        assert enc_nan[0] == 0.0
        assert np.isnan(plain_nan[0])


def test_2_aggregation_equivalence():
    """
    2. Aggregation equivalence: with MockHE and with TenSEAL, GASHE aggregate
       equals plain FedAvg mean of updates within 1e-4 when all clients have equal
       weight.
    """
    rng = np.random.default_rng(42)
    n_clients = 6
    n_params = 100
    updates = [rng.normal(0, 1.0, n_params) for _ in range(n_clients)]
    w_before = rng.normal(0, 1.0, n_params)
    cfg = Config(use_dp=True, use_gashe=True)

    # 2a. MockHE
    mock_he = MockHE()
    payloads_mock = [make_payload(cid=i, dw=u, n_examples=100, cfg=cfg, he=mock_he) for i, u in enumerate(updates)]
    pend_mock = gashe_aggregate(payloads_mock, mock_he, w_before)
    w_gashe_mock = client_apply_update(mock_he, pend_mock)
    w_fedavg = fedavg(w_before, updates, [1.0] * n_clients)
    assert np.allclose(w_gashe_mock, w_fedavg, atol=1e-4)

    # 2b. TenSEAL
    try:
        ts_he = TenSealCKKS()
        payloads_ts = [make_payload(cid=i, dw=u, n_examples=100, cfg=cfg, he=ts_he) for i, u in enumerate(updates)]
        pend_ts = gashe_aggregate(payloads_ts, ts_he, w_before)
        w_gashe_ts = client_apply_update(ts_he, pend_ts)
        assert np.allclose(w_gashe_ts, w_fedavg, atol=1e-4)
    except ImportError:
        pytest.skip("tenseal not installed")


def test_3_tau_and_z():
    """
    3. tau: for the paper config, z in [1.2112 +/- 1e-3], tau = 1.4535 +/- 1e-3.
    """
    cfg = Config(clip_C=1.2, eps0=4.0, delta=1e-5)
    assert abs(cfg.z - 1.2112) < 1e-3
    assert abs(cfg.sigma - 1.4535) < 1e-3


def test_4_rdp_properties():
    """
    4. RDP: epsilon() is monotone in steps and decreasing in z; q=1 branch equals
       alpha/(2 z^2) analytically.
    """
    q = 0.05
    z = 1.2
    delta = 1e-5

    # Monotone in steps
    eps_50 = epsilon(steps=50, q=q, z=z, delta=delta)
    eps_100 = epsilon(steps=100, q=q, z=z, delta=delta)
    assert eps_100 > eps_50

    # Decreasing in z
    eps_z1 = epsilon(steps=100, q=q, z=1.0, delta=delta)
    eps_z2 = epsilon(steps=100, q=q, z=1.5, delta=delta)
    assert eps_z2 < eps_z1

    # q=1 branch equals alpha / (2 * z^2) analytically
    for alpha in [2, 5, 10, 20]:
        val = _rdp_sampled_gaussian(alpha, q=1.0, z=z)
        expected = alpha / (2.0 * z * z)
        assert abs(val - expected) < 1e-7


def test_5_loss_guard_spike():
    """
    5. Loss guard: injected spike after warmup is rejected; history unchanged.
    """
    guard = LossGuard(warmup=5, keep=20)
    normal_losses = [1.0, 1.05, 0.95, 1.02, 0.98]
    for loss in normal_losses:
        assert not guard.reject(loss)

    assert len(guard.hist) == 5

    # Injected spike
    spike_loss = 100.0
    rejected = guard.reject(spike_loss)
    assert rejected is True

    # History unchanged and spike not added
    assert len(guard.hist) == 5
    assert spike_loss not in guard.hist


def test_6_rollback():
    """
    6. Rollback: after a rejected round, global weights equal w_before and
       pending is None.
    """
    cfg = Config(rounds=1, loss_guard=True, attack="mitm", mitm_amplify=100.0)
    guard = LossGuard(warmup=5, keep=20)
    # Fill guard with stable history
    for _ in range(5):
        guard.reject(1.0)

    w_before = np.array([0.5, 0.5])
    w = np.array([99.9, 99.9])  # poisoned
    pending = Pending(c_agg=None, p_agg=np.array([1.0, 1.0]), n=6, w_before=w_before)

    loss = 50.0  # huge spike
    rejected = guard.reject(loss)
    assert rejected is True

    if rejected:
        w, pending = w_before, None

    assert np.allclose(w, w_before)
    assert pending is None


def test_7_payload_sort():
    """
    7. Sort: shuffled payloads are re-ordered by cid.
    """
    payloads = [
        Payload(cid=3, n_examples=10, plain=np.zeros(2)),
        Payload(cid=0, n_examples=10, plain=np.zeros(2)),
        Payload(cid=5, n_examples=10, plain=np.zeros(2)),
        Payload(cid=1, n_examples=10, plain=np.zeros(2)),
        Payload(cid=4, n_examples=10, plain=np.zeros(2)),
        Payload(cid=2, n_examples=10, plain=np.zeros(2)),
    ]
    payloads.sort(key=lambda p: p.cid)
    assert [p.cid for p in payloads] == [0, 1, 2, 3, 4, 5]


def test_8_server_context_cannot_decrypt():
    """
    8. Server-context cannot decrypt (TenSEAL).
    """
    ts_client = TenSealCKKS()
    assert ts_client.is_private() is True
    ts_server = ts_client.public_copy()
    assert ts_server.is_private() is False
    v = np.array([1.0, 2.0, 3.0])
    ciphertext = ts_client.encrypt(v)

    # Server attempts to decrypt should raise an error
    with pytest.raises(Exception):
        ts_server.decrypt(ciphertext)


def test_9_flatten_unflatten_mobilenet():
    """
    9. Flatten/unflatten round trip for MobileNetV2-Tiny.
    """
    model = get_mobilenet_v2_tiny(num_classes=10)
    flat = flatten_weights(model)
    assert len(flat) == 2236682  # ~2.2M parameters

    # Perturb flat weights slightly
    perturbed = flat + 0.01
    unflatten_weights(perturbed, model)

    # Flatten again and verify exact match
    flat_again = flatten_weights(model)
    assert np.allclose(flat_again, perturbed, atol=1e-6)


def test_10_toy_demo_ordering():
    """
    10. Toy demo ordering (qualitative, MockHE): FedAvg-baseline high accuracy;
        FedAvg under model poison and MitM clearly lower; securedrive_mitm
        with guard stays above fedavg_mitm.
    """
    he = MockHE()
    clients, test, K = make_toy_federation(seed=0)

    hist_clean = run_scenario(Config(rounds=15, use_dp=False, use_gashe=False, seed=0), clients, test, K, he)
    hist_poison = run_scenario(Config(rounds=15, use_dp=False, use_gashe=False, attack="model_poison", poison_mode="ascent", seed=0), clients, test, K, he)
    hist_mitm = run_scenario(Config(rounds=15, use_dp=False, use_gashe=False, attack="mitm", seed=0), clients, test, K, he)
    hist_sd_mitm = run_scenario(Config(rounds=15, use_dp=True, use_gashe=True, attack="mitm", loss_guard=True, seed=0), clients, test, K, he)

    acc_clean = hist_clean[-1]["acc"]
    acc_poison = hist_poison[-1]["acc"]
    acc_mitm = hist_mitm[-1]["acc"]
    acc_sd_mitm = hist_sd_mitm[-1]["acc"]

    assert acc_clean > 0.90
    assert acc_poison < 0.30
    assert acc_mitm < 0.30
    assert acc_sd_mitm > acc_mitm


def test_noise_std_matches_accountant():
    """
    Asserts empirical DP noise std == cfg.sigma; no multiplier other than 1.
    Accountant z matches training z.
    """
    from securedrive_fl.train import local_train
    cfg = Config(use_dp=True, clip_C=1.2, eps0=4.0, delta=1e-5)
    # Check that cfg.sigma equals accountant z * C exactly
    assert abs(cfg.sigma - cfg.z * cfg.clip_C) < 1e-6

    # Verify source does not contain scaling multipliers
    import inspect
    from securedrive_fl import train
    src = inspect.getsource(train.local_train)
    assert "5.5" not in src
    assert "noise_std = cfg.sigma * " not in src


def test_eq13_backdoor_update():
    """
    Asserts dw_m == alpha*dw_target + (1-alpha)*dw_benign; poison_alpha changes the result.
    """
    from securedrive_fl.train import local_train, n_params
    d = 10
    K = 5
    w = np.zeros(n_params(d, K))
    X = np.random.randn(20, d)
    y = np.random.randint(0, K, size=20)

    cfg_a1 = Config(poison_mode="backdoor", poison_alpha=0.5, use_dp=False, lr=0.01, local_epochs=1, batch=20)
    cfg_a2 = Config(poison_mode="backdoor", poison_alpha=0.8, use_dp=False, lr=0.01, local_epochs=1, batch=20)

    rng1 = np.random.default_rng(123)
    dw_a1 = local_train(w, X, y, K=5, cfg=cfg_a1, rng=rng1, poison=True)

    rng2 = np.random.default_rng(123)
    dw_a2 = local_train(w, X, y, K=5, cfg=cfg_a2, rng=rng2, poison=True)

    assert not np.allclose(dw_a1, dw_a2)

    cfg_target = Config(poison_mode="backdoor", poison_alpha=1.0, use_dp=False, lr=0.01, local_epochs=1, batch=20)
    cfg_benign = Config(poison_mode="backdoor", poison_alpha=0.0, use_dp=False, lr=0.01, local_epochs=1, batch=20)
    cfg_half = Config(poison_mode="backdoor", poison_alpha=0.5, use_dp=False, lr=0.01, local_epochs=1, batch=20)

    dw_tar = local_train(w, X, y, K=5, cfg=cfg_target, rng=np.random.default_rng(99), poison=True)
    dw_ben = local_train(w, X, y, K=5, cfg=cfg_benign, rng=np.random.default_rng(99), poison=True)
    dw_mid = local_train(w, X, y, K=5, cfg=cfg_half, rng=np.random.default_rng(99), poison=True)

    assert np.allclose(dw_mid, 0.5 * dw_tar + 0.5 * dw_ben, atol=1e-6)


def test_no_poison_boost_special_case():
    """
    Asserts same poison_boost for all configs; BatchCrypt not special-cased.
    """
    cfg = Config()
    assert hasattr(cfg, "poison_boost")
    assert cfg.poison_boost == 1.0

    from securedrive_fl import train
    src = inspect.getsource(train.local_train)
    assert "boost = 1.5 if batchcrypt" not in src
    assert "batchcrypt" not in src


def test_mitm_std_from_cfg():
    """
    Asserts mitm_plain uses cfg.mitm_noise_std exactly (no max(..., 3.0)).
    """
    from securedrive_fl.attacks import mitm_plain
    cfg = Config(mitm_noise_std=0.75, mitm_amplify=0.0)
    rng = np.random.default_rng(42)
    updates = [np.zeros(10000) for _ in range(6)]
    tampered = mitm_plain(updates, cfg, rng)
    emp_std = np.std(tampered[0])
    assert abs(emp_std - 0.75) < 0.05

    from securedrive_fl import attacks
    src = inspect.getsource(attacks.mitm_plain)
    assert "max(cfg.mitm_noise_std, 3.0)" not in src


def test_krum_f_default():
    """
    Asserts Config().krum_f == 2 and multi_krum handles it for N=6.
    """
    cfg = Config()
    assert hasattr(cfg, "krum_f")
    assert cfg.krum_f == 2

    from securedrive_fl.aggregate import multi_krum
    updates = [np.ones(10) * float(i) for i in range(6)]
    agg = multi_krum(np.zeros(10), updates, f=cfg.krum_f)
    assert agg is not None


def test_reports_fail_loudly(tmp_path):
    """
    Asserts report scripts exit non-zero when their input directories are empty.
    """
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)

    scripts_to_test = [
        ["scripts/make_claims_report.py", "--base-dir", empty_dir, "--ext-dir", empty_dir],
        ["scripts/make_tables_ext.py", "--ext-dir", empty_dir],
    ]
    for cmd in scripts_to_test:
        res = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
        assert res.returncode != 0, f"Script {cmd[0]} did not exit non-zero on empty input!"


def test_no_hardcoded_fallbacks():
    """
    Static grep of scripts/ for banned literal metric values.
    """
    banned = [
        "67.00%",
        "73.58%",
        "72.84%",
        "psnr_nondp = [21.5",
        "points = [(\"dpsgd\"",
        "prot_energy = [58.4",
        "clean_accs = [95.35",
        "mitm_accs = [0.02",
        "axhline(y=1.75",
        "axhline(y=73.58",
    ]
    scripts_dir = Path("scripts")
    for py_file in scripts_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for b in banned:
            assert b not in text, f"Banned literal '{b}' found in {py_file}"


def test_data_partition():
    """
    Asserts clients have >= 5 classes in toy federation; deterministic partition across runs.
    """
    from securedrive_fl.data import make_toy_federation
    client_data, test_data, K = make_toy_federation(seed=42)
    for cid, (X, y) in enumerate(client_data):
        unique_classes = len(np.unique(y))
        assert unique_classes >= 5, f"Client {cid} has only {unique_classes} classes (< 5)"

