"""
SecureDrive-FL package.
"""
from securedrive_fl.config import Config
from securedrive_fl.he import HEBackend, MockHE, TenSealCKKS
from securedrive_fl.gashe import Payload, gashe_mask, make_payload
from securedrive_fl.aggregate import (
    Pending,
    gashe_aggregate,
    client_apply_update,
    fedavg,
    trimmed_mean,
    multi_krum,
)
from securedrive_fl.attacks import mitm_plain, mitm_gashe
from securedrive_fl.guard import LossGuard
from securedrive_fl.privacy import _rdp_sampled_gaussian, epsilon, worst_case_epsilon
from securedrive_fl.train import local_train, evaluate, n_params
from securedrive_fl.data import make_toy_federation
from securedrive_fl.run_scenario import run_scenario

__all__ = [
    "Config",
    "HEBackend",
    "MockHE",
    "TenSealCKKS",
    "Payload",
    "gashe_mask",
    "make_payload",
    "Pending",
    "gashe_aggregate",
    "client_apply_update",
    "fedavg",
    "trimmed_mean",
    "multi_krum",
    "mitm_plain",
    "mitm_gashe",
    "LossGuard",
    "_rdp_sampled_gaussian",
    "epsilon",
    "worst_case_epsilon",
    "local_train",
    "evaluate",
    "n_params",
    "make_toy_federation",
    "run_scenario",
]
