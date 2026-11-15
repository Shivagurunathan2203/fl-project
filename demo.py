import math
from securedrive_fl import Config, MockHE, make_toy_federation, run_scenario

def demo(rounds: int = 20, he = None):
    he = he or MockHE()
    clients, test, K = make_toy_federation()
    scenarios = {
        "fedavg_baseline":        dict(use_dp=False, use_gashe=False),
        "fedavg_model_poison":    dict(use_dp=False, use_gashe=False, attack="model_poison"),
        "fedavg_mitm":            dict(use_dp=False, use_gashe=False, attack="mitm"),
        "trimmed_mean_poison":    dict(use_dp=False, use_gashe=False, attack="model_poison", aggregator="trimmed_mean"),
        "gashe_top60_baseline":   dict(use_dp=False, use_gashe=True),
        "securedrive_baseline":   dict(use_dp=True, use_gashe=True),
        "securedrive_poison":     dict(use_dp=True, use_gashe=True, attack="model_poison"),
        "securedrive_mitm":       dict(use_dp=True, use_gashe=True, attack="mitm", loss_guard=True),
    }
    print(f"{'scenario':<24}{'final acc':>10}{'enc frac':>10}{'eps (RDP)':>11}{'rejected':>10}   tau tiers")
    for name, kw in scenarios.items():
        hist = run_scenario(Config(rounds=rounds, **kw), clients, test, K, he)
        h = hist[-1]
        eps = f"{h['eps']:.2f}" if (h["eps"] is not None and math.isfinite(h["eps"])) else "n/a"
        tiers = ",".join(sorted({t for x in hist for t in x["tiers"]})) if h["tiers"][0] != "n/a" else "-"
        rej = sum(x["rejected"] for x in hist)
        print(f"{name:<24}{h['acc']:>10.3f}{h['enc_frac']:>10.2f}{eps:>11}{rej:>10}   {tiers}")

if __name__ == "__main__":
    demo()
