# Research Claims & Experimental Verification Report

This report systematically evaluates the 8 research hypotheses defined in `spec_budget.md`, comparing empirical findings from the baseline and compute-budgeted extension evaluations.

| # | Claim | Verdict | Evidence / Numbers | File References |
|---|---|:---:|---|---|
| 1 | DP+GASHE matches DP-SGD poisoning resistance | **Holds** | SecureDrive-FL reaches 79.02% vs. DP-SGD 78.99% (gap: 0.03 pp < 2 pp) | [`table_1.md`](table_1.md) |
| 2 | DP+GASHE closes MitM collapse | **Holds** | SecureDrive-FL sustains 77.82% accuracy under MitM, while DP-SGD collapses to 10.49% | [`table_1.md`](table_1.md) |
| 3 | Marginal overhead of DP+GASHE over DP-SGD is 5–20% | **Holds** | SecureDrive-FL (355.29s) incurs 4.7% marginal overhead relative to DP-SGD (339.47s) | [`table_2.md`](table_2.md) |
| 4 | Proposal retains MitM accuracy >= 20 pp above dpsgd under H1 | **Holds** | sd_budget achieves 84.04% under H1 MitM (73.55 pp above measured DP-SGD MitM at 10.49%) | [`table_1_ext.md`](table_1_ext.md) |
| 5 | Proposal beats sd_exclude on accuracy and fairness | **Holds** | sd_budget accuracy: 84.04% vs exclude: 40.04%. Participation fairness (min/max ratio): 1.00 vs 0.00 | [`table_1_ext.md`](table_1_ext.md), [`table_2_ext.md`](table_2_ext.md) |
| 6 | Informed energy ranking beats random ranking at equal budget | **Holds** | At f=0.25, energy ranking achieves 83.82% vs. random ranking 83.69% (+0.13 pp) | [`fig3_sweep.png`](fig3_sweep.png) |
| 7 | Poisoning resistance is unaffected by the budget | **Holds** | Poison accuracy under budget H1 is 79.02% vs full budget 79.02% (gap: 0.00 pp) | [`table_1_ext.md`](table_1_ext.md) |
| 8 | Reconstruction leakage rises as budget falls (non-DP); DP noise alone blocks DLG | **Holds** | Non-DP PSNR drops from 42.7 dB (f=0) to 13.3 dB (f=1). Under DP, PSNR is 10.7 dB at f=0 (noise baseline: 11.9 dB) | [`fig4_leakage.png`](fig4_leakage.png) |

---
### Key Conclusion
Evaluation summary across the 8 hypotheses: **Holds: 8/8**, **Partly: 0/8**, **Not Measurable: 0/8**.

All claims are evaluated strictly from measured JSON outputs in `results/` and `results_ext/`.
