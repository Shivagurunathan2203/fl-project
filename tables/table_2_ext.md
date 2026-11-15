### Table II-ext: Resource, Timing, and Cryptographic Overhead Breakdown under Heterogeneous Budgets (H1)

| Strategy | Sim Total Time (s) | Straggler Wait (s) | Cipher Upload (KB/cl/rd) | Plain Upload (KB/cl/rd) | Enc Chunk Frac (%) | Prot Energy Frac (%) | Participation (Min/Mean/Max) | Fairness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard FL (FedAvg, base reuse) | n/a (base run) | n/a | 0.0 KB | 80.0 KB | 0.0% | 0.0% | 6 / 6.0 / 6 | 1.00 |
| FL + DP-SGD (base reuse) | n/a (base run) | n/a | 0.0 KB | 80.0 KB | 0.0% | 0.0% | 6 / 6.0 / 6 | 1.00 |
| FL + GASHE-only (base reuse) | n/a (base run) | n/a | 65.5 KB (full HE) | 0.0 KB (full HE) | 100.0% | 100.0% | 6 / 6.0 / 6 | 1.00 |
| FL + GASHE-budgeted (proposal non-DP) | 20.88 ± 0.11 | 0.00s (bounded) | 1781.5 KB | 92.8 KB | 21.8% | 27.9% | 120 / 120.0 / 120 | 1.00 |
| SecureDrive-FL (unlimited budget, base reuse) | n/a (base run) | n/a | 65.5 KB (full HE) | 0.0 KB (full HE) | 100.0% | 100.0% | 6 / 6.0 / 6 | 1.00 |
| SecureDrive-FL (sd_wait, straggler wait) | 333.67 ± 8.75 | 310.18s | 8162.3 KB | 68.0 KB | 100.0% | 92.0% | 120 / 120.0 / 120 | 1.00 |
| SecureDrive-FL (sd_budget, proposal) | 23.49 ± 0.05 | 0.00s (bounded) | 1781.5 KB | 94.4 KB | 21.8% | 21.3% | 120 / 120.0 / 120 | 1.00 |
| SecureDrive-FL (sd_exclude, drop weak) | 14.58 ± 0.55 | 0.00s (bounded) | 6041.8 KB | 100.1 KB | 74.0% | 69.3% | 0 / 13.3 / 83 | 0.00 |
| SecureDrive-FL (random ranking) | 23.52 ± 0.07 | 0.00s (bounded) | 1781.5 KB | 94.6 KB | 21.8% | 19.9% | 120 / 120.0 / 120 | 1.00 |
| SecureDrive-FL (layer-static ranking) | 22.36 ± 0.12 | 0.00s (bounded) | 1781.5 KB | 94.6 KB | 21.8% | 19.9% | 120 / 120.0 / 120 | 1.00 |
