#!/usr/bin/env python3
"""
Consolidates all project source code, tests, scripts, specifications, and reports
into a single clean project.txt file for comprehensive LLM review.
"""

import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "project.txt"

FILE_GROUPS = [
    ("Core Framework (securedrive_fl/)", [
        "securedrive_fl/__init__.py",
        "securedrive_fl/config.py",
        "securedrive_fl/data.py",
        "securedrive_fl/train.py",
        "securedrive_fl/model.py",
        "securedrive_fl/he.py",
        "securedrive_fl/gashe.py",
        "securedrive_fl/guard.py",
        "securedrive_fl/aggregate.py",
        "securedrive_fl/attacks.py",
        "securedrive_fl/baselines.py",
        "securedrive_fl/privacy.py",
        "securedrive_fl/metrics.py",
        "securedrive_fl/run_scenario.py",
        "securedrive_fl/experiments.py",
        "securedrive_fl/make_report.py",
    ]),
    ("Compute Budget Extension (securedrive_fl/budget/)", [
        "securedrive_fl/budget/__init__.py",
        "securedrive_fl/budget/config.py",
        "securedrive_fl/budget/model.py",
        "securedrive_fl/budget/codec.py",
        "securedrive_fl/budget/aggregate.py",
        "securedrive_fl/budget/scenarios.py",
        "securedrive_fl/budget/time_model.py",
        "securedrive_fl/budget/inversion.py",
        "securedrive_fl/budget/strategies.py",
    ]),
    ("Entry Points & Test Suite", [
        "demo.py",
        "conftest.py",
        "tests/test_securedrive.py",
        "tests/test_budget_invariants.py",
    ]),
    ("Scripts, Tools & Slurm Runners", [
        "scripts/calibrate_he_cost.py",
        "scripts/download_auc_dataset.py",
        "scripts/download_statefarm_dataset.py",
        "scripts/generate_project_txt.py",
        "scripts/make_claims_report.py",
        "scripts/make_figures_ext.py",
        "scripts/make_tables_ext.py",
        "scripts/make_results_md.py",
        "scripts/make_report_ext_md.py",
        "scripts/plot_mitm_analysis.py",
        "scripts/run_audit.py",
        "scripts/run_grid_ext.py",
        "scripts/submit_cpu.sh",
        "scripts/submit_debug.sh",
        "scripts/submit_extension.sh",
        "scripts/submit_gpu.sh",
    ]),
    ("Environment & Configuration", [
        "requirements.txt",
        "environment.yml",
        "data/README.md",
    ]),
    ("Specifications & Architectural Documentation", [
        "final_spec.txt",
        "spec.txt",
        "spec_budget.md",
        "archive/spec.md",
        "hld.txt",
        "DEVIATIONS_EXT.md",
        "guide.md",
        "README.md",
        "report.md",
    ]),
    ("Empirical Benchmark Reports & Tables", [
        "claims_report.md",
        "RESULTS.md",
        "REPORT_EXT.md",
        "table_1.md",
        "table_1.csv",
        "table_1_ext.md",
        "table_1_ext.csv",
        "table_2.md",
        "table_2.csv",
        "table_2_ext.md",
        "table_2_ext.csv",
        "results_ext/calibration.json",
        "results_ext/audit_results.json",
    ]),
]

def build_project_txt():
    print(f"Generating {OUTPUT_FILE}...")
    lines = []
    
    # 1. Header
    lines.append("=" * 80)
    lines.append("PROJECT CONSOLIDATED SOURCE CODE AND SPECIFICATION DUMP")
    lines.append("Project: SecureDrive-FL (sdfl_code)")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("TABLE OF CONTENTS")
    lines.append("-----------------")
    
    total_files = 0
    total_bytes = 0
    total_lines = 0

    file_metadata = []

    for group_name, files in FILE_GROUPS:
        lines.append(f"\n### {group_name}")
        for rel_path in files:
            p = PROJECT_ROOT / rel_path
            if p.exists():
                size = p.stat().st_size
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    n_lines = len(content.splitlines())
                file_metadata.append((group_name, rel_path, n_lines, size, content))
                total_files += 1
                total_bytes += size
                total_lines += n_lines
                lines.append(f"  - {rel_path:<45} ({n_lines:>5} lines, {size:>7} bytes)")
            else:
                lines.append(f"  - [MISSING] {rel_path}")

    lines.append("")
    lines.append("=" * 80)
    lines.append(f"SUMMARY: {total_files} files, {total_lines:,} lines of code/text, {total_bytes / 1024:.1f} KB")
    lines.append("=" * 80)
    lines.append("")

    # 2. File Contents
    for group_name, rel_path, n_lines, size, content in file_metadata:
        sep = "=" * 80
        lines.append(sep)
        lines.append(f"FILE: {rel_path} ({n_lines} lines, {size} bytes)")
        lines.append(f"CATEGORY: {group_name}")
        lines.append(sep)
        lines.append(content)
        if not content.endswith("\n"):
            lines.append("")
        lines.append(f"[END OF FILE: {rel_path}]")
        lines.append("")

    full_text = "\n".join(lines)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Successfully wrote {OUTPUT_FILE}")
    print(f"Total size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"Total lines: {len(full_text.splitlines()):,}")

if __name__ == "__main__":
    build_project_txt()
