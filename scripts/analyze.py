"""Aggregate a run's per-condition episodes.csv into a summary table + paired tests + figure.

Usage: python3 scripts/analyze.py results/<run_name>
Outputs (in the run dir): summary_all.csv, paired_tests.json, failure_breakdown.csv, success_rates.png
"""
import csv
import json
import math
import os
import sys
from collections import Counter

run_dir = os.path.abspath(sys.argv[1])
CONDS = ["C0", "C1", "C2", "C3", "C4", "C5"]
DESC = {"C0": "original", "C1": "+2 ordinary", "C2": "+5 ordinary", "C3": "+5 similar", "C4": "+5 cubes (shape)", "C5": "+5 colors (color)"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """two-sided exact binomial test on discordant pairs"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


data = {}
for c in CONDS:
    f = os.path.join(run_dir, f"eval_{c}", "episodes.csv")
    if os.path.exists(f):
        with open(f) as fh:
            data[c] = {int(r["seed"]): r for r in csv.DictReader(fh)}

rows, fail_rows = [], []
all_types = sorted({r["failure_type"] for d in data.values() for r in d.values() if r["failure_type"]})
for c, d in data.items():
    n = len(d)
    k = sum(int(r["success"]) for r in d.values())
    lo, hi = wilson(k, n)
    wrong = sum(1 for r in d.values() if int(r["wrong_object_steps"] or 0) > 0)
    rows.append({"condition": c, "desc": DESC[c], "episodes": n, "successes": k, "success_rate": round(k / n, 3),
                 "ci95_low": round(lo, 3), "ci95_high": round(hi, 3), "episodes_with_wrong_object_contact": wrong})
    cnt = Counter(r["failure_type"] for r in d.values() if r["failure_type"])
    fail_rows.append({"condition": c, **{t: cnt.get(t, 0) for t in all_types}})

with open(os.path.join(run_dir, "summary_all.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(run_dir, "failure_breakdown.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition"] + all_types)
    w.writeheader()
    w.writerows(fail_rows)

tests = {}
if "C0" in data:
    for c in CONDS[1:]:
        if c not in data:
            continue
        seeds = sorted(set(data["C0"]) & set(data[c]))
        b = sum(1 for s in seeds if data["C0"][s]["success"] == "1" and data[c][s]["success"] == "0")
        cc = sum(1 for s in seeds if data["C0"][s]["success"] == "0" and data[c][s]["success"] == "1")
        tests[f"C0_vs_{c}"] = {"pairs": len(seeds), "C0_only_success": b, f"{c}_only_success": cc,
                               "mcnemar_exact_p": round(mcnemar_exact(b, cc), 4)}
# 2x2 ablation (shape x color): compare conditions that differ in one factor only
for a, b in [("C2", "C4"), ("C2", "C5"), ("C5", "C3"), ("C4", "C3")]:
    if a in data and b in data:
        seeds = sorted(set(data[a]) & set(data[b]))
        x = sum(1 for s in seeds if data[a][s]["success"] == "1" and data[b][s]["success"] == "0")
        y = sum(1 for s in seeds if data[a][s]["success"] == "0" and data[b][s]["success"] == "1")
        tests[f"{a}_vs_{b}"] = {"pairs": len(seeds), f"{a}_only_success": x, f"{b}_only_success": y,
                                "mcnemar_exact_p": round(mcnemar_exact(x, y), 4)}
with open(os.path.join(run_dir, "paired_tests.json"), "w") as f:
    json.dump(tests, f, indent=2)

for r in rows:
    print(r)
print(json.dumps(tests, indent=1))
for r in fail_rows:
    print(r)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.4 * len(rows) + 0.5, 3.6), dpi=150)
    xs = [f"{r['condition']}\n{r['desc']}" for r in rows]
    ys = [r["success_rate"] for r in rows]
    err = [[y - r["ci95_low"] for y, r in zip(ys, rows)], [r["ci95_high"] - y for y, r in zip(ys, rows)]]
    ax.bar(xs, ys, color="#4C72B0", width=0.6)
    ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="#333", capsize=4, lw=1)
    for x, r in zip(xs, rows):
        ax.text(x, r["success_rate"] + 0.03, f"{r['successes']}/{r['episodes']}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Success rate (95% Wilson CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "success_rates.png"))
    print("figure saved")
except ImportError:
    print("matplotlib not available; skipped figure")
