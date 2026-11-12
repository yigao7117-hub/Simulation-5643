"""Aggregate a run's per-condition episodes.csv into a summary table + paired tests + figure.

Usage: python3 scripts/analyze.py results/<run_name>
Outputs (in the run dir): summary_all.csv, paired_tests.json, failure_breakdown.csv, success_rates.png,
  and for instrumented runs: effort_summary.csv, failure_timing.csv, retries.csv
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

# ---- inference effort, failure timing, retries (instrumented runs only) ----
def _f(x):
    return float(x) if x not in ("", None) else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else ""


effort_rows, timing_rows, retry_rows = [], [], []
for c, d in data.items():
    eps = list(d.values())
    if not eps or "infer_ms_total" not in eps[0]:
        continue
    fails = [r for r in eps if r["success"] == "0"]
    succ = [r for r in eps if r["success"] == "1"]
    calls = sum(int(r["policy_calls"]) for r in eps)
    tot = sum(float(r["infer_ms_total"]) for r in eps)
    after_f = sum(float(r["infer_ms_after_last_milestone"]) for r in fails)
    tot_f = sum(float(r["infer_ms_total"]) for r in fails)
    effort_rows.append({
        "condition": c, "episodes": len(eps), "policy_calls": calls,
        "infer_ms_per_call_mean": round(tot / calls, 3),
        "infer_ms_per_call_p95_mean": _mean([_f(r["infer_ms_p95"]) for r in eps]),
        "infer_ms_per_call_max": max(float(r["infer_ms_max"]) for r in eps),
        "envstep_ms_mean": _mean([_f(r["envstep_ms_mean"]) for r in eps]),
        "infer_s_per_episode_success": _mean([_f(r["infer_ms_total"]) / 1000 for r in succ]),
        "infer_s_per_episode_failure": _mean([_f(r["infer_ms_total"]) / 1000 for r in fails]),
        "calls_per_episode_success": _mean([_f(r["policy_calls"]) for r in succ]),
        "calls_per_episode_failure": _mean([_f(r["policy_calls"]) for r in fails]),
        "infer_s_total_failures": round(tot_f / 1000, 1),
        "share_of_failure_inference_after_last_milestone": round(after_f / tot_f, 3) if tot_f else "",
        "wall_s_per_episode_mean": _mean([_f(r["episode_wall_s"]) for r in eps]),
    })
    timing_rows.append({
        "condition": c,
        "step_grasp_red_mean": _mean([_f(r["step_grasp_red"]) for r in eps]),
        "step_stack_red_mean": _mean([_f(r["step_stack_red"]) for r in eps]),
        "step_grasp_green_mean": _mean([_f(r["step_grasp_green"]) for r in eps]),
        "success_steps_mean": _mean([_f(r["steps"]) for r in succ]),
        "fail_last_milestone_step_mean": _mean([_f(r["last_milestone_step"]) for r in fails]),
        "fail_steps_after_last_milestone_mean": _mean([_f(r["steps_after_last_milestone"]) for r in fails]),
        "fail_first_wrong_object_step_mean": _mean([_f(r["step_first_wrong_object"]) for r in fails]),
        "fail_timeout": sum(1 for r in fails if r["outcome"] == "timeout"),
        "fail_drop": sum(1 for r in fails if r["outcome"] == "cube_dropped"),
        **{f"fail_last_milestone={m}": sum(1 for r in fails if r["max_stage"] == m)
           for m in ["none", "grasped_red", "stacked_red", "grasped_green"]},
    })
    cdir = os.path.join(run_dir, f"eval_{c}")
    att = list(csv.DictReader(open(os.path.join(cdir, "attempts.csv")))) if os.path.exists(os.path.join(cdir, "attempts.csv")) else []
    ev = list(csv.DictReader(open(os.path.join(cdir, "events.csv")))) if os.path.exists(os.path.join(cdir, "events.csv")) else []
    retry_rows.append({
        "condition": c, "process_launches": len(att),
        "restarts_after_drop": sum(1 for e in ev if e["event"] == "restart_after_drop"),
        "discarded_invalid_episodes": sum(1 for e in ev if e["event"] == "episode_discarded_invalid_drop"),
        "crashed_launches": sum(1 for a in att if a["exit_code"] != "0"),  # e.g. 139 = segfault; the episode in progress is lost and rerun
        "episodes_not_in_first_launch": sum(1 for r in eps if r.get("attempt", "1") != "1"),
    })

for name, rr in [("effort_summary.csv", effort_rows), ("failure_timing.csv", timing_rows), ("retries.csv", retry_rows)]:
    if rr:
        with open(os.path.join(run_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
            w.writeheader()
            w.writerows(rr)
        print(f"--- {name}")
        for r in rr:
            print(r)

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
