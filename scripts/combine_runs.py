"""Combine repeated evaluation runs (same seeds, same checkpoint) into one analysis.

GPU PhysX is not bit-exact, so the same seed can end differently on a rerun. Each run is treated as a
replicate: an episode is identified by (run, seed), and paired comparisons pair conditions within the same
run and seed.

Usage: python3 scripts/combine_runs.py results/combined results/run1 results/run2 [results/run3 ...]
Outputs (in the output dir): summary.csv, run_agreement.csv, paired_tests.csv, success_rates.png,
  and, from the instrumented runs only: effort.csv, failure_timing.csv, retries.csv
"""
import csv
import math
import os
import sys

out_dir, run_dirs = sys.argv[1], sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)
CONDS = ["C0", "C1", "C2", "C3", "C4", "C5"]
DESC = {"C0": "original", "C1": "+2 ordinary", "C2": "+5 ordinary", "C3": "+5 similar (cube+colour)",
        "C4": "+5 cubes, ordinary colours", "C5": "+5 ordinary shapes, similar colours"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def write(name, rows):
    if not rows:
        return
    with open(os.path.join(out_dir, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"--- {name}")
    for r in rows:
        print(r)


def mean(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return round(sum(xs) / len(xs), 2) if xs else ""


# data[cond][(run, seed)] = row
data = {c: {} for c in CONDS}
runs = [os.path.basename(os.path.normpath(r)) for r in run_dirs]
for run, rd in zip(runs, run_dirs):
    for c in CONDS:
        f = os.path.join(rd, f"eval_{c}", "episodes.csv")
        if os.path.exists(f):
            for r in csv.DictReader(open(f)):
                data[c][(run, r["seed"])] = r
data = {c: d for c, d in data.items() if d}

# success per condition (all runs pooled) + per run
summary = []
for c, d in data.items():
    k, n = sum(r["success"] == "1" for r in d.values()), len(d)
    lo, hi = wilson(k, n)
    row = {"condition": c, "desc": DESC[c], "episodes": n, "successes": k, "success_rate": round(k / n, 3),
           "ci95_low": round(lo, 3), "ci95_high": round(hi, 3)}
    for run in runs:
        ks = [r["success"] == "1" for (rr, _), r in d.items() if rr == run]
        row[f"{run}_rate"] = round(sum(ks) / len(ks), 3) if ks else ""
    summary.append(row)
write("summary.csv", summary)

# run-to-run agreement: same seed, same condition, different run
agree = []
for c, d in data.items():
    seeds = sorted({s for _, s in d})
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            common = [s for s in seeds if (runs[i], s) in d and (runs[j], s) in d]
            same = sum(d[(runs[i], s)]["success"] == d[(runs[j], s)]["success"] for s in common)
            agree.append({"condition": c, "runs": f"{runs[i]} vs {runs[j]}", "seeds": len(common),
                          "same_outcome": same, "agreement": round(same / len(common), 3) if common else ""})
write("run_agreement.csv", agree)

# paired tests within (run, seed)
tests = []
for a, b in [("C0", c) for c in CONDS[1:]] + [("C2", "C4"), ("C2", "C5"), ("C5", "C3"), ("C4", "C3")]:
    if a in data and b in data:
        keys = sorted(set(data[a]) & set(data[b]))
        x = sum(data[a][k]["success"] == "1" and data[b][k]["success"] == "0" for k in keys)
        y = sum(data[a][k]["success"] == "0" and data[b][k]["success"] == "1" for k in keys)
        tests.append({"comparison": f"{a} vs {b}", "pairs": len(keys), f"only_first_success": x,
                      "only_second_success": y, "mcnemar_exact_p": round(mcnemar_exact(x, y), 4)})
write("paired_tests.csv", tests)

# instrumented runs only: inference effort, failure timing, retries
effort, timing, retries = [], [], []
for c, d in data.items():
    eps = [r for r in d.values() if r.get("infer_ms_total")]
    if not eps:
        continue
    fails = [r for r in eps if r["success"] == "0"]
    succ = [r for r in eps if r["success"] == "1"]
    calls = sum(int(r["policy_calls"]) for r in eps)
    tot = sum(float(r["infer_ms_total"]) for r in eps)
    tot_f = sum(float(r["infer_ms_total"]) for r in fails)
    after_f = sum(float(r["infer_ms_after_last_milestone"]) for r in fails)
    effort.append({
        "condition": c, "episodes": len(eps), "infer_ms_per_call": round(tot / calls, 2),
        "infer_ms_p95": mean(r["infer_ms_p95"] for r in eps), "envstep_ms": mean(r["envstep_ms_mean"] for r in eps),
        "calls_per_success": mean(r["policy_calls"] for r in succ),
        "calls_per_failure": mean(r["policy_calls"] for r in fails),
        "infer_s_per_success": mean(float(r["infer_ms_total"]) / 1000 for r in succ),
        "infer_s_per_failure": mean(float(r["infer_ms_total"]) / 1000 for r in fails),
        "failure_inference_share_after_last_milestone": round(after_f / tot_f, 3) if tot_f else "",
    })
    timing.append({
        "condition": c, "failures": len(fails),
        "timeout": sum(r["outcome"] == "timeout" for r in fails),
        "drop": sum(r["outcome"] == "cube_dropped" for r in fails),
        **{f"stuck_after_{m}": sum(r["max_stage"] == m for r in fails)
           for m in ["none", "grasped_red", "stacked_red", "grasped_green"]},
        "step_grasp_red": mean(r["step_grasp_red"] for r in eps),
        "step_stack_red": mean(r["step_stack_red"] for r in eps),
        "step_grasp_green": mean(r["step_grasp_green"] for r in eps),
        "fail_last_milestone_step": mean(r["last_milestone_step"] for r in fails),
        "fail_steps_after_last_milestone": mean(r["steps_after_last_milestone"] for r in fails),
        "fail_first_wrong_object_step": mean(r["step_first_wrong_object"] for r in fails),
    })
    for run, rd in zip(runs, run_dirs):
        cdir = os.path.join(rd, f"eval_{c}")
        ap, ep = os.path.join(cdir, "attempts.csv"), os.path.join(cdir, "events.csv")
        if not os.path.exists(ap):
            continue
        att = list(csv.DictReader(open(ap)))
        ev = list(csv.DictReader(open(ep))) if os.path.exists(ep) else []
        retries.append({
            "run": run, "condition": c, "process_launches": len(att),
            "restarts_after_drop": sum(e["event"] == "restart_after_drop" for e in ev),
            "crashed_launches": sum(a["exit_code"] != "0" for a in att),
            "discarded_invalid_episodes": sum(e["event"] == "episode_discarded_invalid_drop" for e in ev),
        })
write("effort.csv", effort)
write("failure_timing.csv", timing)
write("retries.csv", retries)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    xs = [f"{r['condition']}\n{r['desc'].replace(', ', ',\n')}" for r in summary]
    ys = [r["success_rate"] for r in summary]
    err = [[y - r["ci95_low"] for y, r in zip(ys, summary)], [r["ci95_high"] - y for y, r in zip(ys, summary)]]
    ax.bar(xs, ys, color="#4C72B0", width=0.6)
    ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="#333", capsize=4, lw=1)
    for x, r in zip(xs, summary):
        ax.text(x, r["ci95_high"] + 0.02, f"{r['successes']}/{r['episodes']}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Success rate (95% Wilson CI)")
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "success_rates.png"))
    print("figure saved")
except ImportError:
    print("matplotlib not available; skipped figure")
