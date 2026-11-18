"""Figures for the HOKEA performance runs (log 14).

Usage: python3 scripts/plot_perf.py logs/14_hokea_perf.log jepsen-report/figures

Reads the ELASTICML_PERF lines and draws:
  hokea_commit_timeline.png      committed model version over time per scenario,
                                 grey bands = a worker pod is down,
                                 red bands = voter 2 isolated
  hokea_accuracy_under_churn.png held-out accuracy over time, all scenarios
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
events = [json.loads(line.split("ELASTICML_PERF ", 1)[1])
          for line in log_path.read_text().splitlines() if "ELASTICML_PERF " in line]
summaries = {e["scenario"]: e for e in events if e["event"] == "summary"}
samples = {e["scenario"]: e["samples"] for e in events if e["event"] == "samples"}
faults = {}
for e in events:
    if e["event"] == "fault":
        faults.setdefault(e["scenario"], []).append(e)
order = [s for s in ("healthy", "churn", "churn_partition") if s in summaries]
colors = {"healthy": "#2b6cb0", "churn": "#c05621", "churn_partition": "#9b2c2c"}


def bands(ax, name, label_y):
    down, iso = {}, None
    for f in faults.get(name, []):
        if f["action"] == "kill":
            down[f["node"]] = f["at"]
        elif f["action"] == "start" and f["node"] in down:
            start = down.pop(f["node"])
            ax.axvspan(start, f["at"], color="0.85", lw=0, zorder=0)
            ax.text(start + 1, label_y, f"n{f['node']}", fontsize=7, color="0.35")
        elif f["action"] == "isolate":
            iso = f["at"]
        elif f["action"] == "heal" and iso is not None:
            ax.axvspan(iso, f["at"], color="#f4b6b6", alpha=0.7, lw=0, zorder=0)
            iso = None


fig, axes = plt.subplots(len(order), 1, figsize=(8, 2.5 * len(order)), sharex=True, sharey=True)
axes = [axes] if len(order) == 1 else axes
top = max(s[-1][1] - s[0][1] for s in samples.values())
for ax, name in zip(axes, order):
    data = samples[name]
    base = data[0][1]
    bands(ax, name, top * 0.9)
    ax.step([t for t, _, _ in data], [v - base for _, v, _ in data], where="post",
            color=colors[name])
    s = summaries[name]
    ax.set_title(f"{name}: {s['commits']} commits ({s['commits_per_second']}/s), "
                 f"longest gap {s['max_commit_gap_seconds']} s", fontsize=9, loc="left")
    ax.set_ylabel("commits")
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("seconds since measurement start (grey: worker pod down, red: voter 2 isolated)")
fig.tight_layout()
fig.savefig(out_dir / "hokea_commit_timeline.png", dpi=130)

fig, ax = plt.subplots(figsize=(8, 3))
for name in order:
    points = [(t, a) for t, a in summaries[name]["accuracy"] if a is not None and t > 0]
    ax.plot([t for t, _ in points], [a for _, a in points], marker="o", color=colors[name],
            label=name)
for target in (0.80, 0.85):
    ax.axhline(target, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("seconds since measurement start")
ax.set_ylabel("held-out accuracy")
ax.set_title("fashion_cnn_v5, 6 workers, equal seed", fontsize=9, loc="left")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "hokea_accuracy_under_churn.png", dpi=130)
print("wrote", sorted(p.name for p in out_dir.iterdir()))
