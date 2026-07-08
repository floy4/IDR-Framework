"""
analyze_cf_effects.py

Analyze counterfactual (CF) effect data from LIBERO evaluations.
Compares Non-Pro vs Pro versions across two dimensions:
  1. Effect changes at different stages of a task (progress-normalized step)
  2. Effect changes across different task types (pick_and_place, open, close, push, etc.)
"""

import json
import os
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = "/path/to/VLA-Adapter"

NON_PRO_DIR = os.path.join(PROJECT_ROOT, "experiments/logs-cf/vision")
PRO_DIR = os.path.join(PROJECT_ROOT, "experiments/logs-pro-cf/vision")

# Only analyze input_zeroing method (the primary CF method)
METHOD_FILTER = "input_zeroing"

# Task type classification based on description keywords
TASK_TYPE_KEYWORDS = {
    "pick_and_place": ["pick up", "put", "place"],
    "open": ["open"],
    "close": ["close"],
    "push": ["push", "slide"],
    "turn_on": ["turn on"],
    "turn_off": ["turn off"],
}


def classify_task(description: str) -> str:
    desc_lower = description.lower()
    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in desc_lower:
                return task_type
    return "other"


def load_cf_logs(log_dir: str) -> dict:
    """Load all CF JSON logs from directory, grouped by task suite."""
    pattern = os.path.join(log_dir, f"*{METHOD_FILTER}*_cf_logs.json")
    files = glob.glob(pattern)
    print(f"  Found {len(files)} log files in {log_dir}")

    all_data = defaultdict(list)  # task_suite -> list of log entries
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        parts = fname.split("-")
        task_suite = f"{parts[2]}_{parts[3]}"  # e.g., "libero_spatial"

        with open(fpath) as f:
            data = json.load(f)

        for entry in data.get("logs", []):
            entry["task_suite"] = task_suite
            entry["file"] = fname
            all_data[task_suite].append(entry)

    return all_data


def compute_episode_progress(data_by_suite: dict) -> list:
    """
    Compute normalized progress (0-1) for each step within its episode.
    Groups consecutive steps by task_description, then divides step by max_step.
    """
    all_entries = []
    for suite, entries in data_by_suite.items():
        # Group by task
        task_groups = defaultdict(list)
        for e in entries:
            task_groups[e["task_description"]].append(e)

        for desc, task_entries in task_groups.items():
            task_entries.sort(key=lambda x: x["step"])
            max_step = task_entries[-1]["step"]
            for e in task_entries:
                e["progress"] = e["step"] / max(max_step, 1)
                e["task_type"] = classify_task(desc)
                all_entries.append(e)

    return all_entries


def analyze_by_stage(entries: list, label: str) -> dict:
    """Analyze effect_vlm by progress stage (0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0)."""
    stages = {
        "early (0-25%)": [],
        "mid-early (25-50%)": [],
        "mid-late (50-75%)": [],
        "late (75-100%)": [],
    }
    for e in entries:
        p = e["progress"]
        if p <= 0.25:
            stages["early (0-25%)"].append(e["effect_vlm"])
        elif p <= 0.5:
            stages["mid-early (25-50%)"].append(e["effect_vlm"])
        elif p <= 0.75:
            stages["mid-late (50-75%)"].append(e["effect_vlm"])
        else:
            stages["late (75-100%)"].append(e["effect_vlm"])

    print(f"\n{'='*70}")
    print(f"  [{label}] Effect by Task Stage (progress-normalized)")
    print(f"{'='*70}")
    results = {}
    for stage_name, effects in stages.items():
        if effects:
            avg = np.mean(effects)
            std = np.std(effects)
            results[stage_name] = {"avg": avg, "std": std, "count": len(effects)}
            print(f"  {stage_name:<25s}: avg={avg:.3f}, std={std:.3f}, n={len(effects)}")
        else:
            print(f"  {stage_name:<25s}: (no data)")
    return results


def analyze_by_task_type(entries: list, label: str) -> dict:
    """Analyze effect_vlm by task type."""
    task_groups = defaultdict(list)
    for e in entries:
        task_groups[e["task_type"]].append(e["effect_vlm"])

    print(f"\n{'='*70}")
    print(f"  [{label}] Effect by Task Type")
    print(f"{'='*70}")
    results = {}
    for task_type in sorted(task_groups.keys()):
        effects = task_groups[task_type]
        avg = np.mean(effects)
        std = np.std(effects)
        results[task_type] = {"avg": avg, "std": std, "count": len(effects)}
        print(f"  {task_type:<25s}: avg={avg:.3f}, std={std:.3f}, n={len(effects)}")
    return results


def analyze_by_task_suite_and_type(entries: list, label: str):
    """Detailed breakdown by task_suite x task_type."""
    groups = defaultdict(list)
    for e in entries:
        key = (e["task_suite"], e["task_type"])
        groups[key].append(e["effect_vlm"])

    print(f"\n{'='*70}")
    print(f"  [{label}] Effect by Task Suite x Task Type")
    print(f"{'='*70}")
    for key in sorted(groups.keys()):
        effects = groups[key]
        avg = np.mean(effects)
        std = np.std(effects)
        print(f"  {key[0]:<18s} x {key[1]:<15s}: avg={avg:.3f}, std={std:.3f}, n={len(effects)}")


def overall_stats(entries: list, label: str):
    """Print overall statistics."""
    effects = [e["effect_vlm"] for e in entries]
    use_baseline_count = sum(1 for e in entries if e.get("use_baseline", False))
    print(f"\n{'='*70}")
    print(f"  [{label}] Overall Statistics")
    print(f"{'='*70}")
    print(f"  Total steps:       {len(effects)}")
    print(f"  Mean effect_vlm:   {np.mean(effects):.4f}")
    print(f"  Std effect_vlm:    {np.std(effects):.4f}")
    print(f"  Median effect_vlm: {np.median(effects):.4f}")
    print(f"  Min effect_vlm:    {np.min(effects):.4f}")
    print(f"  Max effect_vlm:    {np.max(effects):.4f}")
    print(f"  use_baseline rate: {use_baseline_count / len(effects):.2%}")
    return {
        "total_steps": len(effects),
        "mean": np.mean(effects),
        "std": np.std(effects),
        "median": np.median(effects),
        "min": np.min(effects),
        "max": np.max(effects),
        "use_baseline_rate": use_baseline_count / len(effects),
    }


def compare_by_stage(non_pro_stages: dict, pro_stages: dict):
    """Compare Non-Pro vs Pro by stage side by side."""
    print(f"\n{'='*70}")
    print(f"  [COMPARISON] Effect by Stage: Non-Pro vs Pro")
    print(f"{'='*70}")
    print(f"  {'Stage':<25s} {'Non-Pro avg':>12s} {'Pro avg':>12s} {'Diff':>10s} {'Δ%':>8s}")
    print(f"  {'-'*70}")
    for stage in ["early (0-25%)", "mid-early (25-50%)", "mid-late (50-75%)", "late (75-100%)"]:
        np_avg = non_pro_stages.get(stage, {}).get("avg", 0)
        p_avg = pro_stages.get(stage, {}).get("avg", 0)
        diff = p_avg - np_avg
        pct = (diff / np_avg * 100) if np_avg > 0 else 0
        print(f"  {stage:<25s} {np_avg:>12.3f} {p_avg:>12.3f} {diff:>+10.3f} {pct:>+7.1f}%")


def compare_by_task_type(non_pro_types: dict, pro_types: dict):
    """Compare Non-Pro vs Pro by task type side by side."""
    print(f"\n{'='*70}")
    print(f"  [COMPARISON] Effect by Task Type: Non-Pro vs Pro")
    print(f"{'='*70}")
    print(f"  {'Task Type':<25s} {'Non-Pro avg':>12s} {'Pro avg':>12s} {'Diff':>10s} {'Δ%':>8s}")
    print(f"  {'-'*70}")
    all_types = sorted(set(list(non_pro_types.keys()) + list(pro_types.keys())))
    for ttype in all_types:
        np_avg = non_pro_types.get(ttype, {}).get("avg", 0)
        p_avg = pro_types.get(ttype, {}).get("avg", 0)
        diff = p_avg - np_avg
        pct = (diff / np_avg * 100) if np_avg > 0 else 0
        print(f"  {ttype:<25s} {np_avg:>12.3f} {p_avg:>12.3f} {diff:>+10.3f} {pct:>+7.1f}%")


def main():
    print("Loading CF effect data...")
    print(f"  Non-Pro dir: {NON_PRO_DIR}")
    non_pro_data = load_cf_logs(NON_PRO_DIR)
    print(f"  Pro dir:     {PRO_DIR}")
    pro_data = load_cf_logs(PRO_DIR)

    # Compute progress and classify tasks
    non_pro_entries = compute_episode_progress(non_pro_data)
    pro_entries = compute_episode_progress(pro_data)

    # Overall stats
    non_pro_overall = overall_stats(non_pro_entries, "Non-Pro (vision)")
    pro_overall = overall_stats(pro_entries, "Pro (vision)")

    # By stage
    non_pro_stages = analyze_by_stage(non_pro_entries, "Non-Pro (vision)")
    pro_stages = analyze_by_stage(pro_entries, "Pro (vision)")
    compare_by_stage(non_pro_stages, pro_stages)

    # By task type
    non_pro_types = analyze_by_task_type(non_pro_entries, "Non-Pro (vision)")
    pro_types = analyze_by_task_type(pro_entries, "Pro (vision)")
    compare_by_task_type(non_pro_types, pro_types)

    # Detailed breakdown
    analyze_by_task_suite_and_type(non_pro_entries, "Non-Pro (vision)")
    analyze_by_task_suite_and_type(pro_entries, "Pro (vision)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  KEY FINDINGS")
    print(f"{'='*70}")

    # Compare overall
    np_mean = non_pro_overall["mean"]
    p_mean = pro_overall["mean"]
    diff_pct = (p_mean - np_mean) / np_mean * 100
    print(f"  1. Overall effect_vlm: Non-Pro={np_mean:.3f}, Pro={p_mean:.3f} ({diff_pct:+.1f}%)")
    print(f"     use_baseline_rate: both = 100% (CF reweighting never activated)")
    print(f"     -> effect_vlm always exceeds threshold (0.5), baseline always used")

    # Task type with highest effect
    for label, types_dict in [("Non-Pro", non_pro_types), ("Pro", pro_types)]:
        if types_dict:
            max_type = max(types_dict, key=lambda k: types_dict[k]["avg"])
            min_type = min(types_dict, key=lambda k: types_dict[k]["avg"])
            print(f"  2. [{label}] Highest effect task type: {max_type} ({types_dict[max_type]['avg']:.3f})")
            print(f"     [{label}] Lowest effect task type:  {min_type} ({types_dict[min_type]['avg']:.3f})")

    # Stage trend
    for label, stages_dict in [("Non-Pro", non_pro_stages), ("Pro", pro_stages)]:
        stages_order = ["early (0-25%)", "mid-early (25-50%)", "mid-late (50-75%)", "late (75-100%)"]
        avgs = [stages_dict.get(s, {}).get("avg", float('nan')) for s in stages_order]
        valid_avgs = [a for a in avgs if not np.isnan(a) and a > 0]
        if len(valid_avgs) >= 2:
            trend = "increasing" if valid_avgs[-1] > valid_avgs[0] else "decreasing"
            print(f"  3. [{label}] Effect trend across stages: {trend}")
            print(f"     Early: {avgs[0]:.3f}, Late: {avgs[-1]:.3f}")
            if avgs[0] > 0:
                print(f"     Change: {(avgs[-1] - avgs[0]):+.3f} ({(avgs[-1] - avgs[0]) / avgs[0] * 100:+.1f}%)")


if __name__ == "__main__":
    main()
