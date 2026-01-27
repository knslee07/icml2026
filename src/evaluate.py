#!/usr/bin/env python3
"""
Evaluation Script for Compensation Consultant Extraction

Computes precision, recall, and F1 metrics comparing model predictions
against gold standard labels.

Paper: "Domain-Specific Alignment for Information Extraction:
        A Framework for Social Science Data Collection"

METRICS COMPUTED:
    1. Instance-based: Each consultant counts as 1 point
       - Overall P/R/F1 across all predictions
       - Category-specific P/R/F1 for RET and SURV separately

    2. Pooled: RET + SURV combined, ignoring categorization
       - Measures entity detection independent of role classification

    3. Traditional (macro-averaged): Per-sample P/R/F1 averaged

USAGE:
    python evaluate.py --predictions results.json --gold gold_labels.json --output eval.csv

GOLD LABEL FORMAT (JSON):
    {
        "entries": [
            {
                "company": "COMPANY_TICKER",
                "year": 2023,
                "ret_consultants": "Pearl Meyer & Partners; Mercer",
                "surv_consultants": "Radford; NO_CONSULTANTS"
            }
        ]
    }
"""

import argparse
import json
import csv
import os
import glob
import re
from collections import defaultdict
from typing import Dict, Tuple, List, Set


def normalize_name(s: str) -> str:
    """Normalize consultant name for comparison."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def parse_consultant_list(s: str) -> List[str]:
    """Parse semicolon-separated consultant names."""
    s = s.strip()
    if not s:
        return []
    if s.upper() == "NO_CONSULTANTS":
        return ["NO_CONSULTANTS"]
    return [name.strip() for name in s.split(";") if name.strip()]


def load_predictions(raw_dir: str, normalize: bool = False) -> Dict[Tuple[str, int], Dict[str, List[str]]]:
    """
    Load predictions from inference output JSON.

    Expected format (from inference.py):
        {
            "company_results": {
                "TICKER": {
                    "consultants_by_year": {
                        "2023": {"RET": [...], "SURV": [...]}
                    }
                }
            }
        }
    """
    # Find latest results file
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No JSON files found in {raw_dir}")
    path = paths[-1]

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out = {}
    company_results = data.get("company_results", {})

    for company, cdata in company_results.items():
        by_year = cdata.get("consultants_by_year", {})
        for year_str, year_data in by_year.items():
            if year_str == "Unknown":
                continue
            try:
                year = int(year_str)
            except ValueError:
                continue

            ret = year_data.get("RET", []) or []
            surv = year_data.get("SURV", []) or []

            if not ret:
                ret = ["NO_CONSULTANTS"]
            if not surv:
                surv = ["NO_CONSULTANTS"]

            if normalize:
                ret = sorted(set(normalize_name(x) for x in ret if x.strip()))
                surv = sorted(set(normalize_name(x) for x in surv if x.strip()))
            else:
                ret = sorted(set(x.strip() for x in ret if x.strip()))
                surv = sorted(set(x.strip() for x in surv if x.strip()))

            out[(company, year)] = {"RET": ret, "SURV": surv}

    return out, path


def load_gold(gold_path: str, normalize: bool = False) -> Dict[Tuple[str, int], Dict[str, List[str]]]:
    """
    Load gold standard labels.

    Supports JSON format with entries list.
    """
    with open(gold_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and 'entries' in data:
        entries = data['entries']
    else:
        raise ValueError("JSON must contain entries list")

    out = {}

    def maybe_norm(x: str) -> str:
        return normalize_name(x) if normalize else x.strip()

    for entry in entries:
        company = entry.get('company', '').strip()
        year = entry.get('year')

        if not company or not year:
            continue

        ret_raw = entry.get('ret_consultants', entry.get('RET', ''))
        surv_raw = entry.get('surv_consultants', entry.get('SURV', ''))

        ret_list = parse_consultant_list(str(ret_raw))
        surv_list = parse_consultant_list(str(surv_raw))

        if normalize:
            ret_list = sorted(set(maybe_norm(x) for x in ret_list))
            surv_list = sorted(set(maybe_norm(x) for x in surv_list))
        else:
            ret_list = sorted(set(ret_list))
            surv_list = sorted(set(surv_list))

        out[(company, year)] = {"RET": ret_list, "SURV": surv_list}

    return out


def prf(pred: set, truth: set) -> Tuple[float, float, float, int]:
    """Calculate precision, recall, F1, and true positives."""
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 1.0 if not truth else 0.0
    r = tp / len(truth) if truth else 1.0 if not pred else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f, tp


def evaluate(pred_map: dict, gold_map: dict, normalize: bool = False):
    """
    Compute all evaluation metrics.

    Returns:
        rows: Per-sample detailed results
        metrics: Aggregate metrics dictionary
    """
    rows = []

    # Aggregate counters
    total_correct = 0
    total_pred = 0
    total_gold = 0

    ret_correct = 0
    ret_pred = 0
    ret_gold = 0

    surv_correct = 0
    surv_pred = 0
    surv_gold = 0

    pooled_correct = 0
    pooled_pred = 0
    pooled_gold = 0

    # Traditional macro counters
    trad_macro = defaultdict(float)
    n = 0

    # Default for missing predictions
    default_val = "no_consultants" if normalize else "NO_CONSULTANTS"
    default_pred = {"RET": [default_val], "SURV": [default_val]}

    # Evaluate each gold sample
    for key in sorted(gold_map.keys()):
        g = gold_map[key]
        p = pred_map.get(key, default_pred)

        g_ret = set(g["RET"])
        g_surv = set(g["SURV"])
        p_ret = set(p["RET"])
        p_surv = set(p["SURV"])

        # Pooled (ignore categorization)
        g_pooled = g_ret | g_surv
        p_pooled = p_ret | p_surv

        # Traditional metrics
        p_ret_m, r_ret_m, f_ret_m, tp_ret = prf(p_ret, g_ret)
        p_surv_m, r_surv_m, f_surv_m, tp_surv = prf(p_surv, g_surv)

        # Pooled metrics
        pooled_tp = len(p_pooled & g_pooled)

        # Instance counts
        inst_correct = tp_ret + tp_surv
        inst_pred = len(p_ret) + len(p_surv)
        inst_gold = len(g_ret) + len(g_surv)

        # Update totals
        total_correct += inst_correct
        total_pred += inst_pred
        total_gold += inst_gold

        ret_correct += tp_ret
        ret_pred += len(p_ret)
        ret_gold += len(g_ret)

        surv_correct += tp_surv
        surv_pred += len(p_surv)
        surv_gold += len(g_surv)

        pooled_correct += pooled_tp
        pooled_pred += len(p_pooled)
        pooled_gold += len(g_pooled)

        # Traditional macro
        for m, v in [("p_ret", p_ret_m), ("r_ret", r_ret_m), ("f_ret", f_ret_m),
                     ("p_surv", p_surv_m), ("r_surv", r_surv_m), ("f_surv", f_surv_m)]:
            trad_macro[m] += v
        n += 1

        # Per-sample metrics
        sample_p = inst_correct / inst_pred if inst_pred > 0 else 1.0
        sample_r = inst_correct / inst_gold if inst_gold > 0 else 1.0
        sample_f1 = 2 * sample_p * sample_r / (sample_p + sample_r) if (sample_p + sample_r) > 0 else 0.0

        rows.append({
            "company": key[0],
            "year": key[1],
            "pred_RET": ";".join(sorted(p["RET"])),
            "true_RET": ";".join(sorted(g["RET"])),
            "pred_SURV": ";".join(sorted(p["SURV"])),
            "true_SURV": ";".join(sorted(g["SURV"])),
            "inst_correct": inst_correct,
            "inst_pred": inst_pred,
            "inst_gold": inst_gold,
            "inst_f1": sample_f1,
        })

    # Compute aggregate metrics
    if n:
        for k in trad_macro:
            trad_macro[k] /= n

    overall_p = total_correct / total_pred if total_pred > 0 else 0.0
    overall_r = total_correct / total_gold if total_gold > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

    ret_p = ret_correct / ret_pred if ret_pred > 0 else 0.0
    ret_r = ret_correct / ret_gold if ret_gold > 0 else 0.0
    ret_f1 = 2 * ret_p * ret_r / (ret_p + ret_r) if (ret_p + ret_r) > 0 else 0.0

    surv_p = surv_correct / surv_pred if surv_pred > 0 else 0.0
    surv_r = surv_correct / surv_gold if surv_gold > 0 else 0.0
    surv_f1 = 2 * surv_p * surv_r / (surv_p + surv_r) if (surv_p + surv_r) > 0 else 0.0

    pooled_p = pooled_correct / pooled_pred if pooled_pred > 0 else 0.0
    pooled_r = pooled_correct / pooled_gold if pooled_gold > 0 else 0.0
    pooled_f1 = 2 * pooled_p * pooled_r / (pooled_p + pooled_r) if (pooled_p + pooled_r) > 0 else 0.0

    metrics = {
        # Instance-based
        "overall_precision": overall_p,
        "overall_recall": overall_r,
        "overall_f1": overall_f1,
        "overall_correct": total_correct,
        "overall_predicted": total_pred,
        "overall_gold": total_gold,
        # RET
        "ret_precision": ret_p,
        "ret_recall": ret_r,
        "ret_f1": ret_f1,
        "ret_correct": ret_correct,
        "ret_gold": ret_gold,
        # SURV
        "surv_precision": surv_p,
        "surv_recall": surv_r,
        "surv_f1": surv_f1,
        "surv_correct": surv_correct,
        "surv_gold": surv_gold,
        # Pooled
        "pooled_precision": pooled_p,
        "pooled_recall": pooled_r,
        "pooled_f1": pooled_f1,
        # Traditional (macro)
        "trad_ret_f1": trad_macro.get("f_ret", 0),
        "trad_surv_f1": trad_macro.get("f_surv", 0),
        # Counts
        "num_samples": len(rows),
    }

    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate consultant extraction")
    parser.add_argument("--predictions", required=True,
                        help="Folder containing prediction JSON files")
    parser.add_argument("--gold", required=True,
                        help="Gold label JSON file")
    parser.add_argument("--output", default=None,
                        help="Output CSV for detailed results")
    parser.add_argument("--normalize", action="store_true",
                        help="Lowercase names for comparison")

    args = parser.parse_args()

    # Load data
    pred_map, src_file = load_predictions(args.predictions, normalize=args.normalize)
    gold_map = load_gold(args.gold, normalize=args.normalize)

    # Evaluate
    rows, metrics = evaluate(pred_map, gold_map, normalize=args.normalize)

    # Print results
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Predictions: {os.path.basename(src_file)}")
    print(f"Gold labels: {os.path.basename(args.gold)}")
    print(f"Samples: {metrics['num_samples']}")
    print("=" * 70)

    print("\nINSTANCE-BASED METRICS:")
    print(f"  Overall: {metrics['overall_correct']}/{metrics['overall_gold']} correct")
    print(f"           P={metrics['overall_precision']:.3f} R={metrics['overall_recall']:.3f} F1={metrics['overall_f1']:.3f}")
    print(f"  RET:     {metrics['ret_correct']}/{metrics['ret_gold']} correct")
    print(f"           P={metrics['ret_precision']:.3f} R={metrics['ret_recall']:.3f} F1={metrics['ret_f1']:.3f}")
    print(f"  SURV:    {metrics['surv_correct']}/{metrics['surv_gold']} correct")
    print(f"           P={metrics['surv_precision']:.3f} R={metrics['surv_recall']:.3f} F1={metrics['surv_f1']:.3f}")

    print(f"\nPOOLED METRICS (RET+SURV combined):")
    print(f"           P={metrics['pooled_precision']:.3f} R={metrics['pooled_recall']:.3f} F1={metrics['pooled_f1']:.3f}")

    print("=" * 70)

    # Save detailed results
    if args.output and rows:
        fieldnames = ["company", "year", "pred_RET", "true_RET", "pred_SURV", "true_SURV",
                      "inst_correct", "inst_pred", "inst_gold", "inst_f1"]
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
