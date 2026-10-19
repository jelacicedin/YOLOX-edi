#!/usr/bin/env python3
"""Compare two result sets and print a markdown table of the differences.

A tag matches bench/results/<tag>.json and bench/results/<tag>_r*.json; with
several repetitions the median per metric is reported, with the min-max spread.

Usage:
    python bench/compare.py A B [--out bench/results/compare.md]
"""

import argparse
import glob
import json
import os
import statistics
import sys

from common import RESULTS_DIR

def collect(blob):
    """Flatten one result file to {metric: (value, higher_is_better)}."""
    rows = {}
    for section, data in blob.items():
        if section == "env":
            continue
        if section.startswith("infer_"):
            for k, v in data.get("latency", {}).items():
                if "median_ms" in v:
                    rows[f"{section} {k} latency (ms)"] = (v["median_ms"], False)
            e2e = data.get("e2e", {})
            for stage in ("preprocess", "model", "postprocess", "total"):
                if stage in e2e:
                    rows[f"{section} e2e {stage} (ms)"] = (e2e[stage]["median_ms"], False)
            if "fps" in e2e:
                rows[f"{section} e2e FPS"] = (e2e["fps"], True)
        elif section.startswith("train_data"):
            rows[f"{section} img/s"] = (data["img_per_s"], True)
        elif section.startswith("train_"):
            rows[f"{section} img/s"] = (data["img_per_s"], True)
            rows[f"{section} peak mem (MB)"] = (data["peak_mem_mb"], False)
            if isinstance(data.get("breakdown"), dict):
                rows[f"{section} assign+loss (ms)"] = (data["breakdown"]["assign_and_loss_ms"], False)
        elif section.startswith("video_"):
            for stage in ("capture", "preprocess", "model", "postprocess", "total"):
                if stage in data:
                    rows[f"{section} {stage} (ms)"] = (data[stage]["median_ms"], False)
            rows[f"{section} FPS"] = (data["fps"], True)
        elif section.startswith("map_"):
            rows[f"{section} AP50:95"] = (data["ap50_95"] * 100, True)
            rows[f"{section} AP50"] = (data["ap50"] * 100, True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("new")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    def load(tag):
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{tag}.json"))
                       + glob.glob(os.path.join(RESULTS_DIR, f"{tag}_r*.json")))
        blobs = []
        for p in paths:
            with open(p) as f:
                blobs.append(json.load(f))
        merged = {}
        for blob in blobs:
            for k, (v, hib) in collect(blob).items():
                merged.setdefault(k, ([], hib))[0].append(v)
        rows = {k: (statistics.median(vs), hib, min(vs), max(vs), len(vs))
                for k, (vs, hib) in merged.items()}
        return blobs[0], rows

    def fmt(v, lo, hi, n):
        return f"{v:.2f}" if n == 1 or hi - lo < 0.005 else f"{v:.2f} ({lo:.1f}–{hi:.1f})"

    a, ra = load(args.base)
    b, rb = load(args.new)
    lines = [
        f"| metric | {args.base} | {args.new} | change |",
        "|---|---:|---:|---:|",
    ]
    for key in ra:
        if key not in rb:
            continue
        (va, hib, la, ha, na), (vb, _, lb, hb, nb) = ra[key], rb[key]
        pct = (vb - va) / va * 100 if va else float("nan")
        if hib is None:
            mark = ""
        else:
            better = (vb > va) if hib else (vb < va)
            mark = " ✅" if better and abs(pct) >= 2 else (" ❌" if not better and abs(pct) >= 2 else "")
        lines.append(f"| {key} | {fmt(va, la, ha, na)} | {fmt(vb, lb, hb, nb)} | {pct:+.1f}%{mark} |")
    for key in rb:
        if key not in ra:
            lines.append(f"| {key} | – | {fmt(*[rb[key][i] for i in (0, 2, 3, 4)])} | new |")
    md = "\n".join(lines)
    md = (f"env {args.base}: {a['env']['git']} / {a['env']['gpu']} / torch {a['env']['torch']}\n\n"
          f"env {args.new}: {b['env']['git']} / {b['env']['gpu']} / torch {b['env']['torch']}\n\n" + md)
    sys.stdout.reconfigure(encoding="utf-8")  # emoji marks on Windows consoles
    print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
