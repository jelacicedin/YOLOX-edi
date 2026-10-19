#!/usr/bin/env python3
# Shared helpers for the YOLOX benchmark harness.

import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS_DIR = os.environ.get("BENCH_RESULTS_DIR", os.path.join(ROOT, "bench", "results"))
WEIGHTS_DIR = os.path.join(ROOT, "weights")
COCO_DIR = os.path.join(ROOT, "datasets", "COCO")


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_rev():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "yolox", "tools"], cwd=ROOT, text=True)
        return rev + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def env_info():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "git": git_rev(),
        "yolox_path": os.path.dirname(__import__("yolox").__file__),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_exp(name, **overrides):
    from yolox.exp import get_exp as _get_exp
    exp = _get_exp(None, name)
    exp.data_dir = COCO_DIR
    for k, v in overrides.items():
        setattr(exp, k, v)
    return exp


def load_model(exp, ckpt=None, device="cuda"):
    exp.__dict__.pop("model", None)  # exp caches the model; always build a fresh one
    model = exp.get_model()
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    return model.to(device)


def default_ckpt(name):
    path = os.path.join(WEIGHTS_DIR, name.replace("-", "_") + ".pth")
    return path if os.path.exists(path) else None


def cuda_time(fn, warmup=20, iters=100):
    """Run fn() repeatedly; return per-call latencies in ms (CUDA-synchronized)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def summarize(times_ms):
    a = np.asarray(times_ms)
    return {
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "std_ms": float(a.std()),
    }


def save_results(tag, section, data):
    """Merge `data` into bench/results/<tag>.json under key `section`."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{tag}.json")
    blob = {}
    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)
    blob.setdefault("env", env_info())
    blob[section] = data
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"[bench] wrote {section} -> {path}")
    return path
