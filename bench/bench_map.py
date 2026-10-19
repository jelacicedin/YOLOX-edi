#!/usr/bin/env python3
"""COCO val2017 accuracy benchmark (mAP via the repo's COCOEvaluator).

Evaluates the inference path actually used for deployment (fused, fp16) so
that inference-side refactors (fusion, re-parameterisation, SPPF ...) are
checked for accuracy as well as speed.

Usage:
    python bench/bench_map.py --tag baseline -n yolox-s
"""

import argparse
import time

import torch

from common import default_ckpt, get_exp, load_model, save_results, seed_all

from yolox.utils import fuse_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("-n", "--name", default="yolox-s")
    parser.add_argument("-c", "--ckpt", default=None)
    parser.add_argument("-b", "--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-fuse", dest="fuse", action="store_false")
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--fast-cocoeval", action="store_true")
    args = parser.parse_args()

    if not args.fast_cocoeval:
        # the fast COCOeval op is JIT-compiled C++ and needs a compiler; force pycocotools
        import yolox.layers
        if hasattr(yolox.layers, "COCOeval_opt"):
            delattr(yolox.layers, "COCOeval_opt")

    seed_all(0)
    torch.backends.cudnn.benchmark = True
    exp = get_exp(args.name, data_num_workers=args.workers)
    model = load_model(exp, args.ckpt or default_ckpt(args.name)).eval()
    if args.fuse:
        model = fuse_model(model)
    evaluator = exp.get_evaluator(batch_size=args.batch, is_distributed=False)

    t0 = time.perf_counter()
    ap50_95, ap50, summary = evaluator.evaluate(model, False, args.half)
    wall = time.perf_counter() - t0
    print(summary)
    save_results(args.tag, f"map_{args.name}", {
        "ap50_95": float(ap50_95), "ap50": float(ap50),
        "fuse": args.fuse, "half": args.half, "eval_wall_s": wall, "summary": summary,
    })


if __name__ == "__main__":
    main()
