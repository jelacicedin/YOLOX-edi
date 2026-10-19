#!/usr/bin/env python3
"""Training throughput benchmark, driving the repo's real Trainer.

Modes
  full : real dataloader (mosaic + mixup) -> images/s as seen in training
  gpu  : K batches are prefetched once and replayed, isolating
         forward + label assignment + loss + backward + optimizer + EMA
  data : dataloader only (no model), images/s the CPU pipeline can deliver

COCO val2017 is used as the training set (we only need realistic images and
boxes, not convergence). Multiscale is off by default so runs are comparable.

Usage:
    python bench/bench_train.py --tag baseline --mode gpu
"""

import argparse
import functools
import itertools
import os
import sys
import time

import numpy as np
import torch

from common import RESULTS_DIR, default_ckpt, get_exp, save_results, seed_all


def patch_dataset_to_val2017():
    import yolox.data as yd
    orig = yd.COCODataset
    yd.COCODataset = functools.partial(orig, name="val2017")


def make_trainer(args):
    from tools.train import make_parser
    from yolox.core import Trainer

    exp = get_exp(args.name, train_ann="instances_val2017.json",
                  output_dir=os.path.join(RESULTS_DIR, "_train_out"),
                  data_num_workers=args.workers, print_interval=10 ** 9)
    if not args.multiscale:
        exp.multiscale_range = 0
    if args.channels_last:
        exp.channels_last = True
    targv = ["-n", args.name, "-b", str(args.batch), "-expn", args.tag]
    if args.fp16:
        targv.append("--fp16")
    ckpt = args.ckpt or default_ckpt(args.name)
    if ckpt:
        targv += ["-c", ckpt]
    targv += args.extra.split() if args.extra else []
    targs = make_parser().parse_args(targv)
    trainer = Trainer(exp, targs)
    return exp, trainer


def timed_iters(trainer, n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses = []
    for i in range(n):
        trainer.iter = i
        trainer.before_iter()
        trainer.train_one_iter()
        trainer.after_iter()
        losses.append(trainer.meter["total_loss"].latest)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, [float(v) for v in losses]


def add_breakdown_hooks(trainer, store):
    """Wrap head.get_losses (label assignment + loss) with synchronized timers."""
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    head = model.head
    orig = head.get_losses

    def wrapped(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        torch.cuda.synchronize()
        store.append((time.perf_counter() - t0) * 1000)
        return out

    head.get_losses = wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("-n", "--name", default="yolox-s")
    parser.add_argument("-c", "--ckpt", default=None)
    parser.add_argument("--mode", choices=["full", "gpu", "data"], default="gpu")
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    # Windows spawns workers slowly: warm up long enough to reach steady state
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--replay", type=int, default=8, help="batches kept in gpu mode")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--extra", default="", help="extra args passed to tools/train.py parser")
    args = parser.parse_args()

    seed_all(0)
    torch.backends.cudnn.benchmark = True
    patch_dataset_to_val2017()
    exp, trainer = make_trainer(args)

    if args.mode == "data":
        loader = exp.get_data_loader(batch_size=args.batch, is_distributed=False)
        it = iter(loader)
        for _ in range(args.warmup):
            next(it)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            next(it)
        dt = time.perf_counter() - t0
        res = {"img_per_s": args.iters * args.batch / dt, "ms_per_batch": dt / args.iters * 1000}
        print(res)
        save_results(args.tag, f"train_data_{args.name}_w{args.workers}", {**vars(args), **res})
        return

    trainer.before_train()
    trainer.epoch = trainer.start_epoch
    trainer.before_epoch()

    if args.mode == "gpu":
        batches = [trainer.prefetcher.next() for _ in range(args.replay)]
        cycle = itertools.cycle(batches)
        trainer.prefetcher.next = lambda: tuple(t.clone() for t in next(cycle))

    _, _ = timed_iters(trainer, args.warmup)
    torch.cuda.reset_peak_memory_stats()
    elapsed, losses = timed_iters(trainer, args.iters)

    res = {
        "img_per_s": args.iters * args.batch / elapsed,
        "ms_per_iter": elapsed / args.iters * 1000,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
        "mean_loss": float(np.mean(losses)),
    }
    if args.breakdown:
        store = []
        add_breakdown_hooks(trainer, store)
        el2, _ = timed_iters(trainer, min(args.iters, 30))
        n = len(store)
        res["breakdown"] = {
            "ms_per_iter_synced": el2 / n * 1000,
            "assign_and_loss_ms": float(np.mean(store)),
            "assign_and_loss_frac": float(np.mean(store) / (el2 / n * 1000)),
        }
    print(res)
    suffix = "_cl" if args.channels_last else ""
    save_results(args.tag, f"train_{args.mode}_{args.name}_bs{args.batch}{suffix}", {**vars(args), **res})
    sys.stdout.flush()
    os._exit(0)  # don't wait on dataloader workers / tensorboard


if __name__ == "__main__":
    main()
