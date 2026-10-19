#!/usr/bin/env python3
"""Numerical equivalence check between two versions of the code.

    # on the reference code (e.g. the baseline commit)
    python bench/check_equiv.py dump --out bench/results/equiv_ref.pt
    # on the modified code
    python bench/check_equiv.py compare --ref bench/results/equiv_ref.pt

Checks (fp32, deterministic cuDNN, TF32 off):
  * eval outputs of the plain and the fused (deploy) model
  * training losses, num_fg, and gradients on a stored mosaic batch, with and
    without the L1 loss -- this exercises SimOTA label assignment end to end
"""

import argparse
import functools

import torch

from common import default_ckpt, get_exp, load_model, seed_all

from yolox.utils import fuse_model

GRAD_PARAMS = ("backbone.backbone.stem", "head.cls_preds.0", "head.reg_preds.2", "head.obj_preds.1")


def setup():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_batch(exp, batch):
    import yolox.data as yd
    yd.COCODataset = functools.partial(yd.COCODataset, name="val2017")
    exp.train_ann = "instances_val2017.json"
    exp.data_num_workers = 0
    loader = exp.get_data_loader(batch_size=batch, is_distributed=False)
    imgs, targets, _, _ = next(iter(loader))
    return imgs.float(), targets.float()


def run_all(exp, ckpt, imgs, targets):
    out = {}
    model = load_model(exp, ckpt)

    model.eval()
    with torch.no_grad():
        x = imgs.cuda()
        out["eval"] = model(x).cpu()
        out["eval_fused"] = fuse_model(load_model(exp, ckpt).eval())(x).cpu()

    for use_l1 in (False, True):
        model = load_model(exp, ckpt).train()
        model.head.use_l1 = use_l1
        res = model(imgs.cuda(), targets.cuda())
        res["total_loss"].backward()
        key = f"train_l1{int(use_l1)}"
        out[key] = {k: float(v) for k, v in res.items()}
        out[key + "_grads"] = {
            n: p.grad.detach().cpu().clone()
            for n, p in model.named_parameters()
            if any(n.startswith(g) for g in GRAD_PARAMS) and p.grad is not None
        }
    return out


def compare(ref, new):
    ok = True

    def report(name, a, b, rtol):
        nonlocal ok
        if isinstance(a, torch.Tensor):
            diff = (a - b).abs().max().item()
            scale = a.abs().max().item() + 1e-12
        else:
            diff, scale = abs(a - b), abs(a) + 1e-12
        good = diff / scale <= rtol
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name:55s} max|diff|={diff:.3e} rel={diff / scale:.3e}")

    report("eval outputs", ref["eval"], new["eval"], 1e-4)
    report("eval outputs (fused)", ref["eval_fused"], new["eval_fused"], 1e-4)
    for key in ("train_l10", "train_l11"):
        for k, v in ref[key].items():
            report(f"{key}.{k}", v, new[key][k], 1e-4)
        for n, g in ref[key + "_grads"].items():
            report(f"{key}.grad.{n}", g, new[key + "_grads"][n], 1e-3)
    print("\nALL EQUIVALENT" if ok else "\nDIFFERENCES FOUND")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["dump", "compare"])
    parser.add_argument("--out", default="bench/results/equiv_ref.pt")
    parser.add_argument("--ref", default="bench/results/equiv_ref.pt")
    parser.add_argument("-n", "--name", default="yolox-s")
    parser.add_argument("-b", "--batch", type=int, default=4)
    args = parser.parse_args()

    setup()
    seed_all(0)
    exp = get_exp(args.name)
    ckpt = default_ckpt(args.name)
    if args.cmd == "dump":
        imgs, targets = make_batch(exp, args.batch)
        res = run_all(exp, ckpt, imgs, targets)
        res["imgs"], res["targets"] = imgs, targets
        torch.save(res, args.out)
        print("saved", args.out, {k: v for k, v in res.items() if k.startswith("train_l") and "grad" not in k})
    else:
        ref = torch.load(args.ref, weights_only=False)
        seed_all(0)
        imgs, targets = make_batch(get_exp(args.name), args.batch)
        data_ok = torch.equal(imgs, ref["imgs"]) and torch.equal(targets, ref["targets"])
        print(f"{'OK ' if data_ok else 'BAD'} data pipeline (mosaic+mixup+hsv batch) bit-exact: {data_ok}")
        seed_all(0)
        new = run_all(exp, ckpt, ref["imgs"], ref["targets"])
        ok = compare(ref, new)
        raise SystemExit(0 if ok and data_ok else 1)


if __name__ == "__main__":
    main()
