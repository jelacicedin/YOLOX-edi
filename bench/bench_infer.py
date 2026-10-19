#!/usr/bin/env python3
"""Inference latency benchmark.

Measures pure-model latency for several execution variants (eager fp32/fp16,
Conv-BN fused, channels-last, torch.compile, optional ONNX Runtime) at batch 1
and batch N, plus an end-to-end single-image pipeline (preprocess -> model ->
decode + NMS) which is what a webcam/video loop would see.

Usage:
    python bench/bench_infer.py --tag baseline -n yolox-s
"""

import argparse
import copy
import os
import tempfile

import cv2
import torch

from common import ROOT, cuda_time, default_ckpt, get_exp, load_model, save_results, seed_all, summarize

from yolox.data.data_augment import ValTransform
from yolox.utils import fuse_model, get_model_info, postprocess


def build_variant(base_model, variant):
    m = copy.deepcopy(base_model).eval()
    dtype = torch.float32
    if "fused" in variant or "compiled" in variant or "cudagraph" in variant:
        m = fuse_model(m)
    if "fp16" in variant:
        m = m.half()
        dtype = torch.float16
    cl = "_cl" in variant or "compiled" in variant
    if cl:
        m = m.to(memory_format=torch.channels_last)
    if "compiled" in variant:
        m = torch.compile(m, mode="max-autotune-no-cudagraphs")
    return m, dtype, cl


def bench_torch(base_model, variant, bs, size, warmup, iters):
    m, dtype, cl = build_variant(base_model, variant)
    x = torch.randn(bs, 3, size, size, device="cuda", dtype=dtype)
    if cl:
        x = x.contiguous(memory_format=torch.channels_last)

    @torch.inference_mode()
    def run():
        return m(x)

    if "cudagraph" in variant:
        # capture the whole forward once, then replay it: removes per-kernel launch overhead
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            m(x)
        run = g.replay

    # extra warmup for compile/autotune
    w = warmup * 3 if "compiled" in variant else warmup
    return summarize(cuda_time(run, warmup=w, iters=iters))


def bench_ort(base_model, bs, size, warmup, iters):
    import onnxruntime as ort
    ort.preload_dlls()

    m = fuse_model(copy.deepcopy(base_model).eval())
    m.head.decode_in_inference = False
    path = os.path.join(tempfile.gettempdir(), f"yolox_bench_{os.getpid()}.onnx")
    x = torch.randn(bs, 3, size, size, device="cuda")
    torch.onnx.export(m, x, path, opset_version=17, input_names=["images"], output_names=["output"],
                      dynamo=False)
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in sess.get_providers():
        return {"error": "CUDAExecutionProvider unavailable"}
    xb = x.cpu().numpy()
    import time

    for _ in range(warmup):
        sess.run(None, {"images": xb})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"images": xb})
        times.append((time.perf_counter() - t0) * 1000)
    os.remove(path)
    return summarize(times)


def bench_e2e(base_model, exp, img_path, warmup, iters, conf=0.25):
    """Single image: cv2 preprocess -> H2D -> fused fp16 model -> decode+NMS."""
    m = fuse_model(copy.deepcopy(base_model).eval()).half()
    preproc = ValTransform(legacy=False)
    img = cv2.imread(img_path)
    size = exp.test_size
    stages = {"preprocess": [], "model": [], "postprocess": [], "total": []}

    @torch.inference_mode()
    def once(record):
        import time
        t0 = time.perf_counter()
        x, _ = preproc(img, None, size)
        x = torch.from_numpy(x).unsqueeze(0).cuda().half()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = m(x)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        dets = postprocess(out, exp.num_classes, conf, exp.nmsthre, class_agnostic=True)
        if dets[0] is not None:
            dets[0].cpu()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if record:
            stages["preprocess"].append((t1 - t0) * 1000)
            stages["model"].append((t2 - t1) * 1000)
            stages["postprocess"].append((t3 - t2) * 1000)
            stages["total"].append((t3 - t0) * 1000)

    for _ in range(warmup):
        once(False)
    for _ in range(iters):
        once(True)
    res = {k: summarize(v) for k, v in stages.items()}
    res["fps"] = 1000.0 / res["total"]["median_ms"]
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("-n", "--name", default="yolox-s")
    parser.add_argument("-c", "--ckpt", default=None)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--variants", default="eager_fp32,eager_fp16,fused_fp16,fused_fp16_cl,cudagraph_fp16,cudagraph_fp16_cl")
    parser.add_argument("--ort", action="store_true", help="also bench ONNX Runtime CUDA EP")
    parser.add_argument("--img", default=os.path.join(ROOT, "assets", "dog.jpg"))
    args = parser.parse_args()

    seed_all(0)
    torch.backends.cudnn.benchmark = True
    exp = get_exp(args.name)
    model = load_model(exp, args.ckpt or default_ckpt(args.name)).eval()

    results = {"model": args.name, "size": args.size, "info": get_model_info(model, (args.size, args.size))}
    print(results["info"])
    lat = {}
    for variant in args.variants.split(","):
        for bs in (1, args.batch):
            key = f"{variant}_bs{bs}"
            try:
                r = bench_torch(model, variant, bs, args.size, args.warmup, args.iters)
                r["img_per_s"] = bs * 1000.0 / r["median_ms"]
            except Exception as e:  # keep the harness running if one variant breaks
                r = {"error": repr(e)[:300]}
            lat[key] = r
            print(f"{key:28s} {r}")
    if args.ort:
        for bs in (1, args.batch):
            try:
                r = bench_ort(model, bs, args.size, args.warmup, args.iters)
                if "median_ms" in r:
                    r["img_per_s"] = bs * 1000.0 / r["median_ms"]
            except Exception as e:
                r = {"error": repr(e)[:300]}
            lat[f"ort_cuda_fp32_bs{bs}"] = r
            print(f"ort_cuda_fp32_bs{bs:<14d} {r}")
    results["latency"] = lat
    results["e2e"] = bench_e2e(model, exp, args.img, args.warmup, args.iters)
    print("e2e", results["e2e"]["total"], "fps", results["e2e"]["fps"])
    save_results(args.tag, f"infer_{args.name}", results)


if __name__ == "__main__":
    main()
