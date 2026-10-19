#!/usr/bin/env python3
"""Live-pipeline benchmark: webcam, video file or image folder.

Times every stage of a real detection loop (capture -> preprocess -> model ->
decode + NMS) and reports FPS. This is what a "look for an object on the
webcam" application experiences, including the camera/decoder cost.

    python bench/bench_video.py --tag improved --source 0                  # webcam 0
    python bench/bench_video.py --tag improved --source clip.mp4 --show
    python bench/bench_video.py --tag improved --source datasets/COCO/val2017 --frames 300
    python bench/bench_video.py --tag improved --source 0 --classes "cell phone,scissors" --show

`--cudagraph` replays the model as a captured CUDA graph (needs a fixed input
size, which the letterboxed preprocessing guarantees).
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch

from common import default_ckpt, get_exp, load_model, save_results, summarize

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, postprocess, vis


def frame_source(source):
    if source.isdigit() or os.path.isfile(source):
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    files = sorted(glob.glob(os.path.join(source, "*.jpg")) + glob.glob(os.path.join(source, "*.png")))
    for f in files:
        yield cv2.imread(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source", default="0", help="webcam index, video file or image folder")
    parser.add_argument("-n", "--name", default="yolox-s")
    parser.add_argument("-c", "--ckpt", default=None)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--classes", default="", help="comma separated COCO class names to keep")
    parser.add_argument("--cudagraph", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    exp = get_exp(args.name)
    model = fuse_model(load_model(exp, args.ckpt or default_ckpt(args.name)).eval()).half()
    preproc = ValTransform(legacy=False)
    size = exp.test_size
    keep = [COCO_CLASSES.index(c.strip()) for c in args.classes.split(",") if c.strip()]

    static_in = torch.zeros(1, 3, *size, device="cuda", dtype=torch.half)
    infer = lambda: model(static_in)  # noqa: E731
    if args.cudagraph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                model(static_in)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            static_out = model(static_in)
        infer = lambda: (graph.replay(), static_out)[1]  # noqa: E731

    stages = {k: [] for k in ("capture", "preprocess", "model", "postprocess", "total")}
    hits = 0
    frames = frame_source(args.source)
    n = 0
    with torch.inference_mode():
        while n < args.frames + args.warmup:
            t0 = time.perf_counter()
            frame = next(frames, None)
            if frame is None:
                break
            t1 = time.perf_counter()
            x, _ = preproc(frame, None, size)
            static_in.copy_(torch.from_numpy(x).to("cuda", non_blocking=True).half()[None])
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            out = infer()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            det = postprocess(out, exp.num_classes, args.conf, exp.nmsthre, class_agnostic=True)[0]
            if det is not None:
                det = det.cpu().numpy()
                if keep:
                    det = det[np.isin(det[:, 6].astype(int), keep)]
            t4 = time.perf_counter()
            if n >= args.warmup:
                for k, v in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                    stages[k].append(v * 1000)
                hits += int(det is not None and len(det) > 0)
            if args.show:
                img = frame
                if det is not None and len(det):
                    r = min(size[0] / frame.shape[0], size[1] / frame.shape[1])
                    img = vis(frame.copy(), det[:, :4] / r, det[:, 4] * det[:, 5], det[:, 6],
                              args.conf, COCO_CLASSES)
                cv2.imshow("bench_video", img)
                if cv2.waitKey(1) == 27:
                    break
            n += 1

    res = {k: summarize(v) for k, v in stages.items() if v}
    res["fps"] = 1000.0 / res["total"]["median_ms"]
    res["frames"] = len(stages["total"])
    res["frames_with_detections"] = hits
    res["source"] = args.source
    res["cudagraph"] = args.cudagraph
    for k in stages:
        if k in res:
            print(f"{k:12s} median {res[k]['median_ms']:.2f} ms")
    print(f"FPS {res['fps']:.1f} over {res['frames']} frames")
    suffix = "_cudagraph" if args.cudagraph else ""
    save_results(args.tag, f"video_{args.name}{suffix}", res)


if __name__ == "__main__":
    main()
