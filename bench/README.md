# YOLOX benchmark harness

Measures speed, memory and accuracy of the code in this tree, and compares two
trees (e.g. `main` vs a branch) with interleaved A/B runs.

| script | what it measures |
|---|---|
| `bench_infer.py` | model latency at bs=1 / bs=N for eager fp32/fp16, Conv-BN fused, channels-last, CUDA graph, (opt.) `torch.compile` and ONNX Runtime; plus an end-to-end single-image pipeline (preprocess → model → NMS) |
| `bench_train.py` | training throughput via the real `Trainer`: `--mode gpu` (replayed batches: forward + SimOTA + loss + backward + optimizer + EMA), `--mode data` (dataloader only), `--mode full` (both); `--breakdown` times label assignment + loss separately |
| `bench_map.py` | COCO val2017 AP50:95 / AP50 of the deploy path (fused, fp16) |
| `bench_video.py` | live loop on a webcam / video / image folder: capture, preprocess, model, NMS, FPS; optional class filter and `--cudagraph` |
| `check_equiv.py` | numerical equivalence of two code versions: data pipeline batch (bit-exact), eval outputs, losses, num_fg, gradients |
| `ab.sh` | interleaved A/B runs of all of the above for two source trees |
| `compare.py` | markdown table of medians (and min–max over repetitions) |

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe numpy opencv-python loguru tqdm thop tabulate psutil tensorboard pycocotools onnx "onnxruntime-gpu==1.22.0"
# weights -> weights/yolox_s.pth (GitHub release 0.1.1rc0), COCO val2017 + annotations -> datasets/COCO
```

COCO val2017 is also used as the *training* set for throughput runs: only realistic
images and boxes are needed, not convergence.

## Typical workflow

```bash
# 1. reference outputs from the baseline code
python bench/check_equiv.py dump --out bench/results/equiv_ref.pt
# 2. ...change code...
python bench/check_equiv.py compare --ref bench/results/equiv_ref.pt
# 3. A/B speed + accuracy (baseline tree e.g. from `git worktree add ../yolox-base main`)
bash bench/ab.sh ../yolox-base . 3 .venv/Scripts/python.exe
python bench/compare.py A B --out bench/results/compare.md
```

## Notes on methodology

* GPU timings use CUDA events / explicit synchronisation, medians over ≥100 runs after warmup.
* Laptop GPUs drift with temperature; `ab.sh` interleaves A and B repetitions so drift hits both.
* Training batch contents are random per run (the dataloader reseeds workers from `uuid4`), so
  losses are not compared across runs; correctness is checked by `check_equiv.py` instead.
* Windows spawns dataloader workers slowly; warm-up defaults are long enough to reach steady state.
* `torch.compile` needs a C++ compiler (MSVC `cl` on Windows); without it the variant reports an error.
