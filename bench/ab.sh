#!/usr/bin/env bash
# Interleaved A/B benchmark of two source trees (A = baseline, B = candidate).
# Interleaving repetitions cancels slow drift (thermals, disk cache, background load).
#
#   bash bench/ab.sh <tree_A> <tree_B> [reps] [python]
#
# Results land in <tree_B>/bench/results/{A,B}_r<i>.json; compare with
#   python bench/compare.py A B
set -u
A=$(cd "$1" && pwd); B=$(cd "$2" && pwd); REPS=${3:-3}; PY=${4:-python}
export BENCH_RESULTS_DIR="$B/bench/results"
run() {  # run <tree> <tag> <script> [args...]
    local tree=$1 tag=$2 script=$3; shift 3
    echo "[ab] $(date +%T) $tag $script $*"
    (cd "$tree" && "$PY" "bench/$script" --tag "$tag" "$@" > "$BENCH_RESULTS_DIR/logs/${tag}_${script%.py}_$*.log" 2>&1) \
        || echo "[ab]   FAILED (see logs)"
}
mkdir -p "$BENCH_RESULTS_DIR/logs"
for i in $(seq 1 "$REPS"); do
    for side in A B; do
        tree=$A; [ $side = B ] && tree=$B
        run "$tree" "${side}_r$i" bench_infer.py --ort
        run "$tree" "${side}_r$i" bench_train.py --mode gpu --breakdown
        run "$tree" "${side}_r$i" bench_train.py --mode data
        run "$tree" "${side}_r$i" bench_train.py --mode full
    done
done
for side in A B; do  # accuracy is deterministic: once is enough
    tree=$A; [ $side = B ] && tree=$B
    run "$tree" "${side}_r1" bench_map.py
done
echo "[ab] done"
