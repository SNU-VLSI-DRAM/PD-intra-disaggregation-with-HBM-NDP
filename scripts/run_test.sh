#!/usr/bin/env bash
#
# Reproduces one workload point of the study -- batch 16 at a 16K context,
# tp=8 pp=1 -- for each model, in the two arrangements worth comparing.
#
#   duopd   the GPU prefills while the NDP decodes against the same HBM. One
#           run reports both the prefill time and the NDP's decode time.
#   decode  the GPU decodes on its own, with the NDP off.
#
# There is deliberately no run with the GPU decoding while the NDP also
# decodes: both devices would be doing the same work and the number would mean
# nothing.
#
# EXPECT THIS TO TAKE HOURS. The prefill runs dominate, and their cost grows
# with batch * context_len for the GEMMs and quadratically in context length
# for attention. To see the thing work before committing a machine to it, run
# one model at a smaller point by hand, for example
#
#   build/simulator/ndp_sim --model LLAMA2-70B --phase decode --no-ndp \
#       --tp 8 --pp 1 --batch 4 --context-len 2048 --decode-steps 1
#
# The simulator is clocked at 1 cycle == 1 ns and reports one decoder layer, so
# a whole-model latency is the reported count times the model's layer depth.
#
# Usage:  scripts/run_test.sh [build-dir]
set -uo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD=${1:-$REPO/build}
SIM=$BUILD/simulator/ndp_sim
NDP_CONFIG=$REPO/simulator/ndp_config.json

BATCH=16
CTX=16384
# Decoder depth per model, to turn a per-layer count into a whole-model one.
declare -A DEPTH=( [GPT-175B]=96 [LLAMA2-70B]=80 [Qwen-235B]=94 )

[[ -x $SIM ]] || { echo "build the simulator first: cmake --build build --target ndp_sim" >&2; exit 1; }

# The simulator opens its DRAM timings through a path relative to the working
# directory, so every run happens somewhere with configs/ beside it.
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/run"
ln -s "$REPO/configs" "$WORK/configs"

ms() { awk -v x="${1:-0}" -v n="$2" 'BEGIN{ if (x == "") print "-"; else printf "%.3f", x*n/1000000 }'; }

# run_one <label> <model> <args...>
run_one() {
    local label=$1 model=$2
    shift 2
    local log="$WORK/run/$model.$label.log"
    ( cd "$WORK/run" && "$SIM" --model "$model" \
        --tp 8 --pp 1 --batch "$BATCH" --context-len "$CTX" \
        --ndp-config "$NDP_CONFIG" "$@" ) > "$log" 2>&1
    local rc=$?
    local depth=${DEPTH[$model]}
    local gpu ndp
    gpu=$(grep -oP 'GPU model done at cycle \K[0-9]+' "$log" | tail -1)
    ndp=$(grep -oP 'average decoding cycles: \K[0-9]+' "$log" | tail -1)
    printf '%-12s %-7s %-6s %12s %9s %13s %9s\n' \
        "$model" "$label" "$([[ $rc -eq 0 ]] && echo ok || echo "rc=$rc")" \
        "${gpu:--}" "$(ms "$gpu" "$depth")" "${ndp:--}" "$(ms "$ndp" "$depth")"
}

echo "batch $BATCH, context $CTX, tp=8 pp=1"
echo
printf '%-12s %-7s %-6s %12s %9s %13s %9s\n' \
    model run status "gpu_ns/layer" "gpu_ms" "ndp_ns/layer" "ndp_ms"
for model in GPT-175B LLAMA2-70B Qwen-235B; do
    # GPU prefill with the NDP decoding alongside it; sp=8 puts one sequence
    # shard on each NDP unit under the GPU. The decode budget only has to
    # outlast the prefill: steps left over cost memory, not time, while running
    # short would leave the NDP idle for part of the window it is measured over.
    run_one duopd  "$model" --phase prefill --sp 8 --ndp    --decode-steps 8192
    # GPU decode on its own.
    run_one decode "$model" --phase decode  --sp 1 --no-ndp --decode-steps 1
done
