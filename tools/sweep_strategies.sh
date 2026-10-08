#!/bin/bash
# sweep_strategies.sh — run Swift benchmark for all strategies on fixture
# and homogeneous workloads, then recommend the default.
#
# Usage:
#   tools/sweep_strategies.sh <tokenizer_folder> <weights_path> [gpu|cpu]
#
# Requires compiled von-benchmark (Release) via xcodebuild.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: tools/sweep_strategies.sh <tokenizer_folder> <weights_path> [gpu|cpu]"
    exit 2
fi

TOKENIZER="$1"
WEIGHTS="$2"
DEVICE="${3:-gpu}"
case "$DEVICE" in
    gpu|cpu) ;;
    *) echo "Error: device must be 'gpu' or 'cpu', got '$DEVICE'"; exit 2 ;;
esac
shift 2 || true

BENCHMARK=".build/xcode-release/Build/Products/Release/von-benchmark"
if [ ! -x "$BENCHMARK" ]; then
    echo "Error: $BENCHMARK not found — run xcodebuild first."
    exit 1
fi

ITERATIONS=30
WARMUP=3
OUTDIR="$(dirname "$0")/../benchmark-results"
mkdir -p "$OUTDIR"

run_bench() {
    local label="$1"
    local strategy="$2"
    local workload="$3"
    local questions="$4"
    local json_file="$OUTDIR/${label// /_}.json"

    echo ""
    echo "=== $label ==="
    "$BENCHMARK" \
        --tokenizer "$TOKENIZER" \
        --weights "$WEIGHTS" \
        --device "$DEVICE" \
        --strategy "$strategy" \
        --workload "$workload" \
        --questions "$questions" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --json | tee "$json_file"

    # Extract mean for comparison
    local mean
    mean=$(python3 -c "import json,sys; d=json.load(sys.stdin); print(d['latency']['mean'])" < "$json_file" 2>/dev/null || echo "0")
    echo "  -> mean: $(python3 -c "print(f'{$mean*1000:.1f} ms')" 2>/dev/null || echo "N/A")"
    MEANS["$label"]="$mean"
}

echo "=== Von Batching Strategy Sweep ==="
echo "Tokenizer: $TOKENIZER"
echo "Weights:   $WEIGHTS"
echo "Device:    $DEVICE"
echo ""

declare -A MEANS

# --- Fixture (4 heterogeneous questions, 5 physical rows) ---
for strategy in sequential automatic batched; do
    label="fixture-$strategy"
    run_bench "$label" "$strategy" fixture 4
done

# --- Homogeneous n=4 ---
for strategy in sequential automatic batched; do
    label="homogeneous-4-$strategy"
    run_bench "$label" "$strategy" homogeneous 4
done

# --- Homogeneous n=8 ---
for strategy in sequential automatic batched; do
    label="homogeneous-8-$strategy"
    run_bench "$label" "$strategy" homogeneous 8
done

# --- Homogeneous n=16 ---
for strategy in sequential automatic batched; do
    label="homogeneous-16-$strategy"
    run_bench "$label" "$strategy" homogeneous 16
done

# --- Summary ---
echo ""
echo "======================================================================"
echo "  Sweep Summary"
echo "======================================================================"
echo ""

report_workload() {
    local wl="$1"
    local questions="$2"
    local seq_mean="${MEANS[$wl-sequential]:-0}"
    local auto_mean="${MEANS[$wl-automatic]:-0}"
    local bat_mean="${MEANS[$wl-batched]:-0}"

    echo "  $wl ($questions questions):"
    printf "    sequential:  %.3f ms\n" "$(python3 -c "print($seq_mean * 1000)" 2>/dev/null || echo 0)"
    printf "    automatic:   %.3f ms\n" "$(python3 -c "print($auto_mean * 1000)" 2>/dev/null || echo 0)"
    printf "    batched:     %.3f ms\n" "$(python3 -c "print($bat_mean * 1000)" 2>/dev/null || echo 0)"

    if python3 -c "exit(0 if $seq_mean > 0 and $auto_mean > 0 else 1)" 2>/dev/null; then
        auto_vs_seq=$(python3 -c "print(f'{$auto_mean / $seq_mean:.2f}x')" 2>/dev/null || echo "N/A")
        echo "    automatic vs sequential: $auto_vs_seq"
    fi
    echo ""
}

report_workload fixture 4
report_workload homogeneous-4 4
report_workload homogeneous-8 8
report_workload homogeneous-16 16

# --- Recommendation ---
echo "======================================================================"
echo "  Recommendation"
echo "======================================================================"
echo ""

# Compute which is faster on fixture: sequential or automatic
FIXTURE_SEQ="${MEANS[fixture-sequential]:-0}"
FIXTURE_AUTO="${MEANS[fixture-automatic]:-0}"
HOMO4_SEQ="${MEANS[homogeneous-4-sequential]:-0}"
HOMO4_AUTO="${MEANS[homogeneous-4-automatic]:-0}"
HOMO8_SEQ="${MEANS[homogeneous-8-sequential]:-0}"
HOMO8_AUTO="${MEANS[homogeneous-8-automatic]:-0}"
HOMO16_SEQ="${MEANS[homogeneous-16-sequential]:-0}"
HOMO16_AUTO="${MEANS[homogeneous-16-automatic]:-0}"

if python3 -c "exit(0 if $FIXTURE_SEQ > 0 and $FIXTURE_AUTO > 0 else 1)" 2>/dev/null; then
    AUTO_VS_SEQ_FIX=$(python3 -c "print($FIXTURE_AUTO / $FIXTURE_SEQ)" 2>/dev/null)
    AUTO_VS_SEQ_H4=$(python3 -c "print($HOMO4_AUTO / $HOMO4_SEQ)" 2>/dev/null)
    AUTO_VS_SEQ_H8=$(python3 -c "print($HOMO8_AUTO / $HOMO8_SEQ)" 2>/dev/null)
    AUTO_VS_SEQ_H16=$(python3 -c "print($HOMO16_AUTO / $HOMO16_SEQ)" 2>/dev/null)

    AUTO_WINS=$(python3 -c "
count=0
if $AUTO_VS_SEQ_FIX < 1.0: count+=1
if $AUTO_VS_SEQ_H4 < 1.0: count+=1
if $AUTO_VS_SEQ_H8 < 1.0: count+=1
if $AUTO_VS_SEQ_H16 < 1.0: count+=1
print(count)
" 2>/dev/null || echo 0)

    echo "  Automatic beats sequential on $AUTO_WINS/4 workloads."
    if [ "$AUTO_WINS" -ge 3 ]; then
        echo "  Recommended default: automatic"
    else
        echo "  Recommended default: sequential"
        echo "  (automatic slower on fixture or too conservative on homogeneous)"
    fi
else
    echo "  Could not compute recommendation (missing benchmark data)."
    echo "  Defaulting to: automatic"
fi

echo ""
echo "  Full results saved to: $OUTDIR/"
echo ""