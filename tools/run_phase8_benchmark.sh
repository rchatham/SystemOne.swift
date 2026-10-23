#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

results_dir="$repo_root/benchmark-results"
mkdir -p "$results_dir"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
iterations="${VON_BENCH_ITERATIONS:-30}"
warmup="${VON_BENCH_WARMUP:-3}"
tokenizer_path="Tests/SystemOneVonTests/Resources/Tokenizer"
weights_path="weights/option_marker.safetensors"

red="$(tput setaf 1 2>/dev/null || echo '')"
green="$(tput setaf 2 2>/dev/null || echo '')"
reset="$(tput sgr0 2>/dev/null || echo '')"

fail() { echo "${red}FAIL${reset} $*" >&2; exit 1; }
ok()   { echo "${green}OK${reset}   $*"; }

if [[ ! -f "$weights_path" ]]; then
  fail "production weights not found: $weights_path (run .venv/bin/python tools/convert_option_marker.py)"
fi
if [[ ! -f "$tokenizer_path/tokenizer.json" ]]; then
  fail "tokenizer resources not found: $tokenizer_path"
fi

echo "=== Phase 8 Benchmark Suite ($timestamp) ==="
echo "  Iterations: $iterations  Warmup: $warmup"
echo ""

# ── Python benchmark ────────────────────────────────────────────────
echo "--- Python (sequential_api, mps) ---"
python_bin="$repo_root/.venv/bin/python"
python_seq_json="$results_dir/python-sequential_api-$timestamp.json"
python_seq_log="$results_dir/python-sequential_api-$timestamp.log"
if "$python_bin" tools/benchmark.py \
    --device mps --mode sequential_api --iterations "$iterations" --warmup "$warmup" --json \
    > "$python_seq_json" 2> "$python_seq_log"; then
  ok "Python sequential_api -> $python_seq_json"
else
  fail "Python sequential_api failed; log: $python_seq_json"
fi

echo "--- Python (batched_model, mps) ---"
python_bat_json="$results_dir/python-batched_model-$timestamp.json"
python_bat_log="$results_dir/python-batched_model-$timestamp.log"
if "$python_bin" tools/benchmark.py \
    --device mps --mode batched_model --iterations "$iterations" --warmup "$warmup" --json \
    > "$python_bat_json" 2> "$python_bat_log"; then
  ok "Python batched_model -> $python_bat_json"
else
  fail "Python batched_model failed; log: $python_bat_json"
fi

# ── Swift benchmark (Release xcodebuild) ──────────────────────────────
echo ""
echo "--- Swift build (Release) ---"
swift_bin="$repo_root/.build/xcode-release/Build/Products/Release/SystemOneBenchmark"

if xcodebuild build \
    -scheme SystemOneBenchmark \
    -destination 'platform=macOS,arch=arm64' \
    -derivedDataPath "$repo_root/.build/xcode-release" \
    -configuration Release >/dev/null 2>&1; then
  ok "Swift SystemOneBenchmark (Release) built"
else
  fail "Swift SystemOneBenchmark build failed"
fi

for strategy in sequential automatic batched; do
  echo "--- Swift (GPU, Release, $strategy) ---"
  swift_json="$results_dir/swift-gpu-release-$strategy-$timestamp.json"
  swift_log="$results_dir/swift-gpu-release-$strategy-$timestamp.log"
  if "$swift_bin" \
      --tokenizer "$tokenizer_path" \
      --weights "$weights_path" \
      --device gpu --strategy "$strategy" \
      --iterations "$iterations" --warmup "$warmup" --json \
      > "$swift_json" 2> "$swift_log"; then
    ok "Swift GPU $strategy -> $swift_json"
  else
    fail "Swift GPU $strategy benchmark failed; log: $swift_json"
  fi
done

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
for f in "$results_dir"/*-"$timestamp".json; do
  if [[ -f "$f" ]]; then
    name="$(basename "$f")"
    # Swift results use "latency" (singular); Python results use "latencies" (plural).
    latency_field="$(python3 -c "
import json
d=json.load(open('$f'))
if 'latency' in d:
    print('latency')
elif 'latencies' in d:
    print('latencies')
else:
    exit(1)
" 2>/dev/null || echo "-")"
    mean="$(python3 -c "
import json
d=json.load(open('$f'))
import sys
field = '$latency_field'
if field in d:
    print(d[field]['mean'])
else:
    sys.exit(1)
" 2>/dev/null || echo "-")"
    mode="$(python3 -c "import json; d=json.load(open('$f')); print(d.get('mode', d.get('strategy','-')))" 2>/dev/null || echo "-")"
    device="$(python3 -c "import json; d=json.load(open('$f')); print(d.get('device','-'))" 2>/dev/null || echo "-")"
    qc="$(python3 -c "import json; d=json.load(open('$f')); print(d.get('question_count', d.get('questionCount','-')))" 2>/dev/null || echo "-")"
    qps="$(python3 -c "
import json, sys
d=json.load(open('$f'))
field = '$latency_field'
if field in d:
    qc = d.get('question_count', d.get('questionCount', 0))
    print(round(qc / d[field]['mean'], 1) if d[field]['mean'] > 0 else '-')
else:
    sys.exit(1)
" 2>/dev/null || echo "-")"
    printf "  %-45s  mode=%-16s  device=%-4s  mean=%-10s  q/s=%s\n" \
      "$name" "$mode" "$device" "$mean" "$qps"
  fi
done

echo ""
ok "Phase 8 benchmark suite complete. Results: $results_dir/"