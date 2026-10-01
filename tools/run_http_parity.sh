#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Port 8014 is assigned to Von in the local registry. Parity deliberately uses
# adjacent, unassigned 8015 and never the workspace-assigned port 8000.
python_port="${VON_PYTHON_PARITY_PORT:-8015}"
if [[ "$python_port" == "8000" || "$python_port" == "8014" ]]; then
  echo "refusing reserved parity port: $python_port" >&2
  exit 2
fi
if lsof -nP -iTCP:"$python_port" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $python_port is already occupied:" >&2
  lsof -nP -iTCP:"$python_port" -sTCP:LISTEN >&2
  exit 2
fi

# Inventory containers and possible orphan listeners before launch. docker may
# not be installed/running on every macOS development host.
echo "Docker port inventory:" >&2
if command -v docker >/dev/null 2>&1; then
  docker_inventory="$(mktemp -t von-docker-ports.XXXXXX)"
  docker ps --format '{{.ID}}\t{{.Ports}}\t{{.Names}}' >"$docker_inventory" 2>&1 &
  docker_pid=$!
  for _ in {1..30}; do
    kill -0 "$docker_pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$docker_pid" 2>/dev/null; then
    kill "$docker_pid" 2>/dev/null || true
    wait "$docker_pid" 2>/dev/null || true
    echo "  docker did not respond within 3 seconds" >&2
  else
    wait "$docker_pid" || true
    cat "$docker_inventory" >&2
  fi
  rm -f "$docker_inventory"
else
  echo "  docker unavailable" >&2
fi
echo "Port $python_port is clear; no listener/orphan found." >&2

: "${VON_PRODUCTION_WEIGHTS_PATH:=$repo_root/weights/option_marker.safetensors}"
: "${VON_TOKENIZER_PATH:=$repo_root/Tests/VonSwiftTests/Resources/Tokenizer}"
: "${VON_API_KEY:=von-http-parity-local}"
[[ -f "$VON_PRODUCTION_WEIGHTS_PATH" ]] || { echo "missing weights: $VON_PRODUCTION_WEIGHTS_PATH" >&2; exit 2; }
[[ -f "$VON_TOKENIZER_PATH/tokenizer.json" ]] || { echo "missing tokenizer resources: $VON_TOKENIZER_PATH" >&2; exit 2; }
[[ -x "$repo_root/.venv/bin/von" ]] || { echo "missing pinned Python CLI: $repo_root/.venv/bin/von" >&2; exit 2; }

python_log="$(mktemp -t von-python-parity.XXXXXX.log)"
python_pid=""
cleanup() {
  if [[ -n "$python_pid" ]] && kill -0 "$python_pid" 2>/dev/null; then
    kill -TERM "$python_pid" 2>/dev/null || true
    wait "$python_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export VON_API_KEY VON_PRODUCTION_WEIGHTS_PATH VON_TOKENIZER_PATH
export VON_CORS_ORIGINS="${VON_CORS_ORIGINS:-http://127.0.0.1}"
export HF_HUB_OFFLINE=1
"$repo_root/.venv/bin/von" serve --host 127.0.0.1 --port "$python_port" --device cpu >"$python_log" 2>&1 &
python_pid=$!

python_url="http://127.0.0.1:$python_port/"
healthy=0
for _ in {1..120}; do
  if ! kill -0 "$python_pid" 2>/dev/null; then
    echo "Python server exited during startup; log: $python_log" >&2
    tail -100 "$python_log" >&2
    exit 1
  fi
  if curl --silent --fail --max-time 1 "${python_url}health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 1
done
if [[ "$healthy" != "1" ]]; then
  echo "Python server did not become healthy; log: $python_log" >&2
  tail -100 "$python_log" >&2
  exit 1
fi

xcodebuild build-for-testing \
  -scheme VonSwift-Package \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$repo_root/.build/xcode"

default_bundle="$repo_root/.build/xcode/Build/Products/Debug/VonHTTPTests.xctest"
bundle="${VON_HTTP_XCTEST_BUNDLE:-$default_bundle}"
[[ -d "$bundle" ]] || { echo "missing Xcode test bundle: $bundle" >&2; exit 1; }

VON_HTTP_PARITY=1 \
VON_PYTHON_SERVER_URL="$python_url" \
VON_API_KEY="$VON_API_KEY" \
VON_PRODUCTION_WEIGHTS_PATH="$VON_PRODUCTION_WEIGHTS_PATH" \
VON_TOKENIZER_PATH="$VON_TOKENIZER_PATH" \
  xcrun xctest \
    -XCTest VonHTTPTests.PythonServerParityTests/testPinnedPythonServerParityWhenConfigured \
    "$bundle"

echo "HTTP parity passed; Python log: $python_log" >&2
