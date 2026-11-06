#!/usr/bin/env python3
"""Write testdata/mlx_affine_quantization_v1.json from MLX's own quantizer.

The fixture pins how MLX packs affine-quantized weights (`mx.quantize`), how it
dequantizes them, and what `mx.quantized_matmul` returns, for every bit width
MLX supports. SystemOne's pure-Swift dequantizer and its MLX loading path are
checked against it.

    python3 -m venv .venv-mlx && .venv-mlx/bin/pip install "mlx==0.32.4"
    .venv-mlx/bin/python tools/mlx_quantization_fixture.py           # write
    .venv-mlx/bin/python tools/mlx_quantization_fixture.py --check   # compare
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

MLX_VERSION = "0.32.4"
OUTPUT = Path(__file__).resolve().parent.parent / "testdata" / "mlx_affine_quantization_v1.json"
CASES = [(2, 32), (3, 64), (4, 32), (4, 64), (5, 32), (6, 64), (8, 64), (8, 128)]
ROWS, COLUMNS, INPUTS = 3, 128, 2


def floats(array) -> list:
    return [float(value) for value in np.asarray(array, dtype=np.float32).reshape(-1)]


def build() -> dict:
    if mx.__version__ != MLX_VERSION:
        sys.exit(f"mlx {mx.__version__} is installed; this fixture pins {MLX_VERSION}")
    mx.set_default_device(mx.cpu)
    mx.random.seed(20261103)
    cases = []
    for bits, group_size in CASES:
        weight = mx.random.normal((ROWS, COLUMNS))
        inputs = mx.random.normal((INPUTS, COLUMNS))
        packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
        dequantized = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
        output = mx.quantized_matmul(
            inputs, packed, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
        cases.append({
            "bits": bits,
            "group_size": group_size,
            "shape": [ROWS, COLUMNS],
            "packed_shape": list(packed.shape),
            "packed": [int(word) for word in np.asarray(packed, dtype=np.uint32).reshape(-1)],
            "scales": floats(scales),
            "biases": floats(biases),
            "dequantized": floats(dequantized),
            "inputs": floats(inputs),
            "output": floats(output),
        })
    return {
        "format": "mlx_affine_quantization",
        "version": 1,
        "metadata": {"mlx": MLX_VERSION, "device": "cpu", "seed": 20261103},
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed fixture")
    args = parser.parse_args()
    fixture = build()
    if args.check:
        committed = json.loads(OUTPUT.read_text())
        if committed["metadata"] != fixture["metadata"]:
            sys.exit("fixture metadata differs")
        for ours, theirs in zip(fixture["cases"], committed["cases"]):
            if ours["packed"] != theirs["packed"]:
                sys.exit(f"packed words differ for bits={ours['bits']} group={ours['group_size']}")
            for key in ("scales", "biases", "dequantized", "output"):
                worst = max(abs(a - b) for a, b in zip(ours[key], theirs[key]))
                if worst > 1e-5:
                    sys.exit(f"{key} differs by {worst} for bits={ours['bits']}")
        print("fixture matches")
        return
    OUTPUT.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
