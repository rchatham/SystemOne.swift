#!/usr/bin/env python3
"""Write testdata/antialiased_resize_v1.json: uint8 resize goldens.

Image preprocessing resizes with two integer kernels that `AntialiasedResize`
reproduces bit for bit: torchvision's (ATen's uint8 antialiased kernel, used by
transformers' torchvision image processors) and Pillow's (`Image.resize`). Each
case resizes a synthetic image both libraries and Swift generate from one
integer formula and records the output's SHA-256 and byte sum.

Pillow 13 resamples the height first when the height shrinks by more than twice
the width's shrink (`ImagingResampleInner`); cases cover both orders.

    .venv/bin/pip install "torch==2.15.0" "torchvision==0.29.2" "pillow==13.0.0" numpy
    .venv/bin/python tools/resize_fixture.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torchvision.transforms.v2 import functional as F

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "antialiased_resize_v1.json"
VERSIONS = {"torch": "2.15.0", "torchvision": "0.29.2", "pillow": "13.0.0"}

# (width, height, salt) -> (width, height): up, down, mixed, and both Pillow pass orders.
CASES = [
    ((37, 23, 0), (64, 41)),
    ((300, 200, 1), (320, 224)),
    ((700, 500, 3), (576, 416)),
    ((61, 47, 5), (17, 13)),
    ((128, 96, 7), (512, 512)),
    ((40, 30, 9), (41, 30)),
    ((33, 40, 11), (33, 71)),
    ((17, 51, 13), (77, 10)),
    ((79, 57, 15), (77, 32)),
    ((66, 85, 17), (16, 10)),
    ((90, 20, 19), (12, 19)),
]


def synthetic(width, height, salt):
    """Interleaved RGB bytes: `((3x + 5y + 85c + salt) & 255) ^ (((7xy + 13c) >> 3) & 63)`."""
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    channels = [((x * 3 + y * 5 + c * 85 + salt) & 255) ^ (((x * y * 7 + c * 13) >> 3) & 63) for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def record(pixels):
    data = np.ascontiguousarray(pixels).tobytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "sum": int(np.asarray(pixels, dtype=np.int64).sum())}


def main():
    installed = {"torch": torch.__version__.split("+")[0], "torchvision": torchvision.__version__.split("+")[0],
                 "pillow": PIL.__version__}
    for package, version in installed.items():
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    cases = []
    for (width, height, salt), (to_width, to_height) in CASES:
        image = synthetic(width, height, salt)
        entry = {"input": [width, height, salt], "output": [to_width, to_height], "source": record(image)}
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        for name, mode in (("cubic", F.InterpolationMode.BICUBIC), ("triangle", F.InterpolationMode.BILINEAR)):
            resized = F.resize(tensor, [to_height, to_width], interpolation=mode, antialias=True)
            entry[f"aten_{name}"] = record(resized.permute(1, 2, 0).numpy())
        for name, mode in (("cubic", Image.Resampling.BICUBIC), ("triangle", Image.Resampling.BILINEAR)):
            entry[f"pillow_{name}"] = record(np.asarray(Image.fromarray(image).resize((to_width, to_height), mode)))
        cases.append(entry)
    out = {"format": "antialiased_resize", "version": 1, "metadata": VERSIONS, "cases": cases}
    OUTPUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
