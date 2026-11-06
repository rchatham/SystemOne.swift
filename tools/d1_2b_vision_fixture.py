#!/usr/bin/env python3
"""Write testdata/d1_2b_vision_tiny/: d1-2B image decisions on a tiny LFM3-VL.

d1-2B (LiquidAI/d1-2B, LFM Open License v1.0) caps every picture at
`d1_vision_max_pixels` with Pillow's bicubic filter (`media.py`, `cap_pixels`),
writes the chat template's `<|vision_start|><|image|><|vision_end|>` per
picture at the head of the user turn, before the escaped state, and expands
`<|image|>` to one token per merged patch block (`processing_lfm3_vl.py`). Its
image processor (`image_processing_lfm3_vl.py`, vendored from transformers'
unreleased LFM3 code, Apache-2.0) smart-resizes to multiples of 32 pixels with
torchvision's bicubic kernel and flattens patches in merge-block order,
repeated over the temporal patch; `smart_resize` and `patchify` below are its
own, and the resize and normalization run on transformers 5.22.1's torchvision
backend, as it does.

The tower, projector, set packing, readout, calibration and answers are mlx-vlm
0.7.11's `lfm3_vl` and `d1_lfm3_vl` (`D1.d1_hidden`, `pack_set`, `readout_table`,
`probabilities`, `answer`), on a tiny random LFM3-VL whose text trunk is
testdata/lfm3_tiny's.

- config.json, processor_config.json (d1-2B's own), model.safetensors:
  transformers' names; mlx/: the same weights under mlx-vlm's names.
- fixture.json: per synthetic picture (tools/resize_fixture.py's formula), its
  capped size and SHA-256, the processor's resized size and SHA-256, and a
  summary of its projected features; per request, the packed rows and answers.

    .venv/bin/pip install "torch==2.15.0" "torchvision==0.29.2" "transformers==5.22.1" \\
        "pillow==13.0.0" "mlx==0.32.4" "mlx-vlm==0.7.11" safetensors numpy
    .venv/bin/python tools/d1_2b_vision_fixture.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx_vlm
import numpy as np
import PIL
import torch
import torchvision
import transformers
from mlx.utils import tree_flatten
from PIL import Image
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_utils import PILImageResampling, SizeDict

from mlx_vlm.models import d1_lfm3_vl
from mlx_vlm.models.d1_lfm3_vl.api import answer
from mlx_vlm.models.d1_lfm3_vl.d1 import LEAD, pack_set
from mlx_vlm.models.d1_lfm3_vl.prompt import (
    DEFAULT_STATE_STYLE,
    DEFAULT_SYSTEM,
    as_question,
    check_question,
    prefix_text,
    probabilities,
    question_header,
    readout_table,
    suffix_layout,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_2b_fixture import ADDED, CALIBRATION, CODES, SPECIAL, WORDS  # noqa: E402
from resize_fixture import synthetic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "d1_2b_vision_tiny"
TRUNK = ROOT / "testdata" / "lfm3_tiny"
VERSIONS = {"torch": "2.15.0", "torchvision": "0.29.2", "transformers": "5.22.1", "pillow": "13.0.0",
            "mlx": "0.32.4", "mlx_vlm": "0.7.11"}
SEED = 20261104
SET_SIZE = 2
MAX_PIXELS = 640 * 640  # the tiny checkpoint's d1_vision_max_pixels, so the large picture is capped
IMAGE, START, END = 120, 121, 122
VISION_TOKENS = {"<|image|>": IMAGE, "<|vision_start|>": START, "<|vision_end|>": END}

# LiquidAI/d1-2B's processor_config.json image processor, verbatim.
PROCESSOR = {
    "image_processor": {
        "do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Lfm3VlImageProcessor", "image_std": [0.5, 0.5, 0.5],
        "merge_size": 2, "patch_size": 16, "resample": 3, "rescale_factor": 0.00392156862745098,
        "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "temporal_patch_size": 2,
    },
    "processor_class": "Lfm3VlProcessor",
}
VISION = {
    "model_type": "lfm3_vl_vision", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3,
    "layer_types": ["conv", "full_attention", "conv"], "num_attention_heads": 4, "num_key_value_heads": 2,
    "in_channels": 3, "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2, "conv_kernel_size": 3,
    "norm_eps": 1e-5, "rope_parameters": {"rope_type": "axial", "rope_theta": 1000000.0},
}

IMAGES = {"small": (200, 150, 21), "large": (900, 700, 22), "wide": (640, 200, 23)}
REQUESTS = [
    ("A photo of the shelf <|image|> here.", ["small"], [
        ("stocked", {"type": "noul", "instructions": "Is the shelf stocked?"}),
        ("aisle", {"type": "choice", "instructions": "Which aisle?",
                   "criteria": {"dairy": "Dairy", "bakery": None, "frozen": "Frozen food"}}),
        ("tidy", {"type": "score", "instructions": "How tidy is it?", "criteria": ["messy", "fine", "neat"]}),
    ]),
    ({"ticket": "Broken lamp", "photos": "2"}, ["large", "wide"], [
        ("broken", {"type": "noul", "instructions": "Is it broken?", "criteria": {"true": "Visible damage"}}),
    ]),
    ("", ["wide"], [
        ("kind", {"type": "choice", "instructions": "What is it?", "criteria": {"lamp": None, "chair": None}}),
        ("light", {"type": "score", "instructions": "How bright?", "criteria": ["dark", "lit"]}),
    ]),
]


class Tokenizer:
    """tools/d1_2b_fixture.py's stand-in with the vision markup's special tokens."""

    bos_token = "<|startoftext|>"
    all_special_tokens = list(SPECIAL) + list(VISION_TOKENS)
    d1_codes = CODES
    d1_numbers = False
    d1_prompt = 1
    d1_box_edges = "median"
    added = {**ADDED, **VISION_TOKENS}

    def encode(self, text, add_special_tokens=False):
        if text in WORDS:
            return [WORDS[text]]
        names = sorted(self.added, key=len, reverse=True)
        ids, i = [], 0
        while i < len(text):
            name = next((n for n in names if text.startswith(n, i)), None)
            if name is not None:
                ids.append(self.added[name])
                i += len(name)
            else:
                ids.extend(b % 100 for b in text[i].encode("utf-8"))
                i += 1
        return ids


# --- image_processing_lfm3_vl.py (Apache-2.0) ---------------------------------------
def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def patchify(images, patch_size, merge_size, temporal_patch_size):
    batch_size, channel, resized_height, resized_width = images.shape
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    patches = images.reshape(batch_size, channel, grid_h // merge_size, merge_size, patch_size,
                             grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flatten_patches = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        .reshape(batch_size, grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    )
    return flatten_patches, grid_h, grid_w
# -------------------------------------------------------------------------------------


class Backend(TorchvisionBackend):
    pass


def process(picture):
    """The image processor on one capped picture: resized bytes, pixel values, grid."""
    settings = PROCESSOR["image_processor"]
    tensor = torch.from_numpy(np.array(picture)).permute(2, 0, 1).contiguous()[None]
    height, width = smart_resize(tensor.shape[-2], tensor.shape[-1], factor=32,
                                 min_pixels=settings["size"]["shortest_edge"],
                                 max_pixels=settings["size"]["longest_edge"])
    backend = Backend()
    resized = backend.resize(image=tensor, size=SizeDict(height=height, width=width),
                             resample=PILImageResampling(settings["resample"]))
    normalized = backend.rescale_and_normalize(resized, True, settings["rescale_factor"], True,
                                               tuple(settings["image_mean"]), tuple(settings["image_std"]))
    patches, grid_h, grid_w = patchify(normalized, 16, 2, 2)
    return resized[0].permute(1, 2, 0).numpy(), patches[0].numpy(), [1, grid_h, grid_w]


def cap_pixels(image, max_pixels):
    """media.py: a picture downscaled to at most `max_pixels`, bicubic."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)


def sha(pixels):
    return hashlib.sha256(np.ascontiguousarray(pixels, dtype=np.uint8).tobytes()).hexdigest()


def summary(features):
    f = np.asarray(features.astype(mx.float32))
    return {"count": f.shape[0], "sum": float(f.sum()), "abs_sum": float(np.abs(f).sum()),
            "first": f[0].tolist(), "last": f[-1].tolist()}


def main():
    installed = {"torch": torch.__version__.split("+")[0], "torchvision": torchvision.__version__.split("+")[0],
                 "transformers": transformers.__version__, "pillow": PIL.__version__, "mlx": mx.__version__,
                 "mlx_vlm": mlx_vlm.__version__}
    for package, version in installed.items():
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(SEED)
    tokenizer = Tokenizer()

    trunk_config = json.loads((TRUNK / "config.json").read_text())
    text = trunk_config["text_config"]
    config = {
        "architectures": ["D1Model"], "model_type": "lfm3_vl", "text_config": text, "vision_config": VISION,
        "downsample_factor": 2, "projector_hidden_size": 48, "projector_hidden_act": "gelu", "projector_bias": True,
        "projector_layer_norm_eps": 1e-6, "image_token_id": IMAGE, "video_token_id": 123,
        "vision_start_token_id": START, "vision_end_token_id": END, "tie_word_embeddings": True,
        "bos_token_id": 100, "d1_attention": "set", "d1_set_size": SET_SIZE, "d1_codes": CODES,
        "d1_vision_max_pixels": MAX_PIXELS,
    }

    # transformers' layout: the tiny LFM3 trunk's text weights, a random tower and projector.
    weights = {n: np.asarray(v.astype(mx.float32)) for n, v in mx.load(str(TRUNK / "model.safetensors")).items()
               if n.startswith("model.language_model.")}
    d, k, inner = VISION["hidden_size"], VISION["conv_kernel_size"], VISION["intermediate_size"]
    heads, kv = VISION["num_attention_heads"], VISION["num_key_value_heads"]
    hd = d // heads

    def normal(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(width):
        return (1 + 0.2 * rng.standard_normal(width)).astype(np.float32)

    tower, projector = "model.vision_tower.", "model.multi_modal_projector."
    weights[f"{tower}patchifier.weight"] = normal((d, 3 * 2 * 16 * 16), 1 / math.sqrt(3 * 16 * 16))
    weights[f"{tower}embedding_norm.weight"] = norm(d)
    for i, kind in enumerate(VISION["layer_types"]):
        p = f"{tower}layers.{i}."
        if kind == "full_attention":
            weights[p + "self_attn.q_proj.weight"] = normal((heads * hd, d), 1 / math.sqrt(d))
            weights[p + "self_attn.k_proj.weight"] = normal((kv * hd, d), 1 / math.sqrt(d))
            weights[p + "self_attn.v_proj.weight"] = normal((kv * hd, d), 1 / math.sqrt(d))
            weights[p + "self_attn.out_proj.weight"] = normal((d, heads * hd), 1 / math.sqrt(d))
            weights[p + "self_attn.q_layernorm.weight"] = norm(hd)
            weights[p + "self_attn.k_layernorm.weight"] = norm(hd)
        else:
            weights[p + "conv.in_proj.weight"] = normal((3 * d, d), 1 / math.sqrt(d))
            weights[p + "conv.spatial_conv.weight"] = normal((d, 1, k, k), 0.5)
            weights[p + "conv.out_proj.weight"] = normal((d, d), 1 / math.sqrt(d))
        weights[p + "operator_norm.weight"] = norm(d)
        weights[p + "ffn_norm.weight"] = norm(d)
        weights[p + "feed_forward.w1.weight"] = normal((inner, d), 1 / math.sqrt(d))
        weights[p + "feed_forward.w3.weight"] = normal((inner, d), 1 / math.sqrt(d))
        weights[p + "feed_forward.w2.weight"] = normal((d, inner), 1 / math.sqrt(inner))
    weights[f"{projector}layer_norm.weight"] = norm(d)
    weights[f"{projector}layer_norm.bias"] = normal(d, 0.1)
    weights[f"{projector}linear_1.weight"] = normal((48, 4 * d), 1 / math.sqrt(4 * d))
    weights[f"{projector}linear_1.bias"] = normal(48, 0.1)
    weights[f"{projector}linear_2.weight"] = normal((text["hidden_size"], 48), 1 / math.sqrt(48))
    weights[f"{projector}linear_2.bias"] = normal(text["hidden_size"], 0.1)

    model_config = d1_lfm3_vl.ModelConfig.from_dict(config)
    model = d1_lfm3_vl.Model(model_config)
    model.load_weights(list(model.sanitize({n: mx.array(v) for n, v in weights.items()}).items()), strict=True)
    model.eval()

    pictures, images, encoded = {}, {}, {}
    for name, (width, height, salt) in IMAGES.items():
        capped = cap_pixels(Image.fromarray(synthetic(width, height, salt)), MAX_PIXELS)
        resized, pixels, grid = process(capped)
        features = model.encode_images(mx.array(pixels), image_grid_thw=mx.array([grid]))
        pictures[name], encoded[name] = capped, (grid, features)
        images[name] = {"source": [width, height, salt], "capped": [capped.width, capped.height],
                        "capped_sha256": sha(capped), "resized": list(resized.shape[:2]), "resized_sha256": sha(resized),
                        "features": summary(features)}

    requests = []
    for state, names, named in REQUESTS:
        for _, q in named:
            check_question(q)
        questions = [as_question(q) for _, q in named]
        markup = "".join(f"<|vision_start|>{'<|image|>' * encoded[n][1].shape[0]}<|vision_end|>" for n in names)
        prefix = prefix_text(tokenizer, state, tokenizer.bos_token, DEFAULT_STATE_STYLE, DEFAULT_SYSTEM, markup)
        layouts = [suffix_layout(tokenizer, q, LEAD) for q in questions]
        header = question_header(tokenizer, state)
        media = [(IMAGE, encoded[n][1]) for n in names]
        rows, answers, read = [], [], 0
        for k0 in range(0, len(questions), SET_SIZE):
            group = questions[k0:k0 + SET_SIZE]
            suffixes = [t for t, _ in layouts[k0:k0 + SET_SIZE]]
            ids, state_len, question, ends = pack_set(
                tokenizer, prefix, suffixes, header, [o for _, o in layouts[k0:k0 + SET_SIZE]],
                tokenizer.encode(prefix + suffixes[0], add_special_tokens=False))
            read += len(ids)
            hidden = model.d1_hidden(ids, state_len, question, media)
            embed = model.language_model.model.embed_tokens
            for q, slots in zip(group, ends):
                tokens, forms, _ = readout_table(tokenizer, q)
                logits = np.array((hidden[mx.array(slots)] @ embed(mx.array(list(tokens))).T).astype(mx.float32))
                best = logits[:, forms].max(-1)[0].tolist()
                answers.append(answer(q, probabilities(q, best, CALIBRATION)))
            rows.append({"ids": ids, "state": state_len, "question": question, "ends": [e[0] for e in ends]})
        requests.append({"images": names, "rows": rows, "tokens": read,
                         "answers": [{"id": n, **a} for (n, _), a in zip(named, answers)]})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(OUTPUT / "model.safetensors"), {n: mx.array(v) for n, v in sorted(weights.items())})
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    (OUTPUT / "processor_config.json").write_text(json.dumps(PROCESSOR, indent=2) + "\n")
    folder = OUTPUT / "mlx"
    folder.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(folder / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (folder / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    (folder / "processor_config.json").write_text(json.dumps(PROCESSOR, indent=2) + "\n")
    out = {"format": "d1_2b_vision_tiny", "version": 1, "metadata": VERSIONS, "set_size": SET_SIZE, "codes": CODES,
           "max_pixels": MAX_PIXELS, "calibration": {"default": CALIBRATION.default, "temperatures": CALIBRATION.temperatures},
           "special": list(SPECIAL) + list(VISION_TOKENS), "added": Tokenizer.added, "words": WORDS,
           "images": images, "requests": requests}
    (OUTPUT / "fixture.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
