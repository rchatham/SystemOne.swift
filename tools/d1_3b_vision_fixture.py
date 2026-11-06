#!/usr/bin/env python3
"""Write testdata/d1_3b_vision_tiny/: d1-3B image decisions on a tiny LFM2-VL.

d1-3B (LiquidAI/d1-3B, LFM Open License v1.0) caps every picture at one
megapixel with Pillow's bicubic filter (`runner.py`, `cap_pixels`), writes the
chat template's `<image>` per picture at the head of the user turn
(`prompt.py`, `prefix_text`), and runs transformers' LFM2-VL: the processor's
smart resize, tiling, and token expansion, the SigLIP2 tower, the projector,
and the features scattered into the `<image>` tokens. This script runs those
pieces of transformers 5.22.1 (torchvision backend) and Pillow 13.0.0 on a tiny
random LFM2-VL; the prompt and readout helpers are the reference's, from
tools/d1_3b_fixture.py.

- config.json, processor_config.json (d1-3B's own), model.safetensors:
  transformers' names; mlx/: the same weights under mlx-vlm 0.7.11's names.
- fixture.json:
  - images: synthetic pictures (the formula of tools/resize_fixture.py), their
    capped size and SHA-256, the processor's layout, each processed view's
    SHA-256 (recovered from `pixel_values`), and a summary of the projected
    features;
  - requests: prompt token ids under a stand-in tokenizer and each question's
    answer from the full prompt's last-token logits.

The stand-in tokenizer splits the image markup's special tokens out of text,
maps a few whole strings (yes/no forms, spaced codes) to their own ids, and
every other text to one token per UTF-8 byte (`byte % 100`), BOS id 1.

    .venv/bin/pip install "torch==2.15.0" "torchvision==0.29.2" "transformers==5.22.1" \\
        "pillow==13.0.0" "mlx==0.32.4" "mlx-vlm==0.7.11" safetensors numpy
    .venv/bin/python tools/d1_3b_vision_fixture.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image
from safetensors.torch import save_file
from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration, Lfm2VlImageProcessor, Lfm2VlProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_3b_fixture import TABLE, aliases, answer, readout, readout_ids, state_block, suffix_text  # noqa: E402
from resize_fixture import synthetic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "d1_3b_vision_tiny"
VERSIONS = {"torch": "2.15.0", "torchvision": "0.29.2", "transformers": "5.22.1", "pillow": "13.0.0",
            "mlx": "0.32.4", "mlx_vlm": "0.7.11"}
SEED = 20261103
BOS = 1
IMAGE_TOKEN = 120
SPECIAL = {"<image>": IMAGE_TOKEN, "<|image_start|>": 121, "<|image_end|>": 122, "<|img_thumbnail|>": 123}
SPECIAL.update({f"<|img_row_{r}_col_{c}|>": 130 + 10 * (r - 1) + (c - 1) for r in range(1, 11) for c in range(1, 11)})
VISION_MAX_PIXELS = 1024 * 1024

# LiquidAI/d1-3B's processor_config.json, verbatim.
PROCESSOR = {
    "image_processor": {
        "data_format": "channels_first", "do_image_splitting": True, "do_normalize": True, "do_pad": True,
        "do_rescale": True, "do_resize": True, "downsample_factor": 2, "encoder_patch_size": 16,
        "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Lfm2VlImageProcessor", "image_std": [0.5, 0.5, 0.5],
        "max_image_tokens": 256, "max_num_patches": 1024, "max_pixels_tolerance": 2.0, "max_tiles": 10,
        "min_image_tokens": 64, "min_tiles": 2, "resample": 3, "rescale_factor": 0.00392156862745098,
        "return_row_col_info": True, "size": {"height": 512, "width": 512}, "tile_size": 512, "use_thumbnail": True,
    },
    "processor_class": "Lfm2VlProcessor",
}
TEXT = {
    "model_type": "lfm2", "vocab_size": 256, "hidden_size": 32, "intermediate_size": 64,
    "block_auto_adjust_ff_dim": False, "num_hidden_layers": 4,
    "layer_types": ["conv", "conv", "full_attention", "conv"],
    "num_attention_heads": 4, "num_key_value_heads": 2, "norm_eps": 1e-5, "conv_L_cache": 3,
    "conv_bias": False, "tie_word_embeddings": True, "max_position_embeddings": 8192,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    "bos_token_id": BOS, "eos_token_id": 2, "pad_token_id": 0,
}
VISION = {
    "model_type": "siglip2_vision_model", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
    "num_attention_heads": 2, "num_channels": 3, "num_patches": 256, "patch_size": 16,
    "hidden_act": "gelu_pytorch_tanh", "layer_norm_eps": 1e-6, "vision_use_head": False,
}
CONFIG = {
    "architectures": ["Lfm2VlForConditionalGeneration"], "model_type": "lfm2_vl", "text_config": TEXT,
    "vision_config": VISION, "image_token_id": IMAGE_TOKEN, "projector_hidden_act": "gelu",
    "projector_hidden_size": 48, "projector_bias": True, "projector_use_layernorm": False, "downsample_factor": 2,
    "tie_word_embeddings": True, "bos_token_id": BOS, "use_image_special_tokens": True,
}

# name: (width, height, salt): a small single view, a tiled picture over one megapixel, a downscaled single view.
IMAGES = {"small": (300, 200, 1), "large": (1280, 960, 2), "medium": (700, 500, 3)}
REQUESTS = [
    ("Is the sign readable?", ["small"], [
        ("readable", {"type": "noul", "instructions": "Can the sign be read?"}),
        ("kind", {"type": "choice", "instructions": "What is shown?",
                  "criteria": {"sign": "A street sign", "poster": None, "screen": "A screen"}}),
    ]),
    ({"ticket": "Damaged parcel", "photos": "2"}, ["large", "medium"], [
        ("damage", {"type": "score", "instructions": "How bad is the damage?", "criteria": ["none", "some", "severe"]}),
    ]),
    ("", ["medium", "small"], [
        ("same", {"type": "noul", "instructions": "Same place?", "criteria": {"true": "Yes, same", "false": "No"}}),
        ("first", {"type": "choice", "instructions": "Which is brighter?", "criteria": {"a": None, "b": None}}),
        ("count", {"type": "score", "instructions": "How many people?", "criteria": ["0", "1", "2", "3+"]}),
    ]),
]


class Tokenizer:
    names = sorted(SPECIAL, key=len, reverse=True)

    def encode(self, text, add_special_tokens=False):
        if text in TABLE:
            return [TABLE[text]]
        ids, i = [], 0
        while i < len(text):
            name = next((n for n in self.names if text.startswith(n, i)), None)
            if name is not None:
                ids.append(SPECIAL[name])
                i += len(name)
            else:
                ids.extend(b % 100 for b in text[i].encode("utf-8"))
                i += 1
        return ids


def cap_pixels(image, max_pixels=VISION_MAX_PIXELS):
    """runner.py: a picture downscaled to at most `max_pixels`, bicubic."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BICUBIC)


def sha(pixels):
    return hashlib.sha256(np.ascontiguousarray(pixels, dtype=np.uint8).tobytes()).hexdigest()


def views(processed, patch=16):
    """Each processed view's RGB bytes, recovered from its normalized patches."""
    out = []
    for values, mask, (rows, cols) in zip(processed["pixel_values"], processed["pixel_attention_mask"],
                                          processed["spatial_shapes"].tolist()):
        patches = values[: int(mask.sum())].numpy().reshape(rows, cols, patch, patch, 3)
        image = patches.transpose(0, 2, 1, 3, 4).reshape(rows * patch, cols * patch, 3)
        out.append(np.rint(image * 127.5 + 127.5).astype(np.uint8))
    return out


def summary(features):
    f = features.float()
    return {"count": f.shape[0], "sum": float(f.sum()), "abs_sum": float(f.abs().sum()),
            "first": f[0].tolist(), "last": f[-1].tolist()}


def randomize(model, generator):
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.ndim == 1:
                if "norm" in name and name.endswith("weight"):
                    value.copy_(1 + 0.2 * torch.randn(value.shape, generator=generator))
                else:
                    value.copy_(0.1 * torch.randn(value.shape, generator=generator))
            elif "conv.conv" in name:
                value.copy_(0.5 * torch.randn(value.shape, generator=generator))
            elif "embed_tokens" in name or "position_embedding" in name:
                value.copy_(torch.randn(value.shape, generator=generator))
            else:
                value.copy_(torch.randn(value.shape, generator=generator) / math.sqrt(value.shape[-1]))


def mlx_export(state, folder):
    """The same weights under mlx-vlm's names, as its loader sanitizes them."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm.models import lfm2_vl

    config = lfm2_vl.ModelConfig.from_dict({**CONFIG, **{k: v for k, v in PROCESSOR["image_processor"].items()
                                                          if k in ("use_thumbnail", "max_tiles", "min_tiles")}})
    # mlx-vlm sizes LFM2 from its block_* and conv_* keys.
    widths = {"block_dim": TEXT["hidden_size"], "conv_dim": TEXT["hidden_size"], "conv_dim_out": TEXT["hidden_size"],
              "block_ff_dim": TEXT["intermediate_size"], "num_heads": TEXT["num_attention_heads"]}
    config.text_config = lfm2_vl.TextConfig.from_dict({**TEXT, **widths})
    config.vision_config = lfm2_vl.VisionConfig.from_dict(VISION)
    model = lfm2_vl.Model(config)
    weights = {name: mx.array(value.float().numpy()) for name, value in state.items()}
    weights = model.sanitize(weights)
    weights = model.vision_tower.sanitize(weights)
    weights = model.language_model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    folder.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(folder / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (folder / "config.json").write_text(json.dumps(CONFIG, indent=1) + "\n")
    (folder / "processor_config.json").write_text(json.dumps(PROCESSOR, indent=2) + "\n")


def main():
    import mlx.core as mx
    import mlx_vlm

    installed = {"torch": torch.__version__.split("+")[0], "torchvision": torchvision.__version__.split("+")[0],
                 "transformers": transformers.__version__, "pillow": PIL.__version__, "mlx": mx.__version__,
                 "mlx_vlm": mlx_vlm.__version__}
    for package, version in installed.items():
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    tokenizer = Tokenizer()

    config = Lfm2VlConfig(**CONFIG)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    model = Lfm2VlForConditionalGeneration(config).eval()
    randomize(model, generator)
    state = {n: v.contiguous() for n, v in model.state_dict().items() if n != "lm_head.weight"}

    processor = Lfm2VlImageProcessor.from_dict(PROCESSOR["image_processor"])
    markup = Lfm2VlProcessor.__new__(Lfm2VlProcessor)
    markup.image_processor = processor
    markup.image_token, markup.image_start_token = "<image>", "<|image_start|>"
    markup.image_end_token, markup.image_thumbnail_token = "<|image_end|>", "<|img_thumbnail|>"

    def process(pictures):
        out = processor(images=[pictures], return_tensors="pt", return_row_col_info=True)
        texts = [markup.replace_image_token(out, i, use_image_special_tokens=True) for i in range(len(pictures))]
        n = int(out["pixel_attention_mask"].sum(1).max())
        vision = {"pixel_values": out["pixel_values"][:, :n], "pixel_attention_mask": out["pixel_attention_mask"][:, :n],
                  "spatial_shapes": out["spatial_shapes"]}
        return out, texts, vision

    pictures, images = {}, {}
    for name, (width, height, salt) in IMAGES.items():
        source = Image.fromarray(synthetic(width, height, salt))
        capped = cap_pixels(source)
        pictures[name] = capped
        out, texts, vision = process([capped])
        with torch.no_grad():
            features = torch.cat(model.model.get_image_features(**vision).pooler_output)
        images[name] = {
            "source": [width, height, salt], "capped": [capped.width, capped.height], "capped_sha256": sha(capped),
            "rows": int(out["image_rows"][0]), "columns": int(out["image_cols"][0]),
            "resized": [int(v) for v in out["image_sizes"][0]], "markup_tokens": texts[0].count("<image>"),
            "views": [{"size": list(v.shape[:2]), "sha256": sha(v)} for v in views(out)],
            "features": summary(features),
        }

    requests = []
    for state_value, names, questions in REQUESTS:
        _, texts, vision = process([pictures[n] for n in names])
        prefix = f"<|im_start|>user\n{''.join(texts)}{state_block(state_value)}\nQUESTION:\n"
        trunk = [BOS] + tokenizer.encode(prefix)
        entries = []
        for name, q in questions:
            suffix = suffix_text(tokenizer, q)
            if len(questions) == 1:
                ids, branch = [BOS] + tokenizer.encode(prefix + suffix), None
            else:
                branch = tokenizer.encode(suffix)
                ids = trunk + branch
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids]), **vision).logits[0, -1].float()
            logz = logits - torch.logsumexp(logits, -1)
            probs = readout(tokenizer, q, logz)
            codes = [c for c, _ in aliases(tokenizer, list(q["criteria"]))] if q["type"] == "choice" else []
            entries.append({"id": name, "question": q, "codes": codes, "groups": readout_ids(tokenizer, q),
                            "ids": branch if branch is not None else ids, "answer": answer(q, probs)})
        requests.append({"state": state_value, "images": names,
                         "prefix": trunk if len(questions) > 1 else [], "questions": entries})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_file(state, str(OUTPUT / "model.safetensors"))
    (OUTPUT / "config.json").write_text(json.dumps(CONFIG, indent=1) + "\n")
    (OUTPUT / "processor_config.json").write_text(json.dumps(PROCESSOR, indent=2) + "\n")
    mlx_export(state, OUTPUT / "mlx")
    out = {"format": "d1_3b_vision_tiny", "version": 1, "metadata": VERSIONS, "bos": BOS, "special": SPECIAL,
           "table": TABLE, "images": images, "requests": requests}
    (OUTPUT / "fixture.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
