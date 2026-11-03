#!/usr/bin/env python3
"""Write testdata/d1_tiny/: a tiny random d1-omni checkpoint in the PyTorch
layout, the MLX float16 and 8-bit exports mlx-vlm converts it to, and the
option logits mlx-vlm computes from each.

LiquidAI's d1-omni-600M-MLX-* exports were written by mlx-vlm's `convert`
(`model_type: d1_omni`), which renames tensors, splits the head's fused
attention projection, moves convolutions to MLX's channels-last layout, casts
to float16, and quantizes every linear outside the vision tower except the
head's scorer and type embedding. This fixture runs that same converter on a
model small enough to commit, so SystemOneD1 can be checked against the export
format and mlx-vlm's numbers without the 1-2 GB checkpoints.

    python3 -m venv .venv-mlx && .venv-mlx/bin/pip install "mlx==0.32.4" "mlx-vlm==0.7.11" tokenizers
    .venv-mlx/bin/python tools/d1_mlx_export_fixture.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx_vlm
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "d1_tiny"
VERSIONS = {"mlx": "0.32.4", "mlx_vlm": "0.7.11"}
SEED = 20261103

CONFIG = {
    "architectures": ["D1OmniModel"],
    "model_type": "d1_omni",
    "dtype": "float32",
    "bos_token_id": 1,
    "pad_token_id": 0,
    "head_layers": 2,
    "max_length": 512,
    "image_text_length": 256,
    "audio_text_length": 256,
    "projector_hidden_size": 64,
    "temperatures": {"choice": 1.0, "score": 1.0, "noul": 1.0},
    "text_config": {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 192,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "layer_types": ["conv", "full_attention", "conv", "full_attention"],
        "norm_eps": 1e-05,
        "conv_L_cache": 3,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 64,
        "max_position_embeddings": 1024,
        "rope_theta": 1000000.0,
    },
    "vision_config": {
        "model_type": "siglip2_vision_model",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_channels": 3,
        "num_hidden_layers": 1,
        "num_patches": 16,
        "patch_size": 16,
        "layer_norm_eps": 1e-06,
        "hidden_act": "gelu_pytorch_tanh",
        "attention_dropout": 0.0,
        "vision_use_head": False,
    },
    "audio_config": {
        "feat_in": 128,
        "n_layers": 1,
        "d_model": 64,
        "subsampling_conv_channels": 16,
        "ff_expansion_factor": 2,
        "n_heads": 2,
        "conv_kernel_size": 9,
        "residual_width": 64,
    },
}

# One text row per question type: ids, marker positions, question type (choice 0, score 1, noul 2).
ROWS = [
    {"ids": [1, 7, 23, 4, 9, 2, 11, 2, 12, 3, 5], "markers": [5, 7], "type": 0},
    {"ids": [1, 30, 31, 32, 2, 40, 2, 41, 2, 42, 6, 8, 9, 10], "markers": [4, 6, 8], "type": 1},
    {"ids": [1, 99, 2, 100, 2, 101, 17], "markers": [2, 4], "type": 2},
]


def feed_forward_size(text):
    scaled = int(text["block_ffn_dim_multiplier"] * (2 * text["intermediate_size"] // 3))
    multiple = text["block_multiple_of"]
    return multiple * ((scaled + multiple - 1) // multiple)


def schema(config):
    """`D1Weights.schema(for:)`: every PyTorch-layout tensor name and shape."""
    out = {}

    def add(name, shape):
        out[name] = shape

    def linear(name, output, inputs, bias=True):
        add(f"{name}.weight", [output, inputs])
        if bias:
            add(f"{name}.bias", [output])

    def norm(name, width):
        add(f"{name}.weight", [width])
        add(f"{name}.bias", [width])

    text = config["text_config"]
    d = text["hidden_size"]
    head = d // text["num_attention_heads"]
    ffn = feed_forward_size(text)
    add("encoder.embed_tokens.weight", [text["vocab_size"], d])
    add("encoder.embedding_norm.weight", [d])
    for index, kind in enumerate(text["layer_types"]):
        prefix = f"encoder.layers.{index}"
        if kind == "full_attention":
            linear(f"{prefix}.self_attn.q_proj", text["num_attention_heads"] * head, d, bias=False)
            linear(f"{prefix}.self_attn.k_proj", text["num_key_value_heads"] * head, d, bias=False)
            linear(f"{prefix}.self_attn.v_proj", text["num_key_value_heads"] * head, d, bias=False)
            linear(f"{prefix}.self_attn.out_proj", d, text["num_attention_heads"] * head, bias=False)
            add(f"{prefix}.self_attn.q_layernorm.weight", [head])
            add(f"{prefix}.self_attn.k_layernorm.weight", [head])
        else:
            add(f"{prefix}.conv.conv.weight", [d, 1, text["conv_L_cache"]])
            linear(f"{prefix}.conv.in_proj", 3 * d, d, bias=False)
            linear(f"{prefix}.conv.out_proj", d, d, bias=False)
        linear(f"{prefix}.feed_forward.w1", ffn, d, bias=False)
        linear(f"{prefix}.feed_forward.w3", ffn, d, bias=False)
        linear(f"{prefix}.feed_forward.w2", d, ffn, bias=False)
        add(f"{prefix}.operator_norm.weight", [d])
        add(f"{prefix}.ffn_norm.weight", [d])

    add("head.type_emb.weight", [3, d])
    for index in range(config["head_layers"]):
        prefix = f"head.head.layers.{index}"
        add(f"{prefix}.self_attn.in_proj_weight", [3 * d, d])
        add(f"{prefix}.self_attn.in_proj_bias", [3 * d])
        linear(f"{prefix}.self_attn.out_proj", d, d)
        linear(f"{prefix}.linear1", 4 * d, d)
        linear(f"{prefix}.linear2", d, 4 * d)
        norm(f"{prefix}.norm1", d)
        norm(f"{prefix}.norm2", d)
    norm("head.scorer.0", d)
    linear("head.scorer.1", d, d)
    linear("head.scorer.3", 1, d)

    vision = config["vision_config"]
    v = vision["hidden_size"]
    tower = "vision.tower.vision_model"
    linear(f"{tower}.embeddings.patch_embedding", v, vision["num_channels"] * vision["patch_size"] ** 2)
    add(f"{tower}.embeddings.position_embedding.weight", [vision["num_patches"], v])
    for index in range(vision["num_hidden_layers"]):
        prefix = f"{tower}.encoder.layers.{index}"
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(f"{prefix}.self_attn.{projection}", v, v)
        linear(f"{prefix}.mlp.fc1", vision["intermediate_size"], v)
        linear(f"{prefix}.mlp.fc2", v, vision["intermediate_size"])
        norm(f"{prefix}.layer_norm1", v)
        norm(f"{prefix}.layer_norm2", v)
    norm(f"{tower}.post_layernorm", v)
    linear("vision.projector.linear_1", config["projector_hidden_size"], 4 * v)
    linear("vision.projector.linear_2", d, config["projector_hidden_size"])

    audio = config["audio_config"]
    a = audio["d_model"]
    channels = audio["subsampling_conv_channels"]
    features = audio["feat_in"]
    for _ in range(3):
        features = (features - 1) // 2 + 1
    linear("audio.encoder.pre_encode.out", a, channels * features)
    add("audio.encoder.pre_encode.conv.0.weight", [channels, 1, 3, 3])
    add("audio.encoder.pre_encode.conv.0.bias", [channels])
    for depthwise in (2, 5):
        add(f"audio.encoder.pre_encode.conv.{depthwise}.weight", [channels, 1, 3, 3])
        add(f"audio.encoder.pre_encode.conv.{depthwise}.bias", [channels])
        add(f"audio.encoder.pre_encode.conv.{depthwise + 1}.weight", [channels, channels, 1, 1])
        add(f"audio.encoder.pre_encode.conv.{depthwise + 1}.bias", [channels])
    for index in range(audio["n_layers"]):
        prefix = f"audio.encoder.layers.{index}"
        for ff in ("feed_forward1", "feed_forward2"):
            linear(f"{prefix}.{ff}.linear1", a * audio["ff_expansion_factor"], a)
            linear(f"{prefix}.{ff}.linear2", a, a * audio["ff_expansion_factor"])
        for name in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            norm(f"{prefix}.{name}", a)
        for projection in ("linear_q", "linear_k", "linear_v", "linear_out"):
            linear(f"{prefix}.self_attn.{projection}", a, a)
        linear(f"{prefix}.self_attn.linear_pos", a, a, bias=False)
        add(f"{prefix}.self_attn.pos_bias_u", [audio["n_heads"], a // audio["n_heads"]])
        add(f"{prefix}.self_attn.pos_bias_v", [audio["n_heads"], a // audio["n_heads"]])
        add(f"{prefix}.conv.pointwise_conv1.weight", [2 * a, a, 1])
        add(f"{prefix}.conv.pointwise_conv1.bias", [2 * a])
        add(f"{prefix}.conv.depthwise_conv.weight", [a, 1, audio["conv_kernel_size"]])
        add(f"{prefix}.conv.depthwise_conv.bias", [a])
        add(f"{prefix}.conv.pointwise_conv2.weight", [a, a, 1])
        add(f"{prefix}.conv.pointwise_conv2.bias", [a])
        norm(f"{prefix}.conv.batch_norm", a)
        add(f"{prefix}.conv.batch_norm.running_mean", [a])
        add(f"{prefix}.conv.batch_norm.running_var", [a])
        out[f"{prefix}.conv.batch_norm.num_batches_tracked"] = None  # int64 scalar
    norm("audio.adapter.norm", a)
    linear("audio.adapter.linear_1", d, a)
    linear("audio.adapter.linear_2", d, d)
    norm("audio.residual.ln", d)
    linear("audio.residual.down", audio["residual_width"], d)
    linear("audio.residual.up", d, audio["residual_width"])
    return out


def random_weights(config):
    rng = np.random.default_rng(SEED)
    weights = {}
    for name, shape in schema(config).items():
        if shape is None:
            weights[name] = mx.array(np.array(7, dtype=np.int64))
        elif name.endswith("running_var"):
            weights[name] = mx.array(rng.uniform(0.5, 1.5, shape).astype(np.float32))
        elif name.endswith(("norm.weight", "layernorm.weight", "norm1.weight", "norm2.weight", ".ln.weight",
                            "post_layernorm.weight", "scorer.0.weight")) or ".norm_" in name and name.endswith(".weight"):
            weights[name] = mx.array((1 + 0.1 * rng.standard_normal(shape)).astype(np.float32))
        else:
            scale = 0.5 if name.endswith(("bias", "running_mean", "pos_bias_u", "pos_bias_v")) else 0.2
            weights[name] = mx.array((scale * rng.standard_normal(shape)).astype(np.float32))
    return weights


def write_tokenizer(folder):
    """A word-level tokenizer: the converter loads one, the fixture never uses it."""
    from tokenizers import Tokenizer, models

    vocab = {f"t{i}": i for i in range(CONFIG["text_config"]["vocab_size"])}
    Tokenizer(models.WordLevel(vocab, unk_token="t0")).save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))


def logits(folder):
    """Raw option logits mlx-vlm computes for `ROWS` from a checkpoint folder."""
    from mlx_vlm.utils import load_model

    model = load_model(folder)
    dtype = model.encoder.embedding_norm.weight.dtype
    length = max(len(row["ids"]) for row in ROWS)
    embeds = mx.stack([
        mx.pad(model.encoder.embed_tokens(mx.array(row["ids"])).astype(dtype), [(0, length - len(row["ids"])), (0, 0)])
        for row in ROWS
    ])
    pad = mx.arange(length)[None] < mx.array([len(row["ids"]) for row in ROWS])[:, None]
    k = max(len(row["markers"]) for row in ROWS)
    positions = mx.array([row["markers"] + [0] * (k - len(row["markers"])) for row in ROWS])
    mask = mx.arange(k)[None] < mx.array([len(row["markers"]) for row in ROWS])[:, None]
    types = mx.array([row["type"] for row in ROWS])
    out = model(embeds, pad, [0] * len(ROWS), [len(row["ids"]) for row in ROWS], positions, mask, types)
    out = out.astype(mx.float32)
    return [[float(x) for x in out[i, : len(row["markers"])].tolist()] for i, row in enumerate(ROWS)]


def main():
    for package, version in (("mlx", mx.__version__), ("mlx_vlm", mlx_vlm.__version__)):
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    mx.set_default_device(mx.cpu)
    from mlx_vlm.convert import convert

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    pytorch = OUTPUT / "pytorch"
    pytorch.mkdir(parents=True)
    (pytorch / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    mx.save_safetensors(str(pytorch / "model.safetensors"), random_weights(CONFIG))

    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "source"
        shutil.copytree(pytorch, source)
        write_tokenizer(source)
        for name, options in (("mlx-fp16", {}), ("mlx-8bit", {"quantize": True, "q_bits": 8, "q_group_size": 64})):
            target = OUTPUT / name
            convert(str(source), mlx_path=str(target), dtype="float16", **options)
            for extra in ("tokenizer.json", "tokenizer_config.json"):
                (target / extra).unlink(missing_ok=True)
        results = {}
        shutil.copy(source / "tokenizer.json", pytorch / "tokenizer.json")
        shutil.copy(source / "tokenizer_config.json", pytorch / "tokenizer_config.json")
        for name in ("pytorch", "mlx-fp16", "mlx-8bit"):
            folder = OUTPUT / name
            if name != "pytorch":
                shutil.copy(source / "tokenizer.json", folder / "tokenizer.json")
                shutil.copy(source / "tokenizer_config.json", folder / "tokenizer_config.json")
            results[name] = logits(folder)
            for extra in ("tokenizer.json", "tokenizer_config.json"):
                (folder / extra).unlink(missing_ok=True)

    names = {}
    for name in ("mlx-fp16", "mlx-8bit"):
        index = json.loads((OUTPUT / name / "model.safetensors.index.json").read_text())
        names[name] = sorted(index["weight_map"])
    fixture = {
        "format": "d1_tiny_mlx_export",
        "version": 1,
        "metadata": {**VERSIONS, "seed": SEED, "device": "cpu"},
        "rows": ROWS,
        "logits": results,
        "tensor_names": names,
    }
    (OUTPUT / "fixture.json").write_text(json.dumps(fixture, indent=1) + "\n")
    print(json.dumps({k: [[round(x, 4) for x in row] for row in v] for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
