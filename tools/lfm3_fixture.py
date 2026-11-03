#!/usr/bin/env python3
"""Write testdata/lfm3_tiny/: a tiny random LFM3 trunk in transformers' LFM3-VL
layout (as d1-2B ships it) and mlx-vlm's hidden states for it.

transformers 5.22.1 has no LFM3, so the reference is mlx-vlm's `lfm3` port
(Apache-2.0), run on the CPU in float32. The model has every LFM3 part:
7-tap convolutions, sliding attention with an 8-token window (shorter than the
rows, so the window binds), gated full attention with half-rotated RoPE, and
sandwich norms.

- rows: three causal rows, each run alone (Swift runs them as one padded batch).
- set: one packed row under d1-2B's set attention (`d1_lfm3_vl.d1.set_masks`):
  full attention reads the state and its own question both ways, sliding
  attention and the convolutions stay causal.
- prefix: a 13-token prefix, then two rows after it; expected states come from
  running each whole sequence alone.

    .venv/bin/pip install "mlx==0.32.4" "mlx-vlm==0.7.11" safetensors numpy
    .venv/bin/python tools/lfm3_fixture.py
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx_vlm
import numpy as np
from mlx.utils import tree_flatten
from mlx_vlm.models.d1_lfm3_vl.d1 import set_masks
from mlx_vlm.models.lfm3.config import ModelConfig
from mlx_vlm.models.lfm3.language import LanguageModel

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "lfm3_tiny"
VERSIONS = {"mlx": "0.32.4", "mlx_vlm": "0.7.11"}
SEED = 20261103

TEXT = {
    "model_type": "lfm3", "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 6,
    "layer_types": ["conv", "sliding_attention", "full_attention", "conv", "sliding_attention", "full_attention"],
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "norm_eps": 1e-5, "conv_L_cache": 7,
    "conv_bias": False, "sliding_window": 8, "use_sandwich_norm": True, "tie_word_embeddings": True,
    "max_position_embeddings": 4096,
    "rope_parameters": {"full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
                        "sliding_attention": {"rope_theta": 10000.0}},
}
CONFIG = {"architectures": ["Lfm3VlForConditionalGeneration"], "model_type": "lfm3_vl", "text_config": TEXT,
          "tie_word_embeddings": True}


def main():
    for package, version in (("mlx", mx.__version__), ("mlx_vlm", mlx_vlm.__version__)):
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    mx.set_default_device(mx.cpu)
    mx.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    tokens = lambda n: rng.integers(3, TEXT["vocab_size"], n).tolist()

    model = LanguageModel(ModelConfig(**{k: v for k, v in TEXT.items() if k != "model_type"}))
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        if value.ndim == 1:
            weights[name] = mx.array((1 + 0.2 * rng.standard_normal(value.shape)).astype(np.float32))
        else:
            scale = 0.5 if "conv.conv" in name else 0.25
            weights[name] = mx.array((scale * rng.standard_normal(value.shape)).astype(np.float32))
    model.load_weights(list(weights.items()))
    backbone = model.model

    def hidden(ids):
        return backbone(mx.array([ids]))[0]

    out = {"format": "lfm3_tiny", "version": 1, "metadata": {**VERSIONS, "seed": SEED}}
    rows = [tokens(30), tokens(12), tokens(21)]
    out["rows"] = {"ids": rows, "hidden": [np.array(hidden(r)).flatten().tolist() for r in rows]}

    # A d1-2B set row: a 9-token state, then three questions of 5, 7, and 6 tokens.
    state, groups = 9, [5, 7, 6]
    ids = tokens(state + sum(groups))
    question = [-1] * state + [k + 1 for k, n in enumerate(groups) for _ in range(n)]
    masks = set_masks(len(ids), state, question, TEXT["sliding_window"])
    h = backbone.embed_tokens(mx.array(ids))[None]
    for layer in backbone.layers:
        h = layer(h, masks[layer.layer_type], None)
    h = backbone.embedding_norm(h)[0]
    visible = np.array(masks["full_attention"]).tolist()
    out["set"] = {"ids": ids, "state": state, "question": question, "visible": visible,
                  "hidden": np.array(h).flatten().tolist()}

    shared, rows = tokens(13), [tokens(9), tokens(4)]
    out["prefix"] = {"prefix": shared, "prefix_hidden": np.array(hidden(shared)).flatten().tolist(), "rows": rows,
                     "hidden": [np.array(hidden(shared + r)[len(shared):]).flatten().tolist() for r in rows]}

    embed = backbone.embed_tokens.weight
    picked = [4, 77, 127]
    last = mx.array(out["rows"]["hidden"][0][-TEXT["hidden_size"]:])
    out["readout"] = {"ids": picked, "rows": np.array(embed[mx.array(picked)]).flatten().tolist(),
                      "logits": np.array(embed[mx.array(picked)] @ last).tolist()}

    # transformers' LFM3-VL names: model.language_model.…, convolutions [channels, 1, kernel].
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stored = {}
    for name, value in tree_flatten(model.parameters()):
        if not name.startswith("model."):
            continue
        if name.endswith("conv.conv.weight"):
            value = value.transpose(0, 2, 1)
        stored["model.language_model." + name[len("model."):]] = value
    stored["model.vision_tower.encoder.post_norm.weight"] = mx.ones(4)
    mx.save_safetensors(str(OUTPUT / "model.safetensors"), stored)
    (OUTPUT / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (OUTPUT / "fixture.json").write_text(json.dumps(out) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
