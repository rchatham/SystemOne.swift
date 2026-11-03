#!/usr/bin/env python3
"""Write testdata/qwen35_tiny/: a tiny random Qwen3.5 text trunk in transformers'
layout and mlx-lm's (float32 and 8-bit), with the reference hidden states
SystemOneDecoders must reproduce.

- transformers/: `Qwen3_5ForConditionalGeneration` tensor names
  (`model.language_model.…`, `lm_head.weight`) plus an `mtp.` tensor and a
  `model.visual.` tensor the trunk must ignore. The config nests `text_config`.
- mlx-fp32/, mlx-8bit/: the same weights through mlx-lm's `sanitize` (norms
  with `+ 1`, the convolution as `[channels, kernel, 1]`), the second quantized
  with group size 32 at 8 bits as `mlx_lm.convert` would.

The cases (see `fixture.json`):
- padded: two right-padded rows, 150 and 70 tokens (three 64-token chunks), causal.
- bidirectional: d3's plan, full-attention layers see the whole row; the
  recurrent layers stay causal.
- tree: one row with restarting positions and block-causal visibility.
- prefix: a 37-token prefix, then two rows after it; expected states come from
  running each whole sequence alone.
- readout: lm_head rows and the logits they give.
- mlx_8bit: mlx-lm's own hidden states for the quantized export.

    .venv/bin/pip install "torch==2.15.0" "transformers==5.22.1" "mlx==0.32.4" "mlx-lm==0.32.2" safetensors
    .venv/bin/python tools/qwen35_fixture.py
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as mnn
import mlx_lm
import torch
import transformers
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5 as mlx_qwen35
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "qwen35_tiny"
VERSIONS = {"torch": "2.15.0", "transformers": "5.22.1", "mlx": "0.32.4", "mlx_lm": "0.32.2"}
SEED = 20261103

TEXT = {
    "model_type": "qwen3_5_text",
    "vocab_size": 96,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 6,
    "full_attention_interval": 3,  # mlx-lm reads only the interval
    "layer_types": ["linear_attention", "linear_attention", "full_attention"] * 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "attn_output_gate": True,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "rope_parameters": {
        "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
        "mrope_section": [1, 1, 0], "mrope_interleaved": True,
    },
}
CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": TEXT,
    "tie_word_embeddings": False,
}


def randomize(model):
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name.endswith("A_log"):
                value.copy_(torch.log(torch.empty_like(value).uniform_(0.5, 8)))
            elif name.endswith("dt_bias") or name.endswith("linear_attn.norm.weight"):
                value.copy_(1 + 0.2 * torch.randn_like(value))
            elif value.ndim == 1:
                value.copy_(0.2 * torch.randn_like(value))  # zero-centred norms
            elif "conv1d" in name:
                value.normal_(0, 0.4)
            else:
                value.normal_(0, 0.2)


def hidden(model, ids, attention_mask=None, position_ids=None):
    with torch.no_grad():
        return model.model(
            input_ids=torch.tensor(ids), attention_mask=attention_mask, position_ids=position_ids
        ).last_hidden_state


def additive(visible):
    return torch.where(torch.tensor(visible), torch.tensor(0.0), torch.tensor(float("-inf")))[:, None]


def main():
    for package, version in (("torch", torch.__version__.split("+")[0]), ("transformers", transformers.__version__),
                             ("mlx", mx.__version__), ("mlx_lm", mlx_lm.__version__)):
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    tokens = lambda n: torch.randint(0, TEXT["vocab_size"], (n,), generator=generator).tolist()

    config = Qwen3_5TextConfig(**TEXT)
    config._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(config).eval()
    randomize(model)
    state = {name: value.detach().contiguous() for name, value in model.state_dict().items()}

    out = {"format": "qwen35_tiny", "version": 1, "metadata": {**VERSIONS, "seed": SEED}}

    # padded: right-padded causal rows.
    rows = [tokens(150), tokens(70)]
    width = max(map(len, rows))
    ids = [r + [0] * (width - len(r)) for r in rows]
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
    h = hidden(model, ids, attention_mask=mask)
    out["padded"] = {"ids": ids, "lengths": [len(r) for r in rows],
                     "hidden": [h[i, : len(r)].flatten().tolist() for i, r in enumerate(rows)]}

    # bidirectional: full-attention layers see every real token of the row.
    rows = [tokens(40), tokens(23)]
    width = max(map(len, rows))
    ids = [r + [0] * (width - len(r)) for r in rows]
    lengths = [len(r) for r in rows]
    visible = [[[k < n if q < n else k == q for k in range(width)] for q in range(width)] for n in lengths]
    padding = torch.tensor([[1] * n + [0] * (width - n) for n in lengths])
    h = hidden(model, ids, attention_mask={"full_attention": additive(visible), "linear_attention": padding})
    out["bidirectional"] = {"ids": ids, "lengths": lengths,
                            "hidden": [h[i, :n].flatten().tolist() for i, n in enumerate(lengths)]}

    # tree: a causal 10-token prefix, then two blocks that see the prefix and
    # themselves causally, both at positions 10…
    prefix, blocks = 10, [6, 5]
    row = tokens(prefix + sum(blocks))
    block = [-1] * prefix + [b for b, n in enumerate(blocks) for _ in range(n)]
    positions = list(range(prefix)) + [prefix + i for n in blocks for i in range(n)]
    visible = [[k <= q and (block[k] == -1 or block[k] == block[q]) for k in range(len(row))] for q in range(len(row))]
    h = hidden(model, [row], attention_mask={"full_attention": additive([visible]), "linear_attention": None},
               position_ids=torch.tensor([positions]))
    out["tree"] = {"ids": row, "positions": positions, "visible": visible, "hidden": h[0].flatten().tolist()}

    # prefix: rows after a cached prefix match each whole sequence run alone.
    shared = tokens(37)
    rows = [tokens(12), tokens(5)]
    out["prefix"] = {
        "prefix": shared,
        "prefix_hidden": hidden(model, [shared])[0].flatten().tolist(),
        "rows": rows,
        "hidden": [hidden(model, [shared + r])[0, len(shared):].flatten().tolist() for r in rows],
    }

    # readout: lm_head rows and the logits at the first padded row's last token.
    picked = [3, 17, 95]
    last = torch.tensor(out["padded"]["hidden"][0][-TEXT["hidden_size"]:])
    out["readout"] = {"ids": picked, "rows": state["lm_head.weight"][picked].flatten().tolist(),
                      "logits": (state["lm_head.weight"][picked] @ last).tolist()}

    # transformers layout, with tensors the trunk must ignore.
    folder = OUTPUT / "transformers"
    folder.mkdir(parents=True, exist_ok=True)
    weights = {("model.language_model." + n[len("model."):] if n.startswith("model.") else n): v for n, v in state.items()}
    weights["mtp.fc.weight"] = torch.zeros(TEXT["hidden_size"], 2 * TEXT["hidden_size"])
    weights["model.visual.merger.linear_fc1.weight"] = torch.zeros(4, 4)
    save_file(weights, folder / "model.safetensors")
    (folder / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    # mlx-lm layouts.
    mlx_model = mlx_qwen35.Model(mlx_qwen35.ModelArgs.from_dict(CONFIG))
    converted = mlx_model.sanitize({n: mx.array(v.numpy()) for n, v in weights.items()})
    mlx_model.load_weights(list(converted.items()))
    for name, quantize in (("mlx-fp32", False), ("mlx-8bit", True)):
        if quantize:
            mnn.quantize(mlx_model, group_size=32, bits=8,
                         class_predicate=lambda _, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % 32 == 0)
        folder = OUTPUT / name
        folder.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(folder / "model.safetensors"), dict(tree_flatten(mlx_model.parameters())))
        config = dict(CONFIG, **({"quantization": {"group_size": 32, "bits": 8}} if quantize else {}))
        (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    row = out["padded"]["ids"][1][:70]
    out["mlx_8bit"] = {"ids": row, "hidden": mlx_model.language_model.model(mx.array([row]))[0].flatten().tolist()}

    (OUTPUT / "fixture.json").write_text(json.dumps(out) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
