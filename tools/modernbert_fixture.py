#!/usr/bin/env python3
"""Write testdata/modernbert_tiny/: tiny random ModernBERT encoders and the hidden
states transformers' `ModernBertModel` computes for them.

Two configurations cover the family's RoPE schemes: ModernBERT-large's (160k on
global layers, 10k on sliding ones) and mmBERT/Ettin's (160k on both). Each runs
a padded batch with default positions, and a tree-masked row in which every
option attends only to the shared prefix and itself, its position ids restarting
after the prefix and the sliding window measured on those ids (von 1.2's
independent options, Bekko, tasksource-decider-nano).

    .venv/bin/pip install "torch==2.15.0" "transformers==5.22.1" safetensors
    .venv/bin/python tools/modernbert_fixture.py
"""

import json
import sys
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import ModernBertConfig, ModernBertModel

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "modernbert_tiny"
VERSIONS = {"torch": "2.15.0", "transformers": "5.22.1"}
SEED = 20261103

BASE = dict(
    vocab_size=64, hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=48,
    global_attn_every_n_layers=3, local_attention=8, max_position_embeddings=128,
    norm_eps=1e-5, norm_bias=False, attention_bias=False, mlp_bias=False,
    hidden_activation="gelu", pad_token_id=0, cls_token_id=1, sep_token_id=2,
    attn_implementation="eager",
)
CONFIGS = {
    "large-rope": {"global_rope_theta": 160000.0, "local_rope_theta": 10000.0},
    "uniform-rope": {"global_rope_theta": 160000.0, "local_rope_theta": 160000.0},
}

PADDED = {"ids": [[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 2], [1, 7, 11, 19, 23, 31, 43, 47, 2, 0, 0, 0]],
          "lengths": [12, 9]}
# A shared prefix of 6 tokens, then two 3-token options.
TREE_IDS = [1, 10, 11, 12, 13, 2, 3, 20, 21, 3, 30, 31]
TREE_SEGMENTS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


def tree_layout():
    positions, counters = [], {}
    prefix = TREE_SEGMENTS.count(0)
    for segment in TREE_SEGMENTS:
        if segment == 0:
            positions.append(len(positions))
        else:
            positions.append(prefix + counters.get(segment, 0))
            counters[segment] = counters.get(segment, 0) + 1
    visible = [[k_seg == 0 or k_seg == q_seg for k_seg in TREE_SEGMENTS] for q_seg in TREE_SEGMENTS]
    return positions, visible


def additive(mask):
    return torch.where(mask, torch.tensor(0.0), torch.tensor(float("-inf")))[None, None]


def main():
    for package, version in (("torch", torch.__version__.split("+")[0]), ("transformers", transformers.__version__)):
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    torch.manual_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixture = {"format": "modernbert_tiny", "version": 1, "metadata": {**VERSIONS, "seed": SEED}, "models": {}}
    positions, visible = tree_layout()
    radius = BASE["local_attention"] // 2
    for name, rope in CONFIGS.items():
        config = ModernBertConfig(**BASE, **rope)
        model = ModernBertModel(config).eval()
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if "norm" in key:
                    value.copy_(1 + 0.1 * torch.randn_like(value))
                else:
                    value.normal_(0, 0.3)
        folder = OUTPUT / name
        folder.mkdir(exist_ok=True)
        config.save_pretrained(folder)
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, folder / "model.safetensors")

        with torch.no_grad():
            ids = torch.tensor(PADDED["ids"])
            mask = (torch.arange(ids.shape[1])[None] < torch.tensor(PADDED["lengths"])[:, None]).long()
            padded = model(input_ids=ids, attention_mask=mask).last_hidden_state

            q = torch.tensor(positions)
            sees = torch.tensor(visible)
            window = (q[:, None] - q[None, :]).abs() <= radius
            tree = model(
                input_ids=torch.tensor([TREE_IDS]), position_ids=q[None],
                attention_mask={"full_attention": additive(sees), "sliding_attention": additive(sees & window)},
            ).last_hidden_state

        fixture["models"][name] = {
            "padded": [[float(x) for x in row.reshape(-1)] for row in padded],
            "tree": [float(x) for x in tree[0].reshape(-1)],
        }
    fixture["padded_inputs"] = PADDED
    fixture["tree_inputs"] = {"ids": TREE_IDS, "positions": positions, "visible": visible}
    (OUTPUT / "fixture.json").write_text(json.dumps(fixture) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
