#!/usr/bin/env python3
"""Write testdata/lfm2_tiny/: a tiny random causal LFM2 trunk in LFM2-VL's layout
(as d1-3B ships it) and transformers' hidden states for it.

- model.safetensors: `model.language_model.…` names with tied embeddings, plus a
  `model.vision_tower.` tensor the trunk must ignore; config.json nests
  `text_config` as `Lfm2VlForConditionalGeneration` does.
- padded: two right-padded causal rows (40 and 17 tokens).
- prefix: a 23-token prefix, then two rows after it; expected states come from
  running each whole sequence alone.
- readout: embedding rows (tied head) and the logits they give.

    .venv/bin/pip install "torch==2.15.0" "transformers==5.22.1" safetensors
    .venv/bin/python tools/lfm2_fixture.py
"""

import json
import sys
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import Lfm2Config, Lfm2ForCausalLM

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "lfm2_tiny"
VERSIONS = {"torch": "2.15.0", "transformers": "5.22.1"}
SEED = 20261103

TEXT = {
    "model_type": "lfm2", "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
    "block_auto_adjust_ff_dim": False, "num_hidden_layers": 4,
    "layer_types": ["conv", "conv", "full_attention", "conv"],
    "num_attention_heads": 4, "num_key_value_heads": 2, "norm_eps": 1e-5, "conv_L_cache": 3,
    "conv_bias": False, "tie_word_embeddings": True, "max_position_embeddings": 4096,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
}
CONFIG = {"architectures": ["Lfm2VlForConditionalGeneration"], "model_type": "lfm2_vl", "text_config": TEXT,
          "tie_word_embeddings": True, "bos_token_id": 1}


def main():
    for package, version in (("torch", torch.__version__.split("+")[0]), ("transformers", transformers.__version__)):
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    tokens = lambda n: torch.randint(3, TEXT["vocab_size"], (n,), generator=generator).tolist()

    config = Lfm2Config(**TEXT)
    config._attn_implementation = "eager"
    model = Lfm2ForCausalLM(config).eval()
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.ndim == 1:
                value.copy_(1 + 0.2 * torch.randn_like(value))
            elif "conv.conv" in name:
                value.normal_(0, 0.5)
            else:
                value.normal_(0, 0.25)

    def hidden(ids, mask=None):
        with torch.no_grad():
            return model.model(input_ids=torch.tensor(ids), attention_mask=mask).last_hidden_state

    out = {"format": "lfm2_tiny", "version": 1, "metadata": {**VERSIONS, "seed": SEED}}
    rows = [tokens(40), tokens(17)]
    width = max(map(len, rows))
    h = hidden([r + [0] * (width - len(r)) for r in rows],
               torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]))
    out["padded"] = {"ids": [r + [0] * (width - len(r)) for r in rows], "lengths": [len(r) for r in rows],
                     "hidden": [h[i, : len(r)].flatten().tolist() for i, r in enumerate(rows)]}

    shared, rows = tokens(23), [tokens(9), tokens(4)]
    out["prefix"] = {"prefix": shared, "prefix_hidden": hidden([shared])[0].flatten().tolist(), "rows": rows,
                     "hidden": [hidden([shared + r])[0, len(shared):].flatten().tolist() for r in rows]}

    embed = model.model.embed_tokens.weight.detach()
    picked = [5, 66, 127]
    last = h[0, len(out["padded"]["ids"][0]) - 1 if out["padded"]["lengths"][0] == width else out["padded"]["lengths"][0] - 1]
    out["readout"] = {"ids": picked, "rows": embed[picked].flatten().tolist(), "logits": (embed[picked] @ last).tolist()}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    weights = {"model.language_model." + n[len("model."):]: v.detach().contiguous()
               for n, v in model.state_dict().items() if n.startswith("model.")}
    weights["model.vision_tower.vision_model.post_layernorm.weight"] = torch.ones(4)
    save_file(weights, OUTPUT / "model.safetensors")
    (OUTPUT / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (OUTPUT / "fixture.json").write_text(json.dumps(out) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
