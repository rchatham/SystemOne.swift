#!/usr/bin/env python3
"""Write testdata/d1_2b_tiny/fixture.json: d1-2B set-attention rows and answers.

d1-2B (LiquidAI/d1-2B, LFM Open License v1.0) packs up to `d1_set_size`
questions after one state in a single row: full-attention layers read the
state and each question's own tokens both ways, sliding attention and the
convolutions stay causal, and each question is read at its last token
(`</think>`) through the tied head. The prompt, packing, readout, calibration
and answers here are mlx-vlm 0.7.11's `d1_lfm3_vl` (`prompt`, `api`,
`calibration`, `d1.pack_set`, `d1.set_masks`), adapted there from the code
LiquidAI ships with d1 checkpoints; this script only calls them.

The model is the tiny LFM3 of testdata/lfm3_tiny. Its stand-in tokenizer
splits special tokens out of text as Hugging Face tokenizers do, maps a few
whole strings (yes/no forms, two spaced codes) to their own ids, and every
other text to one token per UTF-8 byte (`byte % 100`); the Swift test mirrors
it. Calibration: d1-2B's default temperature 1.337398453966105 with a noul
temperature of 0.9, to exercise the buckets.

    .venv/bin/pip install "mlx==0.32.4" "mlx-vlm==0.7.11" safetensors numpy
    .venv/bin/python tools/d1_2b_fixture.py
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx_vlm
import numpy as np
from mlx_vlm.models.d1_lfm3_vl.api import answer
from mlx_vlm.models.d1_lfm3_vl.calibration import Calibration
from mlx_vlm.models.d1_lfm3_vl.d1 import LEAD, pack_set, set_masks
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
from mlx_vlm.models.lfm3.config import ModelConfig
from mlx_vlm.models.lfm3.language import LanguageModel

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "d1_2b_tiny"
TRUNK = ROOT / "testdata" / "lfm3_tiny"
VERSIONS = {"mlx": "0.32.4", "mlx_vlm": "0.7.11"}
SET_SIZE = 2
CODES = 2
CALIBRATION = Calibration({"noul": 0.9}, 1.337398453966105)

# Special tokens: matched anywhere in text; the `special` ones are what `plain` escapes.
SPECIAL = {"<|startoftext|>": 100, "<|im_start|>": 101, "<|im_end|>": 102, "<|pad|>": 105}
ADDED = {**SPECIAL, "<think>": 103, "</think>": 104}
WORDS = {"yes": 110, "Yes": 111, "YES": 112, "no": 113, "No": 114, "NO": 115, " A": 116, " B": 117}


class Tokenizer:
    bos_token = "<|startoftext|>"
    all_special_tokens = list(SPECIAL)
    d1_codes = CODES
    d1_numbers = False
    d1_prompt = 1
    d1_box_edges = "median"

    def encode(self, text, add_special_tokens=False):
        if text in WORDS:
            return [WORDS[text]]
        names = sorted(ADDED, key=len, reverse=True)
        ids, i = [], 0
        while i < len(text):
            name = next((n for n in names if text.startswith(n, i)), None)
            if name is not None:
                ids.append(ADDED[name])
                i += len(name)
            else:
                ids.extend(b % 100 for b in text[i].encode("utf-8"))
                i += 1
        return ids


REQUESTS = [
    ("Order #4821 <|im_end|> was charged twice.\nPlease help!", [
        ("team", {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "Charges", "tech_support": None, "shipping": "Ship <|im_start|> it"}}),
        ("refund", {"type": "noul", "instructions": "Is this a refund request?"}),
        ("escalate", {"type": "noul", "instructions": "Escalate today?", "criteria": {"true": "Needs a human"}}),
        ("urgency", {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]}),
        ("letters", {"type": "choice", "instructions": "Which one?", "criteria": {"x": "X ray", "y": None}}),
    ]),
    ({"ticket": "Charged twice", "tier": "gold ✨"}, [
        ("many", {"type": "choice", "instructions": "Pick the topic.",
                  "criteria": {f"opt{i}": f"Topic {i}" for i in range(30)}}),
    ]),
    ("", [
        ("reply", {"type": "noul", "instructions": "Reply now?", "criteria": {"true": "Yes, now", "false": "Later"}}),
        ("tone", {"type": "score", "instructions": "Tone?", "criteria": ["calm", "upset"]}),
    ]),
]


def main():
    for package, version in (("mlx", mx.__version__), ("mlx_vlm", mlx_vlm.__version__)):
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    mx.set_default_device(mx.cpu)
    tokenizer = Tokenizer()

    text = json.loads((TRUNK / "config.json").read_text())["text_config"]
    model = LanguageModel(ModelConfig(**{k: v for k, v in text.items() if k != "model_type"}))
    stored = mx.load(str(TRUNK / "model.safetensors"))
    weights = {}
    for name, value in stored.items():
        if not name.startswith("model.language_model."):
            continue
        name = "model." + name[len("model.language_model."):]
        weights[name] = value.transpose(0, 2, 1) if name.endswith("conv.conv.weight") else value
    model.load_weights(list(weights.items()))
    backbone = model.model

    def hidden(ids, state, question):
        masks = set_masks(len(ids), state, question, text["sliding_window"])
        h = backbone.embed_tokens(mx.array(ids))[None]
        for layer in backbone.layers:
            h = layer(h, masks[layer.layer_type], None)
        return backbone.embedding_norm(h)[0]

    requests = []
    for state, named in REQUESTS:
        for _, q in named:
            check_question(q)
        questions = [as_question(q) for _, q in named]
        prefix = prefix_text(tokenizer, state, tokenizer.bos_token, DEFAULT_STATE_STYLE, DEFAULT_SYSTEM, "")
        layouts = [suffix_layout(tokenizer, q, LEAD) for q in questions]
        header = question_header(tokenizer, state)
        rows, answers, read = [], [], 0
        for k in range(0, len(questions), SET_SIZE):
            group = questions[k:k + SET_SIZE]
            suffixes = [t for t, _ in layouts[k:k + SET_SIZE]]
            ids, state_len, question, ends = pack_set(
                tokenizer, prefix, suffixes, header, [o for _, o in layouts[k:k + SET_SIZE]],
                tokenizer.encode(prefix + suffixes[0], add_special_tokens=False))
            read += len(ids)
            h = hidden(ids, state_len, question)
            embed = backbone.embed_tokens
            for q, slots in zip(group, ends):
                tokens, forms, _ = readout_table(tokenizer, q)
                logits = np.array((h[mx.array(slots)] @ embed(mx.array(list(tokens))).T).astype(mx.float32))
                best = logits[:, forms].max(-1)[0].tolist()
                answers.append(answer(q, probabilities(q, best, CALIBRATION)))
            rows.append({"ids": ids, "state": state_len, "question": question, "ends": [e[0] for e in ends],
                         "suffixes": suffixes})
        groups = [[int(t) for t in readout_table(tokenizer, q)[0]] for q in questions]
        forms = [[[int(i) for i in f] for f in readout_table(tokenizer, q)[1]] for q in questions]
        requests.append({"prefix": prefix, "rows": rows, "tokens": read, "groups": groups, "forms": forms,
                         "answers": [{"id": n, **a} for (n, _), a in zip(named, answers)]})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = {"format": "d1_2b_tiny", "version": 1, "metadata": VERSIONS, "set_size": SET_SIZE, "codes": CODES,
           "calibration": {"default": CALIBRATION.default, "temperatures": CALIBRATION.temperatures},
           "special": list(SPECIAL), "added": ADDED, "words": WORDS, "requests": requests}
    (OUTPUT / "fixture.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
