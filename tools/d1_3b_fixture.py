#!/usr/bin/env python3
"""Write testdata/d1_3b_tiny/fixture.json: d1-3B prompt goldens and tiny answers.

d1-3B (LiquidAI/d1-3B, LFM Open License v1.0) writes a ChatML prompt per state,
puts each question after it, and reads the answer off the next-token logits:
each option scores its best token form, softmaxed over the options. The prompt,
alias, readout and answer code below is the reference's own (`prompt.py` and
`api.py` at the published defaults: no system turn, `json_only` states, `desc`
options, no lead, no calibration; read 2026-11-03), lightly trimmed.

The answers run the tiny causal LFM2 of testdata/lfm2_tiny through
transformers. Its stand-in tokenizer maps a few whole strings to their own ids
(yes/no forms and some spaced codes) and every other text to one token per
UTF-8 byte (`byte % 100`), with BOS id 1; the Swift test mirrors it.

    .venv/bin/pip install "torch==2.15.0" "transformers==5.22.1" safetensors
    .venv/bin/python tools/d1_3b_fixture.py
"""

import json
import math
import sys
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file
from transformers import Lfm2Config, Lfm2ForCausalLM

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "d1_3b_tiny"
TRUNK = ROOT / "testdata" / "lfm2_tiny"
VERSIONS = {"torch": "2.15.0", "transformers": "5.22.1"}
BOS = 1
TABLE = {"yes": 100, "Yes": 101, "YES": 102, "no": 103, "No": 104, "NO": 105,
         " A": 106, " B": 107, " C": 108, " x": 109}


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [TABLE[text]] if text in TABLE else [b % 100 for b in text.encode("utf-8")]


# --- prompt.py ------------------------------------------------------------------
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")


def option_codes(labels):
    labs = [str(x).strip() for x in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return [chr(ord("A") + i) for i in range(len(labs))]
    return [f"{i:02d}" for i in range(len(labs))]


_FALLBACK_POOL = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    + [chr(a) + chr(b) for a in range(ord("A"), ord("Z") + 1) for b in range(ord("A"), ord("Z") + 1)]
)


def aliases(tokenizer, labels):
    used, out = set(), []

    def take(raw):
        enc = tokenizer.encode(raw, add_special_tokens=False)
        if len(enc) != 1 or enc[0] in used:
            return False
        out.append((raw, enc[0]))
        used.add(enc[0])
        return True

    for code in option_codes(labels):
        if take(code):
            continue
        if not any(take(raw) for raw in _FALLBACK_POOL):
            raise RuntimeError("no alias")
    return out


def _ids(tokenizer, texts):
    out, seen = [], set()
    for t in texts:
        enc = tokenizer.encode(t, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in seen:
            out.append(enc[0])
            seen.add(enc[0])
    return out


def readout_ids(tokenizer, q):
    if q["type"] == "noul":
        return [_ids(tokenizer, YES_FORMS), _ids(tokenizer, NO_FORMS)]
    if q["type"] == "score":
        return [_ids(tokenizer, [str(i)]) for i in range(len(q["criteria"]))]
    groups = []
    for code, tid in aliases(tokenizer, list(q["criteria"].keys())):
        extra = _ids(tokenizer, [f" {code}"])
        groups.append([tid] + [i for i in extra if i != tid])
    return groups


def readout(tokenizer, q, logz):
    scores = [max(float(logz[i]) for i in g) for g in readout_ids(tokenizer, q)]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    return [e / sum(exps) for e in exps]


def state_block(state):
    if isinstance(state, str):
        return f"{state}\n\n"
    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


def question_block(tokenizer, q):
    if q["type"] == "choice":
        labels = list(q["criteria"].keys())
        codes = aliases(tokenizer, labels)
        lines = "\n".join(f"{codes[i][0]} {q['criteria'][lab] or lab.replace('_', ' ')}" for i, lab in enumerate(labels))
        return f"{q['instructions']}\n\nOptions:\n{lines}\n\nReply with the option code only."
    if q["type"] == "noul":
        extra = ""
        if q.get("criteria"):
            extra = f"\nYes: {q['criteria'].get('true')}\nNo: {q['criteria'].get('false')}"
        return f"{q['instructions']}{extra}\n\nReply with yes or no only."
    legend = "\n".join(f"{i} {name}" for i, name in enumerate(q["criteria"]))
    return f"{q['instructions']}\n\n{legend}\n\nReply with a single digit 0-{len(q['criteria']) - 1} only."


def prefix_text(state):
    return f"{IM_START}user\n{state_block(state)}\nQUESTION:\n"


def suffix_text(tokenizer, q):
    return f"{question_block(tokenizer, q)}{IM_END}\n{IM_START}assistant\n"


# --- api.py ---------------------------------------------------------------------
def answer(q, probs):
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs[0]}
    best = max(range(len(probs)), key=probs.__getitem__)
    if q["type"] == "choice":
        names = list(q["criteria"])
        return {"type": "choice", "choice": names[best], "confidence": probs[best],
                "probabilities": dict(zip(names, probs))}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "confidence": probs[best],
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "legend": {str(i): str(text) for i, text in enumerate(q["criteria"])}}


# --- requests ---------------------------------------------------------------------
REQUESTS = [
    ("Order #4821 was charged twice.\nPlease help!", [
        ("team", {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "Charges", "tech_support": None, "shipping": ""}}),
        ("refund", {"type": "noul", "instructions": "Is this a refund request?"}),
        ("escalate", {"type": "noul", "instructions": "Escalate today?", "criteria": {"true": "Needs a human"}}),
        ("urgency", {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]}),
        ("letters", {"type": "choice", "instructions": "Which one?", "criteria": {"x": "X ray", "y": None}}),
    ]),
    ({"ticket": "Charged twice", "tier": "gold ✨"}, [
        ("many", {"type": "choice", "instructions": "Pick the topic.",
                  "criteria": {f"opt{i}": f"Topic {i}" for i in range(28)}}),
    ]),
    ("", [
        ("tone", {"type": "score", "instructions": "Tone?",
                  "criteria": [{"what": "calm", "examples": ["fine, thanks"]}, {"what": "upset", "examples": []}]}),
        ("reply", {"type": "noul", "instructions": "Reply now?", "criteria": {"true": "Yes, now", "false": "Later"}}),
    ]),
]


def main():
    for package, version in (("torch", torch.__version__.split("+")[0]), ("transformers", transformers.__version__)):
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    tokenizer = Tokenizer()
    config = Lfm2Config(**json.loads((TRUNK / "config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    model = Lfm2ForCausalLM(config).eval()
    weights = {n.replace("model.language_model.", "model."): v for n, v in load_file(TRUNK / "model.safetensors").items()
               if n.startswith("model.language_model.")}
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    model.load_state_dict(weights, strict=True)

    requests = []
    for state, questions in REQUESTS:
        prefix = prefix_text(state)
        entries = []
        trunk = [BOS] + tokenizer.encode(prefix)
        for name, q in questions:
            suffix = suffix_text(tokenizer, q)
            if len(questions) == 1:
                ids, branch = [BOS] + tokenizer.encode(prefix + suffix), None
            else:
                branch = tokenizer.encode(suffix)
                ids = trunk + branch
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids])).logits[0, -1].float()
            logz = logits - torch.logsumexp(logits, -1)
            probs = readout(tokenizer, q, logz)
            codes = [c for c, _ in aliases(tokenizer, list(q["criteria"]))] if q["type"] == "choice" else []
            entries.append({"id": name, "question": q, "suffix": suffix, "codes": codes,
                            "groups": readout_ids(tokenizer, q), "tokens": len(branch if branch is not None else ids),
                            "answer": answer(q, probs)})
        requests.append({"state": state, "prefix": prefix,
                         "prefix_tokens": 0 if len(questions) == 1 else len(trunk), "questions": entries})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = {"format": "d1_3b_tiny", "version": 1, "metadata": VERSIONS, "bos": BOS, "table": TABLE, "requests": requests}
    (OUTPUT / "fixture.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
