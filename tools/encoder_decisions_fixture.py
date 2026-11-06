#!/usr/bin/env python3
"""Write testdata/encoder_decisions/: prompt goldens and tiny numeric references
for the encoder decision models SystemOneEncoders serves.

- DecisionTune 1.0 (decision-tune/decisiontune-1.0, engine.py, Apache-2.0)
- Laya (convaiinnovations/laya, rl_common.py and rl_agent_api.py, Apache-2.0)
- Julia-1 (SupersonicLabs/Julia-1, julia/data.py and julia/typed.py, Apache-2.0)
- tasksource-decider-nano (tasksource/tasksource-decider-nano, decider.py, Apache-2.0)

The rendering functions below are those references' own code, lightly trimmed
(Hub files read 2026-11-03), so the goldens are what each model's reference
packs. Prompts use the ModernBERT tokenizer checked in under
Tests/SystemOneVonTests/Resources/Tokenizer. The numeric references run each
reference head on the tiny ModernBERT of testdata/modernbert_tiny/large-rope
with random head weights, on token ids that fit its 64-token vocabulary.

    .venv/bin/pip install "torch==2.15.0" "transformers==5.22.1" safetensors
    .venv/bin/python tools/encoder_decisions_fixture.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from safetensors.torch import load_file, save_file
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "encoder_decisions"
TOKENIZER = ROOT / "Tests" / "SystemOneVonTests" / "Resources" / "Tokenizer"
ENCODER = ROOT / "testdata" / "modernbert_tiny" / "large-rope"
VERSIONS = {"torch": "2.15.0", "transformers": "5.22.1"}
SEED = 20261103


def compact(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


# --- DecisionTune engine.py ---------------------------------------------------
DT_SPECIAL = ("[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]")


def dt_clean(s):
    for t in DT_SPECIAL:
        s = s.replace(t, " ")
    return s


def dt_state_text(s):
    if s is None:
        return ""
    if isinstance(s, str):
        return s
    if isinstance(s, dict):
        return " | ".join(f"{k}: {v if isinstance(v, str) else compact(v)}" for k, v in s.items())
    return compact(s)


def dt_option_text(key, desc):
    if isinstance(desc, str) and desc.strip():
        return desc
    if desc is not None and desc != "":
        return compact(desc)
    return str(key).replace("_", " ")


def dt_encode(tok, state, instructions, opts):
    t = lambda s: tok(dt_clean(s), add_special_tokens=False)["input_ids"]
    CLS, SEP, MARK = (tok.convert_tokens_to_ids(x) for x in ("[CLS]", "[SEP]", "[MASK]"))
    ids, pos = [CLS] + t(compact(instructions)) + [SEP], []
    for o in opts:
        pos.append(len(ids))
        ids += [MARK] + t(o)
    ids += [SEP] + t(dt_state_text(state)) + [SEP]
    return ids, pos


def dt_options(q):
    if q["type"] == "noul":
        return ["yes", "no"]
    if q["type"] == "score":
        return list(q["criteria"])
    return [dt_option_text(k, d) for k, d in q["criteria"].items()]


# --- Laya rl_common.py --------------------------------------------------------
def laya_serialize_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def laya_render_options(q):
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def laya_build_sequence(tok, state, q, max_len, head_max_len):
    mask_tok = tok.mask_token
    opts = laya_render_options(q)
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in range(len(opts)):
        opt_ids.append([tok.mask_token_id] + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(laya_serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


# --- Julia-1 julia/data.py and julia/typed.py --------------------------------
def julia_options(q):
    kind, criteria = q["type"], q.get("criteria")
    if kind == "choice":
        return list(criteria.values())
    if kind == "score":
        return criteria
    return ["false", "true"] if criteria is None else [criteria["false"], criteria["true"]]


def julia_sequence(tok, state, kind, question, options, max_length, head_length):
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    clean = lambda text: text.replace(tok.mask_token, " ")
    encode = lambda text: tok(text, add_special_tokens=False)["input_ids"]
    head = encode(f"{kind} question: {clean(question)}")
    option_ids = [encode(" " + clean(x)) for x in options]
    options = [[tok.mask_token_id] + x[:48] for x in option_ids]
    budget = head_length - sum(map(len, options))
    if budget < 16:
        per_option = max(4, (head_length - 16) // len(options))
        options = [x[:per_option] for x in options]
        budget = head_length - sum(map(len, options))
    ids = [tok.cls_token_id] + head[:max(8, budget)] + [tok.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tok.sep_token_id)
    state_ids = encode(clean(state))
    room = max_length - len(ids) - 1
    return ids + state_ids[:room] + [tok.sep_token_id], markers


# --- tasksource decider.py ("shared" layout, "opt" pooling, "cached" attention)
def ts_state_text(state):
    return compact(state) if state not in ("", None, {}, []) else "(empty)"


def ts_format_option(key, desc):
    desc_str = compact(desc).strip() if desc is not None else ""
    key_str = str(key).strip()
    if not desc_str:
        return key_str
    is_dummy_key = (key_str.lower().startswith(("option_", "k_", "choice_", "item_"))
                    or (len(key_str) == 1 and key_str.isalpha())
                    or (key_str.startswith("(") and key_str.endswith(")") and len(key_str) <= 4))
    if is_dummy_key:
        return desc_str
    if desc_str.lower() == key_str.lower():
        return key_str
    return f"{key_str}: {desc_str}"


def ts_options(q):
    crit = q.get("criteria", {}) or {}
    if q.get("type", "choice") == "choice":
        return [ts_format_option(k, crit[k]) for k in crit]
    descs = (crit.get("false", crit.get("False", "False")), crit.get("true", crit.get("True", "True")))
    return [ts_format_option("false", descs[0]), ts_format_option("true", descs[1])]


def ts_row(tok, state, questions):
    """One row holding every question: ids, positions, and per-token (kind, question, option)."""
    p = tok(ts_state_text(state))["input_ids"]
    ids, pos, kind, qix, oix = list(p), list(range(len(p))), [0] * len(p), [-1] * len(p), [-1] * len(p)
    oid = 0
    for b, (instructions, opts) in enumerate(questions):
        qi = tok(f"Question: {instructions}\nOption:", add_special_tokens=False)["input_ids"]
        ids += qi
        pos += list(range(len(p), len(p) + len(qi)))
        kind += [1] * len(qi); qix += [b] * len(qi); oix += [-1] * len(qi)
        for o in opts:
            x = tok(" " + o, add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
            ids += x
            pos += list(range(len(p) + len(qi), len(p) + len(qi) + len(x)))
            kind += [2] * len(x); qix += [b] * len(x); oix += [oid] * len(x)
            oid += 1
    return {"ids": ids, "positions": pos, "kind": kind, "question": qix, "option": oix}


def ts_allow(kind, qix, oix):
    kind, qix, oix = (torch.tensor(x) for x in (kind, qix, oix))
    ki, kj = kind[:, None], kind[None, :]
    allow = (kj == 0) & (ki >= 0)
    allow |= (kj == 1) & (ki >= 1) & (qix[:, None] == qix[None, :])
    allow |= (kj == 2) & (ki == 2) & (oix[:, None] == oix[None, :])
    return allow | torch.eye(len(kind), dtype=torch.bool)


# --- requests ------------------------------------------------------------------
STATE_TEXT = "I was charged twice for order #4821 and the [MASK] second charge is still pending."
STATE_FIELDS = {"ticket": "Charged twice for one order", "tier": "gold"}
QUESTIONS = [
    {"type": "choice", "instructions": "Which team should handle this?",
     "criteria": {"billing": "Charges, refunds, invoices", "tech_support": "", "a": "App faults", "shipping": "Shipping"}},
    {"type": "noul", "instructions": "Is the customer asking for a refund?", "criteria": None},
    {"type": "noul", "instructions": "Escalate today?", "criteria": {"false": "Routine", "true": "Needs a human today"}},
    {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high"]},
]


def prompts(tok):
    out = {"decisiontune": [], "laya": [], "julia": [], "tasksource": []}
    for state in (STATE_TEXT, STATE_FIELDS):
        for q in QUESTIONS:
            ids, markers = dt_encode(tok, state, q["instructions"], dt_options(q))
            out["decisiontune"].append({"ids": ids, "markers": markers})
            internal = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
            ids, markers = laya_build_sequence(tok, state, internal, 512, 192)
            out["laya"].append({"ids": ids, "markers": markers})
            if q["type"] != "choice" or all(q["criteria"].values()):
                ids, markers = julia_sequence(tok, state, q["type"], q["instructions"], julia_options(q), 8192, 512)
                out["julia"].append({"ids": ids, "markers": markers})
            else:
                out["julia"].append(None)  # Julia needs a description for every choice option
        typed = [q for q in QUESTIONS if q["type"] != "score"]
        out["tasksource"].append(ts_row(tok, state, [(q["instructions"], ts_options(q)) for q in typed]))
    # Laya's option budget: twelve long options shrink evenly.
    many = {"t": "choice", "ins": "Pick the closest topic.",
            "crit": {f"topic_{i}": "a very long description " * 6 for i in range(12)}}
    ids, markers = laya_build_sequence(tok, STATE_TEXT, many, 512, 192)
    out["laya_budget"] = {"ids": ids, "markers": markers}
    return out


# --- tiny numeric references -----------------------------------------------------
class LayaHead(nn.Module):
    """rl_common.DecisionModel without its encoder and act head."""

    def __init__(self, d, layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.1, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, attention_mask, marker_pos, qtype):
        h = h + self.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos[:, :, None].expand(-1, -1, h.size(-1))
        return self.scorer(torch.gather(h, 1, idx)).squeeze(-1)


def randomize(module, scale=0.3):
    with torch.no_grad():
        for name, value in module.state_dict().items():
            if "norm" in name or name.endswith("scorer.0.weight"):
                value.copy_(1 + 0.1 * torch.randn_like(value))
            else:
                value.normal_(0, scale)


def tiny(out):
    config = ModernBertConfig.from_pretrained(ENCODER)
    config._attn_implementation = "eager"
    encoder = ModernBertModel(config).eval()
    encoder.load_state_dict(load_file(ENCODER / "model.safetensors"), strict=True)
    d = config.hidden_size
    rows = [  # ids under the tiny vocabulary; markers point at [MASK]-like slots
        {"ids": [1, 5, 9, 2, 4, 11, 12, 4, 13, 2, 20, 21, 22, 2], "markers": [4, 7], "type": 0},
        {"ids": [1, 6, 7, 8, 2, 4, 30, 4, 31, 4, 32, 2, 40, 2], "markers": [5, 7, 9], "type": 1},
        {"ids": [1, 50, 2, 4, 51, 4, 52, 2, 60, 61, 2], "markers": [3, 5], "type": 2},
    ]
    width = max(len(r["ids"]) for r in rows)
    ids = torch.tensor([r["ids"] + [0] * (width - len(r["ids"])) for r in rows])
    mask = (torch.arange(width)[None] < torch.tensor([len(r["ids"]) for r in rows])[:, None]).long()
    with torch.no_grad():
        hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state

    tune_folder = OUTPUT / "tiny" / "decisiontune"
    tune_folder.mkdir(parents=True, exist_ok=True)
    head = nn.Linear(d, 1)
    randomize(head)
    save_file({"head.weight": head.weight.detach().contiguous(), "head.bias": head.bias.detach().contiguous()},
              tune_folder / "head.safetensors")
    with torch.no_grad():
        out["tiny"]["decisiontune"] = [head(hidden[i, r["markers"]]).squeeze(-1).tolist() for i, r in enumerate(rows)]

    laya_folder = OUTPUT / "tiny" / "laya"
    laya_folder.mkdir(parents=True, exist_ok=True)
    laya = LayaHead(d, 2).eval()
    randomize(laya)
    k = max(len(r["markers"]) for r in rows)
    positions = torch.tensor([r["markers"] + [0] * (k - len(r["markers"])) for r in rows])
    with torch.no_grad():
        z = laya(hidden, mask, positions, torch.tensor([r["type"] for r in rows]))
    out["tiny"]["laya"] = [z[i, : len(r["markers"])].tolist() for i, r in enumerate(rows)]
    weights = {f"encoder.{n}": v for n, v in load_file(ENCODER / "model.safetensors").items()}
    weights.update({n: v.detach().contiguous() for n, v in laya.state_dict().items()})
    save_file(weights, laya_folder / "model.safetensors")

    ts_folder = OUTPUT / "tiny" / "tasksource"
    ts_folder.mkdir(parents=True, exist_ok=True)
    scorer = nn.Linear(d, 1)
    randomize(scorer)
    # Prefix of 5 tokens, two questions of 3 tokens, options of 2-3 tokens each (ending in [SEP]).
    tree = {"ids": [1, 7, 8, 9, 2, 10, 11, 12, 20, 21, 2, 22, 2, 13, 14, 15, 30, 31, 32, 2, 33, 2, 34, 35, 2],
            "kind": [0] * 5 + [1] * 3 + [2] * 3 + [2] * 2 + [1] * 3 + [2] * 4 + [2] * 2 + [2] * 3,
            "question": [-1] * 5 + [0] * 3 + [0] * 5 + [1] * 3 + [1] * 9,
            "option": [-1] * 5 + [-1] * 3 + [0] * 3 + [1] * 2 + [-1] * 3 + [2] * 4 + [3] * 2 + [4] * 3}
    positions = []
    for i, (kind, q, o) in enumerate(zip(tree["kind"], tree["question"], tree["option"])):
        if kind == 0:
            positions.append(i)
        elif kind == 1:
            positions.append(5 + sum(1 for j in range(i) if tree["kind"][j] == 1 and tree["question"][j] == q))
        else:
            positions.append(8 + sum(1 for j in range(i) if tree["option"][j] == o))
    tree["positions"] = positions
    allow = ts_allow(tree["kind"], tree["question"], tree["option"])
    pos = torch.tensor(positions)
    window = (pos[:, None] - pos[None, :]).abs() <= config.sliding_window
    to_mask = lambda m: torch.where(m, torch.tensor(0.0), torch.tensor(float("-inf")))[None, None]
    with torch.no_grad():
        h = encoder(input_ids=torch.tensor([tree["ids"]]), position_ids=pos[None],
                    attention_mask={"full_attention": to_mask(allow), "sliding_attention": to_mask(allow & window)}
                    ).last_hidden_state[0]
        option = torch.tensor(tree["option"])
        logits = [scorer(h[option == o].mean(0)).item() for o in range(5)]
    out["tiny"]["tasksource"] = {"row": tree, "logits": logits}
    weights = {f"backbone.{n}": v for n, v in load_file(ENCODER / "model.safetensors").items()}
    weights.update({"scorer.weight": scorer.weight.detach().contiguous(), "scorer.bias": scorer.bias.detach().contiguous()})
    save_file(weights, ts_folder / "model.safetensors")
    out["tiny"]["rows"] = rows


def main():
    for package, version in (("torch", torch.__version__.split("+")[0]), ("transformers", transformers.__version__)):
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    torch.manual_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tok = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER / "tokenizer.json"), cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]", pad_token="[PAD]", unk_token="[UNK]")
    out = {"format": "encoder_decisions", "version": 1, "metadata": {**VERSIONS, "seed": SEED},
           "state_text": STATE_TEXT, "state_fields": STATE_FIELDS, "questions": QUESTIONS,
           "prompts": prompts(tok), "tiny": {}}
    tiny(out)
    (OUTPUT / "fixture.json").write_text(json.dumps(out) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
