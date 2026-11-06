#!/usr/bin/env python3
"""Generate and verify autotrust/JEV-9B golden vectors.

Two fixtures, both computed on CPU float32 by Hugging Face transformers'
own Qwen3.5 classes plus peft, which are the reference the Swift port
translates:

* ``tiny``: a seeded random Qwen3.5 text model, small enough to commit, saved
  in JEV-9B's exact on-disk layout (two bfloat16 shards behind an index, an
  fp32 peft LoRA adapter, an fp32 24-slot head, judge_config.json,
  calibration.json). Records every layer's hidden states at probe rows, the
  merged LoRA weights, the head logits, and a gated-delta-rule vector. Needs
  no download, so it runs anywhere.
* ``full``: the pinned JEV-9B snapshot. Records tokenizer ids, ``bare-v1``
  prompts, a per-layer trunk probe, and calibrated decision probabilities.

The reference is defined exactly so Swift can match it bit for bit: the base
model loads in bfloat16, the LoRA merges into it as peft's
``merge_and_unload`` does (``bf16(f32(W) + scale * B @ A)``), ``A_log`` and the
gated RMSNorm weights keep their stored float32 values, and everything then
computes in float32.

    .venv-jev/bin/python tools/jev_golden_capture.py tiny            # write tiny model + fixture
    .venv-jev/bin/python tools/jev_golden_capture.py tiny --check    # verify them
    .venv-jev/bin/python tools/jev_golden_capture.py full [--check] [--model-folder weights/JEV-9B]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import d1_golden_capture as golden

SCHEMA_VERSION = 1
HF_REPO = "autotrust/JEV-9B"
HF_REVISION = "7ac669b8fb476ba926491e3d1c39f9330d68dbed"
# Hub LFS hashes; the small JSON files are pinned by content after download.
ARTIFACT_HASHES = {
    "model-00001-of-00005.safetensors": "fdce203bd6767c95690c4695699e9e796fa53caeb06c7f6d43a014f7c513b32c",
    "model-00002-of-00005.safetensors": "64003b88d6e6871f2315e39356188082063ff879994434d5ce8c209b3c736a01",
    "model-00003-of-00005.safetensors": "ef7cf4227fb3f391bb2bcd5d79a3c10cd90d823dbf964f41a0aa34591377ae82",
    "model-00004-of-00005.safetensors": "2265f2fcf900572fa0b7c645a9a64f662d4123daeeeda23e917411e907d66272",
    "model-00005-of-00005.safetensors": "dd63614f1dc80dce2d83be3f3e69af1f8a0ebc8add9b8e0abf3910f563e44344",
    "adapter/adapter_model.safetensors": "796386ace560ec5624bbfe83f5c505121e5c9f9654200a847f032cb333cd1f71",
    "head.safetensors": "d960a6f5989402c6d08d74d8d435a21a031cc7878f3c4ac274a126c900f627e7",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
}
EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.14.0",
    "transformers": "5.17.0",
    "tokenizers": "0.23.2",
    "peft": "0.21.0",
    "safetensors": "0.8.0",
}
SNAPSHOT_PATTERNS = [
    "*.json", "*.jinja", "model-*.safetensors", "head.safetensors", "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
]

REPO_ROOT = golden.REPO_ROOT
TINY_DIR = REPO_ROOT / "testdata" / "jev_tiny_v1"
TINY_FIXTURE = REPO_ROOT / "testdata" / "jev_tiny_golden_v1.json"
FULL_FIXTURE = REPO_ROOT / "testdata" / "jev_golden_v1.json"

LORA_TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
SLOTS = {
    "num_slots": 24,
    "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
    "verbalizers": ["false", "true", "0", "1", "2", "3", "4", "5"] + list("ABCDEFGHIJKLMNOP"),
    "template_version": "bare-v1",
}
LETTERS = "ABCDEFGHIJKLMNOP"


# ---------------------------------------------------------------- shared reference pieces

def runtime_check() -> None:
    from importlib.metadata import version

    for package, expected in EXPECTED_RUNTIME_VERSIONS.items():
        actual = version(package)
        if actual != expected:
            raise SystemExit(f"{package} {actual} != pinned {expected}")


def bare_v1(kind: str, state: str, question: str, options: list[str]) -> str:
    """The ``bare-v1`` template, exactly as the JEV-9B model card renders it."""
    lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
    return f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


def floats(tensor) -> list[float]:
    return [float(x) for x in tensor.detach().to("cpu").float().reshape(-1).tolist()]


def rows(tensor) -> list[list[float]]:
    return [floats(row) for row in tensor]


def load_reference(folder: Path, torch):
    """The exact reference: bf16 base, peft merge, fp32 SSM parameters, fp32 compute."""
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16, attn_implementation="eager")
    stored = {}
    for name in base.state_dict():
        if name.endswith("linear_attn.A_log") or name.endswith("linear_attn.norm.weight"):
            stored[name] = None
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in sorted(set(index.values())):
        for name, value in load_file(folder / shard).items():
            if name in stored:
                stored[name] = value
    model = PeftModel.from_pretrained(base, str(folder), subfolder="adapter").merge_and_unload()
    model = model.float().eval()
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in stored.items():
            assert value is not None and value.dtype == torch.float32, name
            params[name].copy_(value)
    return model


def head_logits(folder: Path, hidden, torch) -> list[float]:
    from safetensors.torch import load_file

    head = load_file(folder / "head.safetensors")
    return floats(head["proj.weight"].float() @ hidden.float() + head["proj.bias"].float())


def distribution(logits: list[float], kind: str, count: int, temperatures: dict) -> list[float]:
    lo, _ = SLOTS["ranges"][kind]
    z = [x / temperatures[kind] for x in logits[lo:lo + count]]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    return [x / sum(e) for x in e]


class LayerRecorder:
    """Forward hooks capturing the embedding, every decoder layer, and the final norm."""

    def __init__(self, model):
        self.outputs: dict[str, Any] = {}
        inner = model.model
        self.handles = [inner.embed_tokens.register_forward_hook(self.hook("embed"))]
        for i, layer in enumerate(inner.layers):
            self.handles.append(layer.register_forward_hook(self.hook(f"layer.{i}")))
        self.handles.append(inner.norm.register_forward_hook(self.hook("norm")))

    def hook(self, name):
        def record(_module, _inputs, output):
            self.outputs[name] = (output[0] if isinstance(output, tuple) else output).detach()
        return record

    def close(self):
        for handle in self.handles:
            handle.remove()


# ---------------------------------------------------------------- tiny model

TINY_TEXT_CONFIG = {
    "vocab_size": 96,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "rope_parameters": {
        "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
        "mrope_interleaved": True, "mrope_section": [2, 1, 1],
    },
    "partial_rotary_factor": 0.25,
    "attention_bias": False,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "full_attention_interval": 4,
    "attn_output_gate": True,
    "max_position_embeddings": 4096,
    "eos_token_id": 95,
    "pad_token_id": None,
    "bos_token_id": None,
}
TINY_LORA = {"r": 4, "alpha": 8}
TINY_TEMPERATURES = {"noul": 1.25, "choice": 0.8, "score": 1.1}
TINY_SEED = 20261105
TINY_LENGTHS = [1, 5, 64, 70]
TINY_CHOICE_OPTIONS = 5
TINY_MERGE_PROBES = [
    "model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.layers.0.linear_attn.out_proj.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "model.layers.3.mlp.down_proj.weight",
]


def probe_rows(length: int) -> list[int]:
    return sorted({0, 1, length // 2, length - 2, length - 1} & set(range(length)))


def write_tiny_model(folder: Path, torch) -> None:
    """Seeded random weights in JEV-9B's layout: bf16 shards, fp32 SSM params, fp32 adapter and head."""
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    generator = torch.Generator().manual_seed(TINY_SEED)

    def normal(shape, std):
        return torch.randn(shape, generator=generator) * std

    def uniform(shape, lo, hi):
        return torch.rand(shape, generator=generator) * (hi - lo) + lo

    config = Qwen3_5TextConfig(**TINY_TEXT_CONFIG)
    config.architectures = ["Qwen3_5ForCausalLM"]
    model = Qwen3_5ForCausalLM(config)
    state = {}
    for name, param in model.named_parameters():
        if name.endswith("A_log"):
            value = uniform(param.shape, 1.0, 8.0).log()
        elif name.endswith("dt_bias"):
            value = uniform(param.shape, -1.0, 1.0)
        elif name.endswith("linear_attn.norm.weight"):
            value = uniform(param.shape, 0.8, 1.2)
        elif "norm" in name:
            value = uniform(param.shape, -0.2, 0.2)  # zero-centred: the layer multiplies by 1 + w
        elif name.endswith("conv1d.weight"):
            value = normal(param.shape, 0.3)
        elif name.endswith("embed_tokens.weight"):
            value = normal(param.shape, 0.5)
        else:
            value = normal(param.shape, 0.1)
        keep_fp32 = name.endswith("A_log") or name.endswith("linear_attn.norm.weight")
        state[name] = value.to(torch.float32 if keep_fp32 else torch.bfloat16).contiguous()

    folder.mkdir(parents=True, exist_ok=True)
    config.dtype = "bfloat16"
    config.save_pretrained(folder)
    first = {k: v for k, v in state.items() if k.startswith("model.embed_tokens") or k.startswith("model.layers.0.")
             or k.startswith("model.layers.1.")}
    second = {k: v for k, v in state.items() if k not in first}
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for shard, tensors in zip(names, [first, second]):
        save_file(dict(sorted(tensors.items())), folder / shard, metadata={"format": "pt"})
    total = sum(v.numel() * v.element_size() for v in state.values())
    weight_map = {k: names[0] if k in first else names[1] for k in sorted(state)}
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")

    # LoRA adapter, saved by peft itself so the format is the real one.
    model.load_state_dict({k: v.float() for k, v in state.items()})
    peft_model = get_peft_model(model, LoraConfig(
        r=TINY_LORA["r"], lora_alpha=TINY_LORA["alpha"], lora_dropout=0.0, target_modules=LORA_TARGETS))
    with torch.no_grad():
        for name, param in sorted(peft_model.named_parameters()):
            if "lora_A" in name or "lora_B" in name:
                param.copy_(normal(param.shape, 0.2))
    peft_model.save_pretrained(folder / "adapter")
    (folder / "adapter" / "README.md").unlink(missing_ok=True)
    # peft keeps target_modules as a set, so its saved order follows the hash seed.
    adapter_config = folder / "adapter" / "adapter_config.json"
    saved = json.loads(adapter_config.read_text())
    saved["target_modules"] = sorted(saved["target_modules"])
    adapter_config.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")

    hidden = TINY_TEXT_CONFIG["hidden_size"]
    save_file({"proj.weight": normal((24, hidden), 0.3).contiguous(), "proj.bias": normal((24,), 0.1).contiguous()},
              folder / "head.safetensors")
    judge = {
        "hidden_size": hidden, "slots": SLOTS, "verbalizer_ids": list(range(64, 88)),
        "kinds": ["noul", "choice", "score"], "model_name": "jev-tiny", "model_version": "0.0.0",
        "lm_head_included": True, "weights_mode": "unmerged", "adapter_subfolder": "adapter",
        "lora": {"r": TINY_LORA["r"], "alpha": TINY_LORA["alpha"], "dropout": 0.0, "target_modules": LORA_TARGETS},
        "softcap": None,
    }
    (folder / "judge_config.json").write_text(json.dumps(judge, indent=2) + "\n")
    (folder / "calibration.json").write_text(json.dumps({"version": 1, "per_kind": TINY_TEMPERATURES}, indent=2) + "\n")


def delta_rule_vector(torch) -> dict:
    """One chunked gated-delta-rule call spanning two chunks, with its recurrent twin."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)

    generator = torch.Generator().manual_seed(TINY_SEED + 1)
    t, heads, dk, dv = 70, 2, 8, 8
    q = torch.randn((1, t, heads, dk), generator=generator)
    k = torch.randn((1, t, heads, dk), generator=generator)
    v = torch.randn((1, t, heads, dv), generator=generator)
    g = -torch.rand((1, t, heads), generator=generator) * 0.5
    beta = torch.rand((1, t, heads), generator=generator)
    chunked, state = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=True,
                                                  use_qk_l2norm_in_kernel=True)
    recurrent, _ = torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    return {
        "shape": [t, heads, dk, dv],
        "q": floats(q), "k": floats(k), "v": floats(v), "g": floats(g), "beta": floats(beta),
        "output": floats(chunked), "final_state": floats(state), "recurrent_output": floats(recurrent),
    }


def tiny_fixture(folder: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    base_tensors = {}
    for shard in sorted(folder.glob("model-*.safetensors")):
        base_tensors.update(load_file(shard))
    adapter = load_file(folder / "adapter" / "adapter_model.safetensors")
    scale = TINY_LORA["alpha"] / TINY_LORA["r"]

    model = load_reference(folder, torch)
    merged = dict(model.named_parameters())
    merge_probes = {}
    for name in TINY_MERGE_PROBES:
        stem = name.removesuffix(".weight")
        lora_a = adapter[f"base_model.model.{stem}.lora_A.weight"]
        lora_b = adapter[f"base_model.model.{stem}.lora_B.weight"]
        expected = (base_tensors[name].float() + scale * (lora_b @ lora_a)).to(torch.bfloat16).float()
        assert torch.equal(merged[name].detach(), expected), f"peft merge differs from bf16(W + s*B@A) for {name}"
        flat = merged[name].detach().reshape(-1)
        indices = sorted({0, 1, 17, flat.numel() // 3, flat.numel() // 2, flat.numel() - 1})
        merge_probes[name] = {
            "indices": indices,
            "base": [float(base_tensors[name].float().reshape(-1)[i]) for i in indices],
            "merged": [float(flat[i]) for i in indices],
        }

    generator = torch.Generator().manual_seed(TINY_SEED + 2)
    cases = []
    for length in TINY_LENGTHS:
        ids = torch.randint(0, TINY_TEXT_CONFIG["vocab_size"], (1, length), generator=generator)
        recorder = LayerRecorder(model)
        with torch.no_grad():
            last = model.model(input_ids=ids).last_hidden_state[0, -1]
        recorder.close()
        probes = probe_rows(length)
        logits = head_logits(folder, last, torch)
        cases.append({
            "ids": [int(i) for i in ids[0]],
            "probe_rows": probes,
            "layers": {name: rows(out[0, probes]) for name, out in recorder.outputs.items()},
            "last_hidden": floats(last),
            "head_logits": logits,
            "probabilities": {
                "noul": distribution(logits, "noul", 2, TINY_TEMPERATURES),
                "score": distribution(logits, "score", 6, TINY_TEMPERATURES),
                "choice": distribution(logits, "choice", TINY_CHOICE_OPTIONS, TINY_TEMPERATURES),
            },
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "lora_scale": scale,
        "temperatures": TINY_TEMPERATURES,
        "choice_options": TINY_CHOICE_OPTIONS,
        "merge_probes": merge_probes,
        "cases": cases,
        "delta_rule": delta_rule_vector(torch),
    }


def run_tiny(check: bool) -> int:
    import torch

    torch.set_num_threads(1)
    if not check:
        if TINY_DIR.exists():
            for path in sorted(TINY_DIR.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
        write_tiny_model(TINY_DIR, torch)
        TINY_FIXTURE.write_bytes(golden.canonical_json_bytes(tiny_fixture(TINY_DIR)))
        size = sum(p.stat().st_size for p in TINY_DIR.rglob("*") if p.is_file())
        print(f"wrote {TINY_DIR.relative_to(REPO_ROOT)} ({size} bytes) and {TINY_FIXTURE.relative_to(REPO_ROOT)} "
              f"({TINY_FIXTURE.stat().st_size} bytes)")
        return 0

    errors = []
    with tempfile.TemporaryDirectory() as scratch:
        regenerated = Path(scratch) / "jev_tiny_v1"
        write_tiny_model(regenerated, torch)
        committed = {p.relative_to(TINY_DIR) for p in TINY_DIR.rglob("*") if p.is_file()}
        fresh = {p.relative_to(regenerated) for p in regenerated.rglob("*") if p.is_file()}
        if committed != fresh:
            errors.append(f"tiny model files differ: {sorted(map(str, committed ^ fresh))}")
        for name in sorted(committed & fresh):
            if golden._sha256(TINY_DIR / name) != golden._sha256(regenerated / name):
                errors.append(f"tiny model file {name} differs")
    generated = json.loads(golden.canonical_json_bytes(tiny_fixture(TINY_DIR)))
    errors += golden.compare_json(generated, json.loads(TINY_FIXTURE.read_text()))
    for error in errors[:50]:
        print(error, file=sys.stderr)
    print("tiny fixture matches" if not errors else f"{len(errors)} mismatches")
    return 1 if errors else 0


# ---------------------------------------------------------------- full model

TOKENIZER_TEXTS = [
    "",
    "hello",
    " hello world",
    "[kind] noul\n[state] x\n[question] y\n[options]\nfalse\ntrue\n[decision]:",
    "A) issue_warning\nB) renegotiate",
    "  leading and trailing  ",
    "tabs\tand\nnewlines\r\n",
    "Ünïcödé façade — naïve café",
    "今天中秋节，还要加班的有吗？来报道下",
    "emoji 👨‍👩‍👧‍👦 and 🇯🇵",
    "numbers 1234567890 3.14159 -42",
    '{"status": "degraded", "region": "Tokyo"}',
    "<|im_start|> literal specials <|endoftext|>",
    "مرحبا بالعالم",
    "a" * 300,
]
FULL_CASES = [
    {"id": "refund", "kind": "noul", "state": "Customer says the parcel arrived damaged and wants their money back.",
     "question": "Is the customer asking for a refund?"},
    {"id": "fit", "kind": "noul", "state": "I'm not happy with the fit. What are my options here?",
     "question": "Is the customer asking for a refund?"},
    {"id": "supplier", "kind": "choice",
     "state": "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
     "question": "Supplier response for this scenario.",
     "options": ["issue_warning", "renegotiate", "dual_source", "maintain"]},
    {"id": "department", "kind": "choice", "state": "Support ticket: I got the wrong item, refund my card please.",
     "question": "Which team handles this?", "options": ["general", "sales", "shipping", "billing", "technical"]},
    {"id": "urgency", "kind": "score", "state": "Production database is down for all customers since 09:12.",
     "question": "How urgent is this incident?"},
    {"id": "json_state", "kind": "noul", "state": '{"status": "degraded", "region": "Tokyo", "error_rate": 0.31}',
     "question": "Is the service healthy?"},
    {"id": "chinese", "kind": "noul", "state": "今天中秋节，还要加班的有吗？来报道下",
     "question": "Does the post promote a product or paid offer?"},
    {"id": "sixteen", "kind": "choice", "state": "The user wants to change the colour of the app to dark blue.",
     "question": "Which settings page should open?",
     "options": [f"page {c}" for c in "abcdefghijklmnop"]},
]
PROBE_DIMENSIONS = [0, 1, 2, 3, 255, 256, 1023, 1024, 2047, 2048, 3071, 4093, 4094, 4095]


def snapshot_path(folder: str | None) -> Path:
    """A local snapshot folder (``hf download --local-dir``), or the pinned revision in the HF cache."""
    if folder:
        return Path(folder).resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_REPO, revision=HF_REVISION, allow_patterns=SNAPSHOT_PATTERNS,
                                  local_files_only=True))


def validate_snapshot(snapshot: Path) -> None:
    for name, expected in ARTIFACT_HASHES.items():
        actual = golden._sha256(snapshot / name)
        if actual != expected:
            raise SystemExit(f"{name} SHA-256 {actual} != pinned {expected}")


def full_fixture(snapshot: Path) -> dict:
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(snapshot)
    temperatures = json.loads((snapshot / "calibration.json").read_text())["per_kind"]
    model = load_reference(snapshot, torch)

    tokenizer_cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in TOKENIZER_TEXTS]
    cases = []
    for spec in FULL_CASES:
        kind = spec["kind"]
        options = {"noul": ["false", "true"], "score": [str(i) for i in range(6)]}.get(kind) or spec["options"]
        prompt = bare_v1(kind, spec["state"], spec["question"], options)
        ids = tok.encode(prompt, add_special_tokens=False)
        recorder = LayerRecorder(model) if spec["id"] == "refund" else None
        with torch.no_grad():
            last = model.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1]
        logits = head_logits(snapshot, last, torch)
        case = {
            "id": spec["id"], "kind": kind, "state": spec["state"], "question": spec["question"],
            "options": options, "prompt": prompt, "ids": ids, "head_logits": logits,
            "probabilities": distribution(logits, kind, len(options), temperatures),
        }
        if recorder is not None:
            recorder.close()
            probes = probe_rows(len(ids))
            case["trunk_probe"] = {
                "rows": probes, "dimensions": PROBE_DIMENSIONS,
                "entries": {name: [[float(out[0, r, d]) for d in PROBE_DIMENSIONS] for r in probes]
                            for name, out in recorder.outputs.items()},
            }
        cases.append(case)
    return {
        "schema_version": SCHEMA_VERSION, "revision": HF_REVISION, "temperatures": temperatures,
        "tokenizer_cases": tokenizer_cases, "cases": cases,
    }


def run_full(check: bool, folder: str | None) -> int:
    snapshot = snapshot_path(folder)
    validate_snapshot(snapshot)
    generated = full_fixture(snapshot)
    if check:
        errors = golden.compare_json(json.loads(golden.canonical_json_bytes(generated)),
                                     json.loads(FULL_FIXTURE.read_text()))
        for error in errors[:50]:
            print(error, file=sys.stderr)
        print("full fixture matches" if not errors else f"{len(errors)} mismatches")
        return 1 if errors else 0
    FULL_FIXTURE.write_bytes(golden.canonical_json_bytes(generated))
    print(f"wrote {FULL_FIXTURE.relative_to(REPO_ROOT)} ({FULL_FIXTURE.stat().st_size} bytes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", choices=["tiny", "full"])
    parser.add_argument("--check", action="store_true", help="compare against the committed fixture")
    parser.add_argument("--model-folder", help="full: a local JEV-9B snapshot instead of the HF cache")
    args = parser.parse_args(argv)
    runtime_check()
    return run_tiny(args.check) if args.fixture == "tiny" else run_full(args.check, args.model_folder)


if __name__ == "__main__":
    raise SystemExit(main())
