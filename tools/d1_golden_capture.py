#!/usr/bin/env python3
"""Generate and verify deterministic d1-omni-600M golden vectors.

The pinned LiquidAI/d1-omni-600M snapshot is the reference oracle. Every
section runs the snapshot's own remote code (prompt.py, encoder.py, vision.py,
audio.py) on CPU float32 and records the intermediate values the Swift port
checks layer by layer: tokenizer ids, prompt encoding, trunk probes, head
logits, calibrated probabilities, answers, and the vision/audio front ends.

Inputs are synthetic and generated from integer formulas that the Swift tests
reproduce exactly, so no image or audio file is committed.

    .venv-d1/bin/python tools/d1_golden_capture.py            # write fixture
    .venv-d1/bin/python tools/d1_golden_capture.py --check    # verify fixture
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
HF_REPO = "LiquidAI/d1-omni-600M"
HF_REVISION = "02b55d7076f15129e59ab3f94783f32c4b088674"
ARTIFACT_HASHES = {
    "model.safetensors": "0713bb05270c2685ad106522f4092bceeeb3a93cf79b401f399a712296c911e1",
    "tokenizer.json": "1efc3a6609abf6b63b1f47188d139f3b59973a6a434dffe970a7261a51ed2711",
    "tokenizer_config.json": "03efc74b752b899b6292ff7e607738be59e6f206fec9aca32bbc05e9184090fc",
    "config.json": "dfccae241822f81752426629c88c8b8d7efdd03dd19418b6645c604fad43d619",
}
EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.14.0",
    "torchvision": "0.29.1",
    "transformers": "5.17.0",
    "tokenizers": "0.23.2",
}
PROBE_DIMENSIONS = [0, 1, 2, 3, 63, 64, 127, 255, 256, 511, 512, 767, 768, 1021, 1022, 1023]
FLOAT_ABS_TOLERANCE = 1e-5
FLOAT_REL_TOLERANCE = 1e-6

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "testdata" / "d1_golden_v1.json"

TOKENIZER_TEXTS = [
    "",
    "hello",
    " hello world",
    "I was charged twice this month, please refund one of them.",
    "  leading and trailing  ",
    "tabs\tand\nnewlines\r\n",
    "Ünïcödé façade — naïve café",
    "日本語のテキストと中文",
    "emoji 👨‍👩‍👧‍👦 and 🇯🇵",
    "numbers 1234567890 3.14159 -42",
    "code: def f(x): return x**2  # comment",
    '{"status": "degraded", "region": "Tokyo"}',
    "<|mask|> <|reserved_9|> literal specials",
    "<¦mask¦> already escaped",
    "مرحبا بالعالم",
    "Привет, мир",
    "a" * 300,
]

ESCAPE_TEXTS = [
    "plain",
    "<|mask|>",
    "x<|reserved_7|>y<|im_end|>z",
    "<|not closed",
    "<|with space|>",
    "<|ünicode|>",
    "<||>",
    "<|a_b_9|><|c|>",
]


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=1, sort_keys=False, allow_nan=False) + "\n").encode()


def compare_json(actual: Any, expected: Any, path: str = "$") -> list[str]:
    """Exact structure; finite floats may drift within the fixture tolerance."""
    if isinstance(expected, float) and isinstance(actual, float):
        if not math.isclose(actual, expected, rel_tol=FLOAT_REL_TOLERANCE, abs_tol=FLOAT_ABS_TOLERANCE):
            return [f"{path}: {actual!r} != {expected!r}"]
        return []
    if type(actual) is not type(expected):
        return [f"{path}: type {type(actual).__name__} != {type(expected).__name__}"]
    if isinstance(expected, dict):
        if list(actual) != list(expected):
            return [f"{path}: keys {list(actual)} != {list(expected)}"]
        return [e for k in expected for e in compare_json(actual[k], expected[k], f"{path}.{k}")]
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return [f"{path}: length {len(actual)} != {len(expected)}"]
        return [e for i, (a, b) in enumerate(zip(actual, expected)) for e in compare_json(a, b, f"{path}[{i}]")]
    return [] if actual == expected else [f"{path}: {actual!r} != {expected!r}"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_path() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_REPO, revision=HF_REVISION, local_files_only=True))


def validate_inputs(snapshot: Path) -> None:
    for name, expected in ARTIFACT_HASHES.items():
        actual = _sha256(snapshot / name)
        if actual != expected:
            raise SystemExit(f"{name} SHA-256 {actual} != pinned {expected}")
    from importlib.metadata import version

    for package, expected in EXPECTED_RUNTIME_VERSIONS.items():
        actual = version(package)
        if actual != expected:
            raise SystemExit(f"{package} {actual} != pinned {expected}")


# ---------------------------------------------------------------- synthetic media

def synthetic_image(width: int, height: int, seed: int) -> list[int]:
    """Interleaved RGB bytes from integer arithmetic only (mirrored in Swift)."""
    pixels = []
    for y in range(height):
        for x in range(width):
            r = (x * 255 // max(1, width - 1) + seed * 37) % 256
            g = (y * 255 // max(1, height - 1) + seed * 91) % 256
            b = ((x // 16 + y // 16) % 2) * 160 + ((x * y + seed) % 64)
            pixels.extend((r, g, b % 256))
    return pixels


def synthetic_audio(samples: int, seed: int) -> list[int]:
    """int16 PCM from an integer LCG and integer triangle waves (mirrored in Swift)."""
    out, state = [], seed & 0xFFFFFFFF
    for n in range(samples):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        noise = (state >> 16) % 2001 - 1000
        period_a, period_b = 73 + seed % 11, 211
        tri_a = abs((n % period_a) * 2 * 6000 // period_a - 6000) - 3000
        tri_b = abs((n % period_b) * 2 * 9000 // period_b - 9000) - 4500
        envelope = (n % 4000) * 4 // 4000 + 1
        out.append(max(-32768, min(32767, (tri_a + tri_b) * envelope // 2 + noise)))
    return out


# ---------------------------------------------------------------- ordered views

def ordered_request(state, questions: dict) -> dict:
    """The request with every order-bearing mapping as a list of pairs, so the
    Swift tests never depend on JSON object key order."""
    if isinstance(state, dict):
        state_view = {"fields": [[k, v] for k, v in state.items()]}
    else:
        state_view = {"text": "" if state is None else state}
    out = []
    for name, q in questions.items():
        criteria = q.get("criteria")
        if q["type"] == "choice":
            criteria = [[k, v] for k, v in criteria.items()]
        out.append({"id": name, "type": q["type"], "instructions": q["instructions"], "criteria": criteria})
    return {"state": state_view, "questions": out}


def ordered_result(result: dict) -> dict:
    answers = []
    for name, answer in result["answers"].items():
        view = {"id": name, **{k: v for k, v in answer.items() if k not in ("probabilities", "legend")}}
        if "probabilities" in answer:
            view["probabilities"] = [[k, v] for k, v in answer["probabilities"].items()]
        if "legend" in answer:
            view["legend"] = [[k, v] for k, v in answer["legend"].items()]
        answers.append(view)
    return {"answers": answers, "usage": result["usage"]}


# ---------------------------------------------------------------- capture

class Oracle:
    def __init__(self, snapshot: Path):
        import torch
        from transformers import AutoModel

        torch.manual_seed(0)
        torch.set_num_threads(1)
        self.torch = torch
        self.model = AutoModel.from_pretrained(str(snapshot), trust_remote_code=True, dtype=torch.float32).eval()
        package = type(self.model).__module__.rsplit(".", 1)[0]
        self.prompt = importlib.import_module(package + ".prompt")
        self.vision = importlib.import_module(package + ".vision")
        self.audio = importlib.import_module(package + ".audio")
        self.tok = self.model.tokenizer

    def special_ids(self) -> dict:
        ids = self.tok.convert_tokens_to_ids
        return {
            "bos": self.tok.bos_token_id,
            "pad": self.tok.pad_token_id,
            "mask": ids(self.prompt.MARKER),
            **{name: ids(token) for name, token in self.prompt.DELIM.items()},
        }

    def tokenizer_cases(self) -> list:
        return [{"text": t, "ids": self.tok(t, add_special_tokens=False)["input_ids"]} for t in TOKENIZER_TEXTS]

    def escape_cases(self) -> list:
        return [{"text": t, "escaped": self.prompt.escape(t)} for t in ESCAPE_TEXTS]

    def encode(self, state, question: dict, max_len: int, mode: str) -> dict:
        q = self.prompt.as_question(question)
        noul = {"false": "no", "true": "yes"} if mode != "text" else None
        ids, markers = self.prompt.encode(self.tok, state, q, max_len, noul, mode == "audio")
        return {"ids": ids, "markers": markers, "options": self.prompt.render_options(q, noul, mode == "audio")}

    def encode_cases(self) -> list:
        long_state = " ".join(f"token{i} lorem ipsum dolor" for i in range(400))
        many = {f"opt{i}": f"Option number {i} with a fairly long description " * 3 for i in range(12)}
        cases = [
            ("text_noul_default", "Hello there", {"type": "noul", "instructions": "Is this a greeting?"}, 16384, "text"),
            ("text_noul_explicit", {"a": "1", "b": "x\"y\nz"},
             {"type": "noul", "instructions": "Is a one?", "criteria": {"true": "a is 1", "false": ""}}, 16384, "text"),
            ("text_choice_fallback", "state", {"type": "choice", "instructions": "Pick",
             "criteria": {"alpha": None, "beta": "", "gamma": "the third <|mask|> one"}}, 16384, "text"),
            ("text_score", "state", {"type": "score", "instructions": "Rate",
             "criteria": ["low", {"what": "mid", "examples": ["x", "y"]}, "high"]}, 16384, "text"),
            ("image_noul", "", {"type": "noul", "instructions": "Is there a cat?"}, 896, "image"),
            ("image_noul_partial", "", {"type": "noul", "instructions": "Cat?", "criteria": {"true": "a cat"}}, 896, "image"),
            ("image_truncated_state", long_state, {"type": "choice", "instructions": "Which?",
             "criteria": {"a": "first", "b": "second"}}, 896, "image"),
            ("image_many_options", "short", {"type": "choice", "instructions": "Which option?",
             "criteria": many}, 896, "image"),
            ("audio_choice", {}, {"type": "choice", "instructions": "Topic?",
             "criteria": {"food": "Food and meals", "travel": "", "weather": None}}, 15360, "audio"),
            ("audio_noul", {}, {"type": "noul", "instructions": "Is it a question?",
             "criteria": {"true": "custom"}}, 15360, "audio"),
        ]
        out = []
        for case_id, state, question, max_len, mode in cases:
            request = ordered_request(state, {case_id: question})
            out.append({"id": case_id, "state": request["state"], "question": request["questions"][0],
                        "max_len": max_len, "mode": mode, **self.encode(state, question, max_len, mode)})
        return out

    def probe_trunk(self, ids: list[int]) -> dict:
        """Re-runs Trunk.forward step by step, capturing every layer boundary."""
        torch, trunk = self.torch, self.model.encoder
        with torch.no_grad():
            h = trunk.embed_tokens(torch.tensor(ids))[None]
            length = h.shape[1]
            pad = torch.ones(1, length, dtype=torch.bool)
            prefix = torch.zeros(1, dtype=torch.long)
            expected = trunk(h, pad, prefix)
            mask = torch.zeros(1, 1, length, length)
            keep_right = torch.ones(1, length)
            padf = pad.float()
            cos, sin = trunk.rope(length, h)
            entries = {"embeddings": h}
            for index, layer in enumerate(trunk.layers):
                x = layer.operator_norm(h)
                x = layer.self_attn(x, cos, sin, mask) if layer.is_attention_layer else layer.conv(x, padf, keep_right)
                h = h + x
                h = h + layer.feed_forward(layer.ffn_norm(h))
                entries[f"layer_{index}"] = h
            entries["final_norm"] = trunk.embedding_norm(h)
            if not torch.equal(entries["final_norm"], expected):
                raise SystemExit("trunk probe diverged from Trunk.forward")
        return {"ids": ids, "dimensions": PROBE_DIMENSIONS,
                "entries": {k: v[0][:, PROBE_DIMENSIONS].tolist() for k, v in entries.items()}}

    def head_logits(self, rows: list) -> list:
        """`_forward` up to the scorer: raw logits per row, before temperature."""
        torch, model = self.torch, self.model
        with torch.no_grad():
            seqs, lengths, offsets = [], [], []
            for prefix, ids, *_ in rows:
                text = model.encoder.embed_tokens(torch.tensor(ids))
                seqs.append(text if prefix is None else torch.cat([prefix[0], text]))
                offsets.append(0 if prefix is None else prefix.shape[1])
                lengths.append(len(ids))
            h = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
            total = torch.tensor([len(s) for s in seqs])
            pad = torch.arange(h.shape[1])[None] < total[:, None]
            h = model.encoder(h, pad, torch.tensor(offsets))
            text = torch.nn.utils.rnn.pad_sequence([h[i, o:o + n] for i, (o, n) in enumerate(zip(offsets, lengths))],
                                                   batch_first=True)
            k = max(len(r[2]) for r in rows)
            text_pad = torch.arange(text.shape[1])[None] < torch.tensor(lengths)[:, None]
            marker_pos = torch.zeros(len(rows), k, dtype=torch.long)
            marker_mask = torch.zeros(len(rows), k, dtype=torch.bool)
            for i, r in enumerate(rows):
                marker_pos[i, :len(r[2])] = torch.tensor(r[2])
                marker_mask[i, :len(r[2])] = True
            qtype = torch.tensor([self.prompt.QTYPES[r[3].type] for r in rows])
            logits = model.head(text, text_pad, marker_pos, marker_mask, qtype)
        return [z[:r[3].options].tolist() for r, z in zip(rows, logits)]

    def request_case(self, case_id: str, state, questions: dict, images=None, audio=None,
                     prefix=None, mode: str = "text") -> dict:
        named = {n: self.prompt.as_question(q) for n, q in questions.items()}
        cfg = self.model.config
        max_len = {"text": cfg.max_length, "image": cfg.image_text_length, "audio": cfg.audio_text_length}[mode]
        p = 0 if prefix is None else prefix.shape[1]
        max_len = min(max_len, cfg.max_length - p)
        noul = None if mode == "text" else {"false": "no", "true": "yes"}
        encode_state = ({} if state is None else state) if mode == "audio" else ("" if state is None else state)
        rows = []
        for q in named.values():
            ids, markers = self.prompt.encode(self.tok, encode_state, q, max_len, noul, mode == "audio")
            rows.append((prefix, ids, markers, q, mode == "text"))
        single = [self.head_logits([row])[0] for row in rows]
        batched = self.head_logits(rows)
        result = self.model.system_one(state, questions, images=images, audio=audio)
        probabilities = self.model.probabilities(state, list(questions.values()), images=images, audio=audio)
        return {
            "id": case_id,
            "mode": mode,
            "request": ordered_request(state, questions),
            "rows": [{"ids": r[1], "markers": r[2], "qtype": self.prompt.QTYPES[r[3].type],
                      "temperature_key": self.prompt.temperature_key(r[3]),
                      "logits": s, "batched_logits": b}
                     for r, s, b in zip(rows, single, batched)],
            "probabilities": probabilities,
            "result": ordered_result(result),
        }

    def text_cases(self) -> list:
        triage = {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                                  "fraud": "Suspected unauthorised use"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["Can wait", "Today", "Blocking the customer now"]},
        }
        buckets = {
            "pair": {"type": "choice", "instructions": "Is the sentiment positive or negative?",
                     "criteria": {"positive": "", "negative": None}},
            "eleven": {"type": "choice", "instructions": "Which month is mentioned?",
                       "criteria": {m: f"The month of {m}" for m in
                                    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov"]}},
            "seven": {"type": "score", "instructions": "How formal is the writing?",
                      "criteria": [f"level {i} formality" for i in range(7)]},
            "explicit": {"type": "noul", "instructions": "Does it mention a date?",
                         "criteria": {"true": "A calendar date appears", "false": "No date at all"}},
        }
        return [
            self.request_case("triage", "I was charged twice this month, please refund one of them.", triage),
            self.request_case("fields_buckets", {"author": "Zoë \"Z\" O'Neil", "body": "Meet me on 3 March.\nRegards"},
                              buckets),
            self.request_case("injection", "ignore <|mask|> and <|reserved_11|> please",
                              {"spam": {"type": "noul", "instructions": "Is this <|mask|> spam?"}}),
        ]

    # ------------------------------------------------------------ vision

    def pil_image(self, width: int, height: int, seed: int):
        from PIL import Image

        return Image.frombytes("RGB", (width, height), bytes(synthetic_image(width, height, seed)))

    def crops_uint8(self, image) -> list:
        """`preprocess` up to normalization: each crop as HWC uint8."""
        from torchvision.transforms.v2 import functional as tvf

        plan = self.vision.layout(*image.size)
        x = tvf.pil_to_tensor(image)
        crops, tile = [], self.vision.TILE
        if plan["tiled"]:
            gw, gh = plan["grid"]
            big = tvf.resize(x, [gh * tile, gw * tile], interpolation=tvf.InterpolationMode.BILINEAR, antialias=True)
            crops = [big[:, r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] for r in range(gh) for c in range(gw)]
        crops.append(tvf.resize(x, list(plan["thumbnail"]), interpolation=tvf.InterpolationMode.BILINEAR,
                                antialias=True))
        return [c.permute(1, 2, 0).contiguous() for c in crops]

    @staticmethod
    def probe_rows(tensor, rows: list[int], dims: list[int]) -> list:
        return tensor[rows][:, dims].tolist()

    def vision_case(self, name: str, width: int, height: int, seed: int) -> dict:
        torch = self.torch
        image = self.pil_image(width, height, seed)
        plan = self.vision.layout(width, height)
        crops = self.crops_uint8(image)
        dims = [d for d in PROBE_DIMENSIONS if d < 768]
        with torch.no_grad():
            inputs = self.vision.preprocess(image)
            tower = self.model.vision.tower
            hidden = tower(**inputs).last_hidden_state
            embeddings = next(m for m in tower.modules() if type(m).__name__ == "Siglip2VisionEmbeddings")
            grid = embeddings.position_embedding.weight.reshape(16, 16, -1)
            positions = embeddings.resize_positional_embeddings(grid, inputs["spatial_shapes"], 1024)
            prefix = self.model.vision([image])[0]
        crop_views = []
        for index, crop in enumerate(crops):
            h, w = crop.shape[0], crop.shape[1]
            n = (h // 16) * (w // 16)
            rows = [0, 1, n // 2, n - 1]
            crop_views.append({
                "height": h, "width": w,
                "sha256": hashlib.sha256(crop.numpy().tobytes()).hexdigest(),
                "sum": int(crop.sum()),
                "position_probe": self.probe_rows(positions[index], rows, dims),
                "tower_probe": self.probe_rows(hidden[index], rows, dims),
                "probe_rows": rows,
            })
        p = prefix.shape[0]
        rows = [0, 1, p // 2, p - 1]
        return {
            "name": name, "width": width, "height": height, "seed": seed,
            "layout": {"grid": list(plan["grid"]), "thumbnail": list(plan["thumbnail"]), "tiled": plan["tiled"]},
            "crops": crop_views,
            "probe_dimensions": dims,
            "prefix_length": p,
            "prefix_rows": rows,
            "prefix_probe": self.probe_rows(prefix, rows, PROBE_DIMENSIONS),
        }

    def vision_cases(self) -> dict:
        specs = {"small": (300, 200, 1), "large": (1100, 700, 2), "tiny": (40, 60, 3)}
        images = {name: self.pil_image(*spec) for name, spec in specs.items()}
        questions = {
            "red": {"type": "noul", "instructions": "Is the image mostly red?"},
            "pattern": {"type": "choice", "instructions": "What pattern is shown?",
                        "criteria": {"checker": "A checkerboard", "stripes": "Stripes", "plain": "A plain colour"}},
            "bright": {"type": "score", "instructions": "How bright is the image?",
                       "criteria": ["Dark", "Medium", "Bright"]},
        }
        requests = []
        for case_id, names, state in [("small", ["small"], None), ("large", ["large"], "A product photo."),
                                      ("pair", ["tiny", "small"], None)]:
            chosen = [images[n] for n in names]
            with self.torch.no_grad():
                prefix = self.model.vision(chosen)
            case = self.request_case(case_id, state, questions, images=chosen, prefix=prefix, mode="image")
            case["images"] = [{"width": specs[n][0], "height": specs[n][1], "seed": specs[n][2]} for n in names]
            requests.append(case)
        return {
            "layouts": [{"width": w, "height": h, **{k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in self.vision.layout(w, h).items()}}
                        for w, h in [(1, 1), (32, 32), (640, 480), (2000, 300), (300, 2000), (512, 512),
                                     (1024, 1024), (4000, 3000), (1100, 700), (700, 1100), (1500, 1500)]],
            "images": [self.vision_case(name, *spec) for name, spec in specs.items()],
            "requests": requests,
        }

    # ------------------------------------------------------------ audio

    def audio_case(self, name: str, samples: int, seed: int) -> dict:
        import numpy as np

        torch = self.torch
        clip = np.array(synthetic_audio(samples, seed), dtype=np.int16)
        with torch.no_grad():
            wave = self.audio.waveform(clip)
            mel, frames = self.model.audio.frontend(wave)
            encoded, lengths = self.model.audio.encoder(mel, frames)
            prefix = self.model.audio(clip)[0]
        t = int(frames[0])
        frame_rows = [0, 1, t // 2, t - 1]
        bins = [0, 1, 2, 31, 64, 100, 126, 127]
        n = int(lengths[0])
        rows = [0, 1, n // 2, n - 1]
        return {
            "name": name, "samples": samples, "seed": seed,
            "clip_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
            "frames": t,
            "mel_frames": frame_rows,
            "mel_bins": bins,
            "mel_probe": mel[0][bins][:, frame_rows].T.tolist(),
            "encoder_rows": rows,
            "encoder_dimensions": [0, 1, 63, 64, 255, 256, 510, 511],
            "encoder_probe": encoded[0][rows][:, [0, 1, 63, 64, 255, 256, 510, 511]].tolist(),
            "prefix_length": prefix.shape[0],
            "prefix_rows": rows,
            "prefix_probe": self.probe_rows(prefix, rows, PROBE_DIMENSIONS),
        }

    def audio_cases(self) -> dict:
        import numpy as np

        specs = {"short": (4800, 5), "speech": (40000, 7)}
        questions = {
            "topic": {"type": "choice", "instructions": "What is the speaker talking about?",
                      "criteria": {"food": "Food and meals", "travel": "Travel and transport", "weather": ""}},
            "question": {"type": "noul", "instructions": "Is the speaker asking a question?"},
        }
        requests = []
        for case_id, name, state in [("speech", "speech", "Voice note from a user."), ("short", "short", None)]:
            clip = np.array(synthetic_audio(*specs[name]), dtype=np.int16)
            with self.torch.no_grad():
                prefix = self.model.audio(clip)
            case = self.request_case(case_id, state, questions, audio=clip, prefix=prefix, mode="audio")
            case["audio"] = {"samples": specs[name][0], "seed": specs[name][1]}
            requests.append(case)
        return {"clips": [self.audio_case(name, *spec) for name, spec in specs.items()], "requests": requests}

    def generate(self) -> dict:
        trunk_ids = self.encode("I was charged twice.", {"type": "noul", "instructions": "Refund?"}, 16384, "text")["ids"]
        return {
            "schema_version": SCHEMA_VERSION,
            "model": {"repo": HF_REPO, "revision": HF_REVISION, "artifact_hashes": ARTIFACT_HASHES},
            "runtime": EXPECTED_RUNTIME_VERSIONS,
            "special_ids": self.special_ids(),
            "temperatures": self.model.config.temperatures,
            "tokenizer_cases": self.tokenizer_cases(),
            "escape_cases": self.escape_cases(),
            "encode_cases": self.encode_cases(),
            "trunk_probe": self.probe_trunk(trunk_ids),
            "text_cases": self.text_cases(),
            "vision": self.vision_cases(),
            "audio": self.audio_cases(),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare against the committed fixture")
    args = parser.parse_args(argv)
    snapshot = snapshot_path()
    validate_inputs(snapshot)
    generated = Oracle(snapshot).generate()
    if args.check:
        expected = json.loads(FIXTURE_PATH.read_text())
        errors = compare_json(json.loads(canonical_json_bytes(generated)), expected)
        for error in errors[:50]:
            print(error, file=sys.stderr)
        print("fixture matches" if not errors else f"{len(errors)} mismatches")
        return 1 if errors else 0
    FIXTURE_PATH.write_bytes(canonical_json_bytes(generated))
    print(f"wrote {FIXTURE_PATH.relative_to(REPO_ROOT)} ({FIXTURE_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
