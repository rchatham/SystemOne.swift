#!/usr/bin/env python3
"""Write testdata/d1_omni_2b_audio_tiny/: d1-Omni-2B audio decisions on a tiny LFM3-Omni.

d1-Omni-2B (LiquidAI/d1-Omni-2B, LFM Open License v1.0) is d1-2B with LFM3-Omni's
speech encoder. Its `media.py` resamples a clip to the feature extractor's 16 kHz
(`resampled`, a torch port of torchaudio's default windowed sinc), writes the chat
template's markup per clip at the head of the user turn, before the escaped state,
and the processor expands `<|audio|>` to one token per encoder position, inside
`<|vision_start|>` … `<|vision_end|>`. The feature extractor
(`feature_extraction_lfm3_omni.py`, vendored from transformers' unreleased LFM3
code, Apache-2.0) right-pads to whole frames and takes a periodic-Hann power
spectrum, Slaney mel filters, `log10` floored at -12 and `(x + 4) / 4`; `resampled`
and `log_mel` below are those two, verbatim but for names.

The encoder (three stride-2 convolutions, a causal LFM2 trunk with a sliding window,
a GELU projector), set packing, readout, calibration and answers are mlx-vlm
0.7.11's `lfm3_audio` and `d1_lfm3_omni` (`D1.d1_hidden`, `pack_set`,
`readout_table`, `probabilities`, `answer`), on a tiny random LFM3-Omni whose text
trunk and vision tower are testdata/d1_2b_vision_tiny's.

- config.json, processor_config.json (d1-Omni-2B's feature extractor), model.safetensors:
  transformers' names (`lin_out` stored as a width-1 Conv1d); mlx/: mlx-vlm's names.
- fixture.json: per synthetic clip (16-bit PCM from an integer formula), its
  resampled samples and log-mel features (summaries), and its projected features;
  per request, its spoken questions' clips, the packed rows and answers. A request
whose state is None (a spoken state, a picture alone, or nothing) writes no state
block and no `QUESTION:` header, as `prefix_text` and `question_header` do.

Videos run through the vendored `video_processing_lfm3_vl.py` (`sample_frames`,
`smart_resize`, `patchify`, transformers' torchvision resize) at a tiny pixel budget, with
`processing_lfm3_omni.py`'s timestamps and markup; mlx-vlm's tower reads each pair of frames.

A spoken question's instructions are its clip: `suffix_layout` writes `media.SPOKEN`
in their place, `Reader.place` swaps in the clip's markup, and the engine fills each
row's `<|audio|>` tokens with the state's clip, then the row's spoken questions'.

    .venv/bin/pip install "torch==2.15.0" "transformers==5.22.1" "mlx==0.32.4" "mlx-vlm==0.7.11" safetensors numpy
    .venv/bin/python tools/d1_omni_2b_audio_fixture.py
"""

import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx_vlm
import numpy as np
import torch
import transformers
from mlx.utils import tree_flatten
from PIL import Image
from transformers.audio_utils import mel_filter_bank

from mlx_vlm.models import d1_lfm3_omni
from mlx_vlm.models.d1_lfm3_vl import media as d1_media
from mlx_vlm.models.d1_lfm3_vl.api import answer
from mlx_vlm.models.d1_lfm3_vl.d1 import LEAD, pack_set
from mlx_vlm.models.d1_lfm3_vl.media import SPOKEN
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
from mlx_vlm.models.lfm3_audio import audio as mlx_audio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_2b_fixture import ADDED, CALIBRATION, CODES, SPECIAL, WORDS  # noqa: E402
from d1_2b_vision_fixture import PROCESSOR as VISION_PROCESSOR  # noqa: E402
from d1_2b_vision_fixture import VISION, VISION_TOKENS, Backend, cap_pixels, process, sha  # noqa: E402
from transformers.image_utils import PILImageResampling, SizeDict  # noqa: E402
from resize_fixture import synthetic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "testdata" / "d1_omni_2b_audio_tiny"
BASE = ROOT / "testdata" / "d1_2b_vision_tiny"
VERSIONS = {"torch": "2.15.0", "transformers": "5.22.1", "mlx": "0.32.4", "mlx_vlm": "0.7.11"}
SEED = 20261105
SET_SIZE = 2
MAX_PIXELS = 640 * 640
AUDIO = 124
AUDIO_TOKENS = {"<|audio|>": AUDIO}

# LiquidAI/d1-Omni-2B's processor_config.json feature extractor, verbatim.
FEATURE_EXTRACTOR = {
    "encoder_stride": 8, "feature_extractor_type": "Lfm3OmniFeatureExtractor", "feature_size": 128,
    "hop_length": 160, "log_floor": -12.0, "n_fft": 400, "normalization_divisor": 4.0,
    "normalization_offset": 4.0, "padding_side": "right", "padding_value": 0.0, "return_attention_mask": True,
    "sampling_rate": 16000, "win_length": 400,
}
# d1-Omni-2B's audio_encoder_config, shrunk: 8 channels, 16 features, 3 layers, a 4-key window.
AUDIO_ENCODER = {
    "conv_channels": 8, "hidden_size": 16, "model_type": "lfm3_omni_encoder", "num_mel_bins": 128,
    "sliding_window": 4,
    "text_config": {
        "block_auto_adjust_ff_dim": True, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 16,
        "conv_L_cache": 3, "conv_bias": False, "hidden_size": 16, "intermediate_size": 40,
        "layer_types": ["conv", "full_attention", "conv"], "max_position_embeddings": 128000,
        "model_type": "lfm2", "norm_eps": 1e-05, "num_attention_heads": 4, "num_hidden_layers": 3,
        "num_key_value_heads": 2, "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        "tie_word_embeddings": True, "vocab_size": 2,
    },
}
DEPTH = {
    "conv_L_cache": 3, "conv_bias": False, "head_dim": 8, "hidden_size": 16, "intermediate_size": 32,
    "layer_types": ["full_attention"], "max_position_embeddings": 3, "model_type": "lfm3", "norm_eps": 1e-05,
    "num_attention_heads": 2, "num_hidden_layers": 1, "num_key_value_heads": 1,
    "rope_parameters": {"full_attention": {"partial_rotary_factor": 1.0, "rope_theta": 1000000.0, "rope_type": "default"}},
    "sliding_window": 256, "tie_word_embeddings": True, "use_sandwich_norm": False, "vocab_size": 1,
}

# name: (sample rate, samples, salt). `short` is under one window; `call`, `phone` and `ask2` are resampled.
# `ask` and `ask2` are spoken questions.
CLIPS = {"short": (16000, 300, 3), "speech": (16000, 9000, 5), "call": (22050, 26000, 7), "phone": (8000, 7000, 11),
         "ask": (16000, 4000, 13), "ask2": (11025, 5000, 17)}
REQUESTS = [
    ("Caller says <|audio|> twice.", "call", None, [
        ("refund", {"type": "noul", "instructions": "Does the caller want a refund?"}),
        ("topic", {"type": "choice", "instructions": "What is the call about?",
                   "criteria": {"billing": "Billing", "outage": None, "sales": "A new plan"}}),
        ("mood", {"type": "score", "instructions": "How upset is the caller?", "criteria": ["calm", "annoyed", "angry"]}),
    ]),
    ({"ticket": "Voicemail", "line": "2"}, "phone", None, [
        ("urgent", {"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "Needs a call back today"}}),
    ]),
    ("", "short", None, [
        ("kind", {"type": "choice", "instructions": "What is it?", "criteria": {"speech": None, "noise": None}}),
        ("loud", {"type": "score", "instructions": "How loud?", "criteria": ["quiet", "loud"]}),
    ]),
    ("A clip.", "speech", None, [
        ("lang", {"type": "choice", "instructions": "Which language?", "criteria": {"en": "English", "fr": "French"}}),
    ]),
    ("A photo of the shelf <|image|> here.", None, "small", [
        ("stocked", {"type": "noul", "instructions": "Is the shelf stocked?"}),
    ]),
    # Spoken questions: their clip's markup sits where their instructions would.
    ("Order 1234 arrived broken.", None, None, [
        ("claim", {"type": "noul", "instructions": {"spoken": "ask"}}),
        ("channel", {"type": "choice", "instructions": "How did they reach us?",
                     "criteria": {"email": None, "phone": "Phone call"}}),
        ("anger", {"type": "score", "instructions": {"spoken": "ask2"}, "criteria": ["calm", "upset"]}),
    ]),
    ("Caller <|audio|> says:", "call", None, [
        ("intent", {"type": "choice", "instructions": {"spoken": "ask"},
                    "criteria": {"cancel": "Cancel the plan", "upgrade": None}}),
        ("vip", {"type": "noul", "instructions": "Is the caller a VIP?"}),
    ]),
    ("A photo of the shelf.", None, "small", [
        ("empty", {"type": "noul", "instructions": {"spoken": "ask2"}}),
    ]),
    # No state (`state=None`): the media alone is the state, and the questions follow it with no
    # `QUESTION:` header. A spoken state is a clip alone.
    (None, "call", None, [
        ("refund", {"type": "noul", "instructions": "Does the caller want a refund?"}),
        ("topic", {"type": "choice", "instructions": "What is the call about?",
                   "criteria": {"billing": "Billing", "outage": None}}),
        ("mood", {"type": "score", "instructions": "How upset is the caller?", "criteria": ["calm", "angry"]}),
    ]),
    (None, "speech", None, [
        ("lang", {"type": "choice", "instructions": {"spoken": "ask"},
                  "criteria": {"en": "English", "fr": "French"}}),
    ]),
    (None, None, "small", [
        ("stocked", {"type": "noul", "instructions": "Is the shelf stocked?"}),
    ]),
    (None, None, None, [
        ("wet", {"type": "noul", "instructions": "Is water wet?"}),
    ]),
    # Videos: an odd frame count (its last frame repeated), a clip over the budget, and an upscaled one
    # under it whose timestamps round half to even (0.25 → "0.2"), here with a spoken question.
    ("A clip of the shelf.", None, "video:clip", [
        ("stocked", {"type": "noul", "instructions": "Is the shelf stocked?"}),
        ("aisle", {"type": "choice", "instructions": "Which aisle?", "criteria": {"dairy": "Dairy", "bakery": None}}),
    ]),
    (None, None, "video:long", [
        ("busy", {"type": "score", "instructions": "How busy is it?", "criteria": ["empty", "busy", "packed"]}),
    ]),
    ("Tiny.", None, "video:tiny", [
        ("moving", {"type": "noul", "instructions": {"spoken": "ask"}}),
    ]),
]
PICTURES = {"small": (200, 150, 21)}
VIDEO = 123
VIDEO_TOKENS = {"<|video|>": VIDEO}
# d1-Omni-2B's video processor, with the tiny checkpoint's pixel budget (8,192 to 200,000 pixels a clip),
# so the clips below take both of `smart_resize`'s branches.
VIDEO_PROCESSOR = {
    "cap_pixels_per_frame": False, "do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True,
    "do_sample_frames": True, "fps": 2.0, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    "max_frames": 384, "max_video_tokens": 768, "merge_size": 2, "min_frames": 4, "patch_size": 16, "resample": 3,
    "rescale_factor": 0.00392156862745098, "return_metadata": False,
    "size": {"longest_edge": 200000, "shortest_edge": 8192}, "temporal_patch_size": 2,
    "video_processor_type": "Lfm3VlVideoProcessor",
}
# name: (width, height, salt, frames, fps); frame i is tools/resize_fixture.py's picture with salt + 7 i.
VIDEOS = {"clip": (160, 120, 31, 25, 10.0), "long": (240, 180, 37, 90, 29.97), "tiny": (20, 20, 41, 4, 2.0)}


def pcm(count, salt):
    """A clip's 16-bit samples: a ramp in five loudness steps plus an LCG's noise, all integer arithmetic."""
    state, out = salt, []
    for i in range(count):
        state = (state * 1103515245 + 12345) % 2147483648
        tone = (i * (37 + salt) % 200) * ((i // 800) % 5 + 1)
        out.append(tone * 24 - 12000 + (state >> 16) % 8192 - 4096)
    return out


# --- media.py (LFM Open License v1.0) ------------------------------------------------
def resampled(wave, rate, target, zeros=6, rolloff=0.99):
    wave = np.asarray(wave, dtype=np.float32)
    if rate == target:
        return wave
    gcd = math.gcd(int(rate), int(target))
    old, new = int(rate) // gcd, int(target) // gcd
    cut = min(old, new) * rolloff
    width = math.ceil(zeros * old / cut)
    t = torch.arange(0, -new, -1, dtype=torch.float32)[:, None, None] / new
    t = ((t + torch.arange(-width, width + old, dtype=torch.float32)[None, None] / old) * cut).clamp(-zeros, zeros)
    window = torch.cos(t * math.pi / zeros / 2) ** 2
    t = t * math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0), t.sin() / t) * (window * (cut / old))
    x = torch.nn.functional.pad(torch.from_numpy(wave)[None, None], (width, width + old))
    out = torch.nn.functional.conv1d(x, kernel, stride=old).transpose(1, 2).reshape(-1)
    return out[:math.ceil(new * len(wave) / old)].numpy()


# --- feature_extraction_lfm3_omni.py (Apache-2.0) -------------------------------------
MEL_FILTERS = mel_filter_bank(num_frequency_bins=201, num_mel_filters=128, min_frequency=0.0, max_frequency=8000.0,
                              sampling_rate=16000, norm="slaney", mel_scale="slaney")


def pad_length(num_samples, n_fft=400, hop_length=160):
    if num_samples < n_fft:
        return n_fft
    return num_samples + (-(num_samples - n_fft) % hop_length)


def log_mel(waveform):
    window = torch.hann_window(400, device=waveform.device)
    spectrogram = torch.stft(waveform, n_fft=400, hop_length=160, win_length=400, window=window, center=False,
                             return_complex=True)
    power = spectrogram.abs().square()
    mel_filters = torch.from_numpy(MEL_FILTERS).to(device=waveform.device, dtype=power.dtype)
    log = (mel_filters.T @ power).log10().clamp(min=-12.0)
    return (log + 4.0) / 4.0
# -------------------------------------------------------------------------------------


# --- video_processing_lfm3_vl.py, processing_lfm3_omni.py (Apache-2.0) ---------------
def sample_frames(total_num_frames, video_fps, fps=2.0, min_frames=4, max_frames=384):
    num_frames = int(total_num_frames / video_fps * fps)
    num_frames = min(max(num_frames, min_frames), max_frames, total_num_frames)
    return np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)


def video_smart_resize(num_frames, height, width, temporal_factor=2, factor=28, min_pixels=112 * 112,
                       max_pixels=14 * 14 * 2 * 2 * 2 * 6144):
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def video_patchify(videos, patch_size, merge_size, temporal_patch_size):
    batch_size, num_frames, channel, resized_height, resized_width = videos.shape
    if pad := -num_frames % temporal_patch_size:
        repeats = videos[:, -1:].expand(-1, pad, -1, -1, -1)
        videos = torch.cat((videos, repeats), dim=1)
        num_frames += pad
    grid_t = num_frames // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    patches = videos.view(batch_size, grid_t, temporal_patch_size, channel, grid_h // merge_size, merge_size,
                          patch_size, grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten_patches = patches.reshape(batch_size, grid_t * grid_h * grid_w,
                                      channel * temporal_patch_size * patch_size * patch_size)
    return flatten_patches, grid_t, grid_h, grid_w


def calculate_timestamps(indices, video_fps, temporal_patch_size):
    indices = list(indices)
    if pad := -len(indices) % temporal_patch_size:
        indices.extend([indices[-1]] * pad)
    return [(indices[i] + indices[i + temporal_patch_size - 1]) / (2 * video_fps)
            for i in range(0, len(indices), temporal_patch_size)]
# -------------------------------------------------------------------------------------


def video_frames(name):
    width, height, salt, count, _ = VIDEOS[name]
    return [synthetic(width, height, salt + 7 * i) for i in range(count)]


def process_video(name):
    """The video processor on one clip: indices, resized frames, pixel values, grid, timestamps."""
    width, height, salt, count, fps = VIDEOS[name]
    v = VIDEO_PROCESSOR
    indices = sample_frames(count, fps, v["fps"], v["min_frames"], v["max_frames"])
    frames = np.stack([video_frames(name)[i] for i in indices])
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    h, w = video_smart_resize(len(indices), height, width, 2, 32, v["size"]["shortest_edge"], v["size"]["longest_edge"])
    backend = Backend()
    resized = backend.resize(image=tensor, size=SizeDict(height=h, width=w), resample=PILImageResampling(v["resample"]))
    normalized = backend.rescale_and_normalize(resized, True, v["rescale_factor"], True,
                                               tuple(v["image_mean"]), tuple(v["image_std"]))
    patches, grid_t, grid_h, grid_w = video_patchify(normalized[None], 16, 2, 2)
    return (indices.tolist(), resized.permute(0, 2, 3, 1).numpy(), patches[0].numpy(), [grid_t, grid_h, grid_w],
            calculate_timestamps(indices.tolist(), fps, 2))


def features_of(samples):
    """The extractor on one 16 kHz clip: (128, frames) float32."""
    clip = torch.from_numpy(np.asarray(samples, np.float32))
    clip = torch.nn.functional.pad(clip, (0, pad_length(clip.shape[0]) - clip.shape[0]))
    return log_mel(clip).numpy()


class Tokenizer:
    """tools/d1_2b_vision_fixture.py's stand-in with the audio placeholder too."""

    bos_token = "<|startoftext|>"
    all_special_tokens = list(SPECIAL) + list(VISION_TOKENS) + list(AUDIO_TOKENS) + list(VIDEO_TOKENS)
    d1_codes = CODES
    d1_numbers = False
    d1_prompt = 1
    d1_box_edges = "median"
    added = {**ADDED, **VISION_TOKENS, **AUDIO_TOKENS, **VIDEO_TOKENS}

    def encode(self, text, add_special_tokens=False):
        if text in WORDS:
            return [WORDS[text]]
        names = sorted(self.added, key=len, reverse=True)
        ids, i = [], 0
        while i < len(text):
            name = next((n for n in names if text.startswith(n, i)), None)
            if name is not None:
                ids.append(self.added[name])
                i += len(name)
            else:
                ids.extend(b % 100 for b in text[i].encode("utf-8"))
                i += 1
        return ids


def summary(values):
    f = np.asarray(values, np.float32)
    return {"count": f.shape[0], "sum": float(f.astype(np.float64).sum()),
            "abs_sum": float(np.abs(f.astype(np.float64)).sum()), "first": f[0].tolist(), "last": f[-1].tolist()}


def main():
    installed = {"torch": torch.__version__.split("+")[0], "transformers": transformers.__version__,
                 "mlx": mx.__version__, "mlx_vlm": mlx_vlm.__version__}
    for package, version in installed.items():
        if version != VERSIONS[package]:
            sys.exit(f"{package} {version} is installed; this fixture pins {VERSIONS[package]}")
    mx.set_default_device(mx.cpu)
    mx.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    tokenizer = Tokenizer()

    base = json.loads((BASE / "config.json").read_text())
    config = {
        **base, "model_type": "lfm3_omni", "audio_encoder_config": AUDIO_ENCODER, "depth_config": DEPTH,
        "audio_token_id": AUDIO, "audio_out_token_id": 125, "audio_start_token_id": 126, "text_end_token_id": 127,
        "mixed_start_token_id": 106, "eos_token_id": [102, 127], "num_codebooks": 3, "codebook_size": 5,
        "audio_eos_code": 4, "maskgit_schedule": [1, 1, 1], "d1_vision_max_pixels": MAX_PIXELS,
    }
    processor = {**VISION_PROCESSOR, "feature_extractor": FEATURE_EXTRACTOR, "video_processor": VIDEO_PROCESSOR,
                 "video_token": "<|video|>", "audio_token": "<|audio|>",
                 "audio_markers": True, "processor_class": "Lfm3OmniProcessor"}

    # transformers' layout: d1_2b_vision_tiny's text and vision weights, a random audio encoder and projector.
    weights = {n: np.asarray(v.astype(mx.float32)) for n, v in mx.load(str(BASE / "model.safetensors")).items()}
    a, c = AUDIO_ENCODER, AUDIO_ENCODER["conv_channels"]
    t = a["text_config"]
    d, heads, kv = t["hidden_size"], t["num_attention_heads"], t["num_key_value_heads"]
    hd = d // heads
    inner = 32  # 2 * 40 // 3 = 26, rounded up to a multiple of 16
    text_hidden = base["text_config"]["hidden_size"]

    def normal(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(width):
        return (1 + 0.2 * rng.standard_normal(width)).astype(np.float32)

    down, lfm, projector = "model.audio_encoder.downsample.", "model.audio_encoder.lfm.", "model.audio_projector."
    weights[down + "conv_in.weight"] = normal((c, 1, 3, 3), 0.5)
    weights[down + "conv_mid.weight"] = normal((c, 1, 3, 3), 0.5)
    weights[down + "conv_end.weight"] = normal((c, 1, 3, 3), 0.5)
    weights[down + "lin_mid.weight"] = normal((c, c, 1, 1), 1 / math.sqrt(c))
    weights[down + "lin_end.weight"] = normal((c, c, 1, 1), 1 / math.sqrt(c))
    for name in ("conv_in", "conv_mid", "conv_end", "lin_mid", "lin_end"):
        weights[down + name + ".bias"] = normal(c, 0.1)
    weights[down + "lin_out.weight"] = normal((d, 16 * c, 1), 1 / math.sqrt(16 * c))
    weights[down + "lin_out.bias"] = normal(d, 0.1)
    for i, kind in enumerate(t["layer_types"]):
        p = f"{lfm}layers.{i}."
        if kind == "full_attention":
            weights[p + "self_attn.q_proj.weight"] = normal((heads * hd, d), 1 / math.sqrt(d))
            weights[p + "self_attn.k_proj.weight"] = normal((kv * hd, d), 1 / math.sqrt(d))
            weights[p + "self_attn.v_proj.weight"] = normal((kv * hd, d), 1 / math.sqrt(d))
            weights[p + "self_attn.out_proj.weight"] = normal((d, heads * hd), 1 / math.sqrt(d))
            weights[p + "self_attn.q_layernorm.weight"] = norm(hd)
            weights[p + "self_attn.k_layernorm.weight"] = norm(hd)
        else:
            weights[p + "conv.in_proj.weight"] = normal((3 * d, d), 1 / math.sqrt(d))
            weights[p + "conv.conv.weight"] = normal((d, 1, 3), 0.5)
            weights[p + "conv.out_proj.weight"] = normal((d, d), 1 / math.sqrt(d))
        weights[p + "operator_norm.weight"] = norm(d)
        weights[p + "ffn_norm.weight"] = norm(d)
        weights[p + "feed_forward.w1.weight"] = normal((inner, d), 1 / math.sqrt(d))
        weights[p + "feed_forward.w3.weight"] = normal((inner, d), 1 / math.sqrt(d))
        weights[p + "feed_forward.w2.weight"] = normal((d, inner), 1 / math.sqrt(inner))
    weights[lfm + "embedding_norm.weight"] = norm(d)
    weights[projector + "layer_norm.weight"] = norm(d)
    weights[projector + "layer_norm.bias"] = normal(d, 0.1)
    weights[projector + "linear_1.weight"] = normal((text_hidden, d), 1 / math.sqrt(d))
    weights[projector + "linear_1.bias"] = normal(text_hidden, 0.1)
    weights[projector + "linear_2.weight"] = normal((text_hidden, text_hidden), 1 / math.sqrt(text_hidden))
    weights[projector + "linear_2.bias"] = normal(text_hidden, 0.1)

    model = d1_lfm3_omni.Model(d1_lfm3_omni.ModelConfig.from_dict(config))
    sanitized = model.sanitize({n: mx.array(v) for n, v in weights.items()})
    parameters = dict(tree_flatten(model.parameters()))
    # Every weight given lands on a parameter of its shape; only the speech decoder's stay random.
    assert set(sanitized) <= set(parameters), set(sanitized) - set(parameters)
    assert all(sanitized[n].shape == parameters[n].shape for n in sanitized)
    assert all(n.startswith(("depth_decoder.", "codebook_embedding.")) for n in set(parameters) - set(sanitized))
    model.load_weights(list(sanitized.items()), strict=False)
    model.eval()

    clips, encoded, waves = {}, {}, {}
    for name, (rate, count, salt) in CLIPS.items():
        source = np.array(pcm(count, salt), np.float32) / 32768
        wave = waves[name] = resampled(source, rate, 16000)
        port = d1_media.resampled(source, rate, 16000)
        assert len(port) == len(wave) and np.abs(port - wave).max() < 1e-5, name
        mel = features_of(wave)
        # mlx-vlm's float64 front end; float32 rounding shows in the near-silent bins of upsampled 8 kHz audio.
        assert np.abs(mlx_audio.log_mel(wave) - mel).max() < 2e-3, name
        frames = mel.shape[1]
        assert frames == mlx_audio.num_frames(len(wave))
        tower = model.audio_tower(mx.array(mel[None]), mx.array([frames]))
        features = model.audio_projector(tower)
        assert features.shape[0] == mlx_audio.num_encoder_tokens(frames)
        encoded[name] = features
        clips[name] = {"source": [rate, count, salt], "resampled": summary(wave), "frames": frames,
                       "mel": summary(mel.T), "features": summary(features.astype(mx.float32))}

    pictures = {}
    for name, (width, height, salt) in PICTURES.items():
        capped = cap_pixels(Image.fromarray(synthetic(width, height, salt)), MAX_PIXELS)
        _, pixels, grid = process(capped)
        pictures[name] = model.encode_images(mx.array(pixels), image_grid_thw=mx.array([grid]))

    videos, video_features = {}, {}
    for name, (width, height, salt, count, fps) in VIDEOS.items():
        indices, resized, pixels, grid, seconds = process_video(name)
        features = model.encode_images(mx.array(pixels), image_grid_thw=mx.array([grid]))
        tokens = grid[1] * grid[2] // 4
        assert features.shape[0] == grid[0] * tokens
        markup = "<|vision_start|>" + "".join(
            f"<{t:.1f} seconds><|vision_start|>{'<|video|>' * tokens}<|vision_end|>" for t in seconds) + "<|vision_end|>"
        video_features[name] = (features, markup)
        videos[name] = {"source": [width, height, salt, count, fps], "indices": indices,
                        "resized": list(resized.shape[1:3]), "resized_sha256": [sha(f) for f in resized],
                        "groups": grid[0], "tokens": tokens, "seconds": seconds, "markup": markup,
                        "features": summary(features.astype(mx.float32))}

    requests = []
    def spoken_markup(name):
        return f"<|vision_start|>{'<|audio|>' * encoded[name].shape[0]}<|vision_end|>"

    for state, clip, picture, named in REQUESTS:
        voices = {n: q["instructions"]["spoken"] for n, q in named if isinstance(q["instructions"], dict)}
        named = [(n, {**q, "instructions": {"type": "audio", "audio": waves[voices[n]], "sampling_rate": 16000}})
                 if n in voices else (n, q) for n, q in named]
        for _, q in named:
            check_question(q)
        questions = [as_question(q) for _, q in named]
        if clip is not None:
            state_media = [(AUDIO, encoded[clip])]
            markup = spoken_markup(clip)
        elif picture is not None and picture.startswith("video:"):
            features, markup = video_features[picture[len("video:"):]]
            state_media = [(VIDEO, features)]
        elif picture is not None:
            state_media = [(VISION_TOKENS["<|image|>"], pictures[picture])]
            markup = f"<|vision_start|>{'<|image|>' * pictures[picture].shape[0]}<|vision_end|>"
        else:
            state_media, markup = [], ""
        prefix = prefix_text(tokenizer, state, tokenizer.bos_token, DEFAULT_STATE_STYLE, DEFAULT_SYSTEM, markup)
        layouts = [suffix_layout(tokenizer, q, LEAD) for q in questions]
        # media.Reader.place: a spoken question's markup in place of SPOKEN.
        texts = [text.replace(SPOKEN, spoken_markup(voices[n])) if n in voices else text
                 for (text, _), (n, _) in zip(layouts, named)]
        header = question_header(tokenizer, state)
        rows, answers, read = [], [], 0
        for k0 in range(0, len(questions), SET_SIZE):
            group = questions[k0:k0 + SET_SIZE]
            suffixes = texts[k0:k0 + SET_SIZE]
            ids, state_len, question, ends = pack_set(
                tokenizer, prefix, suffixes, header, [o for _, o in layouts[k0:k0 + SET_SIZE]],
                tokenizer.encode(prefix + suffixes[0], add_special_tokens=False))
            read += len(ids)
            # The engine's row media: the state's, then each spoken question's, in row order.
            row_media = state_media + [(AUDIO, encoded[voices[n]]) for n, _ in named[k0:k0 + SET_SIZE] if n in voices]
            hidden = model.d1_hidden(ids, state_len, question, row_media)
            embed = model.language_model.model.embed_tokens
            for q, slots in zip(group, ends):
                tokens, forms, _ = readout_table(tokenizer, q)
                logits = np.array((hidden[mx.array(slots)] @ embed(mx.array(list(tokens))).T).astype(mx.float32))
                best = logits[:, forms].max(-1)[0].tolist()
                answers.append(answer(q, probabilities(q, best, CALIBRATION)))
            rows.append({"ids": ids, "state": state_len, "question": question, "ends": [e[0] for e in ends]})
        video = picture[len("video:"):] if picture is not None and picture.startswith("video:") else None
        picture = None if video is not None else picture
        requests.append({"clip": clip, "picture": picture, "video": video, "voices": voices, "rows": rows, "tokens": read,
                         "answers": [{"id": n, **a} for (n, _), a in zip(named, answers)]})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(OUTPUT / "model.safetensors"), {n: mx.array(v) for n, v in sorted(weights.items())})
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    (OUTPUT / "processor_config.json").write_text(json.dumps(processor, indent=2) + "\n")
    folder = OUTPUT / "mlx"
    folder.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(folder / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (folder / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    (folder / "processor_config.json").write_text(json.dumps(processor, indent=2) + "\n")
    out = {"format": "d1_omni_2b_audio_tiny", "version": 1, "metadata": VERSIONS, "set_size": SET_SIZE,
           "codes": CODES, "max_pixels": MAX_PIXELS,
           "calibration": {"default": CALIBRATION.default, "temperatures": CALIBRATION.temperatures},
           "special": Tokenizer.all_special_tokens, "added": Tokenizer.added, "words": WORDS,
           "clips": clips, "pictures": {n: list(v) for n, v in PICTURES.items()}, "videos": videos,
           "requests": requests}
    (OUTPUT / "fixture.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
