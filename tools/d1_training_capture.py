#!/usr/bin/env python3
"""Generate and verify head-only training golden vectors for d1-omni-600M.

The reference ships no training code, so the loss is defined here: the mean
over rows of cross-entropy on the head's raw logits (before any calibration
temperature), with padded option slots masked to -1e4 as `DecisionHead`
masks them. The trunk runs without gradient; every decision-head parameter
requires one. One plain gradient-descent step checks that the update lowers
the loss.

    .venv-d1/bin/python tools/d1_training_capture.py            # write fixture
    .venv-d1/bin/python tools/d1_training_capture.py --check    # verify fixture
"""

from __future__ import annotations

import argparse
import json
import sys

import d1_golden_capture as golden

SCHEMA_VERSION = 1
FIXTURE_PATH = golden.REPO_ROOT / "testdata" / "d1_training_golden_v1.json"
LEARNING_RATE = 1e-3

TRIAGE_STATE = "I was charged twice this month, please refund one of them."
TRIAGE_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                          "fraud": "Suspected unauthorised use"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"]},
}
# Correct option per question: refund true, team fraud (not the model's pick), urgency today.
TRIAGE_TARGETS = [1, 2, 1]


def probe_indices(count: int) -> list[int]:
    """Flat positions every gradient is sampled at, in ascending order."""
    return sorted({0, 1, 7, min(1023, count - 1), count // 3, count // 2, count - 2, count - 1} & set(range(count)))


class TrainingOracle(golden.Oracle):
    def head_inputs(self, state, questions: dict) -> tuple:
        """Frozen trunk output and head inputs for one text request, batched."""
        torch, model = self.torch, self.model
        rows = []
        for q in (self.prompt.as_question(q) for q in questions.values()):
            ids, markers = self.prompt.encode(self.tok, state, q, model.config.max_length, None, False)
            rows.append((ids, markers, q))
        with torch.no_grad():
            seqs = [model.encoder.embed_tokens(torch.tensor(ids)) for ids, _, _ in rows]
            h = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
            lengths = torch.tensor([len(ids) for ids, _, _ in rows])
            pad = torch.arange(h.shape[1])[None] < lengths[:, None]
            text = model.encoder(h, pad, torch.zeros(len(rows), dtype=torch.long))
        k = max(len(markers) for _, markers, _ in rows)
        marker_pos = torch.zeros(len(rows), k, dtype=torch.long)
        marker_mask = torch.zeros(len(rows), k, dtype=torch.bool)
        for i, (_, markers, _) in enumerate(rows):
            marker_pos[i, :len(markers)] = torch.tensor(markers)
            marker_mask[i, :len(markers)] = True
        qtype = torch.tensor([self.prompt.QTYPES[q.type] for _, _, q in rows])
        return rows, (text, pad, marker_pos, marker_mask, qtype)

    def loss(self, inputs: tuple, targets) -> tuple:
        logits = self.model.head(*inputs)
        return logits, self.torch.nn.functional.cross_entropy(logits, targets)

    def head_step_case(self) -> dict:
        torch, head = self.torch, self.model.head
        rows, inputs = self.head_inputs(TRIAGE_STATE, TRIAGE_QUESTIONS)
        targets = torch.tensor(TRIAGE_TARGETS)
        head.zero_grad(set_to_none=True)
        logits, loss = self.loss(inputs, targets)
        loss.backward()

        gradients = []
        for name, parameter in head.named_parameters():
            flat = parameter.grad.reshape(-1)
            indices = probe_indices(flat.numel())
            gradients.append({
                "name": "head." + name,
                "shape": list(parameter.shape),
                "norm": flat.double().norm().item(),
                "sum": flat.double().sum().item(),
                "indices": indices,
                "values": flat[indices].tolist(),
            })

        original = {name: p.detach().clone() for name, p in head.named_parameters()}
        with torch.no_grad():
            for parameter in head.parameters():
                parameter -= LEARNING_RATE * parameter.grad
            _, stepped = self.loss(inputs, targets)
            for name, parameter in head.named_parameters():
                parameter.copy_(original[name])
        head.zero_grad(set_to_none=True)

        return {
            "id": "triage",
            "request": golden.ordered_request(TRIAGE_STATE, TRIAGE_QUESTIONS),
            "rows": [{"ids": ids, "markers": markers, "qtype": self.prompt.QTYPES[q.type]} for ids, markers, q in rows],
            "targets": TRIAGE_TARGETS,
            "logits": [z[:len(r[1])].tolist() for z, r in zip(logits.detach(), rows)],
            "loss": loss.item(),
            "gradients": gradients,
            "learning_rate": LEARNING_RATE,
            "loss_after_step": stepped.item(),
        }

    def generate(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "model": {"repo": golden.HF_REPO, "revision": golden.HF_REVISION,
                      "artifact_hashes": golden.ARTIFACT_HASHES},
            "runtime": golden.EXPECTED_RUNTIME_VERSIONS,
            "head_step": [self.head_step_case()],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare against the committed fixture")
    args = parser.parse_args(argv)
    snapshot = golden.snapshot_path()
    golden.validate_inputs(snapshot)
    generated = TrainingOracle(snapshot).generate()
    if args.check:
        expected = json.loads(FIXTURE_PATH.read_text())
        errors = golden.compare_json(json.loads(golden.canonical_json_bytes(generated)), expected)
        for error in errors[:50]:
            print(error, file=sys.stderr)
        print("fixture matches" if not errors else f"{len(errors)} mismatches")
        return 1 if errors else 0
    FIXTURE_PATH.write_bytes(golden.canonical_json_bytes(generated))
    print(f"wrote {FIXTURE_PATH.relative_to(golden.REPO_ROOT)} ({FIXTURE_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
