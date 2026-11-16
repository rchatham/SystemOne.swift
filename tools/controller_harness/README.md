# SystemOne controller harness

A dependency-free Python harness for testing SystemOne models as tool controllers.
Runs locally or on a worker against a model HTTP server. Tool actions and text
responses are simulated: no real notes or calendar events are created.

## Run a real model

Requires Python 3.10+ and a reachable SystemOne-compatible server. The existing
Swift server runs on an Apple Silicon host; this harness calls it over HTTP.
A Linux Python reference server can also be used if it implements the same API.
This package does not install or serve model weights.

```bash
cd tools/controller_harness
python harness.py \
  --endpoint https://YOUR-SERVER/v1/systemone \
  --model-id 'd1-omni-600M REVISION float32' \
  --repeats 5 --out results/d1-structured-shuffled
```

Set `SYSTEMONE_API_KEY` in the environment if the endpoint requires authentication.
The harness reads it for Bearer authentication and never writes it to results.
Endpoint URLs should not contain credentials. Avoid real personal data in cases:
traces intentionally contain complete requests and responses.

Exit codes: 0 = all cases pass; 1 = behavioral failures; 2 = infrastructure errors
or invalid CLI arguments. HTTP timeouts are 60 seconds per call, with no retries
that could hide errors. Output directories cannot overwrite prior traces.

## Diagnose inconsistent routing

From `tools/controller_harness/`, run these with the same model revision/precision,
suite, and repeat count:

```bash
# Identical options across repeated calls: run-to-run variability
python harness.py --endpoint https://YOUR-SERVER/v1/systemone \
  --model-id YOUR-MODEL --fixed-order --repeats 5 --out results/fixed

# Deterministically shuffled options: position sensitivity plus variability
python harness.py --endpoint https://YOUR-SERVER/v1/systemone \
  --model-id YOUR-MODEL --repeats 5 --out results/shuffled

# Remove prior messages but retain tool observations: history ablation
python harness.py --endpoint https://YOUR-SERVER/v1/systemone \
  --model-id YOUR-MODEL --framing latest --repeats 5 --out results/latest

# Focus on post-tool mistakes independently of earlier model errors
python harness.py --endpoint https://YOUR-SERVER/v1/systemone \
  --model-id YOUR-MODEL --family post_tool --out results/post-tool
```

Latest-message framing deliberately removes information needed by some follow-up
cases. Compare families and individual failures, not just aggregate scores.
Different orderings across seeds do not alone prove stochastic inference.

## What is evaluated

The controller sees messages, prior action observations, current time, and the
available actions. It chooses one action per step via a `choice` question. A fresh
simulated world is created for each episode. Tool results are appended to state,
then the controller chooses again until `finish` or the step limit.

Actions: current time, search/save/list notes, calendar read/create, calculator,
web search, generate text, clarify, and finish. Calculator supports only the fixed
fixture expressions; web search and generation return explicitly simulated data.

83 human-reviewable synthetic cases live in `cases.jsonl`; rebuild them with
`python build_cases.py`. Families include paraphrases, negation/quoted commands,
multiple tasks, history corrections/references, long distractors, empty search,
post-tool continuation, termination, and instruction-like text inside tool results.
Post-tool probes start from synthetic successful observations, localizing the next
decision independently of whether the model could reach that state itself.

Expected action paths are never passed to the model. Fixture arguments are supplied
only at execution time; they are intentionally oracle arguments. This measures
routing and loop control, NOT argument extraction, generated response quality,
real tool reliability, multimodal inputs, or end-to-end production readiness.
The fixed menu is intentionally broad to expose confusions; tool availability
filtering is an experiment to add later, not silently applied here.

Scoring uses an exact expected action sequence, including a final response and
finish. Multi-task cases explicitly request first-mentioned task order. A valid
alternative plan can therefore fail: review traces before interpreting failures
as model limitations. The small synthetic suite is a development diagnostic, not
an independently held-out benchmark for publishing general capability claims.

## Outputs

- `metadata.json`: model label, endpoint, suite/code hashes, framing, seeds, mode.
- `traces.jsonl`: each decision, offered option order, full request/response,
  probability distribution when provided, tool result, latency and expected action.
- `summary.json`: exact-path success, per-family results, repeated-path inconsistency,
  expected/actual action confusions, failure categories and median decision latency.
- `report.md`: failed chat case IDs and observed versus expected paths.

Infrastructure failures are reported separately and excluded from behavioral rate
calculations. Confidence is not assumed comparable across backends. Latency is
HTTP round-trip decision latency, not device kernel latency. There is no automatic
claim that a high score generalizes to unseen chats.

## Validate without a model

From the repository root:

```bash
cd tools/controller_harness
python -m unittest -v
python harness.py --self-test --repeats 3 --out results/my-self-test
```

The test double reads the expected path. This is deliberately labeled
`INFRASTRUCTURE_SELF_TEST`, never model evidence. Unit tests inject wrong tools,
loops, early stopping, malformed probabilities, endpoint errors and position-biased
choices to check failure detection. The HTTP contract test mocks transport so it runs without a listening socket or
model server.

Generated results are ignored and are not committed. The three-repeat self-test
runs 249 scripted episodes across 83 cases. These checks validate harness
infrastructure only; they do not evaluate actual model weights. Run against a real
endpoint to measure model behavior.

The `Controller harness` CI workflow runs the unit tests and scripted self-test
without dependencies, model weights, credentials, or a server.

## Next experiment

Use the same case file and seed for base and edited checkpoints, giving each a
unique revision/precision label and output directory. Inspect confusion pairs and
post-tool cases first. Build a separate held-out corpus before tuning prompts or
weights against this suite. Add actual argument construction and generated-answer
grading as separate stages so routing errors remain distinguishable.
