# Topic sample modes test guide

## Quick usage

Run from the repository root. Live commands require an API key in the environment
and a structured-output-capable model; replace `your-model-name` below.

```sh
# Offline scaling checks: no credentials or services required.
.venv/bin/python -m pytest tests/component/test_topic_sample_modes_scaling.py \
  --junitxml=/tmp/topic-modes-scaling.xml -o junit_family=xunit1

# Live provider checks (real API requests; OPENAI_API_KEY must be set).
export DEEPFABRIC_TOPIC_MODES_LIVE=1
export DEEPFABRIC_TOPIC_MODES_PROVIDERS=openai
export OPENAI_TEST_MODEL='your-model-name'
.venv/bin/python -m pytest tests/integration/test_topic_sample_modes_integration.py \
  -k 'public_modes or close_stream' -v

# Live agent checks: also requires a running Spin service with builtin VFS tools.
export DEEPFABRIC_TOPIC_MODES_SPIN=1
export SPIN_ENDPOINT=http://localhost:3000
.venv/bin/python -m pytest tests/integration/test_topic_sample_modes_integration.py \
  -k agent_with_real_spin -v

# Diversity comparison: writes artifacts for manual quality review.
export DEEPFABRIC_TOPIC_MODES_DIVERSITY=1
export DEEPFABRIC_TOPIC_MODES_ARTIFACT_DIR=/tmp/topic-mode-comparisons
.venv/bin/python -m pytest tests/integration/test_topic_sample_modes_integration.py \
  -k diversity_comparison -v
```

These checks are independent of `prompt-scenarios.csv`, whose expectations are
being reviewed. Public bulk-full generation remains excluded until the separate
truncation fix lands.

## Offline scaling checks

```sh
.venv/bin/python -m pytest tests/component/test_topic_sample_modes_scaling.py \
  --junitxml=/tmp/topic-modes-scaling.xml -o junit_family=xunit1
```

The cases exercise a 500-record single-topic quota and interruption/resume across
60 topics with four records each. They assert bounded workers and requests,
fixed topic groups, exact identities/counts, preservation of saved rows, and
completed-checkpoint no-op behavior. Provider I/O is replaced; scheduling,
assembly, JSONL scans, and checkpoint writes are real. Bulk-full is exercised
only through the internal path already covered by the component acceptance suite.

JUnit properties report request/worker peaks, checkpoint scan counts and rows
read, and elapsed seconds. These are diagnostic measurements, not timing or
memory benchmarks. In particular, saved sibling lookup can repeatedly scan
checkpoint rows; the test reports that cost without asserting a promised
complexity or an unstable wall-clock threshold.

## Live provider checks

Live execution requires explicit opt-in, provider credentials, and an explicitly
chosen model that supports the tested structured-output shapes. With no opt-in,
every case in this file skips before creating a provider client.

```sh
export DEEPFABRIC_TOPIC_MODES_LIVE=1
export DEEPFABRIC_TOPIC_MODES_PROVIDERS=openai
export OPENAI_TEST_MODEL=<your-structured-output-model>
# Set OPENAI_API_KEY through your usual secret-management mechanism.
.venv/bin/python -m pytest tests/integration/test_topic_sample_modes_integration.py \
  -k 'public_modes or close_stream' -v
```

The provider list accepts `openai,gemini,anthropic,openrouter,ollama`. Each selected
provider needs its corresponding `<PROVIDER>_TEST_MODEL`; credentials use the
existing provider environment variables (Gemini accepts `GOOGLE_API_KEY` or
`GEMINI_API_KEY`). Ollama needs a reachable configured service and local model.
Unselected providers or missing model/credential prerequisites skip.

The public mode matrix covers independent, bulk seed, sequential seed, and
sequential full, with basic, CoT, and custom-schema output. Checks include exact
counts, schema validity, topic allocations, unchanged configured output-token
limits, and completed-checkpoint no-op resumes. Separate stream-close/resume
cases preserve previously saved custom records. They do not inject provider
truncation, rate-limit responses, or checkpoint faults; deterministic component
tests cover fault injection, and truncation handling remains separate work.

`DEEPFABRIC_TOPIC_MODES_MAX_TOKENS` sets the per-request output budget (default
2,000). Sample validation allows two retries and provider backoff allows one;
these checks can make additional requests on failure. Use `-k` to select a small
slice before enabling a full provider matrix.

## Live Spin checks

```sh
export DEEPFABRIC_TOPIC_MODES_SPIN=1
export SPIN_ENDPOINT=http://localhost:3000
.venv/bin/python -m pytest tests/integration/test_topic_sample_modes_integration.py \
  -k agent_with_real_spin -v
```

These additionally require the live provider settings above and a Spin service
supporting the builtin VFS `list_files` tool. They use real provider/Spin calls,
observe execution and cleanup without replacing service responses, and check
fresh sessions, cleanup success, and matching transcript tool-call/result IDs.

## Diversity comparison artifacts

```sh
export DEEPFABRIC_TOPIC_MODES_DIVERSITY=1
export DEEPFABRIC_TOPIC_MODES_ARTIFACT_DIR=/tmp/topic-mode-comparisons
export DEEPFABRIC_TOPIC_MODES_SAMPLES=12
export DEEPFABRIC_TOPIC_MODES_QUOTA=4
.venv/bin/python -m pytest tests/integration/test_topic_sample_modes_integration.py \
  -k diversity_comparison -v
```

Each selected coordinated mode is compared with independent generation across
one topic per sample, using the same provider, model, persona, schema, temperature,
and target. The target must be divisible by a quota greater than one. The default
synthetic Python-learning topics and request/answer schema make the comparison
repeatable; they do not replace representative user-selected topics and schemas.

Timestamped JSON artifacts retain configuration, topic paths, rows and slot
identities, elapsed time, adapter-call counts and prompt character counts, and
normalized request uniqueness. Adapter calls do not count underlying HTTP
backoff retries; character counts do not measure tokens or monetary cost. Prompt
bodies are not captured by this suite; separate prompt tracing is still pending.

Passing verifies structural correctness and produces review material. It does
**not** accept the diversity claim: inspect semantic coverage, repetition,
factual correctness, and usefulness against the quota=1 workaround. Repeat with
representative inputs and multiple runs before reaching a quality conclusion.

Manual Rich/simple terminal inspection, real prompt inspection, actual token/cost
measurements, and public bulk-full acceptance remain separate verification.
