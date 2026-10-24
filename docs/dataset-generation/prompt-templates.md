# Prompt templates

Override generation framing without editing Python or changing the built-in defaults:

```yaml
generation:
  system_prompt: You create a CLI risk-classification dataset.
  topic_sample_mode: bulk
  shared_scope: full
  prompt_templates:
    full_sample:
      file: prompts/bulk-full.txt
```

`prompts/bulk-full.txt` (relative to the YAML file):

```text
{{ system_prompt }}
{{ instructions }}
{{ examples }}
{{ subtopics }}

Generate {{ requested }} distinct, complete JSON records.
Return an object with an "items" list matching the supplied schema.
{{ history_section }}
```

For the diverse-versus-distinct experiment, keep everything else identical and
change only that word in two versioned template files. The built-in templates
are packaged as text files under `deepfabric/templates/`; copy the applicable
template rather than changing installed defaults. Replace its
`{{ request_instruction }}` line with your experimental instruction.

## Steps and sources

`generation.prompt_templates` accepts four independent overrides:

| Key | Used for |
|---|---|
| `full_sample` | Independent samples, the first sequential sample, and bulk full-record collections; also the fallback for later sequential samples |
| `sequential_later` | Sequential samples after the first slot within each topic, in either sharing scope |
| `seed` | Bulk seed collections, including conversation user-request seeds |
| `completion` | Bulk completion of a pinned custom-schema seed or conversation user request |

An omitted override retains the built-in template selected for that mode and
output format. An override replaces that step's framing; include the context
and history placeholders you want the model to receive. The response schema,
seed pinning, validation, retries and output assembly remain controlled by code.

### Positioning sequential history

Set `sequential_later` to control later prompts independently. For example:

```yaml
generation:
  topic_sample_mode: sequential
  shared_scope: full
  prompt_templates:
    sequential_later:
      file: prompts/sequential-later.txt
```

`prompts/sequential-later.txt` can place history first:

```text
{{ history_section }}
{{ system_prompt }}
{{ instructions }}
{{ examples }}
{{ subtopics }}

{{ output_instructions }}
```

Move `{{ history_section }}` elsewhere to test another position, or use
`{{ history }}` for the JSON array with your own framing. The first sample
continues to use `full_sample` or the built-in sequential prompt. Later calls
fall back to `full_sample` (including the legacy scalar alias), then the built-in
prompt, if no `sequential_later` override is supplied.

Selection follows the sample's ordinal within its topic, including resume.
If earlier slots failed, a later call still uses `sequential_later`, but its
history may be empty. History contains only earlier successful siblings: seed
values in seed scope, complete records in full scope. Both previews and reports
include the new override; use `--verbose` to preview the later-call case.

Each value is either inline text, `{text: ...}`, or `{file: ...}`. Files must be
UTF-8. Relative paths resolve against the YAML config directory; direct Python
configuration resolves them against the working directory. Sources are loaded
and validated when configuration is created, so editing a file mid-run does
not change subsequent requests. Empty text, unknown steps, unknown placeholders,
missing files and conflicting text/file sources fail before generation.

The formerly ignored `prompt_template` scalar is now a supported shorthand for
`prompt_templates.full_sample`, in YAML and Python. It cannot be combined with
an explicit `full_sample` override. Legacy `{{{{ name }}}}` placeholders in
that scalar are accepted alongside the syntax below.

## Placeholder contract

Use `{{ name }}`. Substitution is literal, single-pass text replacement; it does
not evaluate expressions, Python, loops or conditionals. JSON braces and Docker
Go-template fields such as `{{.ID}}` remain literal. Inserted values are not
parsed again, so braces inside instructions, examples or seed data stay intact.

| Placeholder | Meaning |
|---|---|
| `system_prompt` | Generation prompt for full samples/seeds; dataset prompt for completion |
| `generation_system_prompt` | Text from `generation.system_prompt` |
| `dataset_system_prompt` | `output.system_prompt`, falling back to the generation prompt |
| `instructions` | Formatted configured instructions, or empty text |
| `examples` | Formatted selected examples, or empty text |
| `topic_path` | Raw full root-to-target ancestry joined with ` -> `, including internal nodes |
| `subtopics` | Existing natural-language topic framing; empty on completion unless `completion_includes_topic_path` is enabled |
| `tools` / `tools_section` | Tool descriptions / their formatted section; empty without tools |
| `requested` | Number requested in this call: one for individual samples/completions, remaining quota for bulk |

Full-sample, sequential-later and seed templates additionally accept:

| Placeholder | Meaning |
|---|---|
| `request_instruction` | Default request sentence, including count and singular/plural wording |
| `output_instructions` | Output guidance for the selected step: seed-only collections or complete records/conversations |
| `record_instructions` | Record/conversation guidance used by the built-in bulk full template |
| `history` | JSON array of earlier successful records or seed values; `[]` when empty |
| `history_section` | Formatted history/accepted-sibling section; empty on the first call |
| `accepted_siblings_section` | Bulk accepted-sibling section; empty outside bulk or without accepted siblings |
| `seed_field` | JSON-quoted configured seed-field name, or empty without one |
| `scope` | Configured `seed` or `full` sharing scope |

Completion templates additionally accept:

| Placeholder | Meaning |
|---|---|
| `seed` | JSON-encoded pinned seed value/user request |
| `seed_field` | JSON-quoted custom-schema seed-field name, or empty for conversations |
| `seed_section` | Default pinned-seed/user-request section; empty where the builder pins the request separately |
| `context_guidance` | Default guidance about supplied topic context versus pinned-seed context |
| `output_instructions` | Default completion-output guidance |
| `retry_feedback` | Current custom-schema validation feedback, or empty |

Unavailable step-specific placeholders are errors. Optional values render as
documented empty strings; no sections are automatically appended to a custom
template. `topic_path` deliberately exposes ancestry even when completion's
default `subtopics` framing is disabled: including it explicitly opts into that
context without changing the compatibility flag.

Templates change text, not API message roles. `generation.system_prompt` remains
inline prompt content; this feature does not add API system-role delivery.
`output.system_prompt` and `output.include_system_message` continue to control
exported training messages. Conversation/agent builders retain their existing
request stages and message roles; a completion override customizes their supplied
scenario framing rather than replacing the entire tool loop.

## Topic generation and expansion

`topics.prompt_templates` works for trees and all graph `prompt_style` values:

```yaml
topics:
  prompt: CLI risk classification
  mode: graph
  prompt_style: anchored
  prompt_templates:
    initial:
      file: prompts/topic-initial.txt
    expansion:
      file: prompts/topic-expansion.txt
```

`initial` generates the children of the supplied root; `expansion` generates
children of deeper nodes. The root text comes from `topics.prompt`, rather than
a model call. Each omitted override keeps its own built-in prompt; `initial`
is not a fallback for `expansion`. Sources use the same inline/file formats,
validation and file snapshots as sample templates.

For example, `prompts/topic-expansion.txt`:

```text
{{ system_prompt }}
Expand this topic path: {{ topic_path }}
Generate {{ num_subtopics }} focused subtopics matching the supplied schema.
Use empty connections lists for this anchored graph.
```

Topic templates accept these placeholders (sample-only placeholders are unavailable):

| Placeholder | Meaning |
|---|---|
| `system_prompt` | `topics.system_prompt`, inline |
| `topic_prompt` | Supplied root text |
| `current_topic` | Text of the node being expanded |
| `topic_path` | Full root-to-node ancestry joined with ` -> `; graphs follow first parents |
| `num_subtopics` / `requested` | Requested number of children |
| `examples` | Formatted domain examples from the applicable tree/graph builder |
| `current_graph_summary` | Graph JSON, or the existing size-limit message; empty for trees |
| `mode` | `tree` or `graph` |
| `prompt_style` | Graph's `default`, `isolated` or `anchored`; `tree` for trees |
| `default_prompt` | Complete built-in prompt for this node and style |

To keep the original framing, use `{{ default_prompt }}` and add your own
instructions before or after it. A full replacement must include any desired
context and connection guidance itself. Graph `default` prompts normally permit
cross-connections; `isolated` and `anchored` prompts request empty connections
lists. Templates do not change the response schema or graph processing.

Preview both stages without model calls:

```sh
uv run deepfabric generate config.yaml --topic-only --dry-run --trace-prompt --verbose
```

`--verbose` includes a deeper expansion when depth is greater than one.
`--topics-load` skips topic generation and its templates. Run reports save
resolved topic overrides under `topic_prompt_templates` alongside
sample-template metadata.

## Inspect and reproduce

```sh
uv run deepfabric generate config.yaml --topics-load topics.json --dry-run --trace-prompt --verbose --trace-prompt-file preview.prompts.jsonl
```

Use a fresh trace filename because tracing appends. Preview and real generation
use the same template selection and rendering. Tracing records final requests,
including role-bearing messages at boundaries that use message lists. Enable
`--run-report` to save resolved overrides and content hashes for both overrides
and built-in template assets; a config hash alone cannot capture file edits.
Keep the YAML, template files, topic graph and code revision with each experiment.
Resume warns if resolved templates differ from the checkpoint; use separate
outputs and checkpoints for different template variants.
