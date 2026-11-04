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

`prompts/bulk-full.txt` (relative to the YAML file) contains:

```text
{{ generation_system_prompt }}
{{ instructions }}
{{ examples }}
{{ subtopics }}

Generate {{ num_items }} distinct, complete JSON records.
Return an object with an "items" list matching the supplied schema.
{{ accepted_siblings_section }}
```

For the diverse-versus-distinct experiment, keep everything else identical and
change only that word in two versioned template files. The built-in templates
are packaged as text files under `deepfabric/templates/`; copy the applicable
template rather than changing installed defaults. Replace its
`{{ request_instruction }}` line with your experimental instruction.

## Steps and sources

Custom non-tool conversation templates supply the complete prompt; include any required
message/CoT guidance yourself. Retry feedback, schema validation and exported system messages
remain managed by DeepFabric. Omitted overrides keep the built-in framing.

Put template settings in the same YAML file used by `deepfabric generate`.
The `file` paths below are examples, not required filenames; paths are relative
to that YAML file. Inline text is also supported.

| Stage / use | Configuration key | Example template file |
|---|---|---|
| Expand the topic root | `topics.prompt_templates.initial` | `prompts/topic-initial.txt` |
| Expand deeper topics | `topics.prompt_templates.expansion` | `prompts/topic-expansion.txt` |
| Independent sample or first sequential sample | `generation.prompt_templates.full_sample` | `prompts/full-sample.txt` |
| Later sequential samples | `generation.prompt_templates.sequential_later` | `prompts/sequential-later.txt` |
| Bulk seed/user-request collection | `generation.prompt_templates.seed` | `prompts/seed.txt` |
| Bulk non-tool seed completion | `generation.prompt_templates.completion` | `prompts/completion.txt` |
| Bulk full-record collection | `generation.prompt_templates.full_sample` | `prompts/bulk-full.txt` |

For tool-call conversations, the sample and seed settings customize the user
request, as described below. Bulk/full does not support tool-call conversations.

An omitted override retains the built-in template selected for that mode and
output format. An override replaces that step's framing; include the context
and history placeholders you want the model to receive. The response schema,
seed pinning, validation, retries and output assembly remain controlled by code.

### Tool-call conversations

The agent first obtains a user request, chooses actions using structured LLM
output, executes tools through Spin, and generates a final assistant reply.
Templates customize the request-generation stage, not every call in that flow.

| Agent stage | Template setting or behavior |
|---|---|
| Independent request / first sequential request | `generation.prompt_templates.full_sample` supplies the scenario to a fixed question-generation wrapper |
| Later sequential request | `generation.prompt_templates.sequential_later`; falls back to `full_sample`, then the built-in prompt |
| Bulk user-request collection | `generation.prompt_templates.seed` supplies the collection prompt |
| Start an agent for a collected request | The collected request is pinned; no new question-generation call |
| Choose the next action / tool calls | Fixed agent-builder prompt; no template override key |
| Execute tools | Spin execution; no LLM template |
| Write the final assistant reply | Fixed agent-builder prompt using the dataset system prompt; no template override key |

A bulk tool conversation can accept and render a `completion` override, but the
agent uses its pinned request and fixed action/final-reply builders instead of
that rendered prompt. `generation.prompt_templates.completion` therefore does
not customize those tool-agent calls. It controls non-tool conversation and
custom-schema completions.

The fixed question wrapper still surrounds `full_sample` and `sequential_later`
scenario text in tool mode. Bulk `seed` overrides supply the collection prompt
directly. Including `raw_tools`, `tools`, or `tools_section` chooses how tool
information appears in that template; it does not change execution routing.

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
{{ sequential_history }}
{{ generation_system_prompt }}
{{ instructions }}
{{ examples }}
{{ subtopics }}

{{ output_instructions }}
```

Move `{{ sequential_history }}` elsewhere to test another position, or use
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

Use `{{ name }}`. Substitution is literal and single-pass: no expressions, Python,
loops, or conditionals. JSON braces and Docker Go-template fields such as
`{{.ID}}` remain literal.

All placeholders render as text. Array and object values become JSON with
Unicode preserved; `history` and `seed` already contain JSON text in the current
implementation. Integers become their text value. Whitespace in strings is
preserved. Inserted data is never parsed again as template syntax.

**Run time:** in a Source cell means the value is produced or selected during
generation. **When empty** covers missing, empty, and unavailable values; it is
not a universal blanking rule. Empty replacements leave surrounding template
text in place. Conditional sections are not supported yet.

The raw and framed tables keep two columns visible. Use the buttons above each
table to show Type, related fields, or When empty details.

### Sample field availability

All six new raw fields (`raw_instructions`, `raw_examples`, `raw_tools`,
`raw_seed_field`, `raw_retry_feedback`, and `topic_path_json`) are accepted in all
four sample stages. Existing fields retain the following stage availability:

| Stages | Existing fields |
|---|---|
| All four | `generation_system_prompt`, `dataset_system_prompt`, `instructions`, `examples`, `subtopics`, `topic_path`, `tools`, `tools_section`, `num_items`, `seed_field`, `output_instructions` |
| `full_sample`, `sequential_later`, `seed` | `additional_instructions`, `request_instruction`, `record_instructions`, `history`, `sequential_history`, `accepted_siblings_section`, `scope` |
| `completion` | `seed`, `seed_section`, `context_guidance`, `retry_feedback` |

Accepted does not mean populated in every call. In particular,
`raw_retry_feedback` currently receives errors only on custom-schema bulk
completion retries. Unknown or unavailable stage-specific names are validation
errors.

### Raw and content-only sample fields

These fields expose data without added introductory prose or tags.

<div class="column-toggle-table" data-toggle-columns="3,4,5" data-hidden-columns="3,4,5" markdown="1">

| Placeholder | Source | Type | Related framed field | When empty |
|---|---|---|---|---|
| `generation_system_prompt` | `generation.system_prompt` | Text | — | Blank if empty |
| `dataset_system_prompt` | `output.system_prompt`; fallback: `generation.system_prompt` | Text | — | Empty output prompt → fallback |
| `topic_path` | Run time: full target ancestry, joined with ` -> ` | Text | `subtopics` | Missing or empty → blank |
| `num_items` | Run time: number of output items to generate in this call | Integer | `request_instruction` | Normally positive; `0` stays `0` |
| `history` | Run time: earlier successes for this topic | JSON text | `sequential_history`, `accepted_siblings_section` | No history → `[]` |
| `scope` | `generation.shared_scope` (`seed` or `full`) | Text | `sequential_history`, `request_instruction` | Always set |
| `seed` | Run time: pinned seed or user request | JSON text | `seed_section` | Preserves `""`, `[]`, or `null` |
| `raw_instructions` | `generation.instructions` | Text | `instructions`, `additional_instructions` | `None` / `""` → blank; preserve whitespace |
| `raw_examples` | Run time: the same selected sample examples as `examples` | Array | `examples` | Absent / disabled → blank; empty collection → `[]` |
| `raw_tools` | Run time: resolved `ToolDefinition` dictionaries | Array | `tools`, `tools_section` | No resolved tools → blank |
| `raw_seed_field` | `generation.seed_field` | Text | `seed_field` | Unset → blank |
| `raw_retry_feedback` | Run time: previous custom-schema bulk-completion validation error | Text | `retry_feedback` | First attempt / other paths → blank |
| `topic_path_json` | Run time: full target ancestry as an array | Array | `subtopics` | Missing → blank; empty → `[]` |

</div>

### Framed and formatted sample fields

These convenience fields supply wording, tags, labels, or field-name quoting.
Their Meaning cells describe the purpose rather than reproducing full prompts.

<div class="column-toggle-table" data-toggle-columns="3,4" data-hidden-columns="3,4" markdown="1">

| Placeholder | Meaning | Related raw field | When empty |
|---|---|---|---|
| `instructions` | Formatted additional instructions | `raw_instructions` | `None` / `""` → blank; whitespace keeps framing |
| `additional_instructions` | Additional instructions with a label | `raw_instructions` | Empty → blank; preserve whitespace |
| `examples` | Selected examples with headings and labels | `raw_examples` | Absent / disabled → blank; empty collection keeps headings |
| `subtopics` | Target ancestry with introductory wording | `topic_path`, `topic_path_json` | Missing / completion ancestry off → blank; empty list keeps introduction |
| `tools` | Readable tool names, descriptions, and parameters | `raw_tools` | No resolved tools → blank |
| `tools_section` | Formatted tools inside a tagged section | `raw_tools` | No resolved tools → blank |
| `seed_field` | JSON-quoted seed-field name | `raw_seed_field` | Unset → blank by default; bulk encoding differs |
| `seed_section` | Pinned seed or user-request section | `seed` | Empty seed keeps framing; omitted in agent stages |
| `sequential_history` | Earlier sequential successes with repetition guidance | `history` | No history / outside sequential → blank |
| `accepted_siblings_section` | Accepted bulk outputs with repetition guidance | `history` | No accepted outputs / outside bulk → blank |
| `retry_feedback` | Validation error with retry label | `raw_retry_feedback` | First attempt / conversation completion → blank |

</div>

### Generated sample instructions

These values are generated guidance, rather than raw source data.

| Placeholder | Purpose |
|---|---|
| `request_instruction` | Request seeds or complete items, with count-specific wording |
| `output_instructions` | Required output for this generation stage |
| `record_instructions` | Record or conversation structure guidance |
| `context_guidance` | How completion should use ancestry or its pinned seed |

`generation_system_prompt` and `dataset_system_prompt` keep the same meaning
in every stage. The `system_prompt` name is not accepted in sample templates.
Built-in sample/seed templates use the generation prompt; completion templates
use the dataset prompt, which falls back to the generation prompt.

`topic_path` and `topic_path_json` expose complete ancestry, including interior
graph targets and canonical first-parent ancestry. They remain available when
`completion_includes_topic_path` suppresses completion's `subtopics` framing.

`raw_tools` uses DeepFabric's resolved `ToolDefinition` dictionary shape,
including parameter definitions, returns, category, and component. It is not a
provider-specific tool API schema. Raw and framed examples share one selection.

`sequential_history` includes scope in its tag and uses scope to choose its
description. `request_instruction` uses scope to choose wording, without
inserting the literal scope value. `history_section` is not an accepted name.

Templates change prompt text, not API message roles. `output.system_prompt` and
`output.include_system_message` continue to control exported training messages.
Built-in prompt output is unchanged by choosing a custom template's raw fields.

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

### Raw and content-only topic fields

<div class="column-toggle-table" data-toggle-columns="3,4,5" data-hidden-columns="3,4,5" markdown="1">

| Placeholder | Source | Type | Related framed field | When empty |
|---|---|---|---|---|
| `system_prompt` | `topics.system_prompt` | Text | — | Empty → blank; preserve whitespace |
| `topic_prompt` | `topics.prompt` (configured root text) | Text | — | Root prompt is required |
| `current_topic` | Run time: label of the node being expanded | Text | — | Empty label → blank |
| `topic_path` | Run time: root-to-node ancestry, joined with ` -> ` | Text | — | Empty path → blank |
| `num_subtopics` | Run time: number of children to generate | Integer | — | `0` stays `0` |
| `num_items` | Run time: number of children to generate | Integer | — | `0` stays `0` |
| `prompt_style` | `topics.prompt_style` for graphs; `tree` for trees | Text | — | Always set |
| `mode` | `topics.mode` (`graph` or `tree`) | Text | — | Always set |
| `raw_examples` | Run time: the same selected domain examples as `examples` | Array | `examples` | Empty selection → `[]` |
| `topic_path_json` | Run time: root-to-node ancestry as an array | Array | — | Empty path → `[]` |
| `raw_graph` | Run time: complete graph snapshot; graph mode only | Object | `current_graph_summary` | Tree mode → blank |
| `raw_tree` | Run time: recorded tree snapshot; tree mode only | Object | — | Graph mode → blank; empty paths stay in snapshot |

</div>

### Framed and formatted topic fields

<div class="column-toggle-table" data-toggle-columns="3,4" data-hidden-columns="3,4" markdown="1">

| Placeholder | Meaning | Related raw field | When empty |
|---|---|---|---|
| `examples` | Formatted domain examples | `raw_examples` | Empty selection → blank |
| `current_graph_summary` | Graph snapshot with a display-size cutoff | `raw_graph` | Tree → blank; over 20 nodes → size notice |
| `default_prompt` | Complete built-in prompt for this node and style | — | Normal calls → nonempty |

</div>

All fields in these topic tables are accepted in both `initial` and
`expansion`. Sample-only fields are not accepted here. Graph ancestry follows
first parents.

`raw_graph` uses the existing `Graph.to_json()` schema: nodes, root ID, and
metadata. Its uncapped snapshot is constructed only when explicitly referenced;
`current_graph_summary` retains its 20-node display cutoff.

`raw_tree` uses `Tree.to_dict()`: `tree_paths`, `failed_generations`, and `config`.
It contains recorded completed paths, not branches still being expanded. Empty
collections remain inside the snapshot. `current_topic` and the path fields
separately describe the current expansion.

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
