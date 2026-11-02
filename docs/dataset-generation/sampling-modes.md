# Sample generation modes

`generation.topic_sample_mode` controls how each topic fills its sample quota.
Conversation type (`basic`/`cot`), tools and custom schemas are separate choices.

| Mode | What happens | Shared context |
|---|---|---|
| `independent` (default) | Generate each sample separately. | No earlier samples. |
| `sequential`, scope `seed` | Generate one complete sample at a time. | Earlier successful user questions or custom seed values. |
| `sequential`, scope `full` | Generate one complete sample at a time. | Earlier successful complete samples. |
| `bulk`, scope `seed` | Collect seeds together, then complete each separately. | Collection sees seeds accepted on earlier attempts; each completion gets its pinned seed. |
| `bulk`, scope `full` | Collect complete samples together. | Collection sees samples accepted on earlier attempts. |

Sequential and bulk require a topic model. Sharing stays within one topic; only successful
samples enter history. `shared_scope` defaults to `seed` and cannot be explicitly set for
independent mode. Diversity is requested by prompts, not guaranteed.

The animations below show sample generation for each mode, using non-tool conversations
within one topic as an example. N is the number of samples.

=== "Independent"

    ![Generate each user/assistant pair independently](../images/genmodes/genmodes-full-independent-light.svg#only-light)
    ![Generate each user/assistant pair independently](../images/genmodes/genmodes-full-independent-dark.svg#only-dark)

=== "Sequential / seed"

    ![Generate complete conversations in sequence, sharing earlier user requests](../images/genmodes/genmodes-seed-sequential-light.svg#only-light)
    ![Generate complete conversations in sequence, sharing earlier user requests](../images/genmodes/genmodes-seed-sequential-dark.svg#only-dark)

=== "Sequential / full"

    ![Generate complete conversations in sequence, sharing earlier conversations](../images/genmodes/genmodes-full-sequential-light.svg#only-light)
    ![Generate complete conversations in sequence, sharing earlier conversations](../images/genmodes/genmodes-full-sequential-dark.svg#only-dark)

=== "Bulk / seed"

    ![Collect N user requests in one call, then complete each separately: N plus one calls in total](../images/genmodes/genmodes-seed-bulk-light.svg#only-light)
    ![Collect N user requests in one call, then complete each separately: N plus one calls in total](../images/genmodes/genmodes-seed-bulk-dark.svg#only-dark)

=== "Bulk / full"

    ![Generate N complete user/assistant pairs in one call](../images/genmodes/genmodes-full-bulk-light.svg#only-light)
    ![Generate N complete user/assistant pairs in one call](../images/genmodes/genmodes-full-bulk-dark.svg#only-dark)

## Conversation examples

Add these settings to your generation configuration:

```yaml
generation:
  system_prompt: You are a patient Python tutor.
  topic_sample_mode: sequential
  shared_scope: seed
```

Use `shared_scope: full` to share complete conversations instead of questions.
Sequential still finishes each complete sample before starting the next. History grows
without automatic truncation; large prompts trigger warnings and can increase cost.

For bulk generation, change `topic_sample_mode` to `bulk`:

- `seed` collects `{"items": ["user request", "..."]}` and generates each answer separately.
- `full` collects complete conversations in one request, with top-ups if needed.

CoT seed collection asks for reasoning problems; reasoning is generated during completion.
Completion uses `output.system_prompt`, falling back to `generation.system_prompt`.
For non-tool bulk/seed, `completion_includes_topic_path: true` also supplies the topic path;
the default completion relies on the pinned question.

## Custom-schema seeds

In seed scope, name a required top-level field with `seed_field`. DeepFabric collects its
values first, generates the remaining fields, then assembles each record without changing its seed.

```yaml
generation:
  system_prompt: Create practical command-line exercises.
  topic_sample_mode: bulk
  shared_scope: seed
  seed_field: request
  output_schema:
    type: object
    properties:
      request: {type: string}
      answer: {type: string}
    required: [request, answer]
    additionalProperties: false
```

Sequential seed scope uses the same field to select history, but generates whole records
in each call. Bulk/full needs no seed field and collects complete records.

## Tools and call counts

**Bulk/full is not supported with tools.** Use bulk/seed to collect user requests, then run
each through its own ReAct loop. Sequential history influences question generation only;
ReAct decisions and final answers use the same framing as independent mode.
`completion_includes_topic_path` does not add context to those agent prompts.
Completion-template overrides also do not change the fixed tool-agent completion prompts.

For a topic with N samples, excluding retries and top-ups:

| Mode | Basic / non-tool CoT | Tool conversations, minimum |
|---|---|---|
| Independent or sequential | N calls | 3N calls |
| Bulk/seed | 1 + N calls | 1 + 2N calls |
| Bulk/full | 1 call | Unsupported |

Custom-schema seed completions can be skipped if the seed already forms the entire record.
Tool loops may require more decision calls. Choosing `bulk` does not select a provider's Batch API.

## Budgets, retries and resume

- Bulk collection uses `max_tokens × currently requested items` by default. Set
  `generation.bulk.max_tokens_per_call` for a fixed pooled ceiling. Each completion retains
  `max_tokens`; provider limits still apply without automatic clamping.
- Short or partially invalid collections retain accepted items and retry outstanding slots
  within `sample_retries`. Seed completions retry with the same pinned seed.
- Resume restores accepted seeds and successful sibling history. Use `--resume --retry-failed`
  to reopen terminal failures; successful samples are preserved.
- `output.batch_size` limits active topics, not items per bulk response. Calls within each topic
  run serially, and the next topic group starts after the current group settles.

See [checkpoint configuration](configuration.md#outputcheckpoint-optional) for recovery and
migration, and [prompt templates](prompt-templates.md) for overrides and dry-run inspection.

## Choosing a generation mode

The best choice depends on the topic, model, schema and desired output. These general
guidelines are starting points rather than recommendations for every use case:

- **Independent** preserves the default behavior and is a natural choice when you need
  only one sample per topic node, with no earlier samples to share.
- **Seed scope** can save input tokens compared with full scope by sharing only user
  requests or seed values. In sequential mode, each call still generates a complete sample;
  in bulk mode, seeds are collected before their answers are generated.
- **Bulk/full** can be faster because one call produces several complete samples,
  reducing call overhead. Large responses and retries can offset that advantage.
- **Full scope with sequential or bulk** is useful when considering the other parts of
  a sample may improve quality, or when no single field makes a meaningful seed.
  Sequential/full shows earlier complete samples; bulk/full generates the records together.
- **Sequential** produced higher diversity in our experiments, but the result may depend
  on the topic, model and prompts. Compare a small set of outputs for your task before
  committing to a larger run.
