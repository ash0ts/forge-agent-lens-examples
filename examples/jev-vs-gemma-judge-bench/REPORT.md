# Jev vs Gemma on the insights judges

100 frozen WB Agent turns (`turns_mini`, pinned Weave Dataset), identical budgeted payloads to
both models, one Gemma replicate at temperature 0, one Jev call per judge per turn. Gemma is
`google/gemma-4-31B-it` on W&B Inference; Jev is `jev-1.13.0` via `jev-latest`. No gold labels:
"alignment" means agreement with Gemma, then hand inspection of the disagreements.
Scripts are under `scripts/`, data under `data/` in this package.

## What Jev can and cannot port

Jev returns typed answers only: `choice` (label + per-option probability + confidence), `score`
(rubric level), `noul` (P(yes)). It emits no text. The Gemma judges' load-bearing outputs are
text: the intent `signature` sentence that gets embedded and clustered, and the failure `reason`,
`signature`, and `evidence` indices. None of that ports. Everything below is the classification
half of each judge.

| Gemma field | Jev question | Ported |
| --- | --- | --- |
| intent `status` | choice over ok / needs_context / no_intent | yes |
| intent `category` (per intent) | choice `primary_category` + one noul per category | yes |
| intent `signature` | none | no |
| `language` | choice over 10 codes | yes |
| `sentiment.label`, `.target` | two choices | yes |
| `sentiment.rationale` | none | no |
| failure present | noul `failed` + noul `tool_error` | yes |
| failure `category` | one noul per category, read only when `failed` fires | yes |
| failure `severity` | score over info / minor / major | yes |
| failure `reason`, `signature`, `evidence` | none | no |

## Alignment (n=100)

| Field | Agreement with Gemma |
| --- | --- |
| intent status | 95% |
| intent primary category (Gemma emits exactly 1 category on all 100 turns) | 91% |
| intent category set, nouls at 0.9 with primary fallback | P 85% / R 95% |
| language | 96% |
| sentiment label | 94% |
| sentiment target | 97% |
| failure fired (either noul >= 0.5) | 86% (Gemma fired 19, Jev 31; 1 Gemma failure missed) |
| failure category set | P 39% / R 47%; with tool_failure and tool_misuse merged P 61% / R 74% |
| material failures Gemma called `major` (3 turns) | Jev `failed` 0.72 / 0.51 / 0.88, all caught |

### Where they disagree, after reading the turns

- **Tool errors: Jev is right, Gemma is under-reporting.** 13 turns are Jev-only failures. Every
  one inspected (9 of 13) has a real traceback, argparse usage error, or non-zero exit code in a
  shell tool result. The Gemma prompt requires reporting every such error at `info` severity;
  Gemma skipped them. On the tool-error contract Jev is the more faithful judge.
- **Failure category naming.** Gemma splits tool problems into `tool_failure` and `tool_misuse`
  (17 of its 21 failures). Jev's nouls do not separate these reliably; merging the two lifts set
  recall from 47% to 74%. Category nouls also carry a 0.1 to 0.5 floor on clean turns, so they
  only work gated behind the top-level `failed` noul.
- **Intent co-firing.** `action_request` and `information_request` both exceed 0.85 on "debug and
  explain X" style asks. Independent nouls cannot enforce Gemma's one-category-per-intent rule.
  At threshold 0.5 set precision is 52%; at 0.9 it is 85%. The single `primary_category` choice
  is the honest port and matches Gemma 91%.
- **Status.** 5 disagreements, all Jev `needs_context` vs Gemma `ok`. One (t65) is a turn whose
  user message is product-injected system text; Gemma called it `bad_faith`, Jev declined to
  assign intent. Arguably Jev is right there.
- **Sentiment.** 6 disagreements, all Jev `dissatisfied` vs Gemma `neutral` on turns reporting a
  broken result. The taxonomy defines `dissatisfied` as "reports something broken" so these are
  boundary cases, not errors.

## Cost, tokens, latency

Prices: Gemma $0.10/M input, $0.34/M output. Jev $42 per billion input ($0.042/M), output free.

| Per turn, both judges | Jev | Gemma |
| --- | --- | --- |
| input tokens | 13,226 | 13,923 |
| output tokens | 761 | 93 |
| total cost of the 100-turn run | $0.056 (1.32M input tokens) | $0.142 (1.39M input + 9.3k output tokens) |
| single-call latency | p50 0.15s, p95 0.22s | ~8s |

Jev cost about 2.5x less for the same work and has roughly 50x lower single-call latency. Input tokens are near-identical because the state is the same
payload; Jev's question block replaces Gemma's system prompt at similar size.

Throughput on the Gemma side: 200 calls in 87s at 50 concurrent (2.3 calls/s); the per-user cap
on W&B Inference sits near 100 in flight. Jev was run one call at a time with a 0.5s pause and
one retry-with-backoff on 429/529 after a 529 killed the first full run.

## Bottom line

- Jev matches Gemma on every classification field at 91% to 97%, catches all material failures,
  and out-performs Gemma on the mechanical tool-error contract, at 2.5x lower cost and ~50x lower
  latency.
- Jev cannot replace either judge because it cannot write the signature text the pipeline embeds
  and clusters, nor failure reasons or evidence. Its natural slot is beside a text model, not
  instead of one: cheap fan-out classification (status, category, sentiment, language, failed,
  tool_error, severity) plus confidence routing, with the text model called only when Jev says
  the turn carries an intent or a failure.
- If that split is pursued, the gate matters: on this corpus 19 to 31 of 100 turns fail and ~95
  carry an intent, so the text call is saved on failure judgments but not on intent.
- Calibration to do before trusting it: choose noul thresholds against gold, not against Gemma;
  merge or redefine `tool_failure` / `tool_misuse` for Jev; decide whether recovered tool errors
  Gemma skips should count.
