# Glossary

## Models and API

- **Gemma**: `gemma-4-31B-it`, a generative instruction-tuned model served through an OpenAI-compatible
  inference API. The current Insights judges prompt it to return a JSON object.
- **Jev / System One**: a model class from typesafe.ai. One request carries a JSON `state` and a set of
  named `questions`. The response is one answer per question; there is no free text. Version used:
  `jev-1.13.0` via the `jev-latest` alias.
- **choice**: a question with N named options and a one-line criterion each. Answer: a probability per
  option, the argmax `choice`, and a `confidence`.
- **noul**: a yes/no question. Answer: one number, the probability that the condition holds.
- **score**: an ordered rubric (here info < minor < major). Answer: a probability per level plus a
  `score` in [0, 1], the expected level normalised.
- **state**: the JSON payload the questions are asked about. Identical to the payload the Gemma prompt
  receives, so input token counts are comparable.

## Pipeline terms

- **Insights**: the pipeline that turns each user turn of an agent conversation into an intent record
  and a failure record, then clusters them into topics.
- **turn**: one user message plus everything the agent did in response (assistant messages, tool calls,
  tool results) up to the next user message.
- **intent judge**: reads the conversation history and the current user message; returns status,
  language, a list of intents (category plus a one-sentence signature), and sentiment (label, target,
  rationale).
- **failure judge**: reads the full current turn including tool steps; returns a list of failures
  (category, severity, reason, signature, evidence step indices).
- **signature**: the one-sentence paraphrase Gemma writes for each intent or failure. It is embedded
  and clustered downstream. Jev cannot produce it.
- **status**: `ok` (a clear current intent), `needs_context` (substantive text but no identifiable
  intent), `no_intent` (greeting, thanks, gibberish).
- **fired**: a noul at or above its threshold. Category sets use 0.9; the failure gate uses 0.5.
- **recovered tool error**: a tool call that errored but a later call succeeded. The Gemma prompt asks
  for these at `info` severity; Gemma mostly skipped them and Jev's `tool_error` noul caught them.
- **answerless turn**: the agent made tool calls and ended the turn without answering the user.

## Intent categories

- **action_request**: asks the agent to do, create, change, remove, or execute something, including a requested capability that does not exist yet.
- **information_request**: asks for information, explanation, instructions, comparison, or analysis.
- **problem_report**: reports broken or incorrect behavior without requesting a remedy.
- **positive_feedback**: praises or expresses satisfaction with behavior working as designed, without requesting action.
- **negative_feedback**: complains about behavior working as designed, including naming a capability the product lacks, without requesting action.
- **approval**: authorizes or agrees to a proposed action.
- **rejection**: declines, cancels, stops, or aborts a proposed or in-flight action.
- **refinement**: redirects the agent, by contradicting its output or an earlier request, or by supplying requested or additional detail.
- **bad_faith**: intentional misuse, including jailbreak or prompt-injection attempts, attempts to extract system instructions or another user's data, or sandbox escape.
- **other**: a meaningful intent that fits none of the above.

## Failure categories

- **task_misunderstanding**: misread the current request and solved the wrong problem.
- **context_loss**: forgot or contradicted an established referent, constraint, answer, or instruction.
- **wrong_output**: produced incorrect content, including factual errors, fabricated APIs or data, or code that does not work.
- **requirement_violation**: plausible content that dropped a stated or standing requirement, format, scope, language, or required piece.
- **tool_misuse**: selected the wrong tool, sent malformed arguments, failed to use an available tool, or ignored a tool result.
- **tool_failure**: a well-formed tool call failed inside the tool through an error, timeout, or rate limit the assistant could observe.
- **system_error**: infrastructure outside the assistant failed, such as a crash, model timeout, or truncated response.
- **unproductive_loop**: repeated calls, retries, or approaches without meaningful progress.
- **capability_gap**: the request exceeds what the product can do, even if declined correctly.
- **refusal**: declined, hedged, or evaded a request, whether or not the decline was correct.
- **unsafe_behavior**: took a destructive or unauthorized action or disclosed sensitive information.
- **other**: a real, observable failure that fits none of the above.

## Severity

- **info**: notable behavior that is not necessarily wrong; a refusal or capability gap that left a path forward.
- **minor**: degraded the turn, but the user still got what they asked for, or the shortfall is one correction away.
- **major**: the turn was wasted or misleading; the main task failed outright, or the response asserted something untrue.

## Sentiment

- **frustrated**: explicit exasperation, a repetition callout, profanity, or emphatic insistence.
- **dissatisfied**: reports something broken, wrong, or unwanted, without explicit heat.
- **neutral**: task-focused with no affect. Plain corrections, approvals, and cancellations are neutral.
- **satisfied**: brief thanks, warm agreement, or a short positive acknowledgement.
- **delighted**: emphatic or unprompted praise.

## Sentiment target

- **agent**: the negative affect is pointed at what the assistant did or said in this or the previous turn.
- **product**: a bug, limitation, or missing feature in the product itself.
- **own_work**: disappointment with the user's own results, science, or code.
- **unclear**: cannot be attributed, or the sentiment is not frustrated or dissatisfied.
