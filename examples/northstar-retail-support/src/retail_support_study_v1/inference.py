from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .constants import (
    INFERENCE_PROJECT,
    INTERNAL_SERVER_RETRY_DELAYS_SECONDS,
    OTHER_RETRYABLE_TRANSPORT_DELAYS_SECONDS,
    TRANSPORT_RETRIES,
    WANDB_INFERENCE_BASE_URL,
)
from .pricing import CostBudget, observed_cost_usd
from .schemas import ModelCall, Usage


@dataclass(frozen=True)
class Completion:
    message: dict[str, Any]
    record: ModelCall


class InferenceCallError(RuntimeError):
    """An unusable paid call whose exact evidence must still be frozen."""

    def __init__(self, message: str, *, record: ModelCall) -> None:
        super().__init__(message)
        self.record = record


@runtime_checkable
class SupportInferenceClient(Protocol):
    """Provider-neutral boundary used by the support-agent runner.

    The user simulator deliberately continues to use ``WandbInferenceClient``.
    Support implementations must return the existing normalized ``Completion``
    and ``ModelCall`` evidence rather than leaking a provider response type into
    the conversation runner.
    """

    provider_name: str
    server_address: str
    budget: CostBudget

    def complete(
        self,
        *,
        participant: Literal["support_agent", "user_simulator"],
        model: str,
        temperature: float | None,
        seed: int | None,
        messages: list[dict[str, Any]],
        max_tokens: int,
        enable_thinking: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion: ...


def _retry_delay_seconds(exc: Exception, *, retry_index: int) -> float:
    """Return the frozen delay before the next bounded transport attempt."""

    if retry_index < 0 or retry_index >= TRANSPORT_RETRIES:
        raise ValueError("retry delay requested outside the bounded retry policy")
    if isinstance(exc, APIStatusError) and exc.status_code == 500:
        return INTERNAL_SERVER_RETRY_DELAYS_SECONDS[retry_index]
    return OTHER_RETRYABLE_TRANSPORT_DELAYS_SECONDS[retry_index]


def _message_dict(message: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        value["content"] = message.content
    if getattr(message, "tool_calls", None):
        value["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
    return value


def _usage(
    response: Any,
    *,
    requested_model: str,
    price_lock: dict[str, Any] | None = None,
) -> Usage:
    raw = getattr(response, "usage", None)
    if raw is None:
        return Usage()
    prompt_tokens = getattr(raw, "prompt_tokens", None)
    completion_tokens = getattr(raw, "completion_tokens", None)
    total = getattr(raw, "total_tokens", None)
    if prompt_tokens is None or completion_tokens is None or total is None:
        raise ValueError("successful response returned incomplete token usage")
    input_tokens = int(prompt_tokens)
    prompt_details = getattr(raw, "prompt_tokens_details", None)
    if isinstance(prompt_details, dict):
        cached_input_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
    else:
        cached_input_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
    output_tokens = int(completion_tokens)
    total_tokens = int(total)
    if total_tokens <= 0 or total_tokens != input_tokens + output_tokens:
        raise ValueError("successful response returned inconsistent token usage")
    # Some OpenAI-compatible providers attach an observed dollar cost. When it
    # is absent, price the returned usage with the reviewed public rate lock.
    provider_observed = getattr(raw, "cost", None)
    observed = (
        float(provider_observed)
        if provider_observed is not None
        else observed_cost_usd(
            requested_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            price_lock=price_lock,
        )
    )
    return Usage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        provider_reported_cost_usd=(
            float(provider_observed) if provider_observed is not None else None
        ),
        observed_cost_usd=observed,
    )


def _resolve_client_budget(
    *,
    budget: CostBudget | None,
    cost_cap_usd: float | None,
    observed_cost_usd: float,
    ambiguous_cost_reserve_usd: float,
    uncertainty_reasons: list[str] | None,
    price_lock: dict[str, Any] | None,
) -> tuple[CostBudget, dict[str, Any] | None]:
    if budget is not None:
        if (
            cost_cap_usd is not None
            or observed_cost_usd != 0.0
            or ambiguous_cost_reserve_usd != 0.0
            or uncertainty_reasons is not None
        ):
            raise ValueError(
                "shared budget is mutually exclusive with numeric budget state"
            )
        shared_price_lock = getattr(budget, "_price_lock", None)
        if price_lock is not None and shared_price_lock != price_lock:
            raise ValueError("shared budget and client price locks differ")
        return budget, shared_price_lock
    if cost_cap_usd is None:
        raise ValueError("cost_cap_usd or a shared budget is required")
    created = CostBudget(
        cost_cap_usd,
        observed_usd=observed_cost_usd,
        ambiguous_cost_reserve_usd=ambiguous_cost_reserve_usd,
        uncertainty_reasons=uncertainty_reasons,
        price_lock=price_lock,
    )
    return created, getattr(created, "_price_lock", price_lock)


class WandbInferenceClient:
    provider_name = "wandb"
    server_address = "api.inference.wandb.ai"

    def __init__(
        self,
        *,
        cost_cap_usd: float | None = None,
        observed_cost_usd: float = 0.0,
        ambiguous_cost_reserve_usd: float = 0.0,
        uncertainty_reasons: list[str] | None = None,
        price_lock: dict[str, Any] | None = None,
        budget: CostBudget | None = None,
        inference_project: str = INFERENCE_PROJECT,
    ) -> None:
        api_key = os.environ.get("WANDB_API_KEY", "").strip()
        if not api_key:
            raise ValueError("WANDB_API_KEY is required for paid model execution")
        if not inference_project or inference_project.count("/") != 1:
            raise ValueError("inference_project must be an entity/project path")
        self._client = OpenAI(
            api_key=api_key,
            base_url=WANDB_INFERENCE_BASE_URL,
            max_retries=0,
            timeout=120.0,
            default_headers={"OpenAI-Project": inference_project},
        )
        self.budget, self._price_lock = _resolve_client_budget(
            budget=budget,
            cost_cap_usd=cost_cap_usd,
            observed_cost_usd=observed_cost_usd,
            ambiguous_cost_reserve_usd=ambiguous_cost_reserve_usd,
            uncertainty_reasons=uncertainty_reasons,
            price_lock=price_lock,
        )

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        *,
        participant: Literal["support_agent", "user_simulator"],
        model: str,
        temperature: float | None,
        seed: int | None,
        messages: list[dict[str, Any]],
        max_tokens: int,
        enable_thinking: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if seed is not None:
            request["seed"] = seed
        if enable_thinking is not None:
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
            request["parallel_tool_calls"] = False

        reserve_per_attempt = self.budget.require_capacity(
            model,
            messages=messages,
            tools=tools,
            max_output_tokens=max_tokens,
            max_attempts=TRANSPORT_RETRIES + 1,
        )

        last_error: Exception | None = None
        call_ambiguous_reserve = 0.0
        call_uncertainty_reasons: list[str] = []
        for attempt in range(TRANSPORT_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**request)
                usage = Usage()
                response_id = ""
                returned_model = ""
                system_fingerprint: str | None = None
                finish_reason = "response_parse_error"
                message: dict[str, Any] = {}
                try:
                    usage = _usage(
                        response,
                        requested_model=model,
                        price_lock=getattr(self, "_price_lock", None),
                    )
                    response_id = str(getattr(response, "id", "") or "")
                    returned_model = str(getattr(response, "model", "") or "")
                    raw_fingerprint = getattr(response, "system_fingerprint", None)
                    if raw_fingerprint is not None:
                        system_fingerprint = str(raw_fingerprint)
                    choice = response.choices[0] if len(response.choices) == 1 else None
                    message = (
                        _message_dict(choice.message) if choice is not None else {}
                    )
                    finish_reason = (
                        str(getattr(choice, "finish_reason", "") or "")
                        if choice is not None
                        else ""
                    )
                except Exception as exc:
                    # A provider response may already be billable even when its
                    # OpenAI-compatible envelope is malformed. Freeze a bounded
                    # receipt and stop; never silently lose or retry that exposure.
                    try:
                        response_id = str(getattr(response, "id", "") or "")
                    except Exception:
                        response_id = ""
                    try:
                        returned_model = str(getattr(response, "model", "") or "")
                    except Exception:
                        returned_model = ""
                    reason = (
                        "successful response could not be parsed: "
                        f"response_id={response_id or 'unavailable'}, "
                        f"error={type(exc).__name__}"
                    )
                    if usage.observed_cost_usd is not None:
                        self.budget.record(usage.observed_cost_usd)
                    else:
                        self.budget.record_ambiguous(
                            reserve_per_attempt,
                            reason=reason,
                        )
                        call_ambiguous_reserve = round(
                            call_ambiguous_reserve + reserve_per_attempt,
                            12,
                        )
                        call_uncertainty_reasons.append(reason)
                    record = ModelCall(
                        call_id=str(uuid.uuid4()),
                        participant=participant,
                        requested_model=model,
                        returned_model=returned_model,
                        response_id=response_id,
                        system_fingerprint=system_fingerprint,
                        finish_reason=finish_reason,
                        transport_attempt_count=attempt + 1,
                        ambiguous_cost_reserve_usd=call_ambiguous_reserve,
                        cost_uncertainty_reasons=call_uncertainty_reasons,
                        response_validation_error=reason,
                        temperature=temperature,
                        requested_max_tokens=max_tokens,
                        requested_enable_thinking=enable_thinking,
                        seed=seed,
                        input_messages=messages,
                        output_message=message,
                        usage=usage,
                    )
                    raise InferenceCallError(reason, record=record) from exc
                if usage.observed_cost_usd is None:
                    reason = (
                        "successful response omitted usage: "
                        f"response_id={response_id or 'unavailable'}"
                    )
                    self.budget.record_ambiguous(
                        reserve_per_attempt,
                        reason=reason,
                    )
                    call_ambiguous_reserve = round(
                        call_ambiguous_reserve + reserve_per_attempt,
                        12,
                    )
                    call_uncertainty_reasons.append(reason)
                else:
                    self.budget.record(usage.observed_cost_usd)

                validation_error: str | None = None
                if choice is None:
                    validation_error = (
                        "Forge Inference returned an unusable choice count: "
                        f"expected=1, returned={len(response.choices)}, "
                        f"response_id={response_id or 'unavailable'}"
                    )
                elif returned_model != model:
                    validation_error = (
                        "Forge Inference returned a different model: "
                        f"requested={model!r}, returned={returned_model!r}, "
                        f"response_id={response_id or 'unavailable'}"
                    )
                elif finish_reason not in {"stop", "tool_calls"}:
                    validation_error = (
                        "Forge Inference returned an unusable finish reason: "
                        f"finish_reason={finish_reason!r}, "
                        f"response_id={response_id or 'unavailable'}"
                    )
                record = ModelCall(
                    call_id=str(uuid.uuid4()),
                    participant=participant,
                    requested_model=model,
                    returned_model=returned_model,
                    response_id=response_id,
                    system_fingerprint=system_fingerprint,
                    finish_reason=finish_reason,
                    transport_attempt_count=attempt + 1,
                    ambiguous_cost_reserve_usd=call_ambiguous_reserve,
                    cost_uncertainty_reasons=call_uncertainty_reasons,
                    response_validation_error=validation_error,
                    temperature=temperature,
                    requested_max_tokens=max_tokens,
                    requested_enable_thinking=enable_thinking,
                    seed=seed,
                    input_messages=messages,
                    output_message=message,
                    usage=usage,
                )
                if validation_error is not None:
                    raise InferenceCallError(validation_error, record=record)
                return Completion(message=message, record=record)
            except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
                retryable = not isinstance(exc, APIStatusError) or exc.status_code in {
                    408,
                    409,
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                last_error = exc
                if retryable:
                    reason = (
                        "retryable transport outcome was ambiguous: "
                        f"attempt={attempt + 1}, error={type(exc).__name__}"
                    )
                    self.budget.record_ambiguous(
                        reserve_per_attempt,
                        reason=reason,
                    )
                    call_ambiguous_reserve = round(
                        call_ambiguous_reserve + reserve_per_attempt,
                        12,
                    )
                    call_uncertainty_reasons.append(reason)
                if not retryable or attempt >= TRANSPORT_RETRIES:
                    break
                time.sleep(_retry_delay_seconds(exc, retry_index=attempt))
        assert last_error is not None
        message = (
            f"Forge Inference request unavailable after {attempt + 1} attempts: "
            f"{type(last_error).__name__}"
        )
        record = ModelCall(
            call_id=str(uuid.uuid4()),
            participant=participant,
            requested_model=model,
            returned_model="",
            response_id="",
            finish_reason="transport_error",
            transport_attempt_count=attempt + 1,
            ambiguous_cost_reserve_usd=call_ambiguous_reserve,
            cost_uncertainty_reasons=call_uncertainty_reasons,
            response_validation_error=message,
            temperature=temperature,
            requested_max_tokens=max_tokens,
            requested_enable_thinking=enable_thinking,
            seed=seed,
            input_messages=messages,
            output_message={},
            usage=Usage(),
        )
        raise InferenceCallError(message, record=record) from last_error


ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"


def _anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("Anthropic support tools must use OpenAI function shape")
        function = tool["function"]
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Anthropic support tool requires a non-empty name")
        if not isinstance(parameters, dict):
            raise ValueError("Anthropic support tool requires an object input schema")
        value: dict[str, Any] = {
            "name": name,
            "input_schema": parameters,
        }
        description = function.get("description")
        if isinstance(description, str) and description:
            value["description"] = description
        converted.append(value)
    return converted


def _anthropic_tool_input(raw: Any) -> dict[str, Any]:
    """Translate a prior normalized tool call back to Anthropic content shape.

    A malformed argument string is retained in the frozen ``ModelCall`` and is
    rejected by the runner. On the following call Anthropic still requires an
    object in the historical ``tool_use`` block, so use the same empty-object
    interpretation that the runner used while producing its tool error.
    """

    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    seen_dialogue = False

    def append_message(role: str, blocks: list[dict[str, Any]]) -> None:
        if converted and converted[-1]["role"] == role:
            prior = converted[-1]["content"]
            if not isinstance(prior, list):  # pragma: no cover - invariant
                raise ValueError("Anthropic message content must be a block list")
            prior.extend(blocks)
            return
        converted.append({"role": role, "content": blocks})

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if seen_dialogue:
                raise ValueError("system messages must precede Anthropic dialogue")
            if not isinstance(content, str) or not content:
                raise ValueError("Anthropic system content must be non-empty text")
            system_parts.append(content)
            continue

        seen_dialogue = True
        if role == "user":
            if not isinstance(content, str) or not content:
                raise ValueError("Anthropic user content must be non-empty text")
            append_message("user", [{"type": "text", "text": content}])
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif content is not None and not isinstance(content, str):
                raise ValueError("Anthropic assistant content must be text")
            raw_calls = message.get("tool_calls") or []
            if not isinstance(raw_calls, list):
                raise ValueError("Anthropic assistant tool calls must be a list")
            for raw_call in raw_calls:
                function = (
                    raw_call.get("function") if isinstance(raw_call, dict) else None
                )
                call_id = raw_call.get("id") if isinstance(raw_call, dict) else None
                if not isinstance(function, dict):
                    raise ValueError("Anthropic historical tool call is malformed")
                name = function.get("name")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("Anthropic historical tool call has no id")
                if not isinstance(name, str) or not name:
                    raise ValueError("Anthropic historical tool call has no name")
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": _anthropic_tool_input(function.get("arguments")),
                    }
                )
            if not blocks:
                raise ValueError("Anthropic assistant message cannot be empty")
            append_message("assistant", blocks)
            continue

        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Anthropic tool result requires a tool-call id")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True)
            is_error = False
            try:
                decoded = json.loads(content)
                is_error = isinstance(decoded, dict) and "error" in decoded
            except json.JSONDecodeError:
                pass
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": content,
            }
            if is_error:
                block["is_error"] = True
            append_message("user", [block])
            continue

        raise ValueError(f"unsupported Anthropic message role: {role!r}")

    if not converted:
        raise ValueError("Anthropic request requires dialogue messages")
    return ("\n\n".join(system_parts) if system_parts else None), converted


def _anthropic_output_message(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("role") != "assistant":
        raise ValueError("Anthropic response role must be assistant")
    content = payload.get("content")
    if not isinstance(content, list):
        raise ValueError("Anthropic response content must be a list")

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Anthropic response content block must be an object")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("Anthropic text block has invalid text")
            text_parts.append(text)
            continue
        if block_type == "tool_use":
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Anthropic tool-use block has no id")
            if not isinstance(name, str) or not name:
                raise ValueError("Anthropic tool-use block has no name")
            if not isinstance(arguments, dict):
                raise ValueError("Anthropic tool-use input must be an object")
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            arguments,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                }
            )
            continue
        raise ValueError(
            f"unsupported Anthropic response content block: {block_type!r}"
        )

    message: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        message["content"] = "".join(text_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    if len(message) == 1:
        raise ValueError("Anthropic response contained no usable content")
    return message


def _anthropic_finish_reason(raw: Any) -> str:
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "refusal": "refusal",
        "pause_turn": "pause_turn",
    }.get(str(raw or ""), str(raw or ""))


def _anthropic_usage(
    payload: dict[str, Any],
    *,
    requested_model: str,
    price_lock: dict[str, Any] | None,
) -> Usage:
    raw = payload.get("usage")
    if raw is None:
        return Usage()
    if not isinstance(raw, dict):
        raise ValueError("Anthropic response usage must be an object")
    required = ("input_tokens", "output_tokens")
    if any(raw.get(key) is None for key in required):
        raise ValueError("successful Anthropic response returned incomplete usage")
    base_input_tokens = int(raw["input_tokens"])
    cache_creation_tokens = int(raw.get("cache_creation_input_tokens", 0) or 0)
    cached_input_tokens = int(raw.get("cache_read_input_tokens", 0) or 0)
    output_tokens = int(raw["output_tokens"])
    input_tokens = base_input_tokens + cache_creation_tokens + cached_input_tokens
    total_tokens = input_tokens + output_tokens
    if input_tokens < 0 or output_tokens < 0 or total_tokens <= 0:
        raise ValueError("successful Anthropic response returned invalid usage")
    return Usage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        observed_cost_usd=observed_cost_usd(
            requested_model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            price_lock=price_lock,
        ),
    )


def _anthropic_retry_delay_seconds(
    exc: Exception,
    *,
    retry_index: int,
) -> float:
    if retry_index < 0 or retry_index >= TRANSPORT_RETRIES:
        raise ValueError("retry delay requested outside the bounded retry policy")
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 500:
        return INTERNAL_SERVER_RETRY_DELAYS_SECONDS[retry_index]
    return OTHER_RETRYABLE_TRANSPORT_DELAYS_SECONDS[retry_index]


class AnthropicInferenceClient:
    """Direct Anthropic Messages client normalized to the runner contract."""

    provider_name = "anthropic"
    server_address = "api.anthropic.com"

    def __init__(
        self,
        *,
        cost_cap_usd: float | None = None,
        observed_cost_usd: float = 0.0,
        ambiguous_cost_reserve_usd: float = 0.0,
        uncertainty_reasons: list[str] | None = None,
        price_lock: dict[str, Any] | None = None,
        http_client: httpx.Client | None = None,
        budget: CostBudget | None = None,
    ) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for paid model execution")
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=120.0,
            headers={
                "anthropic-version": ANTHROPIC_API_VERSION,
                "content-type": "application/json",
                "x-api-key": api_key,
            },
        )
        self.budget, self._price_lock = _resolve_client_budget(
            budget=budget,
            cost_cap_usd=cost_cap_usd,
            observed_cost_usd=observed_cost_usd,
            ambiguous_cost_reserve_usd=ambiguous_cost_reserve_usd,
            uncertainty_reasons=uncertainty_reasons,
            price_lock=price_lock,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete(
        self,
        *,
        participant: Literal["support_agent", "user_simulator"],
        model: str,
        temperature: float | None,
        seed: int | None,
        messages: list[dict[str, Any]],
        max_tokens: int,
        enable_thinking: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        if participant != "support_agent":
            raise ValueError("direct Anthropic is reserved for support-agent calls")
        if seed is not None:
            raise ValueError("direct Anthropic calls do not support a request seed")
        if enable_thinking is True:
            raise ValueError("direct Anthropic thinking is not enabled for this study")

        system, converted_messages = _anthropic_messages(messages)
        converted_tools = _anthropic_tools(tools)
        request: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if enable_thinking is False:
            request["thinking"] = {"type": "disabled"}
        if system is not None:
            request["system"] = system
        if converted_tools:
            request["tools"] = converted_tools
            request["tool_choice"] = {
                "type": "auto",
                "disable_parallel_tool_use": True,
            }

        budget_messages = list(converted_messages)
        if system is not None:
            budget_messages.insert(0, {"role": "system", "content": system})
        reserve_per_attempt = self.budget.require_capacity(
            model,
            messages=budget_messages,
            tools=converted_tools,
            max_output_tokens=max_tokens,
            max_attempts=TRANSPORT_RETRIES + 1,
        )

        last_error: Exception | None = None
        call_ambiguous_reserve = 0.0
        call_uncertainty_reasons: list[str] = []
        for attempt in range(TRANSPORT_RETRIES + 1):
            try:
                response = self._client.post(ANTHROPIC_MESSAGES_URL, json=request)
                response.raise_for_status()
                usage = Usage()
                response_id = ""
                returned_model = ""
                finish_reason = "response_parse_error"
                message: dict[str, Any] = {}
                parse_stage = "json"
                try:
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("Anthropic response must be an object")
                    parse_stage = "identity"
                    response_id = str(payload.get("id") or "")
                    returned_model = str(payload.get("model") or "")
                    parse_stage = "stop_reason"
                    finish_reason = _anthropic_finish_reason(payload.get("stop_reason"))
                    parse_stage = "usage"
                    usage = _anthropic_usage(
                        payload,
                        requested_model=model,
                        price_lock=self._price_lock,
                    )
                    parse_stage = "content"
                    message = _anthropic_output_message(payload)
                except Exception as exc:
                    # Report only the bounded parse stage and exception type.
                    # Anthropic content may include hidden reasoning or other
                    # material that must not leak into receipts or telemetry.
                    reason = (
                        "successful Anthropic response could not be parsed: "
                        f"response_id={response_id or 'unavailable'}, "
                        f"stage={parse_stage}, "
                        f"error={type(exc).__name__}"
                    )
                    if usage.observed_cost_usd is not None:
                        self.budget.record(usage.observed_cost_usd)
                    else:
                        self.budget.record_ambiguous(
                            reserve_per_attempt,
                            reason=reason,
                        )
                        call_ambiguous_reserve = round(
                            call_ambiguous_reserve + reserve_per_attempt,
                            12,
                        )
                        call_uncertainty_reasons.append(reason)
                    record = ModelCall(
                        call_id=str(uuid.uuid4()),
                        participant=participant,
                        requested_model=model,
                        returned_model=returned_model,
                        response_id=response_id,
                        finish_reason=finish_reason,
                        transport_attempt_count=attempt + 1,
                        ambiguous_cost_reserve_usd=call_ambiguous_reserve,
                        cost_uncertainty_reasons=call_uncertainty_reasons,
                        response_validation_error=reason,
                        temperature=temperature,
                        requested_max_tokens=max_tokens,
                        requested_enable_thinking=enable_thinking,
                        seed=seed,
                        input_messages=messages,
                        output_message=message,
                        usage=usage,
                    )
                    raise InferenceCallError(reason, record=record) from exc

                if usage.observed_cost_usd is None:
                    reason = (
                        "successful Anthropic response omitted usage: "
                        f"response_id={response_id or 'unavailable'}"
                    )
                    self.budget.record_ambiguous(
                        reserve_per_attempt,
                        reason=reason,
                    )
                    call_ambiguous_reserve = round(
                        call_ambiguous_reserve + reserve_per_attempt,
                        12,
                    )
                    call_uncertainty_reasons.append(reason)
                else:
                    self.budget.record(usage.observed_cost_usd)

                validation_error: str | None = None
                if returned_model != model:
                    validation_error = (
                        "Anthropic returned a different model: "
                        f"requested={model!r}, returned={returned_model!r}, "
                        f"response_id={response_id or 'unavailable'}"
                    )
                elif finish_reason not in {"stop", "tool_calls"}:
                    validation_error = (
                        "Anthropic returned an unusable stop reason: "
                        f"finish_reason={finish_reason!r}, "
                        f"response_id={response_id or 'unavailable'}"
                    )
                elif finish_reason == "tool_calls" and not message.get("tool_calls"):
                    validation_error = (
                        "Anthropic stopped for tool use without a tool call: "
                        f"response_id={response_id or 'unavailable'}"
                    )
                elif finish_reason == "stop" and message.get("tool_calls"):
                    validation_error = (
                        "Anthropic returned a tool call with an end-turn stop reason: "
                        f"response_id={response_id or 'unavailable'}"
                    )
                record = ModelCall(
                    call_id=str(uuid.uuid4()),
                    participant=participant,
                    requested_model=model,
                    returned_model=returned_model,
                    response_id=response_id,
                    finish_reason=finish_reason,
                    transport_attempt_count=attempt + 1,
                    ambiguous_cost_reserve_usd=call_ambiguous_reserve,
                    cost_uncertainty_reasons=call_uncertainty_reasons,
                    response_validation_error=validation_error,
                    temperature=temperature,
                    requested_max_tokens=max_tokens,
                    requested_enable_thinking=enable_thinking,
                    seed=seed,
                    input_messages=messages,
                    output_message=message,
                    usage=usage,
                )
                if validation_error is not None:
                    raise InferenceCallError(validation_error, record=record)
                return Completion(message=message, record=record)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                retryable = True
            except httpx.HTTPStatusError as exc:
                last_error = exc
                retryable = exc.response.status_code in {
                    408,
                    409,
                    429,
                    500,
                    502,
                    503,
                    504,
                }

            assert last_error is not None
            if retryable:
                reason = (
                    "retryable Anthropic transport outcome was ambiguous: "
                    f"attempt={attempt + 1}, error={type(last_error).__name__}"
                )
                self.budget.record_ambiguous(
                    reserve_per_attempt,
                    reason=reason,
                )
                call_ambiguous_reserve = round(
                    call_ambiguous_reserve + reserve_per_attempt,
                    12,
                )
                call_uncertainty_reasons.append(reason)
            if not retryable or attempt >= TRANSPORT_RETRIES:
                break
            time.sleep(_anthropic_retry_delay_seconds(last_error, retry_index=attempt))

        assert last_error is not None
        message = (
            f"Anthropic request unavailable after {attempt + 1} attempts: "
            f"{type(last_error).__name__}"
        )
        record = ModelCall(
            call_id=str(uuid.uuid4()),
            participant=participant,
            requested_model=model,
            returned_model="",
            response_id="",
            finish_reason="transport_error",
            transport_attempt_count=attempt + 1,
            ambiguous_cost_reserve_usd=call_ambiguous_reserve,
            cost_uncertainty_reasons=call_uncertainty_reasons,
            response_validation_error=message,
            temperature=temperature,
            requested_max_tokens=max_tokens,
            requested_enable_thinking=enable_thinking,
            seed=seed,
            input_messages=messages,
            output_message={},
            usage=Usage(),
        )
        raise InferenceCallError(message, record=record) from last_error
