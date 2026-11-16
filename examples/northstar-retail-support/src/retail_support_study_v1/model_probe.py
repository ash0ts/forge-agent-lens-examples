from __future__ import annotations

import json
from typing import Any, Callable

from .canonical import digest_value, write_json_atomic
from .constants import PACKAGE_ROOT
from .inference import (
    AnthropicInferenceClient,
    Completion,
    SupportInferenceClient,
    WandbInferenceClient,
)
from .pricing import pricing_lock
from .schemas import ModelRoute
from .traffic_config import MODEL_PROBE_POLICY, build_model_routes

MODEL_PROBE_PATH = PACKAGE_ROOT / "model-probe-live-v10.json"

_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_order",
        "description": "Read one fictional order for a capability probe.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The fictional order identifier.",
                }
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}
_PROBE_ORDER_ID = "R1234567"


def _client_for_route(route: ModelRoute) -> SupportInferenceClient:
    kwargs = {
        "cost_cap_usd": 5.0,
        "price_lock": pricing_lock(),
    }
    if route.provider_name == "wandb":
        return WandbInferenceClient(**kwargs)
    if route.provider_name == "anthropic":
        return AnthropicInferenceClient(**kwargs)
    raise ValueError(f"unsupported probe provider: {route.provider_name}")


def _probe_request(route: ModelRoute) -> dict[str, Any]:
    return {
        "participant": "support_agent",
        "model": route.model,
        "temperature": route.temperature,
        "enable_thinking": route.enable_thinking,
        "seed": None,
        "messages": [
            {
                "role": "system",
                "content": (
                    "This is a structured tool-calling capability probe. Call "
                    "inspect_order exactly once with the order ID supplied by "
                    "the user. Do not answer the user directly."
                ),
            },
            {
                "role": "user",
                "content": f"Inspect fictional order {_PROBE_ORDER_ID}.",
            },
        ],
        "max_tokens": 128,
        "tools": [_PROBE_TOOL],
    }


def _validate_probe(route: ModelRoute, completion: Completion) -> dict[str, Any]:
    calls = completion.message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("probe did not return exactly one structured tool call")
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict) or function.get("name") != "inspect_order":
        raise ValueError("probe called the wrong tool")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("probe returned malformed tool arguments") from exc
    if arguments != {"order_id": _PROBE_ORDER_ID}:
        raise ValueError("probe returned the wrong tool arguments")
    record = completion.record
    return {
        "route_id": route.route_id,
        "provider_name": route.provider_name,
        "requested_model": route.model,
        "returned_model": record.returned_model,
        "finish_reason": record.finish_reason,
        "tool_name": "inspect_order",
        "arguments_sha256": digest_value(arguments),
        "observed_cost_usd": record.usage.observed_cost_usd,
        "status": "passed",
    }


def probe_model_routes(
    *,
    client_factory: Callable[[ModelRoute], SupportInferenceClient] = (
        _client_for_route
    ),
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for route in build_model_routes():
        client = client_factory(route)
        try:
            completion = client.complete(**_probe_request(route))
            result = _validate_probe(route, completion)
        except Exception as exc:
            result = {
                "route_id": route.route_id,
                "provider_name": route.provider_name,
                "requested_model": route.model,
                "status": "failed",
                "error_type": type(exc).__name__,
                "wandb_fallback_candidates": (
                    list(MODEL_PROBE_POLICY["wandb_fallback_models"])
                    if route.provider_name == "wandb"
                    else []
                ),
            }
        results.append(result)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "probe_policy_sha256": MODEL_PROBE_POLICY["policy_sha256"],
        "required_capability": "structured_tool_calling",
        "routes": results,
        "status": (
            "ready_for_preview"
            if all(result["status"] == "passed" for result in results)
            else "blocked"
        ),
        "probe_sha256": "",
    }
    receipt["probe_sha256"] = digest_value(receipt)
    write_json_atomic(MODEL_PROBE_PATH, receipt)
    return receipt
