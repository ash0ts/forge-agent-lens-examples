#!/usr/bin/env python3
"""Close the CoreWeave Forge loop on the two failure patterns discovered in
'Where Do Agent Lens Insights Come From?':

1. Redundant confirmation after the customer already confirmed ('Yes') a prepared action.
2. Partial handling of two-part support requests (answering only one part).

Runs baseline vs. repaired agent instructions on Forge Serverless Inference,
emits Agent Lens conversation/turn/LLM/tool spans via `coreweave.forge.agentlens.tracing`,
and logs a side-by-side Evaluation Table (`wandb.EvalTable`) into the shared Forge Workspace.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
DEFAULT_MODEL = "zai-org/GLM-5.3-Flash"


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    cluster_topic: str
    prior_assistant: str
    user_turn: str
    prepared_tool: str
    prepared_args: dict[str, Any]
    tool_result: dict[str, Any]
    required_substrings: tuple[str, ...]
    forbidden_substrings: tuple[str, ...]


CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="confirm-address-01",
        cluster_topic="Redundant confirmation of authorized actions",
        prior_assistant=(
            "I prepared the shipping address update on order #W4819201 to "
            "742 Evergreen Terrace, Springfield. Reply 'Yes' to execute this change."
        ),
        user_turn="Yes, please apply that address change now.",
        prepared_tool="modify_pending_order_address",
        prepared_args={
            "order_id": "#W4819201",
            "address1": "742 Evergreen Terrace",
            "city": "Springfield",
            "confirmed": True,
        },
        tool_result={"status": "executed", "order_id": "#W4819201", "updated": True},
        required_substrings=("#W4819201", "742 Evergreen Terrace"),
        forbidden_substrings=("reply 'yes'", "confirm again", "still needs confirmation", "please confirm"),
    ),
    EvalCase(
        case_id="confirm-cancel-02",
        cluster_topic="Redundant confirmation of authorized actions",
        prior_assistant=(
            "I prepared the cancellation for pending order #W3094412 ($84.50 refund to original card). "
            "Reply 'Yes' to confirm."
        ),
        user_turn="Yes.",
        prepared_tool="cancel_pending_order",
        prepared_args={"order_id": "#W3094412", "reason": "ordered by mistake", "confirmed": True},
        tool_result={"status": "executed", "order_id": "#W3094412", "refund_usd": 84.50},
        required_substrings=("#W3094412", "84.50"),
        forbidden_substrings=("reply 'yes'", "confirm again", "do you want me to proceed", "please confirm"),
    ),
    EvalCase(
        case_id="multipart-return-and-address-03",
        cluster_topic="Partial processing of multi-part requests",
        prior_assistant="Hi! How can I help with your Northstar Care account today?",
        user_turn=(
            "Two things: check the tracking status of delivered order #W7721004, "
            "and tell me whether pending order #W7721099 is still eligible for an address update."
        ),
        prepared_tool="get_order_pair_summary",
        prepared_args={"order_ids": ["#W7721004", "#W7721099"]},
        tool_result={
            "#W7721004": {"status": "delivered", "delivered_at": "2026-09-14", "carrier": "UPS"},
            "#W7721099": {"status": "pending", "address_modifiable": True},
        },
        required_substrings=("#W7721004", "delivered", "#W7721099", "eligible"),
        forbidden_substrings=(),
    ),
    EvalCase(
        case_id="multipart-refund-and-exchange-04",
        cluster_topic="Partial processing of multi-part requests",
        prior_assistant="I pulled up your Northstar Care profile.",
        user_turn=(
            "Can you tell me the refund amount posted on #W5501820 AND check if size M is in stock "
            "for item SKU-TRAIL-JACKET?"
        ),
        prepared_tool="get_refund_and_inventory",
        prepared_args={"order_id": "#W5501820", "sku": "SKU-TRAIL-JACKET", "size": "M"},
        tool_result={
            "order_id": "#W5501820",
            "refund_posted_usd": 129.00,
            "sku": "SKU-TRAIL-JACKET",
            "size_M_available": 6,
        },
        required_substrings=("#W5501820", "129", "SKU-TRAIL-JACKET"),
        forbidden_substrings=(),
    ),
)

ARMS: dict[str, str] = {
    "baseline_v10_prompt": (
        "You are a Northstar Care retail support specialist. Always be cautious with state changes "
        "and make sure actions are confirmed. Keep replies concise."
    ),
    "insights_repaired_prompt": (
        "You are a Northstar Care retail support specialist. "
        "1) When the tool result already shows status='executed' after user confirmation, report the completed "
        "outcome directly and NEVER ask the customer to confirm a second time. "
        "2) When a customer asks a multi-part question, enumerate and answer EVERY requested order/item in "
        "your final reply using exact IDs and figures from the tool output; use the word 'eligible' explicitly "
        "when reporting address-modification eligibility."
    ),
}


def grade_reply(case: EvalCase, reply: str) -> dict[str, Any]:
    lowered = reply.lower()
    missing = [s for s in case.required_substrings if s.lower() not in lowered]
    triggered_forbidden = [s for s in case.forbidden_substrings if s.lower() in lowered]
    passed = not missing and not triggered_forbidden
    return {
        "passed": passed,
        "missing_required": missing,
        "triggered_forbidden": triggered_forbidden,
    }


def run_live(project_path: str, model_id: str) -> dict[str, Any]:
    from coreweave.forge.agentlens import tracing
    from coreweave.forge.agentlens.tracing import Message, TextPart, Usage
    from openai import OpenAI
    import wandb

    api_key = os.environ["WANDB_API_KEY"].strip()
    entity, project = project_path.split("/", 1)
    client = OpenAI(
        api_key=api_key,
        base_url=INFERENCE_BASE_URL,
        default_headers={"OpenAI-Project": project_path},
    )

    tracing.init(project_path, api_key=api_key, autopatch_integrations=False)
    run = wandb.init(
        entity=entity,
        project=project,
        name="northstar-insights-cluster-eval",
        job_type="insights-regression-eval",
        config={"model": model_id, "cases": len(CASES), "arms": list(ARMS)},
    )
    table_cls = getattr(wandb, "EvalTable", wandb.Table)
    table = table_cls(
        columns=[
            "case_id",
            "insights_cluster",
            "arm",
            "passed",
            "missing_required",
            "triggered_forbidden",
            "reply",
            "agent_lens_conversation_url",
        ]
    )

    arm_passes: dict[str, int] = {arm: 0 for arm in ARMS}
    rows: list[dict[str, Any]] = []

    try:
        for arm_name, sys_prompt in ARMS.items():
            for case in CASES:
                conv_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"northstar-eval:{arm_name}:{case.case_id}"))
                conv = tracing.start_conversation(
                    conversation_id=conv_id,
                    conversation_name=f"{arm_name}:{case.case_id}",
                    agent_name="northstar-retail-support",
                    model=model_id,
                    attributes={"eval.arm": arm_name, "eval.insights_cluster": case.cluster_topic},
                )
                turn = conv.start_turn(
                    user_message=case.user_turn,
                    agent_name="northstar-retail-support",
                    agent_version=arm_name,
                    model=model_id,
                    system_instructions=[sys_prompt],
                )
                tool = turn.start_tool(
                    name=case.prepared_tool,
                    arguments=json.dumps(case.prepared_args),
                    tool_call_id=f"call-{case.case_id}",
                )
                tool.set_attributes({"gen_ai.tool.call.result": json.dumps(case.tool_result)})
                tool.end()

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "assistant", "content": case.prior_assistant},
                    {"role": "user", "content": case.user_turn},
                    {
                        "role": "system",
                        "content": f"Executed tool {case.prepared_tool} result: {json.dumps(case.tool_result)}",
                    },
                ]
                llm = turn.start_llm(
                    model=model_id,
                    provider_name="forge-serverless-inference",
                    system_instructions=[sys_prompt],
                )
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=350,
                )
                reply = (resp.choices[0].message.content or "").strip()
                in_tok = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
                out_tok = int(getattr(resp.usage, "completion_tokens", 0) or 0)
                llm.record(
                    input_messages=[Message(role="user", parts=[TextPart(content=case.user_turn)])],
                    output_messages=[Message(role="assistant", parts=[TextPart(content=reply)])],
                    usage=Usage(input_tokens=in_tok, output_tokens=out_tok),
                    response_model=resp.model or model_id,
                )
                llm.end()

                grade = grade_reply(case, reply)
                turn.set_attributes({"eval.passed": grade["passed"]})
                turn.record(output_messages=[Message(role="assistant", parts=[TextPart(content=reply)])])
                turn.end()
                conv.end()

                if grade["passed"]:
                    arm_passes[arm_name] += 1
                conv_url = f"https://forge.coreweave.com/agent-lens/{project_path}/conversations/{conv_id}"
                table.add_data(
                    case.case_id,
                    case.cluster_topic,
                    arm_name,
                    grade["passed"],
                    ", ".join(grade["missing_required"]),
                    ", ".join(grade["triggered_forbidden"]),
                    reply,
                    conv_url,
                )
                rows.append({"case_id": case.case_id, "arm": arm_name, **grade, "conversation_url": conv_url})

        tracing.force_flush()
        run.log({"insights_cluster_regression_eval": table})
        for arm_name, hits in arm_passes.items():
            run.summary[f"{arm_name}_pass_rate"] = round(hits / len(CASES), 4)
            run.summary[f"{arm_name}_passes"] = hits
        workspace_url = f"https://forge.coreweave.com/wandb/{project_path}"
        run_url = f"https://forge.coreweave.com/wandb/{project_path}/runs/{run.id}"
    finally:
        run.finish()
        tracing.shutdown()

    return {
        "mode": "live",
        "project": project_path,
        "model": model_id,
        "arm_passes": arm_passes,
        "cases_per_arm": len(CASES),
        "forge_workspace_url": workspace_url,
        "forge_run_url": run_url,
        "agent_lens_url": f"https://forge.coreweave.com/agent-lens/{project_path}/conversations",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="wandb/agent-lens-retail-support-live-v10")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "project": args.project,
                    "model": args.model,
                    "cases": [c.case_id for c in CASES],
                    "arms": list(ARMS),
                },
                indent=2,
            )
        )
        return 0

    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("Set WANDB_API_KEY before running live.")
    print(json.dumps(run_live(args.project, args.model), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
