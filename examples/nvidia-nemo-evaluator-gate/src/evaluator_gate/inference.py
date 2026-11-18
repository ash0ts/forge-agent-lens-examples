"""Forge Serverless Inference client. OpenAI-compatible; the only credential is WANDB_API_KEY."""
from __future__ import annotations

import os
import time

import httpx

INFERENCE_URL = "https://api.inference.wandb.ai/v1"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"


def api_key() -> str:
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        import netrc
        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth and auth[2]:
            return auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    raise SystemExit("Set WANDB_API_KEY (or add api.wandb.ai to ~/.netrc) to call Forge Serverless Inference.")


class ForgeInference:
    """One chat completion per call. Errors are returned, not raised, so the gate can count them."""

    def __init__(
        self,
        project: str,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 120,
        *,
        enable_thinking: bool = False,
    ):
        self.project = project
        self.model = model
        self.enable_thinking = enable_thinking
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=10),
            transport=httpx.HTTPTransport(retries=1),
            headers={"Authorization": "Bearer " + api_key(), "OpenAI-Project": project},
        )

    def close(self):
        self.client.close()

    def ask(self, system: str, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> dict:
        started = time.perf_counter()
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "nemotron" in self.model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        record = {
            "requested_model": self.model,
            "enable_thinking": self.enable_thinking,
            "max_tokens": max_tokens,
            "endpoint": INFERENCE_URL,
        }
        try:
            response = self.client.post(INFERENCE_URL + "/chat/completions", json=payload)
            record["http_status"] = response.status_code
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            record.update(
                text=choice["message"].get("content") or "",
                model=data.get("model"),
                usage=usage,
                finish_reason=choice.get("finish_reason"),
                response_id=data.get("id"),
                error=None,
            )
        except Exception as exc:  # keep failed calls in the denominator with their error class
            record.update(
                text="",
                model=self.model,
                usage={},
                finish_reason=None,
                error=type(exc).__name__,
            )
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return record
