#!/usr/bin/env python3
"""Responses-API MCP namespace <-> flat-function bridge in front of purokishi.

Codex CLI 0.167+ wraps MCP server tools inside OpenAI Responses API
`{"type": "namespace", "name": "mcp__wandb", "tools": [...]}` items and expects
returned `function_call` items to carry `"namespace": "mcp__wandb"` + `"name": "<tool>"`.
`purokishi` v0.2.0 translates top-level `{"type": "function"}` tools between
`/v1/responses` and `/v1/chat/completions` on CoreWeave Forge Serverless Inference.

This lightweight local bridge:
1. Unwraps allowed `namespace` tool groups (default: `mcp__wandb`, `multi_agent_v1`)
   into flat `{"type": "function", "name": "<namespace>__<tool>"}` definitions on
   inbound `/v1/responses` requests (and re-prefixes prior `function_call` items in
   `input` so multi-turn history stays consistent).
2. Rewrites `"name": "<namespace>__<tool>"` back into `"namespace": "<namespace>",
   "name": "<tool>"` on streamed SSE events and JSON responses returning to Codex.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

LISTEN_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("BRIDGE_PORT", "18079"))
UPSTREAM_URL = os.environ.get("PUROKISHI_UPSTREAM", "http://127.0.0.1:18080")
ALLOWED_NAMESPACES = tuple(
    ns.strip()
    for ns in os.environ.get("BRIDGE_ALLOWED_NAMESPACES", "mcp__wandb").split(",")
    if ns.strip()
)
SEP = "__NS__"


def transform_request_body(raw: bytes) -> tuple[bytes, set[str]]:
    known_ns: set[str] = set(ALLOWED_NAMESPACES)
    try:
        payload = json.loads(raw)
    except Exception:
        return raw, known_ns
    if not isinstance(payload, dict):
        return raw, known_ns

    tools = payload.get("tools")
    if isinstance(tools, list):
        flat_tools = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            ttype = t.get("type")
            if ttype == "function":
                flat_tools.append(t)
            elif ttype == "namespace":
                ns_name = t.get("name") or ""
                if ns_name in known_ns:
                    for sub in t.get("tools") or []:
                        if isinstance(sub, dict) and sub.get("type") == "function":
                            cloned = dict(sub)
                            cloned["name"] = f"{ns_name}{SEP}{sub['name']}"
                            flat_tools.append(cloned)
        payload["tools"] = flat_tools

    inp = payload.get("input")
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and item.get("type") == "function_call":
                ns = item.get("namespace")
                name = item.get("name")
                if ns and name and SEP not in name:
                    item["name"] = f"{ns}{SEP}{name}"

    return json.dumps(payload).encode("utf-8"), known_ns


def restore_namespace_in_item(item: object) -> None:
    if not isinstance(item, dict):
        return
    if item.get("type") == "function_call":
        name = item.get("name") or ""
        if SEP in name:
            ns, bare = name.split(SEP, 1)
            item["namespace"] = ns
            item["name"] = bare


def transform_json_obj(obj: object) -> object:
    if isinstance(obj, dict):
        restore_namespace_in_item(obj)
        if isinstance(obj.get("item"), dict):
            restore_namespace_in_item(obj["item"])
        resp = obj.get("response")
        if isinstance(resp, dict) and isinstance(resp.get("output"), list):
            for out_item in resp["output"]:
                restore_namespace_in_item(out_item)
        if isinstance(obj.get("output"), list):
            for out_item in obj["output"]:
                restore_namespace_in_item(out_item)
    return obj


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"[codex-ns-bridge] {fmt % args}\n")
        sys.stderr.flush()

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""
        if self.command == "POST" and self.path.endswith("/responses") and body:
            body, _ = transform_request_body(body)

        parsed = urlparse(UPSTREAM_URL)
        conn = http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port or 18080, timeout=300)
        fwd_headers = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in ("host", "content-length", "transfer-encoding", "connection"):
                continue
            fwd_headers[k] = v
        if body:
            fwd_headers["Content-Length"] = str(len(body))
        fwd_headers["Connection"] = "close"

        conn.request(self.command, self.path, body=body if body else None, headers=fwd_headers)
        resp = conn.getresponse()
        ctype = resp.getheader("Content-Type") or ""

        if "text/event-stream" in ctype:
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                line = resp.fp.readline()
                if not line:
                    break
                if line.startswith(b"data: "):
                    data_bytes = line[6:].strip()
                    if data_bytes and data_bytes != b"[DONE]":
                        try:
                            ev = json.loads(data_bytes)
                            transform_json_obj(ev)
                            line = b"data: " + json.dumps(ev).encode("utf-8") + b"\n"
                        except Exception:
                            pass
                self.wfile.write(line)
                if line in (b"\n", b"\r\n"):
                    self.wfile.flush()
            self.wfile.flush()
            conn.close()
            return

        resp_body = resp.read()
        conn.close()
        if "application/json" in ctype and resp_body:
            try:
                parsed_json = json.loads(resp_body)
                transform_json_obj(parsed_json)
                resp_body = json.dumps(parsed_json).encode("utf-8")
            except Exception:
                pass

        self.send_response(resp.status)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(resp_body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if resp_body:
            self.wfile.write(resp_body)
            self.wfile.flush()

    def do_POST(self) -> None:
        self._proxy()

    def do_GET(self) -> None:
        self._proxy()


if __name__ == "__main__":
    srv = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), BridgeHandler)
    sys.stderr.write(f"[codex-ns-bridge] listening on http://{LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM_URL}\n")
    sys.stderr.flush()
    srv.serve_forever()
