from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .chat_runtime import (
    MAX_MESSAGE_CHARACTERS,
    ChatRuntime,
    ChatRuntimeError,
    SessionBusy,
    SessionNotFound,
)
from .constants import PACKAGE_ROOT


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_id: str | None = None


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARACTERS)


def _translate_error(exc: BaseException) -> HTTPException:
    if isinstance(exc, SessionNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, SessionBusy):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ChatRuntimeError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(
        status_code=502,
        detail="The support agent could not complete this turn.",
    )


def create_app(
    runtime: ChatRuntime | None = None,
    *,
    static_dir: Path | None = None,
) -> FastAPI:
    chat = runtime or ChatRuntime.from_environment()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        chat.close()

    app = FastAPI(
        title="Agent Lens Retail Support Demo",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.chat_runtime = chat

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return chat.health()

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return chat.config()

    @app.post("/api/conversations", status_code=201)
    def create_conversation(
        request: CreateConversationRequest,
    ) -> dict[str, Any]:
        try:
            return chat.create(route_id=request.route_id)
        except Exception as exc:
            raise _translate_error(exc) from exc

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        try:
            return chat.get(conversation_id)
        except Exception as exc:
            raise _translate_error(exc) from exc

    @app.post("/api/conversations/{conversation_id}/messages")
    def send_message(
        conversation_id: str,
        request: SendMessageRequest,
    ) -> dict[str, Any]:
        # This is deliberately a synchronous endpoint. FastAPI runs it in its
        # worker threadpool, so provider calls, MCP stdio, tool latency, and an
        # Agent Lens flush never block the event loop.
        try:
            return chat.send(conversation_id, request.content)
        except Exception as exc:
            raise _translate_error(exc) from exc

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, bool]:
        try:
            return chat.delete(conversation_id)
        except Exception as exc:
            raise _translate_error(exc) from exc

    frontend = (
        static_dir
        or Path(
            os.environ.get(
                "RETAIL_DEMO_STATIC_DIR",
                PACKAGE_ROOT / "src" / "retail_support_study_v1" / "web_static",
            )
        )
    ).resolve()
    if frontend.is_dir():
        index = frontend / "index.html"

        if index.is_file():

            @app.get("/", include_in_schema=False)
            def index_page() -> FileResponse:
                return FileResponse(index)

        app.mount(
            "/static",
            StaticFiles(directory=frontend),
            name="frontend-static",
        )

    return app


app = create_app()


def main() -> None:
    """Run the single-process interactive demo server."""

    import uvicorn

    uvicorn.run(
        "retail_support_study_v1.web:app",
        host=os.environ.get("RETAIL_DEMO_HOST", "127.0.0.1"),
        port=int(os.environ.get("RETAIL_DEMO_PORT", "4180")),
        workers=1,
    )
