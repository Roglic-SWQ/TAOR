"""vLLM ASGI middleware: abort expired MMKGC requests inside the engine."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import math
import re

from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RequestDeadlineMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-mmkgc-request-id", b"").decode("ascii", errors="ignore")
        raw_timeout = headers.get(b"x-mmkgc-timeout-seconds", b"")
        if (scope["type"] != "http" or scope.get("path") != "/v1/chat/completions"
                or not re.fullmatch(r"[0-9a-f]{32}", request_id) or not raw_timeout):
            return await self.app(scope, receive, send)
        try:
            timeout = float(raw_timeout)
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError()
        except ValueError:
            return await JSONResponse({"error": "Invalid MMKGC request deadline"}, status_code=400)(scope, receive, send)
        started = False
        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)
        task = asyncio.create_task(self.app(scope, receive, tracked_send))
        try:
            done, _ = await asyncio.wait({task}, timeout=min(timeout, 600.0))
            if done:
                return await task
            engine = scope["app"].state.engine_client
            aborted = False
            try:
                await asyncio.wait_for(engine.abort("chatcmpl-" + request_id), timeout=3.0)
                aborted = True
            except Exception:
                logger.exception("MMKGC engine abort failed for %s", request_id)
            logger.warning("MMKGC request deadline expired; request=%s aborted=%s", request_id, aborted)
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=2.0)
            if not started:
                response = JSONResponse({"error": {"type": "request_timeout", "request_id": request_id,
                    "message": "MMKGC request deadline exceeded; engine abort requested"}}, status_code=504,
                    headers={"X-MMKGC-Aborted": "true" if aborted else "false"})
                await response(scope, receive, send)
        finally:
            if not task.done():
                task.cancel()
