"""Web server and live-dashboard host for wifi_room_radar.

This module is the *presentation* edge of the system and is deliberately
decoupled from the DSP pipeline: it never imports pipeline classes. Instead
it talks to a duck-typed **provider** object with exactly two members:

``provider.latest_state() -> SensingState | None``
    The most recent fused sensing state. Must be cheap and thread-safe: the
    pipeline thread keeps overwriting it while the asyncio event loop polls
    it here. Returning ``None`` means "no state published yet".

``provider.info -> dict``
    Static metadata about the source and the room (sample rate, antenna
    counts, optionally ``tx_pos`` / ``rx_positions``), shown in the dashboard
    header and used to place the radio markers on the room map.

Transport model: every websocket client receives one ``{"type": "info"}``
message on connect, then a stream of ``{"type": "state"}`` messages. The
stream is throttled to at most :attr:`ServerConfig.ws_fps` messages per
second and de-duplicated on the state timestamp, so a stalled pipeline sends
nothing (the dashboard keeps showing the last state) and a fast pipeline is
decimated rather than buffered.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import ServerConfig
from .types import SensingState

DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


@runtime_checkable
class StateProvider(Protocol):
    """Structural type of the object the server publishes (duck-typed)."""

    @property
    def info(self) -> dict:  # pragma: no cover - protocol declaration
        ...

    def latest_state(self) -> Optional[SensingState]:  # pragma: no cover
        ...


def _json_default(obj: Any) -> Any:
    """``json.dumps`` fallback that collapses numpy values to plain Python.

    The :class:`~wifi_room_radar.types.SensingState` contract says all fields are
    already plain floats/ints/lists, but a stray ``np.float64`` is easy to
    leak from a numpy reduction and would otherwise kill the websocket
    stream. We duck-type instead of importing numpy: numpy scalars expose
    ``item()`` and arrays expose ``tolist()``.
    """
    item = getattr(obj, "item", None)
    if callable(item):
        with contextlib.suppress(Exception):
            return item()
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        with contextlib.suppress(Exception):
            return tolist()
    return float(obj)


def _dumps(payload: Any) -> str:
    """Serialise a websocket/API payload, tolerating numpy leftovers."""
    return json.dumps(payload, default=_json_default, separators=(",", ":"))


def create_app(provider: StateProvider, ws_fps: float = ServerConfig.ws_fps) -> FastAPI:
    """Build the FastAPI app serving the dashboard and the live state feed.

    Args:
        provider: Duck-typed state provider (see module docstring).
        ws_fps: Maximum websocket pushes per second per client. The poll
            period of the state loop is ``1 / ws_fps``; messages are only
            actually sent when ``latest_state().timestamp`` advances.

    Returns:
        A FastAPI application with routes ``/`` (dashboard), ``/static/*``
        (dashboard assets), ``/api/info`` and the ``/ws`` websocket.
    """
    app = FastAPI(title="wifi_room_radar", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

    poll_interval = 1.0 / max(float(ws_fps), 0.1)

    @app.get("/")
    async def index() -> FileResponse:
        """Serve the dashboard single page."""
        return FileResponse(DASHBOARD_DIR / "index.html", media_type="text/html")

    @app.get("/api/info")
    async def api_info() -> Response:
        """Static source/room metadata as JSON (same payload as the WS hello)."""
        return Response(content=_dumps({"info": provider.info}), media_type="application/json")

    async def _stream_states(ws: WebSocket) -> None:
        """Poll the provider and push state messages until the socket dies.

        Runs as a background task per client. Exits silently on any
        send-side failure (client gone); the parent endpoint also cancels it
        when the receive side observes the disconnect.
        """
        last_ts = float("-inf")
        with contextlib.suppress(WebSocketDisconnect, RuntimeError, ConnectionError, OSError):
            while True:
                state = provider.latest_state()
                if state is not None and state.timestamp > last_ts:
                    last_ts = state.timestamp
                    try:
                        payload: Optional[str] = _dumps(
                            {"type": "state", "data": state.to_json_dict()}
                        )
                    except (TypeError, ValueError):
                        payload = None  # unserialisable state: drop it, keep streaming
                    if payload is not None:
                        await ws.send_text(payload)
                await asyncio.sleep(poll_interval)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        """Live state feed: one info message, then throttled state messages."""
        await ws.accept()
        try:
            await ws.send_text(_dumps({"type": "info", "data": provider.info}))
        except (WebSocketDisconnect, RuntimeError, ConnectionError, OSError):
            return
        sender = asyncio.create_task(_stream_states(ws))
        try:
            while True:
                # The dashboard never sends anything; this await is how
                # starlette surfaces a client disconnect promptly even while
                # the sender task is idle between state updates.
                await ws.receive_text()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender

    return app


def serve(provider: StateProvider, config: ServerConfig) -> None:
    """Run the dashboard server (blocking) with uvicorn.

    Typically called on the main thread while the capture/pipeline threads
    run in the background and keep ``provider.latest_state()`` fresh.
    """
    import uvicorn

    uvicorn.run(
        create_app(provider, ws_fps=config.ws_fps),
        host=config.host,
        port=config.port,
        log_level="warning",
    )
