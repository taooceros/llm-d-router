"""Envoy v3 ExternalProcessor serving adapter, not an upstream endpoint picker.

The installed llm-d router uses Envoy 1.33.2 and the standard bidirectional
``envoy.service.ext_proc.v3.ExternalProcessor/Process`` RPC. This adapter uses
SEND request headers + BUFFERED request body and returns one ImmediateResponse
containing the *complete* existing gateway HTTP response. Response processing
is disabled. It deliberately does not set ``x-gateway-destination-endpoint`` or
route client traffic to an engine: only GatewayCore owns placement, migration,
request identity, committed token offsets, and response assembly. Native EPP
routing would bypass that owner and cannot safely reassemble migrated results.

The InferencePool Service reference is gRPC :9002 -> actor :8201; the separate
HTTP :80 -> :8200 port remains an administrative/direct diagnostic surface.
The router must fail closed and have an unreachable-route local 503 fallback,
not an HTTP backend fallback. Non-streaming greedy completions retain the
HTTP API's semantics and bytes, including errors; streaming remains rejected.
The deployed BUFFERED contract permits 16 MiB requests, 64 MiB responses,
and at most 3600 seconds per processing message (Envoy 1.33's supported cap).
Only a completion response through this RPC carries the protocol marker,
including HTTP errors returned by the completion handler.

Runtime: xds-protos==1.84.0 (official generated Envoy protos), grpcio>=1.74,
protobuf>=6.31.1,<8, httpx==0.28.1, plus the HTTP service's FastAPI/Uvicorn.
There is no invented llm-d plugin or independent EPP placement policy here.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import threading
import time
from typing import Any, AsyncIterator, Optional

import grpc
import httpx
from envoy.config.core.v3 import base_pb2
from envoy.service.ext_proc.v3 import external_processor_pb2 as ext
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as ext_grpc
from envoy.type.v3 import http_status_pb2

EXTPROC_PORT = 8201
PROTOCOL_HEADER = "x-omp-gateway-protocol"
PROTOCOL_VERSION = "envoy-ext-proc-v3"
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_HOP_HEADERS = {
    b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
    b"te", b"trailer", b"transfer-encoding", b"upgrade", b"content-length",
}


class GatewayExternalProcessor(ext_grpc.ExternalProcessorServicer):
    """Frames streams on the gRPC loop; runs the existing ASGI app in the gateway's
    event loop and ownership scope."""

    def __init__(self, app: Any, app_loop: asyncio.AbstractEventLoop) -> None:
        self.app_loop = app_loop
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://omp-gateway",
            timeout=None,
            follow_redirects=False,
        )
        # Envoy reports every stream this processor aborts as a bodiless HTTP 500
        # with upstream "-", indistinguishable from a transport failure. Keep the
        # gateway's side of that signature where run artifacts can read it.
        self._aborts_lock = threading.Lock()
        self._aborts: collections.deque = collections.deque(maxlen=64)
        self._abort_count = 0

    def _record_abort(self, code: grpc.StatusCode, exc: BaseException) -> None:
        with self._aborts_lock:
            self._abort_count += 1
            self._aborts.append({"at_s": time.time(), "code": code.name,
                                 "error": f"{type(exc).__name__}: {exc}"[:300]})

    def aborts(self) -> dict:
        """Streams aborted by this processor (Envoy answers them 500, upstream "-")."""
        with self._aborts_lock:
            return {"count": self._abort_count, "recent": list(self._aborts)}

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def _headers(message: Any) -> tuple[str, str, list[tuple[bytes, bytes]]]:
        pseudo: dict[str, str] = {}
        headers = []
        for header in message.headers.headers:
            key = header.key.lower()
            value = header.raw_value if header.raw_value else header.value.encode("utf-8")
            if key.startswith(":"):
                if key in pseudo:
                    raise ValueError("duplicate pseudo-header")
                pseudo[key] = value.decode("ascii")
            elif key.encode("ascii") not in _HOP_HEADERS:
                headers.append((key.encode("ascii"), value))
        method, path = pseudo.get(":method", ""), pseudo.get(":path", "")
        if not method.isalpha() or not method.isupper():
            raise ValueError("missing or invalid :method")
        if not path.startswith("/") or path.startswith("//") or "#" in path:
            raise ValueError("missing or invalid origin-form :path")
        if any(key == b"content-encoding" and value.lower() != b"identity"
               for key, value in headers):
            raise ValueError("compressed request bodies are unsupported")
        return method, path, headers

    @staticmethod
    def _immediate(status: int, body: bytes, headers: list[tuple[bytes, bytes]],
                   *, completion: bool = False) -> Any:
        mutations = []
        for key, value in headers:
            key = key.lower()
            if key in _HOP_HEADERS or key == PROTOCOL_HEADER.encode("ascii"):
                continue
            mutations.append(base_pb2.HeaderValueOption(
                header=base_pb2.HeaderValue(key=key.decode("ascii"), raw_value=value),
                append_action=base_pb2.HeaderValueOption.OVERWRITE_IF_EXISTS_OR_ADD,
            ))
        if completion:
            mutations.append(base_pb2.HeaderValueOption(
                header=base_pb2.HeaderValue(
                    key=PROTOCOL_HEADER, raw_value=PROTOCOL_VERSION.encode("ascii")),
                append_action=base_pb2.HeaderValueOption.OVERWRITE_IF_EXISTS_OR_ADD,
            ))
        return ext.ProcessingResponse(immediate_response=ext.ImmediateResponse(
            status=http_status_pb2.HttpStatus(code=status),
            headers=ext.HeaderMutation(set_headers=mutations),
            body=body,
            details="omp_gateway_ext_proc",
        ))

    async def _dispatch(self, request: tuple[str, str, list[tuple[bytes, bytes]]],
                        body: bytes) -> Any:
        """Runs on the gateway loop."""
        method, path, headers = request
        if len(body) > MAX_REQUEST_BYTES:
            return self._immediate(413, b'{"detail":"request body too large"}',
                                   [(b"content-type", b"application/json")])
        response = await self.client.request(method, path, headers=headers, content=body)
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise RuntimeError("gateway response exceeds the ExtProc message limit")
        return self._immediate(
            response.status_code, response.content, list(response.headers.raw),
            completion=method == "POST" and path.split("?", 1)[0] == "/v1/completions",
        )

    async def _on_app_loop(self, request: tuple[str, str, list[tuple[bytes, bytes]]],
                           body: bytes) -> Any:
        # A gRPC cancellation cancels the gateway-loop task and propagates
        # through ASGI to GatewayCore.dispatch, which already cancels the
        # stable owner safely across migrations.
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(self._dispatch(request, body), self.app_loop))

    async def Process(self, request_iterator: Any, context: Any) -> AsyncIterator[Any]:
        request = None
        try:
            async for message in request_iterator:
                if message.observability_mode:
                    raise ValueError("observability mode cannot serve a response")
                kind = message.WhichOneof("request")
                if request is None:
                    if kind != "request_headers":
                        raise ValueError("request_headers must be the first message")
                    request = self._headers(message.request_headers)
                    if message.request_headers.end_of_stream:
                        yield await self._on_app_loop(request, b"")
                        return
                    yield ext.ProcessingResponse(request_headers=ext.HeadersResponse())
                else:
                    if kind != "request_body" or not message.request_body.end_of_stream:
                        raise ValueError("expected one complete BUFFERED request_body")
                    yield await self._on_app_loop(request, message.request_body.body)
                    return
            raise ValueError("ExtProc stream ended before the complete request")
        except asyncio.CancelledError:
            raise
        except ValueError as exc:
            self._record_abort(grpc.StatusCode.INVALID_ARGUMENT, exc)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            # Never close an incomplete stream with OK: Envoy would then bypass
            # processing. No native state or partial completion is returned.
            import traceback
            from ..evidence import emit_event
            emit_event("GATEWAY_EXTPROC_ERROR", error=f"{type(exc).__name__}: {exc}"[:500],
                       traceback=traceback.format_exc()[-2500:])
            self._record_abort(grpc.StatusCode.INTERNAL, exc)
            await context.abort(grpc.StatusCode.INTERNAL,
                                f"gateway ExtProc processing failed: {type(exc).__name__}")


class ExtProcServer:
    """Lifecycle companion of the HTTP server; no second gateway core is made.

    gRPC runs on its own thread and event loop. grpc.aio accepts one call per
    loop round trip, and gRPC core cancels a call that stays unaccepted for
    30 s (grpc.server_max_unrequested_time_in_server). On a saturated gateway loop,
    Envoy's streams were cancelled that way before any gateway code saw them
    (HTTP 500, upstream "-"). Only stream framing runs on this loop; every
    request still dispatches on the gateway loop that owns GatewayCore.
    """

    def __init__(self, app: Any, *, host: str = "0.0.0.0", port: int = EXTPROC_PORT) -> None:
        self.app = app
        self.address = f"{host}:{port}"
        self.port = 0
        self.processor: Optional[GatewayExternalProcessor] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stopping: Optional[asyncio.Event] = None

    async def start(self) -> None:
        self.processor = GatewayExternalProcessor(self.app, asyncio.get_running_loop())
        bound: concurrent.futures.Future = concurrent.futures.Future()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve(bound)),
                                        name="omp-gateway-extproc", daemon=True)
        self._thread.start()
        self.port = await asyncio.wrap_future(bound)

    async def _serve(self, bound: concurrent.futures.Future) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopping = asyncio.Event()
        try:
            server = grpc.aio.server(options=(
                ("grpc.so_reuseport", 0),
                ("grpc.max_receive_message_length", MAX_REQUEST_BYTES + 64 * 1024),
                ("grpc.max_send_message_length", MAX_RESPONSE_BYTES + 64 * 1024),
            ))
            ext_grpc.add_ExternalProcessorServicer_to_server(self.processor, server)
            port = server.add_insecure_port(self.address)
            if not port:
                raise RuntimeError(f"cannot bind ExtProc to {self.address}")
            await server.start()
        except BaseException as exc:
            bound.set_exception(exc)
            return
        bound.set_result(port)
        await self._stopping.wait()
        # Active handlers are cancelled after the grace period; cancellation
        # reaches the same core as direct HTTP dispatch on the gateway loop.
        await server.stop(grace=30)

    def aborts(self) -> Optional[dict]:
        return self.processor.aborts() if self.processor is not None else None

    async def stop(self) -> None:
        if self._thread is not None:
            if self._loop is not None and self._stopping is not None:
                try:
                    self._loop.call_soon_threadsafe(self._stopping.set)
                except RuntimeError:
                    pass  # _serve already returned (failed start) and closed its loop
            await asyncio.to_thread(self._thread.join)
            self._thread = None
        if self.processor is not None:
            await self.processor.close()
            self.processor = None
