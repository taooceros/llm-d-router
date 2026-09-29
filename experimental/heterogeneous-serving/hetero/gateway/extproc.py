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
from typing import Any, AsyncIterator

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
    """Runs the existing ASGI app in the gateway's event loop and ownership scope."""

    def __init__(self, app: Any) -> None:
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://omp-gateway",
            timeout=None,
            follow_redirects=False,
        )

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
        method, path, headers = request
        if len(body) > MAX_REQUEST_BYTES:
            return self._immediate(413, b'{"detail":"request body too large"}',
                                   [(b"content-type", b"application/json")])
        # A gRPC cancellation propagates through ASGI to GatewayCore.dispatch,
        # which already cancels the stable owner safely across migrations.
        response = await self.client.request(method, path, headers=headers, content=body)
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise RuntimeError("gateway response exceeds the ExtProc message limit")
        return self._immediate(
            response.status_code, response.content, list(response.headers.raw),
            completion=method == "POST" and path.split("?", 1)[0] == "/v1/completions",
        )

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
                        yield await self._dispatch(request, b"")
                        return
                    yield ext.ProcessingResponse(request_headers=ext.HeadersResponse())
                else:
                    if kind != "request_body" or not message.request_body.end_of_stream:
                        raise ValueError("expected one complete BUFFERED request_body")
                    yield await self._dispatch(request, message.request_body.body)
                    return
            raise ValueError("ExtProc stream ended before the complete request")
        except asyncio.CancelledError:
            raise
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:
            # Never close an incomplete stream with OK: Envoy would then bypass
            # processing. No native state or partial completion is returned.
            import traceback
            from ..evidence import emit_event
            emit_event("GATEWAY_EXTPROC_ERROR", error=f"{type(exc).__name__}: {exc}"[:500],
                       traceback=traceback.format_exc()[-2500:])
            await context.abort(grpc.StatusCode.INTERNAL,
                                f"gateway ExtProc processing failed: {type(exc).__name__}")


class ExtProcServer:
    """Lifecycle companion of the HTTP server; no second gateway core is made."""

    def __init__(self, app: Any, *, host: str = "0.0.0.0", port: int = EXTPROC_PORT) -> None:
        self.processor = GatewayExternalProcessor(app)
        self.server = grpc.aio.server(options=(
            ("grpc.so_reuseport", 0),
            ("grpc.max_receive_message_length", MAX_REQUEST_BYTES + 64 * 1024),
            ("grpc.max_send_message_length", MAX_RESPONSE_BYTES + 64 * 1024),
        ))
        ext_grpc.add_ExternalProcessorServicer_to_server(self.processor, self.server)
        self.port = self.server.add_insecure_port(f"{host}:{port}")
        if not self.port:
            raise RuntimeError(f"cannot bind ExtProc to {host}:{port}")

    async def start(self) -> None:
        await self.server.start()

    async def stop(self) -> None:
        # Active handlers are cancelled before closing their ASGI transport;
        # cancellation is handled by the same core as direct HTTP dispatch.
        await self.server.stop(grace=30)
        await self.processor.close()
