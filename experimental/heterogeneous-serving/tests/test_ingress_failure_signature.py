"""A failed request must say which hop answered it.

Two W4 H bf16 reps lost requests in 3-20 ms bursts (router 503, then 500) that
left no gateway record or trace. The client kept only "HTTPStatusError ... for
url", the gateway's ExtProc aborts went to an unretained actor log, and the
discovery 503 named no reason, so Envoy local replies and gateway refusals were
indistinguishable. Separately, the ExtProc path is driven past the client's
admitted cap: 255 in flight is the closed-loop steady state, not a limit.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import grpc
import httpx
from envoy.config.core.v3 import base_pb2
from envoy.service.ext_proc.v3 import external_processor_pb2 as ext
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as ext_grpc

from hetero import experiment
from hetero.gateway.extproc import ExtProcServer, PROTOCOL_HEADER, PROTOCOL_VERSION
from hetero.gateway.service import OmpGateway, build_app

HEADERS = ext.ProcessingRequest(request_headers=ext.HttpHeaders(headers=base_pb2.HeaderMap(headers=[
    base_pb2.HeaderValue(key=":method", value="POST"),
    base_pb2.HeaderValue(key=":path", value="/v1/completions"),
    base_pb2.HeaderValue(key="content-type", value="application/json"),
])))


def body(request_id, *, end_of_stream=True):
    return ext.ProcessingRequest(request_body=ext.HttpBody(
        body=json.dumps({"request_id": request_id, "prompt_token_ids": [1], "max_tokens": 1}).encode(),
        end_of_stream=end_of_stream,
    ))


async def process(stub, *messages):
    async def stream():
        for message in messages:
            yield message
    return [response async for response in stub.Process(stream())]


class ClientFailureSignatureTests(unittest.TestCase):
    def test_failed_records_keep_status_body_and_the_answering_hop(self):
        refusal = {"detail": "logical-instance discovery is not ready: discovery health lease expired"}

        def handler(request):
            if json.loads(request.content)["request_id"] == "ns:gateway-refused":
                return httpx.Response(503, json=refusal, headers={PROTOCOL_HEADER: PROTOCOL_VERSION})
            # Envoy's reply to a failed ExtProc stream: no marker, no body.
            return httpx.Response(500, headers={"server": "envoy"})

        real = httpx.AsyncClient
        payload = [{"request_id": f"ns:{name}", "prompt_token_ids": [1], "max_tokens": 1}
                   for name in ("gateway-refused", "router-500")]
        with mock.patch("httpx.AsyncClient",
                        lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs)):
            records = asyncio.run(experiment._dispatch_http(payload, concurrency=2, timeout_s=30.0))
        by_id = {record["request_id"]: record for record in records}
        gateway, router = by_id["ns:gateway-refused"], by_id["ns:router-500"]
        self.assertTrue(gateway["error"].startswith("HTTPStatusError: Server error '503"))
        self.assertEqual((gateway["http_status"], gateway["response_protocol"]), (503, PROTOCOL_VERSION))
        self.assertEqual(json.loads(gateway["response_body"]), refusal)
        self.assertEqual((router["http_status"], router["response_protocol"], router["response_body"]),
                         (500, None, ""))
        self.assertEqual(router["response_headers"]["server"], "envoy")

        aborts = {"count": 1, "recent": [{"at_s": 1.0, "code": "INTERNAL", "error": "RuntimeError: x"}]}
        summary = experiment._summarise(
            arm="H", workload="W4", repetition=1,
            manifest={"manifest_sha256": "m", "useful_output_tokens": 2,
                      "requests": [{"request_id": "gateway-refused", "max_tokens": 1},
                                   {"request_id": "router-500", "max_tokens": 1}]},
            records=records, makespan_s=1.0,
            status={"core": {"state": {"transactions": [], "counters": {}}}, "extproc_aborts": aborts},
            iteration_stats={"one_to_one": True, "per_instance": {}}, instances=[], request_id_namespace="ns",
        )
        self.assertEqual({row["request_id"]: (row["http_status"], row["response_protocol"])
                          for row in summary["failed_requests"]},
                         {"ns:gateway-refused": (503, PROTOCOL_VERSION), "ns:router-500": (500, None)})
        self.assertEqual(summary["gateway_extproc_aborts"], aborts)


class GatewayFailureSignatureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.patch = mock.patch("hetero.gateway.service.ray.util.get_node_ip_address", return_value="127.0.0.1")
        self.patch.start()
        self.gateway = OmpGateway.__ray_metadata__.modified_class(config={
            "layout_id": "H", "ledger_path": str(Path(self.directory.name) / "ownership.json"),
            "policy": {"small_instances": [], "large_instances": ["large_a", "large_b"],
                       "migration_enabled": False},
        })
        self.server = ExtProcServer(build_app(self.gateway.state), host="127.0.0.1", port=0)
        await self.server.start()
        self.gateway.state._extproc = self.server
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{self.server.port}")
        self.stub = ext_grpc.ExternalProcessorStub(self.channel)

    async def asyncTearDown(self):
        await self.channel.close()
        await self.server.stop()
        self.patch.stop()
        self.directory.cleanup()

    async def test_discovery_refusal_names_its_reason_under_the_gateway_marker(self):
        self.gateway.state._discovery = {"ready": True}
        self.gateway.state.discovery_expires_at = 0.0  # health lease already expired
        responses = await process(self.stub, HEADERS, body("lease-expired"))
        immediate = responses[-1].immediate_response
        marker = {h.header.key: h.header.raw_value.decode() for h in immediate.headers.set_headers}
        self.assertEqual((immediate.status.code, marker[PROTOCOL_HEADER]), (503, PROTOCOL_VERSION))
        self.assertEqual(json.loads(immediate.body)["detail"],
                         "logical-instance discovery is not ready: discovery health lease expired")
        self.assertEqual(self.gateway.status()["extproc_aborts"], {"count": 0, "recent": []})

    async def test_aborted_streams_are_counted_where_run_artifacts_read_them(self):
        with self.assertRaises(grpc.aio.AioRpcError) as raised:
            await process(self.stub, HEADERS, body("partial", end_of_stream=False))
        self.assertEqual(raised.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        aborts = self.gateway.status()["extproc_aborts"]
        self.assertEqual(aborts["count"], 1)
        self.assertEqual((aborts["recent"][0]["code"], aborts["recent"][0]["error"]),
                         ("INVALID_ARGUMENT", "ValueError: expected one complete BUFFERED request_body"))


class ExtProcConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_twice_the_admitted_cap_streams_are_served_concurrently_on_one_connection(self):
        # Guard: no ExtProc stream/RPC cap at or below the client's 256 admitted cap.
        # Every request is held until all of them are inside the app at once.
        streams = 2 * experiment.ADMITTED_CAP
        inside, everyone = 0, asyncio.Event()

        async def app(scope, receive, send):
            nonlocal inside
            more = True
            while more:
                more = (await receive()).get("more_body", False)
            inside += 1
            if inside == streams:
                everyone.set()
            await asyncio.wait_for(everyone.wait(), 30)
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b"{}"})

        server = ExtProcServer(app, host="127.0.0.1", port=0)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                stub = ext_grpc.ExternalProcessorStub(channel)
                results = await asyncio.gather(*[process(stub, HEADERS, body(f"r{index}"))
                                                 for index in range(streams)])
        finally:
            await server.stop()
        self.assertEqual(inside, streams)
        self.assertEqual({responses[-1].immediate_response.status.code for responses in results}, {200})


if __name__ == "__main__":
    unittest.main()
