"""Control-plane lifecycle regressions over the real service/core and gRPC wire.

Native peers below implement RPC acknowledgements only; they model neither
accelerator execution nor performance. No Ray cluster or Kubernetes API runs.
"""

import asyncio
import copy
import inspect
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import grpc
from envoy.config.core.v3 import base_pb2
from envoy.service.ext_proc.v3 import external_processor_pb2 as ext
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as ext_grpc

from hetero.gateway.discovery import DiscoveryBridge, GENERATION_LABEL
from hetero.gateway.extproc import ExtProcServer, PROTOCOL_HEADER, PROTOCOL_VERSION
from hetero.gateway.service import OmpGateway, build_app
from test_gateway_discovery import KubeFixture
from test_gateway_handoff import LocalReference


class Method:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        async def invoke():
            result = self.fn(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        return LocalReference(asyncio.create_task(invoke()))


class Actor:
    def __init__(self, native):
        self.native = native
        self._actor_id = object()

    def __getattr__(self, name):
        return Method(getattr(self.native, name))


class NativePeer:
    def __init__(self, row):
        self.row = row
        self.calls = []
        self.started = asyncio.Event()
        self.drain_started = asyncio.Event()
        self.drain_allowed = asyncio.Event()
        self.drain_allowed.set()
        self.result_allowed = asyncio.Event()
        self.result_allowed.set()
        self.drained = False
        self.drain_ack = True

    def identity(self):
        return {key: self.row[key] for key in ("instance_id", "layout_id", "generation")} | {
            "node_ip": self.row["endpoint_host"], "layout_digest": self.row["config_digest"],
            "state": "drained" if self.drained else "ready",
        }

    def status(self):
        return {"generation": self.row["generation"], "state": "draining" if self.drained else "ready",
                "total_blocks": 32, "free_blocks": 32, "block_size": 4,
                "num_running": 0, "num_waiting": 0}

    async def generate(self, **request):
        if self.drained:
            raise RuntimeError("native admission is fenced")
        self.calls.append(request["request_id"])
        self.started.set()
        await self.result_allowed.wait()
        return {"output_token_ids": [self.row["generation"]], "output_tokens": 1,
                "text": "contract-token", "finish_reason": "length"}

    async def drain(self, timeout_s):
        self.drain_started.set()
        await self.drain_allowed.wait()
        self.drained = True
        return {"instance_id": self.row["instance_id"], "drained": self.drain_ack,
                "inflight": 0 if self.drain_ack else 1}

    async def await_idle(self, timeout_s):
        return {"idle": True, "inflight": 0, "pending_batches": 0, "scheduler_has_work": False}

    def idle_snapshot(self):
        return {"all_blocks_free": True, "num_running": 0, "num_waiting": 0, "inflight": 0}


class Registry:
    def __init__(self, rows):
        self.rows = rows
        self.layout = {"instances": [{"instance_id": row["instance_id"]} for row in rows],
                       "layout_digest": "static-layout"}

    def snapshot(self, layout_id):
        return copy.deepcopy(self.rows)

    def get_layout(self, layout_id):
        return copy.deepcopy(self.layout)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.kube = KubeFixture()
        self.rows = [{"instance_id": name, "layout_id": "M", "generation": 1,
                      "config_digest": "static-layout", "engine_actor_name": f"actor-{name}",
                      "ray_namespace": "omp_hetero", "placement_group_id": f"pg-{name}",
                      "endpoint_host": ip, "endpoint_port": 8000, "endpoint": f"{ip}:8000",
                      "host_chip_rank_map": [{"process_rank": 0, "node_ip": ip}],
                      "state": "ready", "routable": True, "health_age_s": 0,
                      "tp_size": 8, "max_model_len": 128}
                     for name, ip in (("small_a", "10.0.0.1"), ("small_b", "10.0.0.2"))]
        self.registry = Registry(self.rows)
        self.native = {row["instance_id"]: NativePeer(copy.deepcopy(row)) for row in self.rows}
        self.actors = {row["engine_actor_name"]: Actor(self.native[row["instance_id"]]) for row in self.rows}
        self.patches = [
            mock.patch("hetero.gateway.service.ray.util.get_node_ip_address", return_value="127.0.0.1"),
            mock.patch("hetero.gateway.service.find_registry", return_value=Actor(self.registry)),
            mock.patch("hetero.gateway.service.ray.get_actor", side_effect=lambda name, namespace: self.actors[name]),
        ]
        for patch in self.patches:
            patch.start()
        self.gateway = OmpGateway.__ray_metadata__.modified_class(config={
            "layout_id": "M", "ledger_path": str(Path(self.directory.name) / "ownership.json"),
            "policy": {"small_instances": ["small_a", "small_b"], "large_instances": [],
                       "migration_enabled": False},
        })
        self.gateway.state.bridge = DiscoveryBridge(kube=self.kube)
        await self.gateway.reconcile()
        self.server = ExtProcServer(build_app(self.gateway.state), host="127.0.0.1", port=0)
        await self.server.start()
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{self.server.port}")
        self.stub = ext_grpc.ExternalProcessorStub(self.channel)

    async def asyncTearDown(self):
        await self.channel.close()
        await self.server.stop()
        for patch in reversed(self.patches):
            patch.stop()
        self.directory.cleanup()

    async def completion(self, request_id):
        async def messages():
            yield ext.ProcessingRequest(request_headers=ext.HttpHeaders(headers=base_pb2.HeaderMap(headers=[
                base_pb2.HeaderValue(key=":method", value="POST"),
                base_pb2.HeaderValue(key=":path", value="/v1/completions"),
                base_pb2.HeaderValue(key="content-type", value="application/json"),
            ])))
            yield ext.ProcessingRequest(request_body=ext.HttpBody(
                body=json.dumps({"request_id": request_id, "prompt_token_ids": [1], "max_tokens": 1}).encode(),
                end_of_stream=True,
            ))
        responses = [response async for response in self.stub.Process(messages())]
        immediate = responses[-1].immediate_response
        marker = {h.header.key: h.header.raw_value.decode() for h in immediate.headers.set_headers}
        self.assertEqual(marker[PROTOCOL_HEADER], PROTOCOL_VERSION)
        return immediate.status.code, json.loads(immediate.body)

    async def test_peers_serve_through_extproc_during_drain_and_fresh_same_address_restart(self):
        core, scheduler = self.gateway.state.core, self.gateway.state.core.scheduler
        code, first = await self.completion("before")
        self.assertEqual((code, first["omp"]["initial_instance"]), (200, "small_b"))
        original = self.native["small_b"]
        maintenance = await self.gateway.begin_maintenance(["small_b"], reason="scoped qualification")
        self.assertEqual(self.gateway.status()["instances"], ["small_a"])
        with self.assertRaises(RuntimeError):
            await self.gateway.start_capture()
        original.drain_allowed.clear()
        draining = asyncio.create_task(self.gateway.drain_maintenance_instance(
            "small_b", maintenance_id=maintenance["id"], timeout_s=1))
        await original.drain_started.wait()
        # A long native drain must not hold the discovery refresh lock until
        # peer leases expire and turn the whole gateway into a 503.
        await asyncio.wait_for(self.gateway.reconcile(bind_labels=False), timeout=1)
        code, peer = await self.completion("while-draining")
        self.assertEqual((code, peer["omp"]["initial_instance"]), (200, "small_a"))
        self.assertFalse(draining.done())
        original.drain_allowed.set()
        await draining
        self.registry.rows = [self.rows[0]]
        del self.actors["actor-small_b"]
        await self.gateway.reconcile()
        self.assertEqual((await self.completion("withdrawn"))[0], 200)
        replacement = {**self.rows[1], "generation": 2, "placement_group_id": "new-pg"}
        self.registry.rows.append(replacement)
        fresh = NativePeer(replacement)
        self.actors["actor-small_b"] = Actor(fresh)
        await self.gateway.reconcile()
        self.assertEqual(self.gateway.status()["instances"], ["small_a"])
        await self.gateway.finish_maintenance(maintenance_id=maintenance["id"])
        code, restored = await self.completion("restored")
        self.assertEqual((code, restored["omp"]["initial_instance"]), (200, "small_b"))
        self.assertEqual(restored["choices"][0]["token_ids"], [2])
        self.assertEqual(original.calls, ["before"])
        self.assertEqual(fresh.calls, ["restored"])
        self.assertEqual(self.kube.pods[2]["metadata"]["labels"][GENERATION_LABEL], "2")
        self.assertIs(self.gateway.state.core, core)
        self.assertIs(core.scheduler, scheduler)
        self.assertEqual(core.state.owner("before").instance_generation, 1)
        self.assertFalse(self.gateway.status()["maintenance"])

    async def test_gateway_up_waits_for_maintenance_discovery_and_fails_closed(self):
        from types import SimpleNamespace
        from hetero.gateway import lifecycle

        loop = asyncio.get_running_loop()
        proxy = mock.Mock()

        def remote(method):
            def submit(*args, **kwargs):
                async def invoke():
                    result = method(*args, **kwargs)
                    return await result if inspect.isawaitable(result) else result
                return asyncio.run_coroutine_threadsafe(invoke(), loop)
            return submit

        for name in ("status", "finish_maintenance", "set_policy", "reconcile"):
            getattr(proxy, name).remote.side_effect = remote(getattr(self.gateway, name))
        # The real ExtProc server is already running in this fixture. Only
        # starting another listening socket is outside this discovery test.
        proxy.start.remote.side_effect = remote(lambda: None)
        condition = self.kube.pods[2]["status"]["conditions"][0]
        for recovers in (True, False):
            with self.subTest(recovers=recovers):
                maintenance = await self.gateway.begin_maintenance(["small_b"], reason="probe readiness race")
                await self.gateway.drain_maintenance_instance(
                    "small_b", maintenance_id=maintenance["id"], timeout_s=1)
                previous = self.registry.rows[1]
                replacement = {**previous, "generation": previous["generation"] + 1,
                               "placement_group_id": f"replacement-{recovers}"}
                self.registry.rows[1] = replacement
                self.actors[replacement["engine_actor_name"]] = Actor(NativePeer(replacement))
                condition["status"] = "False"
                now = [0.0]

                def sleep(seconds):
                    async def tick():
                        self.assertEqual(self.gateway.status()["maintenance"]["id"], maintenance["id"])
                        self.assertEqual(self.gateway.status()["instances"], ["small_a"])
                        if recovers:
                            condition["status"] = "True"
                    asyncio.run_coroutine_threadsafe(tick(), loop).result(timeout=5)
                    now[0] += seconds

                clock = SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
                with mock.patch.object(lifecycle, "find_gateway", return_value=proxy), \
                     mock.patch.object(lifecycle.ray, "get", side_effect=lambda value, timeout: value.result(timeout=timeout)), \
                     mock.patch.object(lifecycle, "time", clock):
                    operation = asyncio.to_thread(lifecycle.gateway_up, layout_id="M", discovery_timeout_s=10 if recovers else 1)
                    if recovers:
                        await operation
                    else:
                        with self.assertRaises(RuntimeError):
                            await operation
                code, response = await self.completion(f"readiness-{recovers}")
                self.assertEqual(code, 200)
                self.assertEqual(response["omp"]["initial_instance"], "small_b" if recovers else "small_a")
                self.assertEqual(bool(self.gateway.status()["maintenance"]), not recovers)

    async def test_live_responses_capture_and_nonterminal_migration_refuse_maintenance(self):
        native = self.native["small_b"]
        native.result_allowed.clear()
        request = asyncio.create_task(self.completion("inflight"))
        await native.started.wait()
        with self.assertRaises(RuntimeError):
            await self.gateway.begin_maintenance(["small_b"], reason="must refuse live owner")
        native.result_allowed.set()
        self.assertEqual((await request)[0], 200)
        self.gateway.state.capturing = True
        with self.assertRaises(RuntimeError):
            await self.gateway.reconcile(bind_labels=True)
        self.assertTrue(self.gateway.status()["discovery"]["ready"])
        with self.assertRaises(RuntimeError):
            await self.gateway.begin_maintenance(["small_b"], reason="must refuse timed arm")
        self.gateway.state.capturing = False
        state = self.gateway.state.core.state
        state.claim("unresolved", "small_a", 1)
        transaction, _ = state.open_transaction(
            request_id="unresolved", source_instance="small_a", destination_instance="small_b",
            destination_generation=1, mechanism="recompute", blocks=1, growth_blocks=0,
            max_inflight_per_destination=1,
        )
        self.assertIsNotNone(transaction)
        with self.assertRaises(RuntimeError):
            await self.gateway.begin_maintenance(["small_b"], reason="must retain handoff")
        self.assertFalse(self.gateway.status()["maintenance"])
        self.assertEqual(self.gateway.status()["instances"], ["small_a", "small_b"])

    async def test_unacknowledged_drain_and_reused_generation_cannot_promote(self):
        maintenance = await self.gateway.begin_maintenance(["small_b"], reason="failure qualification")
        self.native["small_b"].drain_ack = False
        with self.assertRaises(RuntimeError):
            await self.gateway.drain_maintenance_instance("small_b", maintenance_id=maintenance["id"], timeout_s=1)
        self.assertEqual(self.gateway.status()["maintenance"]["drained"], {})
        with self.assertRaises(RuntimeError):
            await self.gateway.finish_maintenance(maintenance_id=maintenance["id"])
        self.assertEqual((await self.completion("surviving-peer"))[0], 200)
        self.native["small_b"].drain_ack = True
        await self.gateway.drain_maintenance_instance("small_b", maintenance_id=maintenance["id"], timeout_s=1)
        with self.assertRaises(RuntimeError):
            await self.gateway.finish_maintenance(maintenance_id=maintenance["id"])
        self.assertEqual((await self.completion("still-excluded"))[0], 200)

    async def test_watch_fails_closed_instead_of_rebinding_unannounced_generation(self):
        self.rows[1]["generation"] = 2
        with self.assertRaises(RuntimeError):
            await self.gateway._reconcile(bind_labels=False, observe_only=True)
        self.assertEqual((await self.completion("stale-view"))[0], 503)
        with self.assertRaises(RuntimeError):
            await self.gateway.reconcile()
        self.assertEqual(self.gateway.status()["bindings"]["small_b"]["generation"], 1)
        self.assertEqual(self.kube.pods[2]["metadata"]["labels"][GENERATION_LABEL], "1")
        self.actors["actor-small_b"] = Actor(NativePeer(copy.deepcopy(self.rows[1])))
        await self.gateway.reconcile()
        self.assertEqual((await self.completion("fresh-view"))[1]["choices"][0]["token_ids"], [2])

    async def test_same_generation_replacement_actor_is_not_a_fresh_binding(self):
        self.actors["actor-small_b"] = Actor(NativePeer(copy.deepcopy(self.rows[1])))
        with self.assertRaises(RuntimeError):
            await self.gateway._reconcile(bind_labels=False, observe_only=True)
        with self.assertRaises(RuntimeError):
            await self.gateway.reconcile()
        self.assertEqual((await self.completion("ambiguous-generation"))[0], 503)

    async def test_missing_peer_outside_declared_scope_is_never_subset_ready(self):
        await self.gateway.begin_maintenance(["small_b"], reason="one instance only")
        self.registry.rows = []
        with self.assertRaises(RuntimeError):
            await self.gateway.reconcile()
        self.assertEqual((await self.completion("missing-peer"))[0], 503)


class ExtProcLoopIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_are_accepted_while_the_gateway_loop_is_saturated(self):
        # grpc.aio accepts one call per loop round trip and gRPC core cancels
        # calls left unaccepted for 30 s. On the saturated W0 gateway loop that
        # turned Envoy streams into 500s the gateway never saw.
        async def echo(scope, receive, send):
            body, more = b"", True
            while more:
                message = await receive()
                body, more = body + message.get("body", b""), message.get("more_body", False)
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})

        server = ExtProcServer(echo, host="127.0.0.1", port=0)
        await server.start()
        acknowledged = threading.Event()
        payload = json.dumps({"request_id": "saturated-loop"}).encode()

        def envoy():
            def messages():
                yield ext.ProcessingRequest(request_headers=ext.HttpHeaders(headers=base_pb2.HeaderMap(headers=[
                    base_pb2.HeaderValue(key=":method", value="POST"),
                    base_pb2.HeaderValue(key=":path", value="/v1/completions"),
                ])))
                acknowledged.wait(30)
                yield ext.ProcessingRequest(request_body=ext.HttpBody(body=payload, end_of_stream=True))

            with grpc.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                responses = ext_grpc.ExternalProcessorStub(channel).Process(messages(), timeout=30)
                first = next(responses)
                acknowledged.set()
                return first, list(responses)

        try:
            client = asyncio.ensure_future(asyncio.to_thread(envoy))
            await asyncio.sleep(0)
            # Hold the gateway loop (and the GIL, between forced switches).
            deadline = time.monotonic() + 5
            while not acknowledged.is_set() and time.monotonic() < deadline:
                pass
            self.assertTrue(acknowledged.is_set(), "ExtProc stream waited on the gateway loop")
            first, rest = await client
        finally:
            acknowledged.set()
            await server.stop()
        self.assertTrue(first.HasField("request_headers"))
        immediate = rest[-1].immediate_response
        marker = {h.header.key: h.header.raw_value.decode() for h in immediate.headers.set_headers}
        self.assertEqual((immediate.status.code, immediate.body, marker[PROTOCOL_HEADER]),
                         (200, payload, PROTOCOL_VERSION))


class DrainDeletionTests(unittest.TestCase):
    def test_no_destructive_action_without_admission_and_native_idle_ack(self):
        from hetero.deployment import drain_deployment, engine_actor_name
        from hetero.placement import pg_name

        row = {"instance_id": "small_a", "layout_id": "M", "generation": 1,
               "engine_actor_name": engine_actor_name("M", "small_a"),
               "placement_group_name": pg_name("M", "small_a"), "ray_namespace": "omp_hetero"}
        for drained, native_idle, blocks_free in ((False, True, True), (True, False, True), (True, True, False)):
            with self.subTest(drained=drained, native_idle=native_idle, blocks_free=blocks_free):
                actor, registry = mock.Mock(), mock.Mock()
                registry.snapshot.remote.return_value = [row]
                actor.identity.remote.return_value = row
                actor.drain.remote.return_value = {"drained": drained, "inflight": 0 if drained else 1}
                actor.await_idle.remote.return_value = {"idle": native_idle}
                actor.idle_snapshot.remote.return_value = {"all_blocks_free": blocks_free}
                with mock.patch("hetero.registry.find_registry", return_value=registry), \
                     mock.patch("hetero.gateway.lifecycle.find_gateway", return_value=None), \
                     mock.patch("hetero.deployment._find_engine_actor", return_value=actor), \
                     mock.patch("hetero.deployment.ray.get", side_effect=lambda value, **kwargs: value), \
                     mock.patch("hetero.deployment.ray.kill") as kill, \
                     mock.patch("hetero.deployment.release_placement_group") as release:
                    with self.assertRaises(RuntimeError):
                        drain_deployment("M", ["small_a"], timeout_s=1)
                    kill.assert_not_called()
                    release.assert_not_called()
                    actor.shutdown.remote.assert_not_called()
                    registry.withdraw.remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
