"""Gateway HTTP surface and its Ray actor host.

Client traffic reaches the same HTTP application via Envoy's standard gRPC
ExtProc ImmediateResponse bridge. GatewayCore is the stable response owner
across migration; the direct HTTP listener is also available for diagnostics
but never emits the ExtProc protocol qualification marker.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

import ray

from ..evidence import emit_event
from ..registry import REGISTRY_NAMESPACE, find_registry
from fastapi import FastAPI, HTTPException, Request
from .core import GatewayConfig, GatewayCore
from .discovery import DEFAULT_POOL_NAME, DiscoveryBridge
from .extproc import EXTPROC_PORT, PROTOCOL_VERSION, ExtProcServer
from .policy import RoutingPolicy

GATEWAY_ACTOR_NAME = "omp-hetero-gateway"
GATEWAY_PORT = 8200


def build_app(gateway: "GatewayActorState") -> Any:
    app = FastAPI(title="omp-hetero-gateway", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        recovery = gateway.core.recovery_status()
        if not recovery["ready"]:
            raise HTTPException(status_code=503, detail=recovery)
        if not gateway.discovery_snapshot().get("ready", False):
            raise HTTPException(status_code=503, detail="logical-instance discovery is not ready")
        return {"status": "ok", "instances": gateway.core.routable_instances()}

    @app.get("/omp/state")
    async def state() -> Dict[str, Any]:
        return gateway.core.snapshot()

    @app.get("/omp/discovery")
    async def discovery() -> Dict[str, Any]:
        return gateway.discovery_snapshot()

    @app.get("/omp/traces")
    async def traces(limit: int = 500) -> Dict[str, Any]:
        return {"traces": gateway.core.traces(limit)}

    @app.post("/v1/completions")
    async def completions(request: Request) -> Dict[str, Any]:
        if not gateway.discovery_snapshot().get("ready", False):
            raise HTTPException(status_code=503, detail="logical-instance discovery is not ready")
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON request") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="request must be a JSON object")
        if body.get("stream"):
            raise HTTPException(status_code=400, detail="streaming is not supported")
        try:
            record = await gateway.dispatch(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "id": record["request_id"],
            "object": "text_completion",
            "created": int(record["admitted_at_s"]),
            "model": body.get("model", "gemma-4-31b"),
            "choices": [
                {
                    "index": 0,
                    "text": record["text"],
                    "token_ids": record["output_token_ids"],
                    "finish_reason": record["segments"][-1].get("finish_reason"),
                }
            ],
            "usage": {
                "prompt_tokens": record["prompt_tokens"],
                "completion_tokens": record["useful_output_tokens"],
            },
            "omp": {
                "initial_instance": record.get("initial_instance"),
                "segments": [
                    {
                        "instance": s.get("segment_instance") or s.get("instance_id"),
                        "output_tokens": s.get("output_tokens", s.get("new_output_tokens", 0)),
                        "finish_reason": s.get("finish_reason"),
                    }
                    for s in record["segments"]
                ],
                "elapsed_s": record.get("elapsed_s"),
            },
        }

    return app


class GatewayActorState:
    """Holds one core and discovery view shared by the HTTP and ExtProc servers."""

    def __init__(self, config: GatewayConfig, pool_name: str) -> None:
        self.core = GatewayCore(config)
        self.pool_name = pool_name
        self.bridge: Optional[DiscoveryBridge] = None
        self._discovery: Dict[str, Any] = {}
        self._server = None
        self._task = None
        self._refresh_task = None
        self._migration_task = None
        self._extproc: Optional[ExtProcServer] = None
        self._discovery_task = None
        self.discovery_expires_at = 0.0
        self.active_requests = 0
        self.capturing = False
        self.administration_lock = asyncio.Lock()
        self.maintenance: Dict[str, Any] = {}
        self.maintenance_clients: Dict[str, Any] = {}

    def discovery_snapshot(self) -> Dict[str, Any]:
        if self._discovery.get("ready") and time.monotonic() >= self.discovery_expires_at:
            return {**self._discovery, "ready": False, "error": "discovery health lease expired"}
        return self._discovery

    async def dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not self.discovery_snapshot().get("ready"):
            raise RuntimeError("logical-instance discovery is not ready")
        self.active_requests += 1
        try:
            return await self.core.dispatch(request)
        finally:
            self.active_requests -= 1


@ray.remote(num_cpus=1)
class OmpGateway:
    """Detached Ray actor hosting the central gateway serving layer."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        pool_name: str = DEFAULT_POOL_NAME,
        port: int = GATEWAY_PORT,
    ) -> None:
        policy = RoutingPolicy.from_dict(config.pop("policy", None))
        self.config = GatewayConfig(policy=policy, **config)
        self.port = port
        self.state = GatewayActorState(self.config, pool_name)
        self.node_ip = ray.util.get_node_ip_address()
        self.started_at = time.time()

    # ---------------------------------------------------------- binding

    @staticmethod
    def _binding(row: Dict[str, Any]) -> tuple:
        return tuple(row.get(key) for key in (
            "instance_id", "layout_id", "generation", "engine_actor_name", "endpoint",
            "config_digest", "placement_group_id",
        ))

    def _require_idle(self) -> None:
        """Never discard a live owner, pending release, or failed journal decision."""
        snapshot = self.state.core.snapshot()
        if (self.state.active_requests or self.state.capturing
                or snapshot["backlog"]["inflight"] or snapshot["backlog"]["response_tasks"]
                or not snapshot["recovery"]["ready"]
                or any(row["state"] not in ("released", "aborted")
                       for row in snapshot["state"]["transactions"])
                or any(row["state"] not in ("finished", "cancelled")
                       or row["cancellation_pending"]
                       for row in snapshot["state"]["ownership"])):
            raise RuntimeError("administrative transition requires idle responses, terminal migrations, and no capture")

    async def reconcile(self, *, bind_labels: bool = True) -> Dict[str, Any]:
        """Explicit administrative reconcile; the background watch never rebinds."""
        async with self.state.administration_lock:
            if bind_labels and self.state.core.clients and not self.state.maintenance:
                self._require_idle()
            return await self._reconcile(bind_labels=bind_labels)

    async def _reconcile(
        self, *, bind_labels: bool, observe_only: bool = False, restore: bool = False,
    ) -> Dict[str, Any]:
        rows = []
        try:
            registry = find_registry()
            if registry is None:
                raise RuntimeError("serving-instance registry actor is not present")
            rows = await self.state.core._rpc(registry.snapshot.remote(self.config.layout_id))
            sampled_at = time.monotonic()
            layout = await self.state.core._rpc(registry.get_layout.remote(self.config.layout_id))
            if not layout:
                raise RuntimeError("registered static layout is missing")
            expected = {row["instance_id"] for row in layout["instances"]}
            maintenance = self.state.maintenance
            excluded = set(maintenance.get("instances", {})) if not restore else set()
            active = [row for row in rows if row["instance_id"] not in excluded]
            if ({row["instance_id"] for row in active} != expected - excluded
                    or not excluded <= expected
                    or any(row.get("config_digest") != layout["layout_digest"]
                           or row.get("layout_id") != self.config.layout_id for row in rows)):
                raise RuntimeError("registry differs from the declared static layout outside maintenance")
            before = self.state.core.descriptors()
            changed = ({name: self._binding(row) for name, row in before.items()}
                       != {row["instance_id"]: self._binding(row) for row in active})
            if changed and observe_only:
                raise RuntimeError("engine identity changed; explicit idle reconciliation is required")
            if changed and before:
                self._require_idle()
            if maintenance:
                for row in active:
                    name = row["instance_id"]
                    original = maintenance["instances"].get(name)
                    if original is not None:
                        if (not restore or name not in maintenance["drained"]
                                or row["generation"] <= original["generation"]):
                            raise RuntimeError("maintenance requires an acknowledged drain and a fresh generation")
                    elif name not in before or self._binding(row) != self._binding(before[name]):
                        raise RuntimeError("maintenance cannot replace an unselected peer")
            actors, statuses = {}, {}
            for row in active:
                name = row["instance_id"]
                if (not row.get("routable") or row["state"] != "ready"
                        or row.get("ray_namespace") != REGISTRY_NAMESPACE):
                    raise RuntimeError(f"instance is not ready in the owned namespace: {name}")
                actor = ray.get_actor(row["engine_actor_name"], namespace=REGISTRY_NAMESPACE)
                previous = self.state.core.clients.get(name)
                if (previous is not None and actor._actor_id != previous.actor._actor_id
                        and row["generation"] <= previous.generation):
                    raise RuntimeError(f"replacement actor reused a retired generation: {name}")
                identity = await self.state.core._rpc(actor.identity.remote())
                wanted = {"instance_id": name, "layout_id": self.config.layout_id,
                          "generation": row["generation"], "node_ip": row["endpoint_host"],
                          "layout_digest": row["config_digest"], "state": "ready"}
                if any(identity.get(key) != value for key, value in wanted.items()):
                    raise RuntimeError(f"actor and registry identity disagree: {name}")
                status = await self.state.core._rpc(actor.status.remote())
                if status.get("generation") != row["generation"] or status.get("state") != "ready":
                    raise RuntimeError(f"actor generation is not admitting: {name}")
                actors[name], statuses[name] = actor, status

            def discover() -> Dict[str, Any]:
                if self.state.bridge is None:
                    self.state.bridge = DiscoveryBridge(pool_name=self.state.pool_name)
                self.state.bridge.pool(refresh=True)
                labels = (self.state.bridge.bind_labels(rows, maintenance_instances=excluded)
                          if bind_labels else None)
                view = self.state.bridge.resolve(rows, maintenance_instances=excluded)
                if labels is not None:
                    view["labels"] = labels
                return view

            # Kubernetes I/O must not block the shared HTTP / ExtProc event loop.
            discovery = await asyncio.to_thread(discover)
            if not discovery["ready"] and active:
                raise RuntimeError(f"logical discovery has missing, stale, or ambiguous endpoints: {discovery}")
            if not active and discovery["pod_endpoint_count"]:
                raise RuntimeError("maintenance left labelled endpoints without an active instance")
            if changed:
                if before or restore:
                    self._require_idle()
                # A new generation must never inherit the previous KV-capacity sample.
                for name, status in statuses.items():
                    if name not in before or self._binding(before[name]) != self._binding(next(
                            row for row in active if row["instance_id"] == name)):
                        self.state.core.state.observe_capacity(
                            name, total_blocks=int(status["total_blocks"]),
                            free_blocks=int(status["free_blocks"]), block_size=int(status["block_size"]),
                            running=int(status.get("num_running", 0)), waiting=int(status.get("num_waiting", 0)),
                            resident_growth_allowance=self.state.core._growth_allowance(status),
                            sample_started_at=sampled_at,
                        )
                self.state.core.bind_instances(active, actors)
            self.state.discovery_expires_at = sampled_at + min(
                [self.config.health_lease_s]
                + [max(0.0, self.config.health_lease_s - row.get("health_age_s", 0.0)) for row in active]
            )
        except Exception as exc:
            self.state._discovery = {"ready": False, "error": f"{type(exc).__name__}: {str(exc)[:1200]}"}
            raise
        self.state._discovery = discovery
        return {"instances": self.state.core.routable_instances(), "registry_rows": len(rows),
                "discovery": discovery}

    async def begin_maintenance(self, instance_ids: List[str], *, reason: str) -> Dict[str, Any]:
        """Fence selected admissions between workloads; peers keep the same core."""
        async with self.state.administration_lock:
            names = set(instance_ids)
            if not names or not reason.strip():
                raise ValueError("maintenance requires explicit instances and an operator reason")
            if self.state.maintenance:
                if names != set(self.state.maintenance["instances"]):
                    raise RuntimeError("another maintenance scope is active")
                await self._reconcile(bind_labels=True)
                return dict(self.state.maintenance)
            self._require_idle()
            if not self.state.discovery_snapshot().get("ready"):
                raise RuntimeError("maintenance must begin from a fully reconciled fabric")
            clients = self.state.core.clients
            if not names <= clients.keys():
                raise ValueError("maintenance target is not owned by this gateway")
            self.state.maintenance = {
                "id": uuid.uuid4().hex, "reason": reason,
                "instances": {name: dict(clients[name].descriptor) for name in sorted(names)},
                "drained": {},
            }
            self.state.maintenance_clients = {name: clients[name] for name in names}
            # No await between the idle check and exclusion: not even a queued
            # HTTP or direct RPC dispatch may acquire a retiring generation.
            self.state.core.clients = {name: client for name, client in clients.items() if name not in names}
            await self._reconcile(bind_labels=True)
            return dict(self.state.maintenance)

    async def drain_maintenance_instance(
        self, instance_id: str, *, maintenance_id: str, timeout_s: float = 600.0,
    ) -> Dict[str, Any]:
        """Acknowledge both actor admission drain and native scheduler/KV idle."""
        async with self.state.administration_lock:
            maintenance = self.state.maintenance
            if maintenance.get("id") != maintenance_id or instance_id not in maintenance["instances"]:
                raise RuntimeError("maintenance scope does not own this instance")
            if instance_id in maintenance["drained"]:
                return maintenance["drained"][instance_id]
            client = self.state.maintenance_clients[instance_id]
        # The admission fence already owns this generation. Do not hold the
        # administration lock across a native drain: the discovery lease must
        # keep refreshing while peers serve during a long shutdown barrier.
        identity = await self.state.core._rpc(client.actor.identity.remote())
        if identity.get("generation") != client.generation:
            raise RuntimeError("retiring actor generation changed before drain")
        drained = await self.state.core._rpc(client.actor.drain.remote(timeout_s=timeout_s),
                                             timeout_s=timeout_s + self.config.rpc_timeout_s)
        if not drained.get("drained") or drained.get("inflight") != 0:
            raise RuntimeError(f"native admission drain was not acknowledged: {drained}")
        idle = await self.state.core._rpc(client.actor.await_idle.remote(timeout_s=timeout_s),
                                          timeout_s=timeout_s + self.config.rpc_timeout_s)
        snapshot = await self.state.core._rpc(client.actor.idle_snapshot.remote())
        if not idle.get("idle") or not snapshot.get("all_blocks_free"):
            raise RuntimeError(f"native scheduler/KV is not idle: {idle}, {snapshot}")
        receipt = {"generation": client.generation, "drain": drained,
                   "native_idle": idle, "idle_snapshot": snapshot}
        async with self.state.administration_lock:
            if self.state.maintenance.get("id") != maintenance_id:
                raise RuntimeError("maintenance scope changed during native drain")
            maintenance["drained"][instance_id] = receipt
        return receipt

    async def finish_maintenance(self, *, maintenance_id: str) -> Dict[str, Any]:
        """Publish fresh generations only after drain and full discovery validation."""
        async with self.state.administration_lock:
            if self.state.maintenance.get("id") != maintenance_id:
                raise RuntimeError("maintenance scope does not match")
            self._require_idle()
            try:
                result = await self._reconcile(bind_labels=True, restore=True)
            except Exception:
                # A failed promotion must not advertise the target or lose peers.
                await self._reconcile(bind_labels=True)
                raise
            self.state.maintenance = {}
            self.state.maintenance_clients = {}
            return result

    # ------------------------------------------------------------ server

    async def start(self) -> Dict[str, Any]:
        import uvicorn

        if self.state._task is not None:
            return {"started": False, "reason": "already_running", "port": self.port,
                    "extproc_port": EXTPROC_PORT}
        await self.state.core.recover()
        app = build_app(self.state)
        config = uvicorn.Config(
            app, host="0.0.0.0", port=self.port, log_level="warning",
            access_log=False, timeout_keep_alive=600,
        )
        try:
            self.state._extproc = ExtProcServer(app)
            self.state._server = uvicorn.Server(config)
            self.state._task = asyncio.create_task(self.state._server.serve())
            deadline = time.monotonic() + 60
            while not getattr(self.state._server, "started", False):
                if self.state._task.done():
                    raise RuntimeError(f"gateway failed to start: {self.state._task.exception()}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("gateway HTTP listener did not start")
                await asyncio.sleep(0.1)
            await self.state._extproc.start()
        except BaseException:
            await self.stop()
            raise
        self.state._refresh_task = asyncio.create_task(self._refresh_loop())
        self.state._discovery_task = asyncio.create_task(self._discovery_loop())
        self.state._migration_task = asyncio.create_task(self._migration_loop())
        return {"started": True, "host": self.node_ip, "port": self.port,
                "extproc_port": EXTPROC_PORT, "protocol": PROTOCOL_VERSION}

    async def stop(self) -> Dict[str, Any]:
        if self.state._extproc is not None:
            await self.state._extproc.stop()
            self.state._extproc = None
        for attr in ("_refresh_task", "_migration_task", "_discovery_task"):
            task = getattr(self.state, attr)
            setattr(self.state, attr, None)
            if task is not None:
                task.cancel()
        if self.state._server is not None:
            self.state._server.should_exit = True
        task = self.state._task
        self.state._task = None
        self.state._server = None
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=30)
            except Exception:  # pragma: no cover
                task.cancel()
        return {"stopped": True}

    async def _refresh_loop(self) -> None:
        while True:
            try:
                await self.state.core.refresh_capacity()
            except Exception:  # pragma: no cover
                pass
            await asyncio.sleep(self.config.capacity_refresh_s)

    async def _discovery_loop(self) -> None:
        while True:
            await asyncio.sleep(min(5.0, self.config.health_lease_s / 3))
            try:
                async with self.state.administration_lock:
                    await self._reconcile(bind_labels=False, observe_only=True)
            except Exception:
                # _reconcile publishes a fail-closed diagnostic, never silently
                # swaps a generation or mutates labels during a workload.
                pass

    async def _migration_loop(self) -> None:
        while True:
            try:
                async with self.state.administration_lock:
                    actions = ([] if self.state.maintenance or not self.config.policy.migration_enabled
                               else await self.state.core.poll_and_migrate())
                self.state.core.record_sweep(actions)
                for action in actions:
                    if action.get("committed"):
                        emit_event(
                            "MIGRATION_COMMITTED",
                            request_id=action["request_id"],
                            source=action["source"],
                            destination=action["destination"],
                            transaction=action["transaction_id"],
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never kill the loop
                # A silently swallowed sweep failure is indistinguishable from
                # "the policy never wanted to migrate"; record it instead.
                self.state.core.record_sweep_error(exc)
            await asyncio.sleep(self.config.migration_poll_s)

    # ------------------------------------------------------------ control

    def status(self) -> Dict[str, Any]:
        descriptors = self.state.core.descriptors()
        return {
            "node_ip": self.node_ip,
            "port": self.port,
            "uptime_s": round(time.time() - self.started_at, 2),
            "serving": self.state._task is not None and not self.state._task.done(),
            "extproc_port": EXTPROC_PORT,
            "extproc_serving": self.state._extproc is not None,
            "protocol": PROTOCOL_VERSION,
            "layout_id": self.config.layout_id,
            "maintenance": dict(self.state.maintenance),
            "capturing": self.state.capturing,
            "bindings": descriptors,
            "instances": sorted(descriptors),
            "total_chips": sum(
                int(d.get("chips") or d.get("tp_size") or 0) for d in descriptors.values()
            ),
            "core": self.state.core.snapshot(),
            "discovery": self.state.discovery_snapshot(),
        }

    async def await_idle_instances(self, timeout_s: float = 300.0) -> Dict[str, Any]:
        """Blocks until every engine reports an idle scheduler."""
        return await self.state.core.await_idle_instances(timeout_s)

    async def resume_admission(self) -> Dict[str, Any]:
        """Clears a latched drain on every engine."""
        return await self.state.core.resume_admission()

    async def start_capture(self) -> Dict[str, Any]:
        """Arm a timed workload only on the complete static fabric."""
        async with self.state.administration_lock:
            if self.state.maintenance or not self.state.discovery_snapshot().get("ready"):
                raise RuntimeError("cannot start workload capture without a complete static fabric")
            self._require_idle()
            self.state.capturing = True
            return await self.state.core.start_capture()

    async def stop_capture(self) -> Dict[str, List[Dict[str, Any]]]:
        """Returns and disarms per-iteration telemetry from every engine."""
        result = await self.state.core.stop_capture()
        self.state.capturing = False
        return result

    def set_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """Freezes a new static policy. Only legal between workloads."""
        if self.state.maintenance:
            raise RuntimeError("cannot change the static policy during maintenance")
        self._require_idle()
        new_policy = RoutingPolicy.from_dict(policy)
        self.config.policy = new_policy
        self.state.core.config.policy = new_policy
        self.state.core.scheduler.policy = new_policy
        return new_policy.to_dict()

    async def reset_records(self) -> Dict[str, Any]:
        if self.state.maintenance or self.state.capturing:
            raise RuntimeError("cannot reset the journal during maintenance or capture")
        await self.state.core.reset_records()
        return {"reset": True}

    async def cancel(self, request_id: str) -> Dict[str, Any]:
        return await self.state.core.cancel(request_id)

    def records(self) -> List[Dict[str, Any]]:
        return self.state.core.request_records()

    def traces(self, limit: int = 4000) -> List[Dict[str, Any]]:
        return self.state.core.traces(limit)

    async def migrate_once(self) -> List[Dict[str, Any]]:
        """Runs one bounded migration sweep on demand (qualification path)."""
        async with self.state.administration_lock:
            if self.state.maintenance:
                raise RuntimeError("migration is suspended during declared maintenance")
            return await self.state.core.poll_and_migrate()

    async def dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Direct dispatch used by qualification harnesses and load drivers."""
        return await self.state.dispatch(request)

    async def dispatch_many(
        self, requests: List[Dict[str, Any]], concurrency: int = 64
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(item: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.state.dispatch(item)
                except Exception as exc:
                    return {"request_id": item.get("request_id"), "error": str(exc)[:300]}

        self.state.active_requests += 1
        try:
            return list(await asyncio.gather(*[_one(r) for r in requests]))
        finally:
            self.state.active_requests -= 1
