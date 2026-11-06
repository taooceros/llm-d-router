"""C6: engines as gateway endpoints.

A switch ``drain``s the source engines (stop new work, wait for in-flight work),
``withdraw``s them, and after the target engines are built ``publish``es those.
``Endpoints`` is the controller's only view of the gateway.

What the gateway sees of an engine is its *frontend actor* (``hetero/vproc/frontend.py``),
an actor in namespace ``omp_hetero`` that implements the HeteroEngine protocol (identity,
status, generate, drain, await_idle, idle_snapshot). Host kernels are never registered.

- ``InMemoryEndpoints``: records every call; used by tests and local runs.
- ``GatewayEndpoints``: the existing gateway/registry protocol, unchanged
  (``hetero/switch.py`` ``RayFabric``: scoped maintenance drain, registry withdraw,
  ``retarget_layout`` + ``join_instances``). Cluster-only.

Layout identity (``hetero/gateway/service.py:243-306``): the gateway requires
``layout["layout_digest"] == descriptor["config_digest"]`` for every registered row of
the layout, and each actor's ``identity()`` to report that layout id and digest. The
controller therefore computes one ``layout_digest`` per published layout
(``layout_digest()`` below), hands ``{"layout_id", "layout_digest"}`` to every frontend it
builds, adopts the new identity on kept frontends, and passes the same ``layout`` mapping
to ``publish``; ``GatewayEndpoints`` writes it into the layout and into every new and
retained descriptor.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

from .types import EngineSpec

FRONTEND_NAMESPACE = "omp_hetero"  # hetero.registry.REGISTRY_NAMESPACE; the gateway only looks here


def _spec_body(spec: EngineSpec) -> Dict[str, Any]:
    return {"engine_id": spec.engine_id, "generation": spec.generation, "kind": spec.kind,
            "hosts": list(spec.hosts), "tp_size": spec.tp_size, "model": spec.model}


def layout_digest(layout_id: str, specs: Sequence[EngineSpec]) -> str:
    """One digest for a whole layout: its id and every engine (id, generation, hosts, TP, options)."""
    body = {"layout_id": layout_id, "engines": sorted((_spec_body(s) for s in specs),
                                                      key=lambda b: b["engine_id"])}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def layout_info(layout_id: str, specs: Sequence[EngineSpec]) -> Dict[str, Any]:
    """The ``layout`` argument of ``Endpoints.publish`` (``specs`` = every engine of the layout)."""
    return {"layout_id": layout_id, "layout_digest": layout_digest(layout_id, specs),
            "engines": sorted(s.engine_id for s in specs)}


class Endpoints(Protocol):
    def drain(self, ids: Sequence[str]) -> Dict[str, Any]:
        """Stop admission to ``ids`` and wait until they are idle."""

    def withdraw(self, ids: Sequence[str]) -> Dict[str, Any]:
        """Remove ``ids`` from discovery. Must work without a prior drain
        (the controller withdraws everything when it tears a failed block down)."""

    def publish(self, specs: Sequence[EngineSpec], frontends: Optional[Mapping[str, Mapping]] = None,
                layout: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Make the built engines routable. ``frontends[engine_id]`` = ``{"actor_name",
        "namespace", "identity"}`` of the engine's frontend actor; ``layout`` = ``layout_info``
        of the layout being published (kept engines included)."""


class EndpointError(RuntimeError):
    pass


class InMemoryEndpoints:
    """Gateway stand-in. ``events`` is the ordered call log ``(op, ids)``;
    ``published`` maps live engine id -> spec; ``draining`` holds drained ids;
    ``frontends`` the frontend info of published engines; ``layouts`` every ``layout`` seen."""

    def __init__(self, fail_on: Optional[Dict[str, BaseException]] = None):
        self.events: List[Tuple[str, Tuple[str, ...]]] = []
        self.published: Dict[str, EngineSpec] = {}
        self.frontends: Dict[str, Mapping] = {}
        self.layouts: List[Mapping[str, Any]] = []
        self.draining: Set[str] = set()
        self.fail_on = dict(fail_on or {})
        self._lock = threading.Lock()

    def _enter(self, op: str, ids: Sequence[str]) -> None:
        self.events.append((op, tuple(ids)))
        failure = self.fail_on.pop(op, None)
        if failure is not None:
            raise failure

    def drain(self, ids: Sequence[str]) -> Dict[str, Any]:
        with self._lock:
            self._enter("drain", ids)
            unknown = [i for i in ids if i not in self.published]
            if unknown:
                raise EndpointError(f"drain of unpublished engines {unknown}")
            self.draining.update(ids)
            return {"drained": list(ids)}

    def withdraw(self, ids: Sequence[str]) -> Dict[str, Any]:
        with self._lock:
            self._enter("withdraw", ids)
            gone = [i for i in ids if self.published.pop(i, None) is not None]
            for i in ids:
                self.frontends.pop(i, None)
            self.draining.difference_update(ids)
            return {"withdrawn": gone, "missing": [i for i in ids if i not in gone]}

    def publish(self, specs: Sequence[EngineSpec], frontends: Optional[Mapping[str, Mapping]] = None,
                layout: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            ids = [s.engine_id for s in specs]
            self._enter("publish", ids)
            live = [i for i in ids if i in self.published]
            if live:
                raise EndpointError(f"publish of engines that are still published: {live}")
            self.published.update({s.engine_id: s for s in specs})
            self.frontends.update({i: f for i, f in (frontends or {}).items()})
            if layout is not None:
                self.layouts.append(dict(layout))
            return {"published": ids}

    def ops(self) -> List[str]:
        return [op for op, _ in self.events]


# --------------------------------------------------------------------------- gateway

def engine_descriptor(spec: EngineSpec, layout_id: str, config_digest: str, frontend: Mapping[str, Any], *,
                      endpoint_port: int = 0, placement_group: str = "", model: str = "",
                      **capability: Any) -> Dict[str, Any]:
    """``registry.InstanceDescriptor`` body of one engine, addressing its frontend actor.

    ``frontend`` = ``{"actor_name", "namespace", "identity": {"node_ip", ...}}``. The gateway
    resolves ``engine_actor_name`` with ``ray.get_actor(name, namespace="omp_hetero")`` and
    compares ``identity()`` with this row, so ``endpoint_host`` is the frontend's ``node_ip``.
    The port is ``frontend["endpoint_port"]``, else the identity's, else ``endpoint_port``.
    Capability keys present in the identity (``max_model_len``, ``kv_block_size``, ...) pass
    through, explicit ``capability`` wins.
    """
    identity = dict(frontend.get("identity") or {})
    host = identity.get("node_ip")
    port = frontend.get("endpoint_port") or identity.get("endpoint_port") or endpoint_port
    if not host or not port:
        raise EndpointError(f"{spec.engine_id}: frontend identity has no node_ip/endpoint port ({identity})")
    known = {k: identity[k] for k in ("max_model_len", "kv_block_size", "kv_usable_blocks",
                                      "kv_usable_tokens", "max_num_seqs", "model_build") if k in identity}
    return {"instance_id": spec.engine_id, "layout_id": layout_id, "generation": spec.generation,
            "config_digest": config_digest, "ray_namespace": frontend["namespace"],
            "engine_actor_name": frontend["actor_name"],
            "placement_group_name": placement_group, "placement_group_id": placement_group,
            "host_chip_rank_map": [{"rank": rank, "host": host_rank} for rank, host_rank in enumerate(spec.hosts)],
            "endpoint_host": str(host), "endpoint_port": int(port),
            "model": model or str(spec.model.get("name", "")), "tp_size": spec.tp_size, "state": "ready",
            **known, **capability}


RPC_TIMEOUT_S = 3600.0


def _ray_rpc(target: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    import ray

    return ray.get(getattr(target, method).remote(*args, **kwargs), timeout=RPC_TIMEOUT_S)


class GatewayEndpoints:
    """``Endpoints`` over the existing ``OmpGateway`` actor and the serving-instance registry.

    - ``drain``: ``begin_maintenance(ids)`` then ``drain_maintenance_instance`` per id.
    - ``withdraw``: ``registry.withdraw(id, generation)`` per id (generation as published).
    - ``publish`` (needs ``layout``): ``put_layout`` with ``layout_digest`` and every instance;
      then, by case,
        * no maintenance scope (first deploy, or a rebuild of an undrained block): register the
          descriptors, ``reconcile``;
        * maintenance scope open and the published ids are exactly the drained ones (rebuild of
          the source layout after a failed switch): register under the gateway's current layout,
          ``finish_maintenance``;
        * otherwise a switch: re-register the kept engines under the new layout and digest,
          ``retarget_layout(layout, new ids)``, register the new descriptors, ``join_instances``.
      Every descriptor of the layout, new and retained, carries ``config_digest == layout_digest``.

    ``endpoint_port`` is the registry port when the frontend identity names none.
    ``descriptor_for(spec, layout_id, digest, frontend)`` replaces ``engine_descriptor``.
    ``rpc(target, method, *args, **kwargs)`` calls an actor method (default ``ray.get``);
    tests pass a direct-call function.
    """

    def __init__(self, gateway: Any, registry: Any, *, endpoint_port: int = 0,
                 descriptor_for: Optional[Callable[..., Dict[str, Any]]] = None,
                 rpc: Optional[Callable[..., Any]] = None, policy_after: Optional[Dict[str, Any]] = None,
                 reason: str = "vproc layout switch", drain_timeout_s: float = 600.0):
        self.gateway, self.registry = gateway, registry
        self.rpc = rpc or _ray_rpc
        self.descriptor_for = descriptor_for or (
            lambda spec, layout_id, digest, frontend: engine_descriptor(
                spec, layout_id, digest, frontend, endpoint_port=endpoint_port))
        self.policy_after, self.reason, self.drain_timeout_s = policy_after, reason, drain_timeout_s
        self.generations: Dict[str, int] = {}
        self.live: Set[str] = set()
        self.maintenance_id: Optional[str] = None
        self.maintained: Set[str] = set()
        self.layout_id: Optional[str] = None  # the layout the gateway currently runs

    def drain(self, ids: Sequence[str]) -> Dict[str, Any]:
        ids = list(ids)
        maintenance = self.rpc(self.gateway, "begin_maintenance", ids, reason=self.reason)
        self.maintenance_id, self.maintained = maintenance["id"], set(ids)
        for name, descriptor in maintenance.get("instances", {}).items():
            self.generations[name] = descriptor["generation"]
        receipts = {i: self.rpc(self.gateway, "drain_maintenance_instance", i,
                                maintenance_id=self.maintenance_id, timeout_s=self.drain_timeout_s)
                    for i in ids}
        return {"maintenance_id": self.maintenance_id, "receipts": receipts}

    def withdraw(self, ids: Sequence[str]) -> Dict[str, Any]:
        withdrawn, missing = [], []
        for i in ids:
            done = self.rpc(self.registry, "withdraw", i, self.generations.get(i))
            (withdrawn if done else missing).append(i)
            self.live.discard(i)
            self.generations.pop(i, None)
        return {"withdrawn": withdrawn, "missing": missing}

    def publish(self, specs: Sequence[EngineSpec], frontends: Optional[Mapping[str, Mapping]] = None,
                layout: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if layout is None or not frontends:
            raise EndpointError("GatewayEndpoints.publish needs layout=layout_info(...) and the frontends' info")
        specs = list(specs)
        joiners = [s.engine_id for s in specs]
        missing = [i for i in joiners if i not in frontends]
        if missing:
            raise EndpointError(f"no frontend for engines {missing}")
        layout_id, digest = layout["layout_id"], layout["layout_digest"]
        kept = sorted(self.live - set(joiners))
        if set(layout["engines"]) != set(kept) | set(joiners):
            raise EndpointError(f"layout {layout_id} names {layout['engines']}, endpoints hold kept {kept} "
                                f"and publish {joiners}")
        self.rpc(self.registry, "put_layout", {
            "layout_id": layout_id, "layout_digest": digest,
            "instances": [{"instance_id": i} for i in sorted(layout["engines"])]})
        result: Dict[str, Any] = {"layout_id": layout_id, "layout_digest": digest, "joined": joiners, "kept": kept}
        if self.maintenance_id is None:
            self._register_all(specs, frontends, layout_id, digest)
            result["reconcile"] = self.rpc(self.gateway, "reconcile", bind_labels=True)
        elif set(joiners) == self.maintained and not kept:
            if layout_id != self.layout_id:
                raise EndpointError(f"engines {joiners} were drained under layout {self.layout_id}; "
                                    f"cannot restore them as {layout_id}")
            self._register_all(specs, frontends, layout_id, digest)
            result["finish_maintenance"] = self.rpc(self.gateway, "finish_maintenance",
                                                    maintenance_id=self.maintenance_id)
        else:
            if kept:
                from ..registry import descriptor_for_layout
            for name in kept:  # kept engines are adopted by the new layout (hetero/transition.py adopt_kept)
                row = self.rpc(self.registry, "get", name)
                if row is not None:
                    self.rpc(self.registry, "register", descriptor_for_layout(row, layout_id, digest))
            result["retarget"] = self.rpc(self.gateway, "retarget_layout", layout_id, joiners)
            self._register_all(specs, frontends, layout_id, digest)
            result["join"] = self.rpc(self.gateway, "join_instances", joiners, self.policy_after)
        self.maintenance_id, self.maintained = None, set()
        self.layout_id = layout_id
        return result

    def _register_all(self, specs: Sequence[EngineSpec], frontends: Mapping[str, Mapping],
                      layout_id: str, digest: str) -> None:
        for spec in specs:
            body = self.descriptor_for(spec, layout_id, digest, frontends[spec.engine_id])
            answer = self.rpc(self.registry, "register", body)
            if not answer.get("accepted"):
                raise EndpointError(f"registry refused {spec.engine_id}: {answer}")
            self.generations[spec.engine_id] = spec.generation
            self.live.add(spec.engine_id)
