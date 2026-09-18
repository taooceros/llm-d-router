"""Gateway lifecycle operations invoked from the remote CLI."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import ray

from ..evidence import emit_event, emit_log
from ..registry import REGISTRY_NAMESPACE, find_registry
from .policy import RoutingPolicy
from .service import GATEWAY_ACTOR_NAME, GATEWAY_PORT, OmpGateway
from .discovery import DEFAULT_POOL_NAME


def find_gateway() -> Optional[Any]:
    try:
        return ray.get_actor(GATEWAY_ACTOR_NAME, namespace=REGISTRY_NAMESPACE)
    except ValueError:
        return None


def gateway_up(
    *,
    layout_id: str,
    policy: Optional[Dict[str, Any]] = None,
    pool_name: str = DEFAULT_POOL_NAME,
    port: int = GATEWAY_PORT,
    head_node_ip: Optional[str] = None,
    bind_labels: bool = True,
    discovery_timeout_s: float = 420.0,
) -> Dict[str, Any]:
    """Creates or attaches the detached gateway actor and binds discovery."""
    actor = find_gateway()
    created = False
    deadline = time.monotonic() + discovery_timeout_s

    def await_discovery(operation):
        while True:
            try:
                return ray.get(operation(), timeout=300)
            except Exception as exc:
                if time.monotonic() >= deadline:
                    raise
                emit_log(f"discovery not settled yet, retrying: {str(exc)[:160]}")
                time.sleep(5.0)

    if actor is None:
        options: Dict[str, Any] = {
            "name": GATEWAY_ACTOR_NAME,
            "namespace": REGISTRY_NAMESPACE,
            "lifetime": "detached",
            "max_concurrency": 4096,
        }
        if head_node_ip:
            options["resources"] = {f"node:{head_node_ip}": 0.001}
        else:
            options["resources"] = {"node:__internal_head__": 0.001}
        actor = OmpGateway.options(**options).remote(
            config={"layout_id": layout_id, "policy": policy or {}},
            pool_name=pool_name,
            port=port,
        )
        created = True
    else:
        existing = ray.get(actor.status.remote(), timeout=60)
        if existing["layout_id"] != layout_id:
            raise RuntimeError(
                f"gateway owns layout {existing['layout_id']!r}, not {layout_id!r}; "
                "stop the existing gateway explicitly before attaching another layout"
            )
        if existing.get("maintenance"):
            await_discovery(lambda: actor.finish_maintenance.remote(
                maintenance_id=existing["maintenance"]["id"]))
        if policy is not None:
            ray.get(actor.set_policy.remote(policy), timeout=60)
    # Engine readiness precedes the kubelet readiness probe that discovery
    # resolves, so reconcile is retried under a bounded deadline instead of
    # failing a correct deployment on a probe-interval race.
    bind = await_discovery(lambda: actor.reconcile.remote(bind_labels=bind_labels))
    start = ray.get(actor.start.remote(), timeout=300)
    status = ray.get(actor.status.remote(), timeout=60)
    emit_event(
        "GATEWAY_READY",
        created=created,
        instances=status["core"]["instances"],
        endpoint=f"{status['node_ip']}:{status['port']}",
    )
    return {
        "created": created,
        "bind": bind,
        "start": start,
        "status": status,
    }


def gateway_down(*, remove: bool = True) -> Dict[str, Any]:
    actor = find_gateway()
    if actor is None:
        return {"found": False}
    result: Dict[str, Any] = {"found": True}
    try:
        result["stop"] = ray.get(actor.stop.remote(), timeout=120)
    except Exception as exc:  # pragma: no cover
        result["stop_error"] = str(exc)[:200]
    if remove:
        ray.kill(actor, no_restart=True)
        result["removed"] = True
    return result


def gateway_status() -> Dict[str, Any]:
    actor = find_gateway()
    if actor is None:
        return {"present": False}
    return {"present": True, "status": ray.get(actor.status.remote(), timeout=60)}
