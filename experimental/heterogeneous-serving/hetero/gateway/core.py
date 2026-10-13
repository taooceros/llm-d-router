"""The llm-d gateway serving layer: central scheduler, dispatcher, coordinator.

One process owns cluster-level destination selection, admission, destination
reservations, ownership commits and the stable request/response identity.
Source engines report migration candidates; they never choose a destination and
never submit a continuation to a peer.  KV payloads (when a transport arm is
enabled) move engine-to-engine; only scheduling messages traverse this service.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..evidence import emit_event
from .policy import Candidate, CentralScheduler, RoutingPolicy, blocks_for
from .state import (
    COMMITTED_STATES, TERMINAL_STATES, GatewayState, OwnershipState, TransactionState,
)


@dataclasses.dataclass
class GatewayConfig:
    layout_id: str = "M"
    policy: RoutingPolicy = dataclasses.field(default_factory=RoutingPolicy)
    max_inflight_requests: int = 256
    capacity_refresh_s: float = 0.25
    migration_poll_s: float = 0.5
    quiesce_timeout_s: float = 30.0
    prepare_timeout_s: float = 120.0
    health_lease_s: float = 45.0
    admission_poll_s: float = 0.05
    admission_timeout_s: float = 3600.0
    rpc_timeout_s: float = 30.0
    result_timeout_s: float = 3600.0
    # Same-node, process-crash journal. Use a persistent local path if needed;
    # relocating the coordinator without its ledger is not supported recovery.
    ledger_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        body = dataclasses.asdict(self)
        body["policy"] = self.policy.to_dict()
        return body


class InstanceClient:
    """Typed handle to one engine actor, resolved from the registry."""

    def __init__(self, descriptor: Dict[str, Any], actor: Any) -> None:
        self.descriptor = descriptor
        self.actor = actor

    @property
    def instance_id(self) -> str:
        return self.descriptor["instance_id"]

    @property
    def generation(self) -> int:
        return int(self.descriptor["generation"])


class GatewayCore:
    """Authoritative scheduling, ownership and migration coordination."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.state = GatewayState(
            config.ledger_path or f"/tmp/omp-hetero-gateway-{config.layout_id}.json"
        )
        self.scheduler = CentralScheduler(config.policy)
        self.clients: Dict[str, InstanceClient] = {}
        self._dispatch_semaphore = asyncio.Semaphore(config.max_inflight_requests)
        self._migrations_done: Dict[str, int] = {}
        self._traces: List[Dict[str, Any]] = []
        self._request_records: Dict[str, Dict[str, Any]] = {}
        self._migration_task: Optional[asyncio.Task] = None
        self._capacity_task: Optional[asyncio.Task] = None
        self._started_at = time.time()
        self._backlog_admitted = 0
        self._backlog_completed = 0
        self._last_sweep: Dict[str, Any] = {}
        self._sweeps = 0
        self._sweep_errors: List[Dict[str, Any]] = []
        self._last_probe: Dict[str, Dict[str, Any]] = {}
        self._last_refresh_s = 0.0
        self._projected: Dict[str, Dict[str, int]] = {}
        self._transaction_locks: Dict[str, asyncio.Lock] = {}
        self._dispatch_tasks: Dict[str, asyncio.Task] = {}
        self._recovery_lock = asyncio.Lock()
        self._recovery_pending = self.state.recovered
        self._orphaned_requests = {
            row["request_id"] for row in self.state.snapshot()["ownership"]
            if row["state"] != "finished" and (row["state"] != "cancelled" or row["cancellation_pending"])
        }
        self._paused_requests: Dict[str, str] = {}

    # ------------------------------------------------------- discovery

    def bind_instances(self, descriptors: List[Dict[str, Any]], actors: Dict[str, Any]) -> None:
        clients: Dict[str, InstanceClient] = {}
        for descriptor in descriptors:
            actor = actors.get(descriptor["instance_id"])
            if actor is None:
                continue
            clients[descriptor["instance_id"]] = InstanceClient(descriptor, actor)
        self.clients = clients

    def routable_instances(self) -> List[str]:
        return sorted(self.clients)

    # -------------------------------------------------------- capacity

    async def _maybe_refresh(self) -> None:
        """Refreshes observed capacity at most once per refresh interval.

        Hundreds of concurrently waiting admissions must not each fan out a
        status RPC to every engine.
        """
        now = time.time()
        if now - self._last_refresh_s < self.config.capacity_refresh_s:
            return
        self._last_refresh_s = now
        await self.refresh_capacity()

    async def refresh_capacity(self) -> Dict[str, Any]:
        rows = {}
        for instance_id, client in self.clients.items():
            try:
                sample_started_at = time.monotonic()
                status = await self._rpc(client.actor.status.remote())
                if int(status["generation"]) != client.generation:
                    raise RuntimeError("instance generation changed")
            except Exception as exc:  # pragma: no cover - instance vanished
                rows[instance_id] = {"error": str(exc)[:200]}
                continue
            self.state.observe_capacity(
                instance_id,
                total_blocks=int(status.get("total_blocks", 0)),
                free_blocks=int(status.get("free_blocks", 0)),
                block_size=int(status.get("block_size", 0)),
                running=int(status.get("num_running", 0)),
                waiting=int(status.get("num_waiting", 0)),
                resident_growth_allowance=self._growth_allowance(status),
                sample_started_at=sample_started_at,
            )
            rows[instance_id] = status
        return rows

    def _growth_allowance(self, status: Dict[str, Any]) -> int:
        """Blocks held back for the decode growth of already-resident requests."""
        block_size = max(1, int(status.get("block_size", 1)))
        running = int(status.get("num_running", 0))
        # One block of growth per resident request per accounting interval.
        return min(int(status.get("free_blocks", 0)), running)

    def projected_usage(self) -> Dict[str, Dict[str, int]]:
        """Conservative endpoint projection for requests this gateway owns.

        Engine-observed free blocks lag admission by up to one capacity refresh,
        so a burst of concurrent dispatches would otherwise all score against
        the same stale snapshot and herd onto a single instance.  Every
        unfinished request is charged its conservative terminal footprint --
        ``ceil((prompt + max_tokens) / block_size)`` -- against its current
        owner until it completes or is migrated away.

        Maintained incrementally: recomputing it per admission poll is O(n) in
        the backlog and starves the gateway's event loop on a deep manifest.
        """
        return {
            name: dict(self._projected.get(name, {"blocks": 0, "requests": 0}))
            for name in self.clients
        }

    def _project_add(self, instance_id: str, blocks: int) -> None:
        row = self._projected.setdefault(instance_id, {"blocks": 0, "requests": 0})
        row["blocks"] += blocks
        row["requests"] += 1

    def _project_remove(self, instance_id: str, blocks: int) -> None:
        row = self._projected.setdefault(instance_id, {"blocks": 0, "requests": 0})
        row["blocks"] = max(0, row["blocks"] - blocks)
        row["requests"] = max(0, row["requests"] - 1)

    def _project_move(self, source: str, destination: str, blocks: int) -> None:
        self._project_remove(source, blocks)
        self._project_add(destination, blocks)

    def candidates(self) -> List[Candidate]:
        rows = []
        inflight = self._inflight_migrations_by_destination()
        projected = self.projected_usage()
        for instance_id, client in self.clients.items():
            capacity = self.state.capacity(instance_id)
            if capacity is None:
                continue
            observed_used = capacity.total_blocks - capacity.free_blocks
            projected_used = projected.get(instance_id, {}).get("blocks", 0)
            # Take the pessimistic view of occupancy: whichever of the engine's
            # own accounting and the gateway's projection is larger.
            effective_free = max(
                0, capacity.total_blocks - max(observed_used, projected_used)
            )
            rows.append(
                Candidate(
                    instance_id=instance_id,
                    tp_size=int(client.descriptor.get("tp_size", 0)),
                    max_model_len=int(client.descriptor.get("max_model_len", 0)),
                    block_size=capacity.block_size,
                    total_blocks=capacity.total_blocks,
                    free_blocks=effective_free,
                    admissible_blocks=min(capacity.admissible_blocks, effective_free),
                    running=capacity.running,
                    waiting=capacity.waiting
                    + max(0, projected.get(instance_id, {}).get("requests", 0)
                          - capacity.running),
                    inflight_migrations=inflight.get(instance_id, 0),
                )
            )
        return rows

    def _inflight_migrations_by_destination(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for txn in self.state.transactions():
            if txn["state"] not in ("released", "aborted"):
                counts[txn["destination_instance"]] = (
                    counts.get(txn["destination_instance"], 0) + 1
                )
        return counts

    # ----------------------------------------------- stable request owner

    async def dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """One result task per request identity, including duplicate client calls."""
        self._validate_request(request)
        await self._ensure_ready()
        request_id = request.get("request_id") or f"omp-{uuid.uuid4().hex[:16]}"
        task = self._dispatch_tasks.get(request_id)
        if task is None:
            if self.state.owner(request_id) is not None:
                raise RuntimeError("request identity belongs to a previous coordinator/run")
            task = asyncio.create_task(self._dispatch_request({**request, "request_id": request_id}))
            self._dispatch_tasks[request_id] = task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await self.cancel(request_id)
            raise
        except Exception as exc:
            owner = self.state.owner(request_id)
            if owner is not None and owner.state not in (OwnershipState.FINISHED, OwnershipState.CANCELLED):
                self._paused_requests[request_id] = f"response unresolved: {type(exc).__name__}: {exc}"
            raise

    @staticmethod
    def _validate_request(request: Dict[str, Any]) -> None:
        # This serving contract is greedy single-completion inference. Never
        # silently drop sampling/stop/structured-output semantics at migration.
        supported = {"request_id", "model", "prompt", "prompt_token_ids", "max_tokens",
                     "ignore_eos", "temperature", "stream"}
        unsupported = set(request) - supported
        if unsupported:
            raise ValueError(f"unsupported request fields: {sorted(unsupported)}")
        if request.get("temperature", 0.0) != 0.0:
            raise ValueError("only greedy temperature=0 requests are supported")
        if request.get("stream"):
            raise ValueError("streaming is not supported")

    async def _dispatch_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Owns one logical request end to end, across any migration."""
        request_id = request.get("request_id") or f"omp-{uuid.uuid4().hex[:16]}"
        prompt_token_ids = request.get("prompt_token_ids")
        prompt = request.get("prompt")
        max_tokens = int(request.get("max_tokens", 16))
        prompt_tokens = (
            len(prompt_token_ids)
            if prompt_token_ids is not None
            else _estimate_prompt_tokens(prompt)
        )
        record = {
            "request_id": request_id,
            "admitted_at_s": time.time(),
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "segments": [],
            "useful_output_tokens": 0,
            "output_token_ids": [],
            "text": "",
            "migrations": [],
            "ignore_eos": bool(request.get("ignore_eos", False)),
        }
        self._request_records[request_id] = record

        async with self._dispatch_semaphore:
            # Bulk backlog: a request that does not fit yet waits for capacity
            # instead of failing.  The wait is part of the measured makespan.
            deadline = time.time() + self.config.admission_timeout_s
            waits = 0
            while True:
                await self._ensure_ready()
                await self._maybe_refresh()
                instance_id, trace = self.scheduler.select_initial(
                    self.candidates(),
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                )
                if instance_id is not None:
                    break
                if trace.get("refusal") == "no_instance_supports_context":
                    break
                if time.time() >= deadline:
                    break
                waits += 1
                await asyncio.sleep(self.config.admission_poll_s)
            trace["request_id"] = request_id
            trace["admission_waits"] = waits
            trace["admission_wait_s"] = round(
                time.time() - record["admitted_at_s"], 4
            )
            self._traces.append(trace)
            if instance_id is None:
                record["refused"] = trace.get("refusal")
                raise RuntimeError(f"gateway refused {request_id}: {trace.get('refusal')}")
            record["admission_wait_s"] = trace["admission_wait_s"]
            client = self.clients[instance_id]
            self.state.claim(request_id, instance_id, client.generation)
            capacity = self.state.capacity(instance_id)
            projected_blocks = blocks_for(
                prompt_tokens + max_tokens, capacity.block_size if capacity else 256
            )
            record["projected_blocks"] = projected_blocks
            self._project_add(instance_id, projected_blocks)
            self._backlog_admitted += 1
            record["initial_instance"] = instance_id

            remaining = max_tokens
            current = instance_id
            payload_prompt_ids = prompt_token_ids
            payload_prompt = prompt
            while True:
                segment = await self._run_segment(
                    request_id,
                    current,
                    prompt=payload_prompt,
                    prompt_token_ids=payload_prompt_ids,
                    max_tokens=remaining,
                    record=record,
                )
                transaction_id = segment.get("migrated_transaction")
                if transaction_id:
                    txn = self.state.transaction(transaction_id)
                    if txn is None or txn.request_id != request_id or txn.state not in COMMITTED_STATES:
                        raise RuntimeError("source returned an uncommitted migration")
                    # Scheduler-fenced prefix is authoritative, even if the
                    # asynchronously delivered source stream was behind it.
                    segment["output_token_ids"] = list(txn.snapshot["generated_token_ids"])
                    segment["output_tokens"] = len(segment["output_token_ids"])
                    segment["text"] = txn.snapshot["generated_text"]
                record["segments"].append(segment)
                # The committed prefix
                # or not the request went on to migrate.  Dropping it here
                # would make a migrated request short by exactly the prefix.
                produced = list(segment.get("output_token_ids") or [])
                record["useful_output_tokens"] += len(produced)
                record["output_token_ids"].extend(produced)
                record["text"] += segment.get("text") or ""
                remaining -= len(produced)
                transaction_id = segment.get("migrated_transaction")
                if not transaction_id:
                    break
                owner = self.state.owner(request_id)
                current = owner.instance_id
                continuation = await self._await_continuation(
                    request_id, transaction_id, current
                )
                record["segments"].append(continuation)
                new_ids = list(continuation["new_output_token_ids"])
                record["useful_output_tokens"] += len(new_ids)
                record["output_token_ids"].extend(new_ids)
                record["text"] = continuation.get("full_output_text", record["text"] + (continuation.get("text") or ""))
                remaining -= len(new_ids)
                record["migrations"].append(
                    {
                        "transaction_id": transaction_id,
                        "source": segment["segment_instance"],
                        "destination": current,
                        "committed_prefix_tokens": len(produced),
                        "replayed_tokens": continuation["replayed_tokens"],
                        "continuation_tokens": len(new_ids),
                    }
                )
                break

        record["finished_at_s"] = time.time()
        record["elapsed_s"] = round(record["finished_at_s"] - record["admitted_at_s"], 4)
        owner = self.state.owner(request_id)
        if owner is not None and owner.state is OwnershipState.CANCELLED:
            raise asyncio.CancelledError()
        if owner is not None:
            self._project_remove(owner.instance_id, record.get("projected_blocks", 0))
        self.state.finish(request_id, record["useful_output_tokens"])
        self._backlog_completed += 1
        return record

    async def _run_segment(
        self,
        request_id: str,
        instance_id: str,
        *,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]],
        max_tokens: int,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        client = self.clients[instance_id]
        future = client.actor.generate.remote(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=0.0,
            request_id=request_id,
            ignore_eos=bool(record.get("ignore_eos", False)),
        )
        result = await self._rpc(future, timeout_s=self.config.result_timeout_s)
        result["segment_instance"] = instance_id
        return result

    async def _await_continuation(
        self, request_id: str, transaction_id: str, destination: str
    ) -> Dict[str, Any]:
        # Source stream completion can race the final release ACK. Join the
        # coordinator before observing continuation output, never release here.
        action = await self._execute(transaction_id)
        if action.get("unresolved"):
            raise RuntimeError(action["unresolved"])
        txn = self.state.transaction(transaction_id)
        client = self._transaction_client(txn, destination=True)
        return await self._rpc(
            client.actor.await_continuation.remote(
                transaction_id=transaction_id, timeout_s=self.config.result_timeout_s
            ), timeout_s=self.config.result_timeout_s,
        )

    # ------------------------------------------------ migration coordinator

    async def poll_and_migrate(self) -> List[Dict[str, Any]]:
        """One bounded migration sweep: propose, reserve, quiesce, prepare, commit."""
        await self._ensure_ready()
        policy = self.config.policy
        if not policy.migration_enabled:
            return []
        await self.refresh_capacity()
        actions: List[Dict[str, Any]] = []
        for source_id in policy.small_instances:
            client = self.clients.get(source_id)
            if client is None:
                continue
            proposal = await self._rpc(
                client.actor.migration_candidates.remote(
                    min_generated_tokens=policy.min_generated_tokens,
                    min_context_tokens=policy.min_context_tokens,
                    min_remaining_tokens=policy.min_remaining_tokens,
                )
            )
            if int(proposal["generation"]) != client.generation:
                raise RuntimeError("stale migration proposal generation")
            pressure = proposal["pressure"]
            free_fraction = (
                pressure["free_blocks"] / max(1, pressure["total_blocks"])
            )
            # Always record why a source produced nothing; an empty sweep is
            # otherwise indistinguishable from a disabled policy.
            self._last_probe[source_id] = {
                "at": time.time(),
                "free_fraction": round(free_fraction, 4),
                "pressure_threshold": policy.source_pressure_free_block_fraction,
                "under_pressure": free_fraction
                <= policy.source_pressure_free_block_fraction,
                "num_candidates": len(proposal["candidates"]),
                "running": pressure.get("num_running"),
                "rejected": proposal.get("rejected", [])[:5],
            }
            skips: List[str] = []
            for candidate in proposal["candidates"]:
                action = await self._consider(source_id, candidate, free_fraction)
                actions.append(action)
                if action.get("skipped"):
                    skips.append(f"{candidate['request_id']}:{action['skipped']}")
                if action.get("committed"):
                    break
            self._last_probe[source_id]["skips"] = skips[:5]
            self._last_probe[source_id]["actions"] = [
                a.get("declined") or a.get("skipped") or ("committed" if a.get("committed") else "other")
                for a in actions[-len(proposal["candidates"]) :]
            ][:5]
        return actions

    async def _consider(
        self, source_id: str, candidate: Dict[str, Any], free_fraction: float
    ) -> Dict[str, Any]:
        request_id = candidate["request_id"]
        owner = self.state.owner(request_id)
        if owner is None or owner.instance_id != source_id:
            return {"request_id": request_id, "skipped": "not_owned_here"}
        if owner.state is not OwnershipState.OWNED:
            return {"request_id": request_id, "skipped": f"state_{owner.state.value}"}

        destination, trace, reserve = self.scheduler.select_migration_destination(
            self.candidates(),
            source_instance=source_id,
            prompt_tokens=candidate["prompt_tokens"],
            generated_tokens=candidate["generated_tokens"],
            remaining_tokens=candidate["remaining_tokens"],
            source_free_fraction=free_fraction,
            migrations_done=self._migrations_done.get(request_id, 0),
            total_inflight_migrations=sum(
                self._inflight_migrations_by_destination().values()
            ),
            unsupported=candidate.get("unsupported_continuation") or [],
        )
        trace["request_id"] = request_id
        self._traces.append(trace)
        if destination is None:
            return {"request_id": request_id, "declined": trace.get("refusal"), "trace": trace}

        txn, reason = self.state.open_transaction(
            request_id=request_id,
            source_instance=source_id,
            destination_instance=destination,
            mechanism=self.config.policy.migration_mechanism,
            blocks=reserve["blocks"],
            growth_blocks=reserve["growth_blocks"],
            destination_generation=self.clients[destination].generation,
            max_inflight_per_destination=self.config.policy.max_inflight_migrations_per_destination,
            max_context_tokens=int(candidate["prompt_tokens"]) + int(candidate["generated_tokens"]) + int(candidate["remaining_tokens"]),
        )
        if txn is None:
            return {"request_id": request_id, "declined": reason, "trace": trace}

        return await self._execute(txn.transaction_id)

    async def _rpc(self, reference: Any, *, timeout_s: Optional[float] = None) -> Any:
        """Bounded local wait; timing out never cancels an uncertain remote RPC."""
        future = asyncio.wrap_future(reference.future())
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.wait_for(
            asyncio.shield(future), timeout=self.config.rpc_timeout_s if timeout_s is None else timeout_s
        )

    def _transaction_client(self, txn: Any, *, destination: bool) -> InstanceClient:
        name = txn.destination_instance if destination else txn.source_instance
        generation = txn.destination_generation if destination else txn.source_generation
        client = self.clients.get(name)
        if client is None or client.generation != generation:
            raise RuntimeError(f"transaction endpoint unavailable or replaced: {name}@{generation}")
        return client

    async def _receipt_rpc(
        self, client: InstanceClient, receipt_transaction_id: str, method: str, receipt_key: str,
        *, timeout_s: Optional[float] = None, **arguments: Any,
    ) -> Dict[str, Any]:
        try:
            receipt = await self._rpc(
                getattr(client.actor, method).remote(**arguments), timeout_s=timeout_s
            )
        except Exception as original:
            # Lost ACK is not rejection. Consult the engine's idempotent receipt
            # while the fsynced ledger remains the sole ownership authority.
            status = await self._rpc(
                client.actor.continuation_status.remote(transaction_id=receipt_transaction_id)
            )
            if int(status["generation"]) != client.generation:
                raise RuntimeError("receipt belongs to a different engine generation") from original
            receipt = status.get(receipt_key)
            if not isinstance(receipt, dict):
                raise RuntimeError(f"{method} decision unresolved: {original}") from original
        if (not isinstance(receipt, dict) or receipt.get("generation") != client.generation
                or receipt.get("transaction_id") != receipt_transaction_id):
            raise RuntimeError(f"{method} receipt identity does not match the transaction generation")
        return receipt

    @staticmethod
    def _source_arguments(txn: Any) -> Dict[str, Any]:
        return {
            "request_id": txn.request_id, "transaction_id": txn.transaction_id,
            "source_generation": txn.source_generation, "source_epoch": txn.source_epoch,
        }

    async def _abort_precommit(self, txn: Any, reason: str, report: Dict[str, Any]) -> None:
        self.state.ensure_healthy()
        if txn.state in COMMITTED_STATES:
            await self._finish_commit(txn, report)
            return
        if txn.state is TransactionState.ABORTED:
            report["aborted"] = txn.decline_reason
            return
        self.state.abort(txn.transaction_id, reason)
        destination = self._transaction_client(txn, destination=True)
        source = self._transaction_client(txn, destination=False)
        abort = await self._receipt_rpc(
            destination, txn.transaction_id, "abort_continuation", "abort",
            transaction_id=txn.transaction_id,
        )
        if not abort.get("aborted"):
            raise RuntimeError(f"native abort not acknowledged: {abort}")
        # Including a never-quiesced source installs a resume tombstone, fencing
        # a delayed quiesce. Never resume before destination cleanup is proven.
        resume = await self._receipt_rpc(
            source, txn.transaction_id, "unquiesce", "resume", **self._source_arguments(txn)
        )
        if not resume.get("resumed"):
            raise RuntimeError(f"source resume not acknowledged: {resume}")
        self.state.finish_abort(txn.transaction_id, abort, resume)
        report["aborted"] = txn.decline_reason
        report["abort"] = abort
        report["resume"] = resume

    async def _finish_commit(self, txn: Any, report: Dict[str, Any]) -> None:
        self.state.ensure_healthy()
        if txn.state not in COMMITTED_STATES:
            raise RuntimeError("activation requires durable committed ownership")
        if txn.state is not TransactionState.RELEASED:
            destination = self._transaction_client(txn, destination=True)
            source = self._transaction_client(txn, destination=False)
            if not txn.activation_receipt:
                activation = await self._receipt_rpc(
                    destination, txn.transaction_id, "commit_continuation", "activation",
                    transaction_id=txn.transaction_id, ownership=txn.ownership_certificate,
                )
                self.state.mark_activated(txn.transaction_id, activation)
            release = await self._receipt_rpc(
                source, txn.transaction_id, "release_source", "release",
                request_id=txn.request_id, transaction_id=txn.transaction_id,
                ownership=txn.ownership_certificate,
            )
            self.state.release(txn.transaction_id, release)
        self._migrations_done[txn.request_id] = txn.destination_epoch
        report.update({
            "committed": True, "activation": txn.activation_receipt, "release": txn.release_receipt,
            "t_committed_s": txn.committed_at, "t_released_s": txn.released_at,
        })

    async def _execute(self, transaction_id: str, *, recovering: bool = False) -> Dict[str, Any]:
        """Reserve native storage, fence, prepare, fsync ownership, then activate."""
        lock = self._transaction_locks.setdefault(transaction_id, asyncio.Lock())
        async with lock:
            txn = self.state.transaction(transaction_id)
            if txn is None:
                raise RuntimeError("unknown migration transaction")
            report: Dict[str, Any] = {
                "transaction_id": transaction_id, "request_id": txn.request_id,
                "source": txn.source_instance, "destination": txn.destination_instance,
                "mechanism": txn.mechanism, "t_open_s": txn.created_at,
            }
            try:
                self.state.ensure_healthy()
                if txn.state in COMMITTED_STATES:
                    await self._finish_commit(txn, report)
                    return report
                if recovering or txn.state in (TransactionState.ABORTING, TransactionState.ABORTED):
                    await self._abort_precommit(txn, "coordinator_recovery", report)
                    return report
                source = self._transaction_client(txn, destination=False)
                destination = self._transaction_client(txn, destination=True)
                identity = {
                    **self._source_arguments(txn), "source_instance": txn.source_instance,
                    "destination_epoch": txn.destination_epoch,
                    "destination_generation": txn.destination_generation,
                    "max_context_tokens": txn.max_context_tokens, "mechanism": txn.mechanism,
                }
                reservation = await self._receipt_rpc(
                    destination, transaction_id, "reserve_continuation", "reservation", payload=identity,
                )
                if not reservation.get("accepted"):
                    raise RuntimeError(f"native destination reservation refused: {reservation}")
                self.state.mark_reserved(transaction_id, reservation)
                report["reservation"] = reservation
                snapshot = await self._rpc(
                    source.actor.quiesce.remote(
                        **self._source_arguments(txn), timeout_s=self.config.quiesce_timeout_s
                    ), timeout_s=self.config.quiesce_timeout_s + self.config.rpc_timeout_s,
                )
                for key, expected in self._source_arguments(txn).items():
                    if snapshot.get(key) != expected:
                        raise RuntimeError(f"source snapshot identity mismatch: {key}")
                if not isinstance(snapshot.get("generated_text"), str):
                    raise RuntimeError("source snapshot omitted exact committed text")
                snapshot_tokens = len(snapshot["prompt_token_ids"]) + len(snapshot["generated_token_ids"])
                self.state.mark_quiesced(transaction_id, snapshot_tokens, snapshot)
                report.update({"t_quiesced_s": time.time(), "snapshot_tokens": snapshot_tokens})
                payload = {**snapshot, **identity}
                if txn.mechanism == "kv_host":
                    # Deliberately nested Ray ref: only the destination resolves
                    # export tensors. Never await, inspect, persist or log it here.
                    payload["kv_export"] = {"object_ref": source.actor.export_request_kv.remote(
                        **self._source_arguments(txn)
                    )}
                elif txn.mechanism != "recompute":
                    raise RuntimeError(f"unsupported migration mechanism: {txn.mechanism}")
                verdict = await self._receipt_rpc(
                    destination, transaction_id, "prepare_continuation", "preparation",
                    payload=payload, timeout_s=self.config.prepare_timeout_s,
                )
                report["prepare"] = verdict
                if not verdict.get("accepted"):
                    raise RuntimeError(f"destination preparation refused: {verdict}")
                replay = int(verdict["rebuild_tokens"])
                transferred = int(verdict["transferred_bytes"])
                if txn.mechanism == "kv_host" and transferred <= 0:
                    raise RuntimeError("kv_host prepared no transferred native KV")
                if txn.mechanism == "recompute" and transferred != 0:
                    raise RuntimeError("recompute preparation reported KV transfer")
                self.state.mark_prepared(
                    transaction_id, replay_tokens=replay, transferred_bytes=transferred, receipt=verdict
                )
                report["t_prepared_s"] = time.time()
                if self.state.commit(transaction_id, destination.generation) is None:
                    raise RuntimeError("authoritative ownership commit rejected")
                record = self._request_records.get(txn.request_id)
                if record is not None:
                    self._project_move(txn.source_instance, txn.destination_instance,
                                       int(record.get("projected_blocks", 0)))
                await self._finish_commit(txn, report)
            except asyncio.CancelledError:
                try:
                    await self._abort_precommit(txn, "coordinator_cancelled", report)
                except Exception as exc:
                    self._record_unresolved(txn, exc, report)
                raise
            except Exception as exc:
                try:
                    # Reads the durable decision first; a committed transaction
                    # can only advance, including when activation's ACK was lost.
                    await self._abort_precommit(txn, f"{type(exc).__name__}: {exc}"[:300], report)
                except Exception as cleanup:
                    self._record_unresolved(txn, cleanup, report)
            return report

    def _record_unresolved(self, txn: Any, exc: Exception, report: Dict[str, Any]) -> None:
        reason = f"handoff unresolved; coordinator paused: {type(exc).__name__}: {exc}"[:500]
        report["unresolved"] = reason
        if self.state.storage_error is None:
            self.state.mark_unresolved(txn.transaction_id, reason)

    async def recover(self) -> Dict[str, Any]:
        """Reconcile the ledger before traffic; interrupted HTTP responses fail closed."""
        async with self._recovery_lock:
            if self._recovery_pending:
                for row in self.state.transactions():
                    if row["state"] not in ("released", "aborted"):
                        await self._execute(row["transaction_id"], recovering=True)
                for row in self.state.snapshot()["ownership"]:
                    if row["cancellation_pending"]:
                        try:
                            await self.cancel(row["request_id"])
                        except Exception:
                            pass  # cancel records the unresolved decision, never resumes traffic
                self._recovery_pending = False
        return self.recovery_status()

    def recovery_status(self) -> Dict[str, Any]:
        unresolved = [row["transaction_id"] for row in self.state.transactions() if row["unresolved"]]
        return {
            "ready": not (self._recovery_pending or self.state.storage_error or unresolved
                          or self._orphaned_requests or self._paused_requests),
            "pending": self._recovery_pending, "storage_error": self.state.storage_error,
            "unresolved_transactions": unresolved,
            "interrupted_responses": sorted(self._orphaned_requests),
            "paused_requests": dict(self._paused_requests),
        }

    async def _ensure_ready(self) -> None:
        self.state.ensure_healthy()
        await self.recover()
        for row in self.state.transactions():
            if row["unresolved"]:
                emit_event("GATEWAY_RECOVERING_TRANSACTION", transaction=row["transaction_id"],
                           state=row["state"], unresolved=str(row["unresolved"])[:500])
                await self._execute(row["transaction_id"], recovering=True)
        status = self.recovery_status()
        if not status["ready"]:
            raise RuntimeError(f"gateway fails closed: {status}")

    async def cancel(self, request_id: str) -> Dict[str, Any]:
        """Terminal cancellation is idempotent and is never ownership rollback."""
        try:
            return await self._cancel(request_id)
        except Exception as exc:
            self._paused_requests[request_id] = f"cancellation unresolved: {type(exc).__name__}: {exc}"
            raise

    async def _cancel(self, request_id: str) -> Dict[str, Any]:
        self.state.cancel(request_id)
        owner = self.state.owner(request_id)
        if owner is None:
            task = self._dispatch_tasks.get(request_id)
            if task is not None:
                task.cancel()
            return {"request_id": request_id, "cancelled": True}
        if owner.state is OwnershipState.FINISHED:
            return {"request_id": request_id, "cancelled": False, "finished": True}
        if owner.cancellation_receipt:
            self._orphaned_requests.discard(request_id)
            self._paused_requests.pop(request_id, None)
            return owner.cancellation_receipt
        transactions = [row for row in self.state.transactions() if row["request_id"] == request_id]
        for row in transactions:
            action = await self._execute(row["transaction_id"], recovering=True)
            if action.get("unresolved"):
                return {"request_id": request_id, "cancelled": False, "unresolved": action["unresolved"]}
        committed = next((row for row in reversed(transactions) if row["state"] in ("committed", "released")), None)
        if committed is not None:
            txn = self.state.transaction(committed["transaction_id"])
            client = self._transaction_client(txn, destination=True)
            receipt = await self._receipt_rpc(
                client, txn.transaction_id, "cancel_continuation", "cancellation",
                transaction_id=txn.transaction_id, ownership=txn.ownership_certificate,
            )
        else:
            client = self.clients.get(owner.instance_id)
            if client is None or client.generation != owner.instance_generation:
                raise RuntimeError("cancel owner generation unavailable")
            receipt = await self._rpc(client.actor.cancel_request.remote(
                request_id=request_id, epoch=owner.epoch, generation=owner.instance_generation
            ))
        if not receipt.get("cancelled"):
            raise RuntimeError(f"terminal cancellation not acknowledged: {receipt}")
        self.state.record_cancellation(request_id, receipt)
        self._orphaned_requests.discard(request_id)
        self._paused_requests.pop(request_id, None)
        task = self._dispatch_tasks.get(request_id)
        if task is not None:
            task.cancel()
        record = self._request_records.get(request_id)
        if record is not None and not record.get("cancelled"):
            record["cancelled"] = True
            self._project_remove(owner.instance_id, int(record.get("projected_blocks", 0)))
            self._backlog_completed += 1
        return receipt

    # ------------------------------------------------------------ status

    def snapshot(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "recovery": self.recovery_status(),
            "instances": self.routable_instances(),
            "migration_sweeps": self._sweeps,
            "last_sweep": self._last_sweep,
            "sweep_errors": self._sweep_errors,
            "last_probe": self._last_probe,
            "projected_usage": self.projected_usage(),
            "state": self.state.snapshot(),
            "backlog": {
                "admitted": self._backlog_admitted,
                "completed": self._backlog_completed,
                "inflight": self._backlog_admitted - self._backlog_completed,
                "response_tasks": sum(not task.done() for task in self._dispatch_tasks.values()),
            },
            "uptime_s": round(time.time() - self._started_at, 2),
        }

    def traces(self, limit: int = 2000) -> List[Dict[str, Any]]:
        return self._traces[-limit:]

    def request_records(self) -> List[Dict[str, Any]]:
        return list(self._request_records.values())



    def record_sweep(self, actions: List[Dict[str, Any]]) -> None:
        self._last_sweep = {
            "at": time.time(),
            "actions": actions[-20:],
            "count": len(actions),
        }
        self._sweeps += 1

    def record_sweep_error(self, exc: BaseException) -> None:
        self._sweep_errors.append(
            {"at": time.time(), "error": f"{type(exc).__name__}: {exc}"[:400]}
        )
        del self._sweep_errors[:-20]

    def descriptors(self) -> Dict[str, Dict[str, Any]]:
        return {name: dict(c.descriptor) for name, c in self.clients.items()}

    async def await_idle_instances(self, timeout_s: float = 300.0) -> Dict[str, Any]:
        """Non-mutating idle barrier across every engine."""
        names = sorted(self.clients)
        results = await asyncio.gather(
            *[
                self._rpc(self.clients[name].actor.await_idle.remote(timeout_s),
                          timeout_s=timeout_s + self.config.rpc_timeout_s)
                for name in names
            ],
            return_exceptions=True,
        )
        rows = {}
        for name, result in zip(names, results):
            rows[name] = (
                {"error": str(result)[:200]}
                if isinstance(result, BaseException)
                else result
            )
        return rows

    async def resume_admission(self) -> Dict[str, Any]:
        """Clears any latched drain left by an operator action."""
        names = sorted(self.clients)
        results = await asyncio.gather(
            *[self._rpc(self.clients[n].actor.resume_admission.remote()) for n in names]
        )
        return dict(zip(names, results))

    async def start_capture(self) -> Dict[str, Any]:
        await asyncio.gather(
            *[
                self._rpc(c.actor.start_iteration_capture.remote(), timeout_s=30.0 + self.config.rpc_timeout_s)
                for c in self.clients.values()
            ]
        )
        return {"capturing": sorted(self.clients)}

    async def stop_capture(self) -> Dict[str, List[Dict[str, Any]]]:
        names = sorted(self.clients)
        results = await asyncio.gather(
            *[
                self._rpc(self.clients[name].actor.stop_iteration_capture.remote(),
                          timeout_s=300.0 + self.config.rpc_timeout_s)
                for name in names
            ]
        )
        return dict(zip(names, results))

    async def reset_records(self) -> None:
        # An explicit new run is the only recovery for lost response tasks. Do
        # not discard decisions, receipts or projections while work can execute.
        await self.recover()
        if any(not task.done() for task in self._dispatch_tasks.values()):
            raise RuntimeError("cannot reset live gateway response tasks")
        for row in self.state.transactions():
            if row["state"] not in ("released", "aborted"):
                await self._execute(row["transaction_id"], recovering=True)
        for request_id in self._orphaned_requests:
            owner = self.state.owner(request_id)
            client = self.clients.get(owner.instance_id)
            if client is None or client.generation != owner.instance_generation:
                raise RuntimeError("cannot reset an interrupted response on a replaced engine generation")
        idle = await self.await_idle_instances()
        if not idle or any(not row.get("idle") for row in idle.values()):
            raise RuntimeError(f"cannot reset until every engine is idle: {idle}")
        self.state.reset()
        # Fresh-routing share is workload state, not lifetime traffic debt.
        self.scheduler = CentralScheduler(self.config.policy)
        self._dispatch_tasks.clear()
        self._transaction_locks.clear()
        self._orphaned_requests.clear()
        self._paused_requests.clear()
        self._traces.clear()
        self._request_records.clear()
        self._migrations_done.clear()
        self._backlog_admitted = 0
        self._backlog_completed = 0
        self._last_sweep: Dict[str, Any] = {}
        self._sweeps = 0
        self._sweep_errors: List[Dict[str, Any]] = []
        self._last_probe: Dict[str, Dict[str, Any]] = {}
        self._last_refresh_s = 0.0
        self._projected = {}


def _estimate_prompt_tokens(prompt: Optional[str]) -> int:
    if not prompt:
        return 0
    # Conservative upper estimate used only for eligibility before tokenisation;
    # the engine reports the real count back in the segment record.
    return max(1, len(prompt) // 3)
