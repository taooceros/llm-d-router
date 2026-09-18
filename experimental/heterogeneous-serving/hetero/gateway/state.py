"""Authoritative ownership epochs, native reservations and durable handoff decisions.

A single local writer appends changed records and fsyncs each recoverability
barrier before sending an RPC. Claim/finish cost one owner record, independent of
backlog history. Only bootstrap and an explicitly idle reset atomically replace
the base snapshot. Recovery replays complete journal lines; a torn final append
cannot have authorized an RPC and is discarded. Complete malformed records fail
closed. File/write/fsync errors latch uncertainty, never permission to roll back.

Supported fault envelope: coordinator process loss with the same journal/lock
path and unchanged live engine generations; delayed/duplicate RPCs and lost ACKs.
The flock is not a distributed lease. Missing journal beside its lock marker,
corruption, unavailable receipts, or replaced engines refuse new traffic. Loss
of the entire local directory/node is outside automatic recovery: an operator
must retire/drain the old fabric before bootstrapping a new ledger. The default
/tmp path survives process loss only, not machine replacement. Interrupted HTTP
responses are not reconstructed; cancellation or an explicit idle-proven reset
is required before new traffic. Native tensors never enter this journal.
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class OwnershipState(str, Enum):
    OWNED = "owned"
    QUIESCING = "quiescing"
    PREPARED = "prepared"
    COMMITTED = "committed"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class TransactionState(str, Enum):
    RESERVED = "reserved"
    NATIVE_RESERVED = "native_reserved"
    QUIESCED = "quiesced"
    PREPARED = "prepared"
    COMMITTED = "committed"
    RELEASED = "released"
    ABORTING = "aborting"
    ABORTED = "aborted"
    DECLINED = "declined"


COMMITTED_STATES = (TransactionState.COMMITTED, TransactionState.RELEASED)
TERMINAL_STATES = (TransactionState.RELEASED, TransactionState.ABORTED)


@dataclasses.dataclass
class Ownership:
    request_id: str
    instance_id: str
    epoch: int = 0
    state: OwnershipState = OwnershipState.OWNED
    committed_offset: int = 0
    instance_generation: int = 0
    updated_at: float = dataclasses.field(default_factory=time.time)
    transaction_id: Optional[str] = None
    cancellation_pending: bool = False
    cancellation_receipt: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        body = dataclasses.asdict(self)
        body["state"] = self.state.value
        return body


@dataclasses.dataclass
class Reservation:
    """Conservative gateway credits; not a substitute for native reservation."""

    transaction_id: str
    request_id: str
    instance_id: str
    blocks: int
    growth_blocks: int
    created_at: float = dataclasses.field(default_factory=time.time)
    converted: bool = False
    native_reserved_blocks: int = 0
    native_reserved_at: float = 0.0
    credit_observed: bool = False

    @property
    def total_blocks(self) -> int:
        return self.blocks + self.growth_blocks

    def to_dict(self) -> Dict[str, Any]:
        body = dataclasses.asdict(self)
        body["total_blocks"] = self.total_blocks
        return body


@dataclasses.dataclass
class MigrationTransaction:
    transaction_id: str
    request_id: str
    source_instance: str
    destination_instance: str
    source_epoch: int
    destination_epoch: int
    mechanism: str
    source_generation: int = 0
    destination_generation: int = 0
    max_context_tokens: int = 0
    state: TransactionState = TransactionState.RESERVED
    created_at: float = dataclasses.field(default_factory=time.time)
    updated_at: float = dataclasses.field(default_factory=time.time)
    committed_at: Optional[float] = None
    released_at: Optional[float] = None
    snapshot_tokens: int = 0
    replay_tokens: int = 0
    transferred_bytes: int = 0
    decline_reason: Optional[str] = None
    snapshot: Dict[str, Any] = dataclasses.field(default_factory=dict)
    reservation_receipt: Dict[str, Any] = dataclasses.field(default_factory=dict)
    preparation_receipt: Dict[str, Any] = dataclasses.field(default_factory=dict)
    activation_receipt: Dict[str, Any] = dataclasses.field(default_factory=dict)
    release_receipt: Dict[str, Any] = dataclasses.field(default_factory=dict)
    abort_receipt: Dict[str, Any] = dataclasses.field(default_factory=dict)
    resume_receipt: Dict[str, Any] = dataclasses.field(default_factory=dict)
    ownership_certificate: Dict[str, Any] = dataclasses.field(default_factory=dict)
    unresolved: Optional[str] = None
    timeline: List[Tuple[str, float]] = dataclasses.field(default_factory=list)

    def mark(self, state: TransactionState) -> None:
        self.state = state
        self.updated_at = time.time()
        self.timeline.append((state.value, self.updated_at))

    def to_dict(self) -> Dict[str, Any]:
        body = dataclasses.asdict(self)
        body["state"] = self.state.value
        body["timeline"] = [[name, ts] for name, ts in self.timeline]
        return body


@dataclasses.dataclass
class InstanceCapacity:
    instance_id: str
    total_blocks: int = 0
    free_blocks: int = 0
    block_size: int = 0
    reserved_blocks: int = 0
    native_reserved_blocks: int = 0
    staging_blocks: int = 0
    resident_growth_allowance: int = 0
    observed_at: float = 0.0
    running: int = 0
    waiting: int = 0

    @property
    def admissible_blocks(self) -> int:
        return max(0, self.free_blocks - self.reserved_blocks
                   - self.staging_blocks - self.resident_growth_allowance)

    def to_dict(self) -> Dict[str, Any]:
        body = dataclasses.asdict(self)
        body["admissible_blocks"] = self.admissible_blocks
        return body


class GatewayState:
    """Single-writer state. No-path mode is only for isolated protocol checks."""

    def __init__(self, ledger_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._ownership: Dict[str, Ownership] = {}
        self._reservations: Dict[str, Reservation] = {}
        self._transactions: Dict[str, MigrationTransaction] = {}
        self._capacity: Dict[str, InstanceCapacity] = {}
        self._unobserved_reservations: Dict[str, Dict[str, Reservation]] = {}
        self._ledger_path = Path(ledger_path) if ledger_path else None
        self._ledger_lock = None
        self._journal = None
        self.storage_error: Optional[str] = None
        self.recovered = False
        self._counters = dict.fromkeys(
            ("proposals", "approved", "declined", "committed", "aborted", "released"), 0
        )
        if self._ledger_path is not None:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = Path(str(self._ledger_path) + ".lock")
            existed = lock_path.exists()
            self._ledger_lock = open(lock_path, "a")
            try:
                fcntl.flock(self._ledger_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if self._ledger_path.exists():
                    self._load()
                    self._journal = open(self._ledger_path, "a", encoding="utf-8")
                elif existed:
                    raise RuntimeError("ownership ledger missing beside its persistent lock marker")
                else:
                    self._compact()
            except Exception:
                self.close()
                raise

    def ensure_healthy(self) -> None:
        if self.storage_error is not None:
            raise RuntimeError(f"ownership ledger unavailable: {self.storage_error}")

    def _persist(self, *records: Any) -> None:
        """Append only the changed records, never serialize the growing ledger."""
        self.ensure_healthy()
        if self._ledger_path is None:
            return
        body = {"ownership": [], "transactions": [], "reservations": [],
                "counters": dict(self._counters)}
        for record in records:
            key = ("ownership" if isinstance(record, Ownership) else
                   "transactions" if isinstance(record, MigrationTransaction) else "reservations")
            body[key].append(record.to_dict())
        try:
            self._journal.write(json.dumps(body, separators=(",", ":"), allow_nan=False) + "\n")
            self._journal.flush()
            os.fsync(self._journal.fileno())
        except Exception as exc:
            self.storage_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError("ownership journal append uncertain; coordinator paused") from exc

    def _compact(self) -> None:
        """Only bootstrap/idle reset rewrites the base snapshot atomically."""
        self.ensure_healthy()
        if self._ledger_path is None:
            return
        path = self._ledger_path
        temporary = path.with_name(path.name + ".pending")
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump({"version": 2, "snapshot": self.snapshot()}, stream,
                          separators=(",", ":"), allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if self._journal is not None:
                self._journal.close()
            self._journal = open(path, "a", encoding="utf-8")
        except Exception as exc:
            self.storage_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError("ownership snapshot replacement uncertain; coordinator paused") from exc

    def _apply_record(self, body: Dict[str, Any]) -> None:
        for row in body["ownership"]:
            row["state"] = OwnershipState(row["state"])
            owner = Ownership(**row)
            self._ownership[owner.request_id] = owner
        for row in body["transactions"]:
            row["state"] = TransactionState(row["state"])
            txn = MigrationTransaction(**row)
            self._transactions[txn.transaction_id] = txn
        for row in body["reservations"]:
            row.pop("total_blocks", None)
            reservation = Reservation(**row)
            self._reservations[reservation.transaction_id] = reservation
        self._counters.update(body["counters"])

    def _load(self) -> None:
        with open(self._ledger_path, "rb+") as stream:
            header = stream.readline()
            if not header.endswith(b"\n"):
                raise RuntimeError("incomplete ownership journal base snapshot")
            body = json.loads(header)
            if body["version"] != 2:
                raise RuntimeError("unsupported ownership journal version")
            self._apply_record(body["snapshot"])
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # A torn final append cannot have passed flush/fsync, so no
                    # RPC was authorized by it. Keep the last complete decision.
                    stream.truncate(offset)
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                self._apply_record(json.loads(line))
        # Observations expire at restart; reconstruct outstanding credits only.
        for reservation in self._reservations.values():
            if not reservation.converted:
                capacity = self._capacity.setdefault(
                    reservation.instance_id, InstanceCapacity(reservation.instance_id)
                )
                reservation.credit_observed = False
                reservation.native_reserved_at = 0.0
                capacity.reserved_blocks += reservation.native_reserved_blocks or reservation.total_blocks
                capacity.native_reserved_blocks += reservation.native_reserved_blocks
                self._unobserved_reservations.setdefault(reservation.instance_id, {})[
                    reservation.transaction_id
                ] = reservation
        self.recovered = bool(self._ownership or self._transactions)

    def close(self) -> None:
        if self._journal is not None:
            self._journal.close()
            self._journal = None
        if self._ledger_lock is not None:
            self._ledger_lock.close()
            self._ledger_lock = None

    def claim(self, request_id: str, instance_id: str, generation: int) -> Ownership:
        with self._lock:
            self.ensure_healthy()
            existing = self._ownership.get(request_id)
            if existing is not None:
                if (existing.instance_id, existing.instance_generation) != (instance_id, generation):
                    raise RuntimeError("request identity already has a different owner")
                return existing
            owner = Ownership(request_id, instance_id, instance_generation=generation)
            self._ownership[request_id] = owner
            self._persist(owner)
            return owner

    def owner(self, request_id: str) -> Optional[Ownership]:
        with self._lock:
            return self._ownership.get(request_id)

    def finish(self, request_id: str, committed_offset: int) -> None:
        with self._lock:
            self.ensure_healthy()
            owner = self._ownership.get(request_id)
            if owner is None or owner.state is OwnershipState.CANCELLED:
                return
            if owner.transaction_id:
                txn = self._transactions[owner.transaction_id]
                if txn.state not in TERMINAL_STATES:
                    raise RuntimeError("cannot finish an unresolved handoff")
            owner.state = OwnershipState.FINISHED
            owner.committed_offset = committed_offset
            owner.updated_at = time.time()
            self._persist(owner)

    def cancel(self, request_id: str) -> None:
        """Records cancellation intent; native cleanup must still be acknowledged."""
        with self._lock:
            self.ensure_healthy()
            owner = self._ownership.get(request_id)
            if owner is None or owner.state in (OwnershipState.CANCELLED, OwnershipState.FINISHED):
                return
            owner.state = OwnershipState.CANCELLED
            owner.cancellation_pending = True
            owner.updated_at = time.time()
            changed = []
            for txn in self._transactions.values():
                if txn.request_id == request_id and txn.state not in COMMITTED_STATES:
                    self._abort_locked(txn, "request_cancelled")
                    changed.append(txn)
            self._persist(owner, *changed)

    def record_cancellation(self, request_id: str, receipt: Dict[str, Any]) -> None:
        with self._lock:
            self.ensure_healthy()
            owner = self._ownership[request_id]
            if owner.state is not OwnershipState.CANCELLED or not receipt.get("cancelled"):
                raise RuntimeError("terminal cancellation not acknowledged")
            owner.cancellation_receipt = dict(receipt)
            owner.cancellation_pending = False
            self._persist(owner)

    def observe_capacity(
        self, instance_id: str, *, total_blocks: int, free_blocks: int,
        block_size: int, running: int, waiting: int, resident_growth_allowance: int,
        sample_started_at: Optional[float] = None,
    ) -> InstanceCapacity:
        with self._lock:
            record = self._capacity.setdefault(instance_id, InstanceCapacity(instance_id))
            record.total_blocks = total_blocks
            record.free_blocks = free_blocks
            record.block_size = block_size
            record.running = running
            record.waiting = waiting
            record.resident_growth_allowance = resident_growth_allowance
            record.observed_at = time.time()
            # Only a status call issued after the native ACK can include those
            # held pages. Earlier in-flight samples retain the gateway credit.
            observed_after = time.monotonic() if sample_started_at is None else sample_started_at
            pending = self._unobserved_reservations.get(instance_id, {})
            for transaction_id, reservation in list(pending.items()):
                if reservation.native_reserved_blocks and reservation.native_reserved_at <= observed_after:
                    record.reserved_blocks = max(0, record.reserved_blocks - reservation.native_reserved_blocks)
                    reservation.credit_observed = True
                    del pending[transaction_id]
            return record

    def capacity(self, instance_id: str) -> Optional[InstanceCapacity]:
        with self._lock:
            return self._capacity.get(instance_id)

    def capacities(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [c.to_dict() for c in self._capacity.values()]

    def open_transaction(
        self, *, request_id: str, source_instance: str, destination_instance: str,
        mechanism: str, blocks: int, growth_blocks: int, destination_generation: int,
        max_inflight_per_destination: int, max_context_tokens: int = 0,
    ) -> Tuple[Optional[MigrationTransaction], Optional[str]]:
        with self._lock:
            self.ensure_healthy()
            self._counters["proposals"] += 1
            owner = self._ownership.get(request_id)
            if owner is None:
                return None, "unknown_request"
            if owner.instance_id != source_instance:
                return None, "stale_owner"
            if any(t.request_id == request_id and t.state not in TERMINAL_STATES
                   for t in self._transactions.values()):
                return None, "duplicate_proposal"
            if owner.state is not OwnershipState.OWNED:
                return None, f"request_not_owned_stable:{owner.state.value}"
            capacity = self._capacity.get(destination_instance)
            if capacity is None:
                return None, "destination_unknown"
            inflight = sum(t.destination_instance == destination_instance
                           and t.state not in TERMINAL_STATES
                           for t in self._transactions.values())
            if inflight >= max_inflight_per_destination:
                self._counters["declined"] += 1
                return None, "destination_migration_limit"
            if capacity.admissible_blocks < blocks + growth_blocks:
                self._counters["declined"] += 1
                return None, "insufficient_destination_capacity"
            transaction_id = f"mig-{uuid.uuid4().hex}"
            capacity.reserved_blocks += blocks + growth_blocks
            self._reservations[transaction_id] = Reservation(
                transaction_id, request_id, destination_instance, blocks, growth_blocks
            )
            self._unobserved_reservations.setdefault(destination_instance, {})[transaction_id] = (
                self._reservations[transaction_id]
            )
            txn = MigrationTransaction(
                transaction_id, request_id, source_instance, destination_instance,
                owner.epoch, owner.epoch + 1, mechanism,
                source_generation=owner.instance_generation,
                destination_generation=destination_generation,
                max_context_tokens=max_context_tokens or (blocks + growth_blocks) * capacity.block_size,
            )
            txn.mark(TransactionState.RESERVED)
            self._transactions[transaction_id] = txn
            owner.state = OwnershipState.QUIESCING
            owner.transaction_id = transaction_id
            owner.updated_at = time.time()
            self._counters["approved"] += 1
            self._persist(owner, txn, self._reservations[transaction_id])
            return txn, None

    def mark_reserved(self, transaction_id: str, receipt: Dict[str, Any]) -> MigrationTransaction:
        with self._lock:
            self.ensure_healthy()
            txn = self._transactions[transaction_id]
            if txn.state is not TransactionState.RESERVED:
                return txn
            if not receipt.get("accepted") or int(receipt.get("native_reserved_blocks", 0)) <= 0:
                raise RuntimeError("native destination reservation not acknowledged")
            txn.reservation_receipt = dict(receipt)
            reservation = self._reservations[transaction_id]
            reservation.native_reserved_blocks = int(receipt["native_reserved_blocks"])
            reservation.native_reserved_at = time.monotonic()
            capacity = self._capacity[txn.destination_instance]
            capacity.reserved_blocks += reservation.native_reserved_blocks - reservation.total_blocks
            capacity.native_reserved_blocks += reservation.native_reserved_blocks
            txn.mark(TransactionState.NATIVE_RESERVED)
            self._persist(txn, self._reservations[transaction_id])
            return txn

    def mark_quiesced(
        self, transaction_id: str, snapshot_tokens: int,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> MigrationTransaction:
        with self._lock:
            self.ensure_healthy()
            txn = self._transactions[transaction_id]
            if txn.state is TransactionState.QUIESCED:
                return txn
            if txn.state is not TransactionState.NATIVE_RESERVED:
                raise RuntimeError("quiescence requires acknowledged native reservation")
            txn.snapshot_tokens = snapshot_tokens
            txn.snapshot = dict(snapshot or {})
            txn.mark(TransactionState.QUIESCED)
            self._persist(txn)
            return txn

    def mark_prepared(
        self, transaction_id: str, *, replay_tokens: int = 0, transferred_bytes: int = 0,
        receipt: Optional[Dict[str, Any]] = None,
    ) -> MigrationTransaction:
        with self._lock:
            self.ensure_healthy()
            txn = self._transactions[transaction_id]
            if txn.state is TransactionState.PREPARED:
                return txn
            if txn.state is not TransactionState.QUIESCED:
                raise RuntimeError("preparation requires a quiesced source")
            if replay_tokens < 0 or transferred_bytes < 0:
                raise ValueError("negative preparation accounting")
            txn.replay_tokens = replay_tokens
            txn.transferred_bytes = transferred_bytes
            txn.preparation_receipt = dict(receipt or {})
            txn.mark(TransactionState.PREPARED)
            self._ownership[txn.request_id].state = OwnershipState.PREPARED
            self._persist(txn, self._ownership[txn.request_id])
            return txn

    def commit(self, transaction_id: str, destination_generation: int) -> Optional[MigrationTransaction]:
        """Fsync the ownership decision before destination activation/source release."""
        with self._lock:
            self.ensure_healthy()
            txn = self._transactions.get(transaction_id)
            if txn is None or destination_generation != txn.destination_generation:
                return None
            if txn.state in COMMITTED_STATES:
                return txn
            if txn.state is not TransactionState.PREPARED:
                return None
            owner = self._ownership.get(txn.request_id)
            if owner is None or (owner.instance_id, owner.instance_generation, owner.epoch) != (
                txn.source_instance, txn.source_generation, txn.source_epoch
            ) or owner.state is not OwnershipState.PREPARED:
                return None
            owner.instance_id = txn.destination_instance
            owner.epoch = txn.destination_epoch
            owner.state = OwnershipState.OWNED
            owner.instance_generation = destination_generation
            owner.committed_offset = len(txn.snapshot.get("generated_token_ids", []))
            owner.updated_at = time.time()
            txn.committed_at = time.time()
            txn.mark(TransactionState.COMMITTED)
            # Stable certificate: later receipts/timeline updates do not change it.
            txn.ownership_certificate = {
                key: getattr(txn, key) for key in (
                    "transaction_id", "request_id", "source_instance", "destination_instance",
                    "source_epoch", "destination_epoch", "source_generation", "destination_generation",
                    "mechanism", "committed_at",
                )
            }
            txn.ownership_certificate["state"] = "committed"
            self._convert_reservation(txn)
            self._counters["committed"] += 1
            self._persist(txn, owner, self._reservations[transaction_id])
            return txn

    def mark_activated(self, transaction_id: str, receipt: Dict[str, Any]) -> None:
        with self._lock:
            self.ensure_healthy()
            txn = self._transactions[transaction_id]
            if txn.state not in COMMITTED_STATES or not receipt.get("activated"):
                raise RuntimeError("activation not acknowledged under committed ownership")
            txn.activation_receipt = dict(receipt)
            self._persist(txn)

    def release(self, transaction_id: str, receipt: Dict[str, Any]) -> MigrationTransaction:
        with self._lock:
            self.ensure_healthy()
            txn = self._transactions[transaction_id]
            if txn.state is TransactionState.RELEASED:
                return txn
            if txn.state is not TransactionState.COMMITTED or not txn.activation_receipt or not receipt.get("released"):
                raise RuntimeError("source release requires activation and release acknowledgements")
            txn.release_receipt = dict(receipt)
            txn.released_at = time.time()
            txn.unresolved = None
            txn.mark(TransactionState.RELEASED)
            self._counters["released"] += 1
            self._persist(txn)
            return txn

    def abort(self, transaction_id: str, reason: str) -> MigrationTransaction:
        with self._lock:
            self.ensure_healthy()
            txn = self._transactions[transaction_id]
            self._abort_locked(txn, reason)
            self._persist(txn)
            return txn

    def _abort_locked(self, txn: MigrationTransaction, reason: str) -> None:
        if txn.state in COMMITTED_STATES or txn.state in (TransactionState.ABORTING, TransactionState.ABORTED):
            return
        txn.decline_reason = reason
        txn.mark(TransactionState.ABORTING)

    def finish_abort(
        self, transaction_id: str, abort_receipt: Dict[str, Any], resume_receipt: Dict[str, Any],
    ) -> MigrationTransaction:
        with self._lock:
            self.ensure_healthy()
            txn = self._transactions[transaction_id]
            if txn.state is TransactionState.ABORTED:
                return txn
            if txn.state is not TransactionState.ABORTING or not abort_receipt.get("aborted") or not resume_receipt.get("resumed"):
                raise RuntimeError("abort cleanup/resume not acknowledged")
            txn.abort_receipt = dict(abort_receipt)
            txn.resume_receipt = dict(resume_receipt)
            self._convert_reservation(txn)
            owner = self._ownership[txn.request_id]
            if owner.transaction_id == transaction_id:
                if owner.state not in (OwnershipState.CANCELLED, OwnershipState.FINISHED):
                    owner.state = OwnershipState.OWNED
                owner.transaction_id = None
                owner.updated_at = time.time()
            txn.unresolved = None
            txn.mark(TransactionState.ABORTED)
            self._counters["aborted"] += 1
            self._persist(txn, owner, self._reservations[transaction_id])
            return txn

    def _convert_reservation(self, txn: MigrationTransaction) -> None:
        reservation = self._reservations.get(txn.transaction_id)
        if reservation is None or reservation.converted:
            return
        capacity = self._capacity.get(txn.destination_instance)
        if capacity is not None:
            if not reservation.credit_observed:
                credit = reservation.native_reserved_blocks or reservation.total_blocks
                capacity.reserved_blocks = max(0, capacity.reserved_blocks - credit)
            capacity.native_reserved_blocks = max(0, capacity.native_reserved_blocks - reservation.native_reserved_blocks)
        self._unobserved_reservations.get(txn.destination_instance, {}).pop(txn.transaction_id, None)
        reservation.converted = True

    def mark_unresolved(self, transaction_id: str, reason: str) -> None:
        with self._lock:
            self.ensure_healthy()
            self._transactions[transaction_id].unresolved = reason
            self._persist(self._transactions[transaction_id])

    def transaction(self, transaction_id: str) -> Optional[MigrationTransaction]:
        with self._lock:
            return self._transactions.get(transaction_id)

    def transactions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [t.to_dict() for t in self._transactions.values()]

    def counters(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ownership": [o.to_dict() for o in self._ownership.values()],
                "reservations": [r.to_dict() for r in self._reservations.values()],
                "transactions": [t.to_dict() for t in self._transactions.values()],
                "capacity": [c.to_dict() for c in self._capacity.values()],
                "counters": dict(self._counters),
            }

    def reset(self) -> None:
        """Explicit measurement reset; caller must first prove all engines idle."""
        with self._lock:
            self.ensure_healthy()
            if any(t.state not in TERMINAL_STATES for t in self._transactions.values()):
                raise RuntimeError("cannot reset unresolved ownership transactions")
            self._ownership.clear()
            self._reservations.clear()
            self._unobserved_reservations.clear()
            self._transactions.clear()
            for capacity in self._capacity.values():
                capacity.reserved_blocks = 0
                capacity.native_reserved_blocks = 0
            for key in self._counters:
                self._counters[key] = 0
            self.recovered = False
            self._compact()
