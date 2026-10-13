"""Local control-plane fault checks; no accelerator or performance simulation."""

import asyncio
import concurrent.futures
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from hetero.gateway.core import GatewayConfig, GatewayCore
from hetero.gateway.policy import Candidate, CentralScheduler, RoutingPolicy
from hetero.gateway.state import GatewayState, OwnershipState, TransactionState


def ledger_view(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    base = records[0]["snapshot"]
    owners = {row["request_id"]: row for row in base["ownership"]}
    transactions = {row["transaction_id"]: row for row in base["transactions"]}
    for record in records[1:]:
        owners.update((row["request_id"], row) for row in record["ownership"])
        transactions.update((row["transaction_id"], row) for row in record["transactions"])
    return {"ownership": list(owners.values()), "transactions": list(transactions.values())}


class LocalReference:
    def __init__(self, task):
        self.result = concurrent.futures.Future()

        def complete(done):
            if done.cancelled():
                self.result.cancel()
            elif done.exception() is not None:
                self.result.set_exception(done.exception())
            else:
                self.result.set_result(done.result())

        task.add_done_callback(complete)

    def future(self):
        return self.result


class ExportReference:
    def future(self):
        raise AssertionError("gateway must never resolve the tensor export")


class RemoteMethod:
    def __init__(self, engine, name):
        self.engine = engine
        self.name = name

    def remote(self, *args, **kwargs):
        if self.name == "export_request_kv":
            return self.engine.export

        async def invoke():
            return await getattr(self.engine, self.name)(*args, **kwargs)

        return LocalReference(asyncio.create_task(invoke()))


class LocalActor:
    def __init__(self, engine):
        self.engine = engine

    def __getattr__(self, name):
        return RemoteMethod(self.engine, name)


class ControlPeer:
    """A fault-injectable native RPC contract, not a model/allocator emulator."""

    def __init__(self, fabric, name):
        self.fabric = fabric
        self.name = name
        self.receipts = {}
        self.reserved = False
        self.fenced = False
        self.activated = False
        self.activation_count = 0
        self.cancel_count = 0
        self.generate_count = 0
        self.refuse_prepare = False
        self.lose_abort = False
        self.lose_commit = False
        self.lose_status = False
        self.commit_ack = asyncio.Event()
        self.export = ExportReference()
        self.started = asyncio.Event()
        self.source_result = asyncio.get_running_loop().create_future()

    def receipt(self, txn, **fields):
        return {"transaction_id": txn, "generation": 1, "instance_id": self.name, **fields}

    async def status(self):
        return {"generation": 1, "total_blocks": 32, "free_blocks": 32,
                "block_size": 4, "num_running": 0, "num_waiting": 0}

    async def generate(self, **request):
        self.generate_count += 1
        self.started.set()
        return await self.source_result

    async def reserve_continuation(self, *, payload):
        self.fabric.events.append("reserve")
        self.reserved = True
        result = self.receipt(payload["transaction_id"], accepted=True, native_reserved_blocks=2)
        self.receipts["reservation"] = result
        return result

    async def quiesce(self, **identity):
        if not self.fabric.destination.reserved:
            raise AssertionError("source fenced before native reservation")
        self.fabric.events.append("quiesce")
        self.fenced = True
        identity.pop("timeout_s")
        return {**identity, "source_instance": self.name, "prompt_token_ids": [1, 2],
                "generated_token_ids": [11, 12], "generated_text": "AB", "remaining_tokens": 2,
                "prompt_len": 2, "generated_len": 2, "sampling": {"max_tokens": 4}}

    async def prepare_continuation(self, *, payload):
        self.fabric.events.append("prepare")
        if payload["mechanism"] == "kv_host":
            assert payload["kv_export"]["object_ref"] is self.fabric.source.export
        result = self.receipt(payload["transaction_id"], accepted=not self.refuse_prepare,
                              rebuild_tokens=1 if payload["mechanism"] == "kv_host" else 4,
                              transferred_bytes=128 if payload["mechanism"] == "kv_host" else 0,
                              evidence={"contract_fixture": True})
        self.receipts["preparation"] = result
        return result

    async def commit_continuation(self, *, transaction_id, ownership):
        disk = ledger_view(self.fabric.path)
        transaction = next(row for row in disk["transactions"] if row["transaction_id"] == transaction_id)
        assert transaction["state"] in ("committed", "released")
        assert disk["ownership"][0]["instance_id"] == "large"
        assert transaction["ownership_certificate"] == ownership
        if not self.activated:
            self.activation_count += 1
            self.activated = True
            self.fabric.events.append("activate")
        result = self.receipt(transaction_id, activated=True)
        self.receipts["activation"] = result
        if self.lose_commit:
            await self.commit_ack.wait()
        return result

    async def release_source(self, *, request_id, transaction_id, ownership):
        assert self.fabric.destination.activated
        self.fabric.events.append("release")
        self.fenced = False
        result = self.receipt(transaction_id, released=True)
        self.receipts["release"] = result
        if not self.source_result.done():
            # Deliberately lag the fenced snapshot by one output token.
            self.source_result.set_result({"output_token_ids": [11], "text": "A",
                                           "migrated_transaction": transaction_id})
        return result

    async def await_continuation(self, *, transaction_id, timeout_s):
        return {"new_output_token_ids": [13, 14], "text": "CD", "full_output_text": "ABCD",
                "replayed_tokens": self.receipts["preparation"]["rebuild_tokens"], "finish_reason": "length"}

    async def abort_continuation(self, *, transaction_id):
        if self.lose_abort:
            raise TimeoutError("abort unavailable")
        assert not self.activated
        self.fabric.events.append("abort")
        self.reserved = False
        result = self.receipt(transaction_id, aborted=True)
        self.receipts["abort"] = result
        return result

    async def unquiesce(self, **identity):
        assert not self.fabric.destination.reserved
        assert not self.fabric.destination.activated
        self.fabric.events.append("resume")
        self.fenced = False
        result = self.receipt(identity["transaction_id"], resumed=True)
        self.receipts["resume"] = result
        return result

    async def continuation_status(self, *, transaction_id):
        if self.lose_status:
            raise TimeoutError("receipt unavailable")
        return self.receipt(transaction_id, **self.receipts)

    async def cancel_request(self, *, request_id, epoch, generation):
        self.cancel_count += 1
        return {"request_id": request_id, "epoch": epoch, "generation": generation, "cancelled": True}

    async def cancel_continuation(self, *, transaction_id, ownership):
        assert ownership["destination_epoch"] == 1 and self.activated
        result = await self.cancel_request(request_id=ownership["request_id"], epoch=1, generation=1)
        result["transaction_id"] = transaction_id
        self.receipts["cancellation"] = result
        return result

    async def await_idle(self, timeout_s=300.0, **kwargs):
        return {"idle": not self.fenced}


class GatewayHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "ownership.json"
        self.events = []
        self.source = ControlPeer(self, "small")
        self.destination = ControlPeer(self, "large")
        self.config = GatewayConfig(
            ledger_path=str(self.path), rpc_timeout_s=0.01,
            policy=RoutingPolicy(small_instances=("small",), large_instances=("large",),
                                 migration_enabled=True, large_fresh_traffic_share=0,
                                 reserve_large_headroom_blocks=0),
        )
        self.core = self.new_core()
        await self.core.refresh_capacity()

    def new_core(self):
        core = GatewayCore(self.config)
        core.bind_instances([
            {"instance_id": name, "generation": 1, "max_model_len": 32, "tp_size": size}
            for name, size in (("small", 8), ("large", 16))
        ], {"small": LocalActor(self.source), "large": LocalActor(self.destination)})
        return core

    async def asyncTearDown(self):
        self.destination.commit_ack.set()
        for task in self.core._dispatch_tasks.values():
            if not task.done():
                task.cancel()
        self.core.state.close()
        self.directory.cleanup()

    async def test_reset_restores_fresh_routing_share(self):
        candidates = self.core.candidates()
        for _ in range(8):
            chosen, _ = self.core.scheduler.select_initial(
                candidates, prompt_tokens=2, max_tokens=4)
            self.assertEqual(chosen, "small")
        policy = RoutingPolicy.from_dict({
            **self.config.policy.to_dict(), "large_fresh_traffic_share": 0.5})
        self.core.config.policy = policy
        self.core.scheduler.policy = policy
        await self.core.reset_records()
        placements = [self.core.scheduler.select_initial(
            candidates, prompt_tokens=2, max_tokens=4)[0] for _ in range(8)]
        self.assertEqual(placements, ["large", "small"] * 4)

    def open_transaction(self, mechanism="recompute"):
        self.core.state.claim("r", "small", 1)
        txn, reason = self.core.state.open_transaction(
            request_id="r", source_instance="small", destination_instance="large",
            mechanism=mechanism, blocks=1, growth_blocks=1, max_context_tokens=6,
            destination_generation=1, max_inflight_per_destination=2,
        )
        self.assertIsNone(reason)
        return txn

    async def test_kv_transport_and_exactly_once_response(self):
        request = {"request_id": "r", "prompt_token_ids": [1, 2], "max_tokens": 4}
        first = asyncio.create_task(self.core.dispatch(request))
        duplicate = asyncio.create_task(self.core.dispatch(request))
        await self.source.started.wait()
        txn = self.open_transaction("kv_host")
        report = await self.core._execute(txn.transaction_id)
        self.assertTrue(report["committed"])
        self.assertEqual((txn.transferred_bytes, txn.replay_tokens), (128, 1))
        results = await asyncio.gather(first, duplicate)
        self.assertEqual(results[0]["output_token_ids"], [11, 12, 13, 14])
        self.assertEqual(results[0]["text"], "ABCD")
        self.assertEqual(results[0]["useful_output_tokens"], 4)
        self.assertEqual(results[1], results[0])
        await self.core._execute(txn.transaction_id)
        self.assertEqual(self.source.generate_count, 1)
        self.assertEqual(self.destination.activation_count, 1)
        self.assertEqual(self.events, ["reserve", "quiesce", "prepare", "activate", "release"])

    async def test_unknown_abort_keeps_source_fenced_and_credit_held(self):
        txn = self.open_transaction()
        self.destination.refuse_prepare = True
        self.destination.lose_abort = True
        report = await self.core._execute(txn.transaction_id)
        self.assertIn("unresolved", report)
        self.assertTrue(self.source.fenced)
        self.assertTrue(self.destination.reserved)
        self.assertEqual(txn.state, TransactionState.ABORTING)
        self.assertEqual(self.core.state.capacity("large").reserved_blocks, 2)
        self.destination.lose_abort = False
        report = await self.core._execute(txn.transaction_id, recovering=True)
        self.assertIn("aborted", report)
        self.assertEqual(self.events[-2:], ["abort", "resume"])
        self.assertEqual(self.core.state.owner("r").state, OwnershipState.OWNED)
        self.assertEqual(self.core.state.capacity("large").reserved_blocks, 0)

    async def test_lost_commit_ack_is_never_rolled_back(self):
        txn = self.open_transaction()
        self.destination.lose_commit = self.destination.lose_status = True
        report = await self.core._execute(txn.transaction_id)
        self.assertIn("unresolved", report)
        self.assertEqual(txn.state, TransactionState.COMMITTED)
        self.assertTrue(self.destination.activated)
        self.assertTrue(self.source.fenced)
        self.assertNotIn("abort", self.events)
        self.assertNotIn("resume", self.events)
        self.destination.lose_status = False
        report = await self.core._execute(txn.transaction_id, recovering=True)
        self.assertTrue(report["committed"])
        self.assertEqual(txn.state, TransactionState.RELEASED)
        self.assertEqual(self.destination.activation_count, 1)

    async def test_uncertain_disk_commit_recovers_forward_before_new_traffic(self):
        txn = self.open_transaction()
        from hetero.gateway import state as state_module
        fsync = state_module.os.fsync

        def fail_after_commit_sync(descriptor):
            fsync(descriptor)
            body = ledger_view(self.path)
            if body["transactions"][0]["state"] == "committed":
                raise OSError("journal fsync decision unavailable")

        with mock.patch.object(state_module.os, "fsync", side_effect=fail_after_commit_sync):
            report = await self.core._execute(txn.transaction_id)
        self.assertIn("unresolved", report)
        self.assertTrue(self.source.fenced)
        self.assertFalse(self.destination.activated)
        self.assertNotIn("resume", self.events)
        self.core.state.close()
        self.core = self.new_core()
        await self.core.recover()
        self.assertEqual(self.core.state.transaction(txn.transaction_id).state, TransactionState.RELEASED)
        self.assertEqual(self.destination.activation_count, 1)
        with self.assertRaises(RuntimeError):
            await self.core.dispatch({"request_id": "new", "prompt_token_ids": [1]})
        self.assertEqual(self.core.recovery_status()["interrupted_responses"], ["r"])

    async def test_unsupported_sampling_is_rejected_before_admission(self):
        with self.assertRaises(ValueError):
            await self.core.dispatch({"prompt_token_ids": [1], "stop": ["end"]})
        with self.assertRaises(ValueError):
            await self.core.dispatch({"prompt_token_ids": [1], "temperature": 0.8})
        self.assertEqual(self.source.generate_count, 0)
        self.assertEqual(self.core.state.snapshot()["ownership"], [])

    async def test_replacement_generation_cannot_receive_delayed_commit(self):
        txn = self.open_transaction()
        self.core.clients["large"].descriptor["generation"] = 2
        report = await self.core._execute(txn.transaction_id)
        self.assertIn("unresolved", report)
        self.assertEqual(self.events, [])
        self.assertEqual(self.core.state.owner("r").instance_id, "small")
        self.assertFalse(self.core.recovery_status()["ready"])

    async def test_cancellation_is_terminal_not_commit_rollback(self):
        txn = self.open_transaction()
        await self.core._execute(txn.transaction_id)
        first = await self.core.cancel("r")
        second = await self.core.cancel("r")
        self.assertEqual(first, second)
        self.assertEqual(self.destination.cancel_count, 1)
        owner = self.core.state.owner("r")
        self.assertEqual((owner.instance_id, owner.epoch, owner.state), ("large", 1, OwnershipState.CANCELLED))
        self.assertEqual(txn.state, TransactionState.RELEASED)
        self.assertNotIn("resume", self.events)

    async def test_admission_polls_do_not_scale_with_retained_handoffs(self):
        # Every waiting admission re-checks readiness and candidates at 20 Hz.
        # Copying each retained handoff's token snapshot per poll saturated the
        # gateway event loop after the first W0 migration; Envoy's ExtProc
        # streams then sat unaccepted until gRPC core cancelled them (500s).
        quiesce = self.source.quiesce

        async def long_context(**identity):
            return {**await quiesce(**identity), "generated_token_ids": list(range(200_000))}

        self.source.quiesce = long_context
        txn = self.open_transaction()

        def inflight():
            return {c.instance_id: c.inflight_migrations for c in self.core.candidates()}

        self.assertEqual(inflight(), {"small": 0, "large": 1})
        self.assertTrue((await self.core._execute(txn.transaction_id))["committed"])
        self.assertEqual(len(txn.snapshot["generated_token_ids"]), 200_000)
        self.assertEqual(inflight(), {"small": 0, "large": 0})
        started = time.perf_counter()
        for _ in range(100):
            await self.core._ensure_ready()
            self.core.candidates()
        self.assertLess(time.perf_counter() - started, 0.5)


class GatewayLedgerTests(unittest.TestCase):
    def test_single_writer_and_native_ack_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "ledger.json")
            state = GatewayState(path)
            try:
                with self.assertRaises(BlockingIOError):
                    GatewayState(path)
                state.observe_capacity("large", total_blocks=8, free_blocks=8, block_size=4,
                                       running=0, waiting=0, resident_growth_allowance=0)
                state.claim("r", "small", 1)
                txn, _ = state.open_transaction(
                    request_id="r", source_instance="small", destination_instance="large",
                    mechanism="recompute", blocks=1, growth_blocks=1,
                    destination_generation=1, max_inflight_per_destination=1,
                )
                with self.assertRaises(RuntimeError):
                    state.mark_quiesced(txn.transaction_id, 4)
                state.cancel("r")
                self.assertEqual(state.capacity("large").reserved_blocks, 2)
                state.finish_abort(txn.transaction_id, {"aborted": True}, {"resumed": True})
                self.assertEqual(state.owner("r").state, OwnershipState.CANCELLED)
                self.assertEqual(state.capacity("large").reserved_blocks, 0)
            finally:
                state.close()

    def test_native_reservation_is_counted_once_after_fresh_observation(self):
        state = GatewayState()
        observed = dict(total_blocks=8, free_blocks=8, block_size=4,
                        running=0, waiting=0, resident_growth_allowance=0)
        state.observe_capacity("large", **observed)
        state.claim("r", "small", 1)
        txn, _ = state.open_transaction(
            request_id="r", source_instance="small", destination_instance="large",
            mechanism="recompute", blocks=1, growth_blocks=1,
            destination_generation=1, max_inflight_per_destination=1,
        )
        state.mark_reserved(txn.transaction_id, {"accepted": True, "native_reserved_blocks": 2})
        # This response started before the reservation and cannot cover its credit.
        state.observe_capacity("large", sample_started_at=0, **observed)
        self.assertEqual(state.capacity("large").admissible_blocks, 6)
        observed["free_blocks"] = 6
        state.observe_capacity("large", **observed)
        capacity = state.capacity("large")
        self.assertEqual((capacity.reserved_blocks, capacity.native_reserved_blocks), (0, 2))
        self.assertEqual(capacity.admissible_blocks, 6)
        state.abort(txn.transaction_id, "declined")
        state.finish_abort(txn.transaction_id, {"aborted": True}, {"resumed": True})
        self.assertEqual(capacity.native_reserved_blocks, 0)

    def test_torn_final_append_cannot_change_recovered_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            state = GatewayState(str(path))
            state.claim("r", "small", 1)
            state.close()
            with path.open("ab") as stream:
                stream.write(b'{"ownership":[{"request_id":"r","instance_id":"large"')
            recovered = GatewayState(str(path))
            try:
                self.assertEqual((recovered.owner("r").instance_id, recovered.owner("r").epoch), ("small", 0))
                recovered.finish("r", 4)
            finally:
                recovered.close()
            final = GatewayState(str(path))
            self.assertEqual(final.owner("r").state, OwnershipState.FINISHED)
            final.close()

    def test_m_arms_share_the_fresh_admission_boundary(self):
        candidate = Candidate("large", 16, 32, 4, 16, 5, 5, 0, 0, 0)
        for enabled in (False, True):
            policy = RoutingPolicy(large_instances=("large",), reserve_large_headroom_blocks=4,
                                   migration_enabled=enabled)
            scheduler = CentralScheduler(policy)
            selected, _ = scheduler.select_initial([candidate], prompt_tokens=4, max_tokens=4)
            self.assertIsNone(selected)
            candidate.admissible_blocks = candidate.free_blocks = 6
            selected, _ = scheduler.select_initial([candidate], prompt_tokens=4, max_tokens=4)
            self.assertEqual(selected, "large")
            candidate.admissible_blocks = candidate.free_blocks = 5


if __name__ == "__main__":
    unittest.main()
