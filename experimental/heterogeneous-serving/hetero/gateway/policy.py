"""Central placement and migration policy for the llm-d gateway serving layer.

The installed llm-d endpoint picker (`llm-d-router-endpoint-picker:v0.9.0`)
ranks pod endpoints with queue/KV/prefix scorers.  It cannot reserve
destination memory, cannot throttle a saturated destination against in-flight
migrations and has no notion of a continuation.  This module is the extended
scheduling stage: one eligibility, selection and admission pipeline used by both
initial placement and migration, with phase-specific constraints.

Scoring reuses the installed profile's signals and weights (queue depth and KV
utilisation) so the placement behaviour stays comparable to the stock router.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Weights mirror the deployed `optimized-baseline-plugins.yaml` scheduling
# profile (queue-scorer 2, kv-cache-utilization-scorer 2).  Prefix scoring is
# omitted because every engine runs with prefix caching disabled.
QUEUE_SCORER_WEIGHT = 2.0
KV_SCORER_WEIGHT = 2.0


@dataclasses.dataclass(frozen=True)
class RoutingPolicy:
    """Frozen static routing and migration rules for one arm."""

    # Initial placement
    small_instances: Tuple[str, ...] = ()
    large_instances: Tuple[str, ...] = ()
    # Prompts longer than this are long requests; see long_reserve_blocks.
    small_prompt_token_limit: int = 4096
    large_fresh_traffic_share: float = 0.25
    reserve_large_headroom_blocks: int = 64
    # KV tokens on every large instance that short requests may not take, so a
    # long request is not starved by a backlog of short ones.  0 disables.
    # Tokens, not blocks: block size differs between engines.
    long_reserve_tokens: int = 0
    # When the preferred pool has no admissible room, try the other pool
    # instead of waiting for the preferred one.
    work_conserving_fallback: bool = False
    # Migration
    migration_enabled: bool = False
    migration_mechanism: str = "recompute"
    min_generated_tokens: int = 128
    min_context_tokens: int = 1024
    min_remaining_tokens: int = 128
    max_migrations_per_request: int = 1
    max_inflight_migrations_per_destination: int = 2
    max_inflight_migrations_total: int = 3
    destination_growth_allowance_blocks: int = 8
    source_pressure_free_block_fraction: float = 0.15
    # Decode-length-blind scheduling: prompt length is known and used, but never the decode budget
    # (max_tokens) for placement, admission accounting or migration. The
    # engine still enforces max_tokens as a stop condition only.
    decode_length_blind: bool = False
    admission_growth_blocks: int = 1
    # Late binding (work steering): requests wait in one gateway queue and are
    # bound to an engine only when one has room.  "arrival" keeps today's
    # order; "longest_prompt_first" serves the longest known prompt first.
    dispatch_order: str = "arrival"
    # Hold KV back on a large instance only while a long request that does not
    # yet fit is waiting, sized to that request, instead of the static
    # long_reserve_tokens.  Requires late binding.
    dynamic_long_reserve: bool = False
    # --- late-binding knobs (owner: late-binding work) ---
    # --- work-stealing knobs (owner: work-stealing work) ---
    # Move requests an engine is holding but not running to an engine with room.
    steal_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RoutingPolicy":
        data = dict(data or {})
        for key in ("small_instances", "large_instances"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown routing policy fields: {sorted(unknown)}")
        return cls(**data)


@dataclasses.dataclass
class Candidate:
    """One eligible destination with its current observed load."""

    instance_id: str
    tp_size: int
    max_model_len: int
    block_size: int
    total_blocks: int
    free_blocks: int
    admissible_blocks: int
    running: int
    waiting: int
    inflight_migrations: int

    def queue_score(self, worst_queue: int) -> float:
        if worst_queue <= 0:
            return 1.0
        return max(0.0, 1.0 - (self.waiting + self.running) / worst_queue)

    def kv_score(self) -> float:
        if self.total_blocks <= 0:
            return 0.0
        return max(0.0, min(1.0, self.free_blocks / self.total_blocks))

    def score(self, worst_queue: int) -> float:
        return (
            QUEUE_SCORER_WEIGHT * self.queue_score(worst_queue)
            + KV_SCORER_WEIGHT * self.kv_score()
        ) / (QUEUE_SCORER_WEIGHT + KV_SCORER_WEIGHT)


def blocks_for(tokens: int, block_size: int) -> int:
    if block_size <= 0:
        return 0
    return -(-tokens // block_size)


class CentralScheduler:
    """Single authority for initial placement and migration destinations."""

    def __init__(self, policy: RoutingPolicy) -> None:
        self.policy = policy
        self._round_robin = 0
        self._fresh_dispatched = 0
        self._fresh_to_large = 0

    def reset_fresh_traffic(self) -> None:
        """Zero the fresh-traffic share counters (a membership change starts a new quota)."""
        self._fresh_dispatched = 0
        self._fresh_to_large = 0

    # -------------------------------------------------- initial placement

    def select_initial(
        self,
        candidates: Sequence[Candidate],
        *,
        prompt_tokens: int,
        max_tokens: int,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Chooses the instance for a fresh request, or refuses with a reason."""
        policy = self.policy
        if policy.decode_length_blind:
            # Only the prompt is known; charge one growth allowance on top.
            context = prompt_tokens + policy.admission_growth_blocks * _block_size(candidates)
        else:
            context = prompt_tokens + max_tokens
        by_id = {c.instance_id: c for c in candidates}
        eligible = [c for c in candidates if c.max_model_len >= context]
        trace: Dict[str, Any] = {
            "phase": "initial",
            "prompt_tokens": prompt_tokens,
            "max_tokens": None if policy.decode_length_blind else max_tokens,
            "decode_length_blind": policy.decode_length_blind,
            "context": context,
            "considered": [c.instance_id for c in candidates],
            "eligible": [c.instance_id for c in eligible],
        }
        if not eligible:
            trace["refusal"] = "no_instance_supports_context"
            return None, trace

        small = [c for c in eligible if c.instance_id in policy.small_instances]
        large = [c for c in eligible if c.instance_id in policy.large_instances]
        fits_small = [
            c
            for c in small
            if blocks_for(context, c.block_size) <= max(1, c.total_blocks - 1)
        ]
        trace["fits_small"] = [c.instance_id for c in fits_small]

        # Work-conserving rule for Large: a fixed share of fresh traffic goes to
        # the large instance so half the cluster is not idle awaiting outliers.
        want_large = False
        if large and policy.large_fresh_traffic_share > 0.0:
            target = policy.large_fresh_traffic_share * (self._fresh_dispatched + 1)
            want_large = self._fresh_to_large < target
        trace["work_conserving_large"] = want_large

        if not fits_small:
            pools = [large or eligible]
            trace["reason"] = "prompt_does_not_fit_a_small_instance"
        elif want_large:
            pools = [large or fits_small, fits_small]
            trace["reason"] = "large_fresh_traffic_share"
        else:
            pools = [fits_small, large]
            trace["reason"] = "prefer_small"
        if not policy.work_conserving_fallback:
            pools = pools[:1]

        long_request = prompt_tokens > policy.small_prompt_token_limit
        pool: List[Candidate] = []
        for index, option in enumerate(pools):
            pool = [c for c in option if _has_room(c, context, policy, long_request)]
            if pool:
                if index:
                    trace["fallback"] = True
                break
        if not pool:
            trace["refusal"] = "no_destination_has_admissible_capacity"
            return None, trace

        worst_queue = max(1, max(c.running + c.waiting for c in candidates))
        scored = sorted(
            ((c.score(worst_queue), c.instance_id) for c in pool), reverse=True
        )
        chosen = scored[0][1]
        self._fresh_dispatched += 1
        if chosen in policy.large_instances:
            self._fresh_to_large += 1
        trace["scores"] = {name: round(score, 5) for score, name in scored}
        trace["selected"] = chosen
        trace["selected_free_blocks"] = by_id[chosen].free_blocks
        return chosen, trace

    # ------------------------------------------------------- migration

    def select_migration_destination(
        self,
        candidates: Sequence[Candidate],
        *,
        source_instance: str,
        prompt_tokens: int,
        generated_tokens: int,
        remaining_tokens: int,
        source_free_fraction: float,
        migrations_done: int,
        total_inflight_migrations: int,
        unsupported: Sequence[str],
    ) -> Tuple[Optional[str], Dict[str, Any], Dict[str, int]]:
        """Applies the frozen bounded small-to-large migration rule."""
        policy = self.policy
        trace: Dict[str, Any] = {
            "phase": "migration",
            "source": source_instance,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "remaining_tokens": None if policy.decode_length_blind else remaining_tokens,
            "source_free_fraction": round(source_free_fraction, 5),
        }
        if not policy.migration_enabled:
            trace["refusal"] = "migration_disabled"
            return None, trace, {}
        if unsupported:
            trace["refusal"] = "unsupported_continuation"
            trace["unsupported"] = list(unsupported)
            return None, trace, {}
        if migrations_done >= policy.max_migrations_per_request:
            trace["refusal"] = "per_request_migration_limit"
            return None, trace, {}
        if total_inflight_migrations >= policy.max_inflight_migrations_total:
            trace["refusal"] = "global_migration_limit"
            return None, trace, {}
        if generated_tokens < policy.min_generated_tokens:
            trace["refusal"] = "below_generated_threshold"
            return None, trace, {}
        context = prompt_tokens + generated_tokens
        if context < policy.min_context_tokens:
            trace["refusal"] = "below_context_threshold"
            return None, trace, {}
        if not policy.decode_length_blind and remaining_tokens < policy.min_remaining_tokens:
            trace["refusal"] = "insufficient_remaining_budget"
            return None, trace, {}
        if source_free_fraction > policy.source_pressure_free_block_fraction:
            trace["refusal"] = "source_not_under_pressure"
            return None, trace, {}

        destinations = [
            c
            for c in candidates
            if c.instance_id in policy.large_instances
            and c.instance_id != source_instance
            and c.max_model_len >= context + (1 if policy.decode_length_blind else remaining_tokens)
        ]
        trace["destinations_considered"] = [c.instance_id for c in destinations]
        if not destinations:
            trace["refusal"] = "no_eligible_destination"
            return None, trace, {}

        blocks = 0
        growth = 0
        viable = []
        for candidate in destinations:
            need = blocks_for(context, candidate.block_size)
            if policy.decode_length_blind:
                grow = policy.destination_growth_allowance_blocks
            else:
                grow = max(
                    policy.destination_growth_allowance_blocks,
                    blocks_for(remaining_tokens, candidate.block_size),
                )
            if candidate.admissible_blocks < need + grow:
                continue
            if candidate.inflight_migrations >= policy.max_inflight_migrations_per_destination:
                continue
            viable.append((candidate, need, grow))
        if not viable:
            trace["refusal"] = "no_destination_capacity_or_migration_slot"
            return None, trace, {}

        worst_queue = max(1, max(c.running + c.waiting for c in candidates))
        viable.sort(key=lambda row: row[0].score(worst_queue), reverse=True)
        candidate, blocks, growth = viable[0]
        trace["selected"] = candidate.instance_id
        trace["reserve_blocks"] = blocks
        trace["reserve_growth_blocks"] = growth
        trace["destination_admissible_blocks"] = candidate.admissible_blocks
        return candidate.instance_id, trace, {"blocks": blocks, "growth_blocks": growth}


def _block_size(candidates: Sequence["Candidate"]) -> int:
    return max((c.block_size for c in candidates), default=256)


def _has_room(candidate: Candidate, context: int, policy: RoutingPolicy,
              long_request: bool) -> bool:
    need = blocks_for(context, candidate.block_size)
    headroom = 0
    if candidate.instance_id in policy.large_instances:
        headroom = policy.reserve_large_headroom_blocks
        if not long_request:
            headroom += blocks_for(policy.long_reserve_tokens, candidate.block_size)
    return candidate.admissible_blocks - headroom >= need
