"""Length-blind scheduling never lets the decode budget (max_tokens) steer decisions."""

import unittest

from hetero.gateway.policy import Candidate, CentralScheduler, RoutingPolicy

BLOCK = 256


def candidate(name, tp, total, free, running=0):
    # instance_id, tp_size, max_model_len, block_size, total_blocks,
    # free_blocks, admissible_blocks, running, waiting, inflight_migrations
    return Candidate(name, tp, 16384, BLOCK, total, free, free, running, 0, 0)


def policy(**overrides):
    base = dict(small_instances=("small_a", "small_b"), large_instances=("large",),
                large_fresh_traffic_share=0.0, reserve_large_headroom_blocks=16,
                migration_enabled=True, source_pressure_free_block_fraction=0.12,
                destination_growth_allowance_blocks=8, decode_length_blind=True)
    base.update(overrides)
    return RoutingPolicy.from_dict(base)


class LengthBlindPlacement(unittest.TestCase):
    def pool(self):
        return [candidate("small_a", 8, 103, 20), candidate("small_b", 8, 103, 20),
                candidate("large", 16, 283, 200)]

    def test_placement_ignores_decode_budget(self):
        decisions = []
        for max_tokens in (256, 4000, 16000):
            chosen, trace = CentralScheduler(policy()).select_initial(
                self.pool(), prompt_tokens=300, max_tokens=max_tokens)
            decisions.append((chosen, trace["context"], tuple(trace["fits_small"])))
            self.assertIsNone(trace["max_tokens"])
        self.assertEqual(len(set(decisions)), 1)
        # prompt (2 blocks) plus one growth block, not prompt + max_tokens
        self.assertEqual(decisions[0][1], 300 + BLOCK)

    def test_aware_policy_still_uses_budget(self):
        _, trace = CentralScheduler(policy(decode_length_blind=False)).select_initial(
            self.pool(), prompt_tokens=300, max_tokens=16000)
        self.assertEqual(trace["context"], 16300)
        self.assertEqual(trace["max_tokens"], 16000)


class LengthBlindMigration(unittest.TestCase):
    def propose(self, remaining, pol=None):
        pool = [candidate("small_a", 8, 103, 5), candidate("large", 16, 283, 40)]
        return CentralScheduler(pol or policy()).select_migration_destination(
            pool, source_instance="small_a", prompt_tokens=200, generated_tokens=3000,
            remaining_tokens=remaining, source_free_fraction=5 / 103, migrations_done=0,
            total_inflight_migrations=0, unsupported=[])

    def test_reservation_is_context_plus_growth_whatever_the_budget(self):
        results = {r: self.propose(r) for r in (1, 500, 12000)}
        for remaining, (destination, trace, reserve) in results.items():
            self.assertEqual(destination, "large", trace)
            self.assertEqual(reserve, {"blocks": 13, "growth_blocks": 8})
            self.assertIsNone(trace["remaining_tokens"])

    def test_aware_policy_reserves_full_remaining_budget(self):
        destination, trace, reserve = self.propose(12000, policy(decode_length_blind=False))
        self.assertIsNone(destination)
        self.assertEqual(trace["refusal"], "no_destination_capacity_or_migration_slot")


if __name__ == "__main__":
    unittest.main()
