"""Mixed long/short placement: work-conserving fallback and the long-request reserve."""

import unittest

from hetero.gateway.policy import Candidate, CentralScheduler, RoutingPolicy

BLOCK = 64


def candidate(name, tp, max_len, total, free, block=BLOCK):
    return Candidate(name, tp, max_len, block, total, free, free, 0, 0, 0)


def policy(**overrides):
    base = dict(small_instances=("small_a", "small_b"), large_instances=("large",),
                small_prompt_token_limit=16384, large_fresh_traffic_share=0.5,
                reserve_large_headroom_blocks=16, decode_length_blind=True)
    base.update(overrides)
    return RoutingPolicy.from_dict(base)


def select(pol, pool, prompt_tokens):
    return CentralScheduler(pol).select_initial(pool, prompt_tokens=prompt_tokens, max_tokens=256)


class WorkConservingFallback(unittest.TestCase):
    def test_full_large_does_not_stall_shorts_owed_to_it(self):
        # First dispatch is owed to large by the 0.5 share, but large is full.
        pool = [candidate("small_a", 8, 16384, 412, 300), candidate("small_b", 8, 16384, 412, 300),
                candidate("large", 16, 36864, 1134, 10)]
        chosen, trace = select(policy(), pool, 500)
        self.assertIsNone(chosen)
        self.assertEqual(trace["refusal"], "no_destination_has_admissible_capacity")
        chosen, trace = select(policy(work_conserving_fallback=True), pool, 500)
        self.assertIn(chosen, ("small_a", "small_b"))
        self.assertTrue(trace["fallback"])

    def test_full_smalls_spill_to_large(self):
        pool = [candidate("small_a", 8, 16384, 412, 2), candidate("small_b", 8, 16384, 412, 2),
                candidate("large", 16, 36864, 1134, 900)]
        pol = policy(large_fresh_traffic_share=0.0, work_conserving_fallback=True)
        self.assertEqual(select(pol, pool, 500)[0], "large")


class LongReserve(unittest.TestCase):
    # The deployed TP16 engine at max_model_len 36864 uses 256-token blocks:
    # 282 blocks = 72192 tokens, with 150 blocks free here.
    pool = [candidate("large", 16, 36864, 282, 150, block=256)]
    homogeneous = dict(small_instances=(), large_instances=("large",), large_fresh_traffic_share=0.0)

    def test_short_request_cannot_take_the_reserve(self):
        # 150 free - 16 headroom - 144 reserved blocks leaves nothing for a short request.
        pol = policy(long_reserve_tokens=36864, **self.homogeneous)
        chosen, trace = select(pol, self.pool, 500)
        self.assertIsNone(chosen)
        self.assertEqual(trace["refusal"], "no_destination_has_admissible_capacity")

    def test_reserve_is_counted_in_tokens_not_blocks(self):
        # 36864 tokens is 144 blocks of 256, so 282 - 16 - 144 = 122 blocks stay usable.
        idle = [candidate("large", 16, 36864, 282, 282, block=256)]
        pol = policy(long_reserve_tokens=36864, **self.homogeneous)
        self.assertEqual(select(pol, idle, 500)[0], "large")

    def test_long_request_may_use_the_reserve(self):
        pol = policy(long_reserve_tokens=36864, **self.homogeneous)
        self.assertEqual(select(pol, self.pool, 30000)[0], "large")

    def test_long_prompt_goes_to_large_in_mixed_layout(self):
        pool = [candidate("small_a", 8, 16384, 412, 412), candidate("small_b", 8, 16384, 412, 412),
                candidate("large", 16, 36864, 282, 282, block=256)]
        chosen, trace = select(policy(large_fresh_traffic_share=0.0, long_reserve_tokens=36864),
                               pool, 30000)
        self.assertEqual(chosen, "large")
        self.assertEqual(trace["reason"], "prompt_does_not_fit_a_small_instance")


if __name__ == "__main__":
    unittest.main()
