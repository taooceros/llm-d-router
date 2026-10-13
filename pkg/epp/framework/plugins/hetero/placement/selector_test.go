/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package placement

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
)

// The expected values below are the outputs of hetero.gateway.policy.CentralScheduler
// for the same inputs.

func candidate(name string, role capacity.Role, maxLen, total, free, block int) capacity.EndpointCapacity {
	return capacity.EndpointCapacity{
		Name: name, Role: role, MaxModelLen: maxLen, BlockSize: block,
		TotalBlocks: total, FreeBlocks: free, AdmissibleBlocks: free,
	}
}

func mixedPolicy(edit func(*capacity.Policy)) capacity.Policy {
	p := capacity.DefaultPolicy()
	p.SmallPromptTokenLimit = 16384
	p.LargeFreshTrafficShare = 0.5
	p.ReserveLargeHeadroomBlocks = 16
	p.DecodeLengthBlind = true
	if edit != nil {
		edit(&p)
	}
	return p
}

func TestSelectMixedLength(t *testing.T) {
	small, large := capacity.RoleSmall, capacity.RoleLarge
	fullLarge := []capacity.EndpointCapacity{
		candidate("small_a", small, 16384, 412, 300, 64),
		candidate("small_b", small, 16384, 412, 300, 64),
		candidate("large", large, 36864, 1134, 10, 64),
	}
	longReserve := func(p *capacity.Policy) { p.LongReserveTokens = 36864 }
	largeOnly := func(p *capacity.Policy) {
		p.LargeFreshTrafficShare = 0
		p.LongReserveTokens = 36864
	}
	tp16 := []capacity.EndpointCapacity{candidate("large", large, 36864, 282, 150, 256)}

	tests := []struct {
		name         string
		policy       capacity.Policy
		snap         []capacity.EndpointCapacity
		promptTokens int
		wantSelected string
		wantRefusal  string
		wantReason   string
		wantFallback bool
		wantContext  int
		wantScores   map[string]float64
	}{
		{
			name:         "full large owed the request waits without fallback",
			policy:       mixedPolicy(nil),
			snap:         fullLarge,
			promptTokens: 500,
			wantRefusal:  RefusalNoCapacity,
			wantReason:   ReasonLargeFreshTraffic,
			wantContext:  564,
		},
		{
			name:         "full large owed the request falls back to small",
			policy:       mixedPolicy(func(p *capacity.Policy) { p.WorkConservingFallback = true }),
			snap:         fullLarge,
			promptTokens: 500,
			wantSelected: "small_b",
			wantReason:   ReasonLargeFreshTraffic,
			wantFallback: true,
			wantContext:  564,
			wantScores:   map[string]float64{"small_a": 0.86408, "small_b": 0.86408},
		},
		{
			name: "full smalls spill to large",
			policy: mixedPolicy(func(p *capacity.Policy) {
				p.LargeFreshTrafficShare = 0
				p.WorkConservingFallback = true
			}),
			snap: []capacity.EndpointCapacity{
				candidate("small_a", small, 16384, 412, 2, 64),
				candidate("small_b", small, 16384, 412, 2, 64),
				candidate("large", large, 36864, 1134, 900, 64),
			},
			promptTokens: 500,
			wantSelected: "large",
			wantReason:   ReasonPreferSmall,
			wantFallback: true,
			wantContext:  564,
			wantScores:   map[string]float64{"large": 0.89683},
		},
		{
			// 150 free - 16 headroom - 144 reserved blocks leaves nothing.
			name:         "short request cannot take the long reserve",
			policy:       mixedPolicy(largeOnly),
			snap:         tp16,
			promptTokens: 500,
			wantRefusal:  RefusalNoCapacity,
			wantReason:   ReasonLongPrompt,
			wantContext:  756,
		},
		{
			// 36864 tokens is 144 blocks of 256: 282 - 16 - 144 = 122 usable.
			name:         "reserve is counted in tokens",
			policy:       mixedPolicy(largeOnly),
			snap:         []capacity.EndpointCapacity{candidate("large", large, 36864, 282, 282, 256)},
			promptTokens: 500,
			wantSelected: "large",
			wantReason:   ReasonLongPrompt,
			wantContext:  756,
			wantScores:   map[string]float64{"large": 1},
		},
		{
			name:         "long request may use the reserve",
			policy:       mixedPolicy(largeOnly),
			snap:         tp16,
			promptTokens: 30000,
			wantSelected: "large",
			wantReason:   ReasonLongPrompt,
			wantContext:  30256,
			wantScores:   map[string]float64{"large": 0.76596},
		},
		{
			name: "long prompt goes to large in mixed layout",
			policy: mixedPolicy(func(p *capacity.Policy) {
				p.LargeFreshTrafficShare = 0
				longReserve(p)
			}),
			snap: []capacity.EndpointCapacity{
				candidate("small_a", small, 16384, 412, 412, 64),
				candidate("small_b", small, 16384, 412, 412, 64),
				candidate("large", large, 36864, 282, 282, 256),
			},
			promptTokens: 30000,
			wantSelected: "large",
			wantReason:   ReasonLongPrompt,
			wantContext:  30256,
			wantScores:   map[string]float64{"large": 1},
		},
		{
			name:         "no endpoint supports the context",
			policy:       mixedPolicy(nil),
			snap:         []capacity.EndpointCapacity{candidate("small_a", small, 16384, 412, 412, 64)},
			promptTokens: 20000,
			wantRefusal:  RefusalNoContextSupport,
			wantContext:  20064,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision, ok := NewSelector(tt.policy).Select(tt.snap, Request{ID: "r", PromptTokens: tt.promptTokens, MaxTokens: 256}, Options{})
			trace := decision.Trace
			assert.Equal(t, tt.wantSelected != "", ok)
			assert.Equal(t, tt.wantSelected, decision.Endpoint)
			assert.Equal(t, tt.wantSelected, trace.Selected)
			assert.Equal(t, tt.wantRefusal, trace.Refusal)
			assert.Equal(t, tt.wantReason, trace.Reason)
			assert.Equal(t, tt.wantFallback, trace.Fallback)
			assert.Equal(t, tt.wantContext, trace.Context)
			assert.Equal(t, tt.wantContext, decision.ContextTokens)
			assert.Equal(t, tt.wantScores, trace.Scores)
		})
	}
}

func TestSelectLengthBlind(t *testing.T) {
	snap := func() []capacity.EndpointCapacity {
		return []capacity.EndpointCapacity{
			candidate("small_a", capacity.RoleSmall, 16384, 103, 20, 256),
			candidate("small_b", capacity.RoleSmall, 16384, 103, 20, 256),
			candidate("large", capacity.RoleLarge, 16384, 283, 200, 256),
		}
	}
	policy := func(blind bool) capacity.Policy {
		p := capacity.DefaultPolicy()
		p.LargeFreshTrafficShare = 0
		p.ReserveLargeHeadroomBlocks = 16
		p.DecodeLengthBlind = blind
		return p
	}

	t.Run("placement ignores the decode budget", func(t *testing.T) {
		for _, maxTokens := range []int{256, 4000, 16000} {
			decision, ok := NewSelector(policy(true)).Select(snap(), Request{PromptTokens: 300, MaxTokens: maxTokens}, Options{})
			require.True(t, ok)
			assert.Equal(t, "small_b", decision.Endpoint)
			// prompt plus one growth block of the largest block size
			assert.Equal(t, 300+256, decision.Trace.Context)
			assert.Equal(t, []string{"small_a", "small_b"}, decision.Trace.FitsSmall)
			assert.Nil(t, decision.Trace.MaxTokens)
			// blocks_for(prompt) + admission_growth_blocks
			assert.Equal(t, 3, decision.ChargeBlocks)
		}
	})

	t.Run("aware policy uses the budget", func(t *testing.T) {
		decision, ok := NewSelector(policy(false)).Select(snap(), Request{PromptTokens: 300, MaxTokens: 16000}, Options{})
		assert.False(t, ok)
		assert.Equal(t, 16300, decision.Trace.Context)
		require.NotNil(t, decision.Trace.MaxTokens)
		assert.Equal(t, 16000, *decision.Trace.MaxTokens)
		assert.Equal(t, RefusalNoCapacity, decision.Trace.Refusal)
	})
}

// TestSelectQuotaAdvancesOnDispatchOnly replays a sequence where a refusal
// must leave the fresh-traffic quota untouched.
func TestSelectQuotaAdvancesOnDispatchOnly(t *testing.T) {
	withLoad := func(ep capacity.EndpointCapacity, running int) capacity.EndpointCapacity {
		ep.Running = running
		return ep
	}
	room := []capacity.EndpointCapacity{
		withLoad(candidate("small_a", capacity.RoleSmall, 16384, 412, 300, 64), 2),
		withLoad(candidate("small_b", capacity.RoleSmall, 16384, 412, 250, 64), 1),
		withLoad(candidate("large", capacity.RoleLarge, 36864, 282, 200, 256), 3),
	}
	fullLarge := []capacity.EndpointCapacity{
		candidate("small_a", capacity.RoleSmall, 16384, 412, 300, 64),
		candidate("small_b", capacity.RoleSmall, 16384, 412, 250, 64),
		candidate("large", capacity.RoleLarge, 36864, 282, 10, 256),
	}
	p := capacity.DefaultPolicy()
	p.LargeFreshTrafficShare = 0.25
	p.ReserveLargeHeadroomBlocks = 16
	p.DecodeLengthBlind = true

	steps := []struct {
		snap         []capacity.EndpointCapacity
		wantSelected string
		wantReason   string
		wantCounts   [2]int
	}{
		{fullLarge, "", ReasonLargeFreshTraffic, [2]int{0, 0}},
		{room, "large", ReasonLargeFreshTraffic, [2]int{1, 1}},
		{room, "small_b", ReasonPreferSmall, [2]int{2, 1}},
		{fullLarge, "small_a", ReasonPreferSmall, [2]int{3, 1}},
		{room, "small_b", ReasonPreferSmall, [2]int{4, 1}},
		{room, "large", ReasonLargeFreshTraffic, [2]int{5, 2}},
		{room, "small_b", ReasonPreferSmall, [2]int{6, 2}},
		{fullLarge, "small_a", ReasonPreferSmall, [2]int{7, 2}},
		{room, "small_b", ReasonPreferSmall, [2]int{8, 2}},
	}
	selector := NewSelector(p)
	for i, step := range steps {
		decision, ok := selector.Select(step.snap, Request{PromptTokens: 1000}, Options{})
		assert.Equal(t, step.wantSelected != "", ok, "step %d", i)
		assert.Equal(t, step.wantSelected, decision.Endpoint, "step %d", i)
		assert.Equal(t, step.wantReason, decision.Trace.Reason, "step %d", i)
		dispatched, toLarge := selector.Counters()
		assert.Equal(t, step.wantCounts, [2]int{dispatched, toLarge}, "step %d", i)
	}
}

func TestSelectLateBinding(t *testing.T) {
	small, large := capacity.RoleSmall, capacity.RoleLarge
	p := mixedPolicy(nil)
	fullLarge := []capacity.EndpointCapacity{
		candidate("small_a", small, 16384, 412, 300, 64),
		candidate("large", large, 36864, 1134, 10, 64),
	}

	t.Run("quota never makes a request wait", func(t *testing.T) {
		decision, ok := NewSelector(p).Select(fullLarge, Request{PromptTokens: 500}, Options{LateBinding: true})
		require.True(t, ok)
		assert.Equal(t, "small_a", decision.Endpoint)
		assert.True(t, decision.Trace.Fallback)
	})

	t.Run("extra held blocks refuse a short request on large", func(t *testing.T) {
		q := mixedPolicy(func(p *capacity.Policy) {
			p.LargeFreshTrafficShare = 0
			p.DynamicLongReserve = true
			p.LongReserveTokens = 36864
		})
		snap := []capacity.EndpointCapacity{candidate("large", large, 36864, 282, 150, 256)}
		// The static reserve is replaced by the dynamic one: 150 - 16 >= 3.
		_, ok := NewSelector(q).Select(snap, Request{PromptTokens: 500}, Options{LateBinding: true})
		assert.True(t, ok)
		// Holding 132 blocks leaves 2 < 3.
		decision, ok := NewSelector(q).Select(snap, Request{PromptTokens: 500}, Options{LateBinding: true, ExtraHeld: map[string]int{"large": 132}})
		assert.False(t, ok)
		assert.Equal(t, RefusalNoCapacity, decision.Trace.Refusal)
		assert.False(t, decision.Long)
		assert.Equal(t, 756, decision.ContextTokens)
	})
}
