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

package latebinding

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	contractmocks "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/placement"
)

// lateBindingController runs a real FlowController whose saturation detector is a late-binding gate, with unbounded
// queue wait as deployed for bulk backlogs.
type lateBindingController struct {
	ctx       context.Context
	fc        *controller.FlowController
	reg       *registry.FlowRegistry
	placement *placement.Plugin

	mu       sync.Mutex
	outcomes map[string]types.QueueOutcome
}

func newLateBindingController(t *testing.T, policy capacity.Policy, ordering flowcontrol.OrderingPolicy,
	endpoints []datalayer.Endpoint) *lateBindingController {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	fairness, err := globalstrict.GlobalStrictFairnessPolicyFactory("global-strict", nil, nil)
	require.NoError(t, err)
	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: ordering,
		FairnessPolicy: fairness.(flowcontrol.FairnessPolicy),
	}
	band, err := registry.NewPriorityBandConfig(0, defaults,
		registry.WithBandMaxRequests(10_000), registry.WithBandDefaultRequestTTL(0))
	require.NoError(t, err)
	regCfg, err := registry.NewConfig(defaults, registry.WithPriorityBand(band))
	require.NoError(t, err)
	reg := registry.NewFlowRegistry(regCfg, logr.Discard())
	go reg.RunMaintenanceLoop(ctx)

	pl := placement.New("placement", policy)
	fc := controller.NewFlowController(ctx, "pool", &controller.Config{
		ExpiryCleanupInterval:    10 * time.Millisecond,
		EnqueueChannelBufferSize: 100,
	}, controller.Deps{
		Registry:           reg,
		SaturationDetector: NewGate("gate", pl),
		EndpointCandidates: &contractmocks.MockEndpointCandidates{Candidates: endpoints},
		UsageLimitPolicy:   usagelimits.DefaultPolicy(),
	})
	return &lateBindingController{ctx: ctx, fc: fc, reg: reg, placement: pl, outcomes: map[string]types.QueueOutcome{}}
}

// submit enqueues a request and records its outcome when EnqueueAndWait returns.
func (c *lateBindingController) submit(id string, promptTokens int) {
	go func() {
		outcome, _ := c.fc.EnqueueAndWait(c.ctx, flowRequest(id, promptTokens))
		c.mu.Lock()
		c.outcomes[id] = outcome
		c.mu.Unlock()
	}()
}

func (c *lateBindingController) dispatched() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var ids []string
	for id, outcome := range c.outcomes {
		if outcome == types.QueueOutcomeDispatched {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}

func (c *lateBindingController) finished() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.outcomes)
}

func (c *lateBindingController) queued() int { return int(c.reg.Stats().Global.Len) }

func TestController_ConcurrentArrivalsFillWithoutOverbooking(t *testing.T) {
	t.Parallel()
	policy := testPolicy()
	endpoints := mixedLayout(smallTotal, largeTotal)
	c := newLateBindingController(t, policy, fcfsPolicy(t), endpoints)

	const total = 200
	for i := range total {
		c.submit(fmt.Sprintf("r-%03d", i), 2000)
	}
	// A 2000-token request needs 36 and is charged 33 blocks on TP8 (12 fit in 412), and needs and is charged 9 on
	// TP16 (29 fit in 282 - 16).
	const fit = 12 + 12 + 29
	require.Eventually(t, func() bool { return len(c.dispatched()) == fit && c.queued() == total-fit },
		5*time.Second, time.Millisecond)

	tracker := c.placement.Tracker()
	for _, ep := range endpoints {
		name := capacity.EndpointName(ep.GetMetadata())
		blocks, _ := tracker.Projected(name)
		limit := smallTotal
		if name == "large" {
			limit = largeTotal - policy.ReserveLargeHeadroomBlocks
		}
		assert.LessOrEqual(t, blocks, limit, "endpoint %s is overbooked", name)
	}
	for _, id := range c.dispatched() {
		endpoint, _, ok := tracker.Charged(id)
		assert.True(t, ok, "dispatched request %s holds a charge", id)
		assert.NotEmpty(t, endpoint)
	}

	// Completing five requests on large frees room for exactly five more.
	released := 0
	for _, id := range c.dispatched() {
		if endpoint, _, _ := tracker.Charged(id); endpoint == "large" && released < 5 {
			tracker.Release(id)
			released++
		}
	}
	require.Eventually(t, func() bool { return len(c.dispatched()) == fit+5 }, 5*time.Second, time.Millisecond)

	// With no queue-wait TTL the rest stay queued.
	time.Sleep(100 * time.Millisecond)
	assert.Equal(t, fit+5, c.finished(), "no queued request is shed or rejected")
	assert.Equal(t, total-fit-5, c.queued())
}

func TestController_DispatchOrderUnderContention(t *testing.T) {
	t.Parallel()
	dynamic := testPolicy()
	dynamic.DynamicLongReserve = true
	cases := []struct {
		name     string
		policy   capacity.Policy
		ordering func(*testing.T) flowcontrol.OrderingPolicy
		want     []string
	}{
		{
			// short-1 takes 3 blocks, leaving 132 of the 135 the long request needs; short-2 backfills.
			name: "arrival order backfills shorts past the long request", policy: testPolicy(), ordering: fcfsPolicy,
			want: []string{"short-1", "short-2"},
		},
		{
			// Once the long request fits nowhere, short-2 may not take its 119 blocks on large.
			name: "arrival order with dynamic reserve stops shorts behind the long request", policy: dynamic,
			ordering: fcfsPolicy, want: []string{"short-1"},
		},
		{
			name: "longest prompt first binds the long request before the shorts", policy: testPolicy(),
			ordering: longestPromptFirstPolicy, want: []string{"long"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			c := newLateBindingController(t, tc.policy, tc.ordering(t), []datalayer.Endpoint{large("large", largeTotal)})
			tracker := c.placement.Tracker()
			tracker.Charge("occupant", "large", largeTotal)

			for i, r := range [][2]any{{"short-1", 500}, {"long", 30000}, {"short-2", 500}} {
				c.submit(r[0].(string), r[1].(int))
				require.Eventually(t, func() bool { return c.queued() == i+1 }, time.Second, time.Millisecond)
			}
			// The occupant finishing part of its work leaves 135 blocks free, 119 above the headroom.
			tracker.Charge("occupant", "large", 147)

			require.Eventually(t, func() bool { return len(c.dispatched()) == len(tc.want) }, 5*time.Second,
				time.Millisecond)
			time.Sleep(50 * time.Millisecond)
			assert.Equal(t, tc.want, c.dispatched())
		})
	}
}

func fcfsPolicy(t *testing.T) flowcontrol.OrderingPolicy {
	p, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, nil)
	require.NoError(t, err)
	return p.(flowcontrol.OrderingPolicy)
}

func longestPromptFirstPolicy(t *testing.T) flowcontrol.OrderingPolicy {
	p, err := LongestPromptFirstFactory("", nil, nil)
	require.NoError(t, err)
	return p.(flowcontrol.OrderingPolicy)
}
