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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/placement"
)

type admission struct {
	verdict  flowcontrol.DispatchVerdict
	endpoint string
	trace    placement.Trace
	token    string
}

// runPass opens one pass over endpoints and admits requests (id -> prompt tokens) in the given order.
func runPass(t *testing.T, gate *Gate, endpoints []datalayer.Endpoint, requests [][2]any) []admission {
	t.Helper()
	pass := gate.BeginDispatchPass(context.Background(), endpoints, true)
	require.NotNil(t, pass, "a pass should open while some endpoint has room")
	now := time.Now()
	out := make([]admission, 0, len(requests))
	for i, r := range requests {
		item := queueItem(r[0].(string), r[1].(int), now.Add(time.Duration(i)*time.Millisecond))
		a := admission{verdict: pass.Admit(context.Background(), item)}
		a.token = placement.ReservationToken(item.OriginalRequest().InferenceRequest())
		if d, ok := fwksched.ReadRequestAttribute[placement.Decision](item.OriginalRequest().InferenceRequest(),
			placement.BoundDecisionKey); ok {
			a.endpoint, a.trace = d.Endpoint, d.Trace
		}
		out = append(out, a)
	}
	return out
}

func mixedLayout(smallFree, largeFree int) []datalayer.Endpoint {
	return []datalayer.Endpoint{small("small_a", smallFree), small("small_b", smallFree), large("large", largeFree)}
}

func TestGate_NoOwedWait(t *testing.T) {
	t.Parallel()
	// The 0.5 share owes the first dispatch to large, which is full; late binding places it on a small endpoint.
	pl := placement.New("placement", testPolicy())
	got := runPass(t, NewGate("gate", pl), mixedLayout(300, 10), [][2]any{{"short", 500}})

	require.Equal(t, flowcontrol.DispatchAdmit, got[0].verdict)
	assert.Contains(t, []string{"small_a", "small_b"}, got[0].endpoint)
	assert.Equal(t, placement.ReasonLargeFreshTraffic, got[0].trace.Reason)
	assert.True(t, got[0].trace.Fallback)
	endpoint, blocks, ok := pl.Tracker().Charged(got[0].token)
	require.True(t, ok, "the gate charges the endpoint it binds")
	assert.Equal(t, got[0].endpoint, endpoint)
	assert.Equal(t, capacity.BlocksFor(500, smallBlock)+1, blocks)
}

func TestGate_BackfillPastUnplacedLongRequest(t *testing.T) {
	t.Parallel()
	// 30000 tokens fit no TP8 engine and need 119+16 blocks on large, which has 100 free.
	pl := placement.New("placement", testPolicy())
	got := runPass(t, NewGate("gate", pl), mixedLayout(smallTotal, 100),
		[][2]any{{"long", 30000}, {"short-1", 500}, {"short-2", 500}})

	assert.Equal(t, flowcontrol.DispatchSkip, got[0].verdict)
	assert.Equal(t, flowcontrol.DispatchAdmit, got[1].verdict)
	assert.Equal(t, flowcontrol.DispatchAdmit, got[2].verdict)
	_, _, charged := pl.Tracker().Charged(got[0].token)
	assert.False(t, charged, "a skipped request holds no charge")
}

func TestGate_MissingTelemetryKeepsRequestQueued(t *testing.T) {
	t.Parallel()
	// Large does not report free blocks, so the long request it alone could serve waits instead of being refused.
	pl := placement.New("placement", testPolicy())
	got := runPass(t, NewGate("gate", pl), mixedLayout(smallTotal, noMetrics),
		[][2]any{{"long", 30000}, {"short", 500}})

	assert.Equal(t, flowcontrol.DispatchSkip, got[0].verdict)
	assert.Equal(t, flowcontrol.DispatchAdmit, got[1].verdict)
	assert.Contains(t, []string{"small_a", "small_b"}, got[1].endpoint)
}

func TestGate_DynamicLongReserve(t *testing.T) {
	t.Parallel()
	dynamic := testPolicy()
	dynamic.DynamicLongReserve = true

	cases := []struct {
		name      string
		policy    capacity.Policy
		endpoints []datalayer.Endpoint
		requests  [][2]any
		// want is the endpoint each request binds to, "" for skipped.
		want []string
		// wantHeld is the reserve recorded on the last bound request.
		wantHeld map[string]int
	}{
		{
			name:      "no reserve without a waiting long request",
			policy:    dynamic,
			endpoints: mixedLayout(smallTotal, 120),
			requests:  [][2]any{{"short", 500}},
			want:      []string{"large"},
		},
		{
			name:      "an unplaced long request holds its need on large and shorts go to TP8",
			policy:    dynamic,
			endpoints: mixedLayout(smallTotal, 120),
			// 30000 + 256 tokens is 119 blocks of 256; 119 + 16 headroom exceeds the 120 free.
			requests: [][2]any{{"long", 30000}, {"short", 500}},
			want:     []string{"", "small_b"},
			wantHeld: map[string]int{"large": 119},
		},
		{
			name:      "a short request waits when only large has room below the reserve",
			policy:    dynamic,
			endpoints: mixedLayout(0, 120),
			requests:  [][2]any{{"long", 30000}, {"short", 500}},
			want:      []string{"", ""},
		},
		{
			name:      "without dynamicLongReserve the short request takes large",
			policy:    testPolicy(),
			endpoints: mixedLayout(0, 120),
			requests:  [][2]any{{"long", 30000}, {"short", 500}},
			want:      []string{"", "large"},
		},
		{
			name:      "a long request that fits binds and holds nothing",
			policy:    dynamic,
			endpoints: mixedLayout(0, largeTotal),
			requests:  [][2]any{{"long", 30000}, {"short", 500}},
			want:      []string{"large", "large"},
		},
		{
			name:      "the reserve is sized to the first unplaced long request and other long requests may use it",
			policy:    dynamic,
			endpoints: mixedLayout(0, 150),
			// 36000 + 256 tokens is 142 blocks, over the 134 admissible; 20000 + 256 is 80 blocks.
			requests: [][2]any{{"long-36k", 36000}, {"long-20k", 20000}, {"short", 500}},
			want:     []string{"", "large", ""},
		},
		{
			name:      "a long request no endpoint's max-model-len admits is released unbound and reserves nothing",
			policy:    dynamic,
			endpoints: mixedLayout(0, 120),
			// 40000 tokens exceed max-model-len 36864 on every endpoint.
			requests: [][2]any{{"too-long", 40000}, {"short", 500}},
			want:     []string{unbound, "large"},
		},
		{
			name:      "a released unservable request leaves the reserve to the next unplaced long request",
			policy:    dynamic,
			endpoints: mixedLayout(smallTotal, 120),
			requests:  [][2]any{{"too-long", 40000}, {"long", 30000}, {"short", 500}},
			want:      []string{unbound, "", "small_b"},
			wantHeld:  map[string]int{"large": 119},
		},
		{
			name:   "a long request no large endpoint can ever hold leaves the reserve to the next one",
			policy: dynamic,
			endpoints: []datalayer.Endpoint{small("small_a", smallTotal), small("small_b", smallTotal),
				endpoint("large", capacity.RoleLarge, largeMaxLen, largeBlock, 150, 120)},
			// 36000 + 256 tokens is 142 blocks, which with the headroom exceeds the 150 total.
			requests: [][2]any{{"never-fits", 36000}, {"long", 30000}, {"short", 500}},
			want:     []string{"", "", "small_b"},
			wantHeld: map[string]int{"large": 119},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			pl := placement.New("placement", tc.policy)
			got := runPass(t, NewGate("gate", pl), tc.endpoints, tc.requests)
			endpoints := make([]string, len(got))
			var lastBound *admission
			for i := range got {
				endpoints[i] = got[i].endpoint
				switch {
				case got[i].verdict == flowcontrol.DispatchAdmit && got[i].endpoint == "":
					endpoints[i] = unbound
				case got[i].verdict == flowcontrol.DispatchAdmit:
					lastBound = &got[i]
				default:
					assert.Equal(t, flowcontrol.DispatchSkip, got[i].verdict)
				}
			}
			assert.Equal(t, tc.want, endpoints)
			if lastBound != nil && tc.policy.DynamicLongReserve {
				assert.Equal(t, tc.wantHeld, lastBound.trace.ExtraHeld)
			}
		})
	}
}

// unbound marks a request the gate admits without binding it, leaving the refusal to placement.
const unbound = "(unbound)"

func TestGate_PassBindsOnlyWhatFits(t *testing.T) {
	t.Parallel()
	pl := placement.New("placement", testPolicy())
	gate := NewGate("gate", pl)
	endpoints := []datalayer.Endpoint{large("large", largeTotal)}

	// Each 33700-token request needs and is charged 133 blocks; two fill the 266 blocks above the headroom.
	got := runPass(t, gate, endpoints, [][2]any{{"long-1", 33700}, {"long-2", 33700}, {"short", 500}})
	assert.Equal(t, []flowcontrol.DispatchVerdict{flowcontrol.DispatchAdmit, flowcontrol.DispatchAdmit,
		flowcontrol.DispatchStop}, []flowcontrol.DispatchVerdict{got[0].verdict, got[1].verdict, got[2].verdict})
	blocks, requests := pl.Tracker().Projected("large")
	assert.Equal(t, 266, blocks)
	assert.Equal(t, 2, requests)

	assert.Nil(t, gate.BeginDispatchPass(context.Background(), endpoints, true),
		"no pass opens while no endpoint has room for the smallest request")
}

func TestGate_ShortRequestsFillWithoutOverbooking(t *testing.T) {
	t.Parallel()
	pl := placement.New("placement", testPolicy())
	endpoints := []datalayer.Endpoint{large("large", largeTotal)}
	requests := make([][2]any, 100)
	for i := range requests {
		requests[i] = [2]any{fmt.Sprintf("short-%d", i), 500}
	}

	got := runPass(t, NewGate("gate", pl), endpoints, requests)
	admitted := 0
	for _, a := range got {
		if a.verdict == flowcontrol.DispatchAdmit {
			admitted++
		}
	}
	// A 500-token request needs 3 blocks and is charged 3; 88 fit in 282 - 16.
	assert.Equal(t, 88, admitted)
	blocks, _ := pl.Tracker().Projected("large")
	assert.Equal(t, 264, blocks)
}

func TestGate_BeginDispatchPass(t *testing.T) {
	t.Parallel()
	pl := placement.New("placement", testPolicy())
	gate := NewGate("gate", pl)
	ep := large("large", largeTotal)
	endpoints := []datalayer.Endpoint{ep}
	ctx := context.Background()

	require.NotNil(t, gate.BeginDispatchPass(ctx, endpoints, false), "the first pass always opens")
	assert.Nil(t, gate.BeginDispatchPass(ctx, endpoints, false), "nothing changed")
	assert.NotNil(t, gate.BeginDispatchPass(ctx, endpoints, true), "the queues changed")

	setFree(ep, 200)
	assert.NotNil(t, gate.BeginDispatchPass(ctx, endpoints, false), "a metrics refresh changed free blocks")
	assert.Nil(t, gate.BeginDispatchPass(ctx, endpoints, false))

	pl.Tracker().Charge("running", "large", 100)
	assert.NotNil(t, gate.BeginDispatchPass(ctx, endpoints, false), "a charge changed projected capacity")
	pl.Tracker().Release("running")
	assert.NotNil(t, gate.BeginDispatchPass(ctx, endpoints, false), "a completion released projected capacity")

	setFree(ep, 0)
	assert.Nil(t, gate.BeginDispatchPass(ctx, endpoints, true), "no endpoint has room")
}

func TestGate_UnknownPromptIsAdmittedUnbound(t *testing.T) {
	t.Parallel()
	pl := placement.New("placement", testPolicy())
	pass := NewGate("gate", pl).BeginDispatchPass(context.Background(), mixedLayout(smallTotal, largeTotal), true)
	require.NotNil(t, pass)
	req := &mocks.MockFlowControlRequest{IDV: "text", InferenceRequestV: &fwksched.InferenceRequest{
		RequestID: "text",
		Body:      &fwkrh.InferenceRequestBody{Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}}},
	}}

	assert.Equal(t, flowcontrol.DispatchAdmit, pass.Admit(context.Background(), &mocks.MockQueueItemAccessor{OriginalRequestV: req}))
	_, bound := fwksched.ReadRequestAttribute[placement.Decision](req.InferenceRequestV, placement.BoundDecisionKey)
	assert.False(t, bound)
	assert.Zero(t, projectedTotal(pl), "an unbound request holds no reservation")
}

func TestGate_RevertReleasesLeaseAndQuota(t *testing.T) {
	t.Parallel()
	pl := placement.New("placement", testPolicy())
	pass := NewGate("gate", pl).BeginDispatchPass(context.Background(), mixedLayout(smallTotal, largeTotal), true)
	require.NotNil(t, pass)
	item := queueItem("cancelled", 500, time.Now())

	require.Equal(t, flowcontrol.DispatchAdmit, pass.Admit(context.Background(), item))
	dispatched, _ := pl.Selector().Counters()
	require.Equal(t, 1, dispatched)
	pass.Revert(item)
	assert.Zero(t, projectedTotal(pl))
	dispatched, _ = pl.Selector().Counters()
	assert.Zero(t, dispatched, "reverting a bind rolls back the fresh-traffic quota")
}

func TestGate_UnscheduledBindExpires(t *testing.T) {
	t.Parallel()
	pl := placement.New("placement", testPolicy())
	now := time.Now()
	pl.Tracker().SetClock(func() time.Time { return now })
	gate := NewGate("gate", pl)
	endpoints := []datalayer.Endpoint{large("large", largeTotal)}

	// Both requests are bound but never reach PreRequest.
	got := runPass(t, gate, endpoints, [][2]any{{"lost", 33700}, {"lost-2", 33700}})
	require.Equal(t, flowcontrol.DispatchAdmit, got[1].verdict)
	assert.Nil(t, gate.BeginDispatchPass(context.Background(), endpoints, true))

	now = now.Add(placement.DefaultBindLease)
	got = runPass(t, gate, endpoints, [][2]any{{"next", 33700}})
	assert.Equal(t, flowcontrol.DispatchAdmit, got[0].verdict, "expired leases free their blocks for queued requests")
	blocks, requests := pl.Tracker().Projected("large")
	assert.Equal(t, 133, blocks)
	assert.Equal(t, 1, requests)
}

func projectedTotal(pl *placement.Plugin) int {
	total := 0
	for _, name := range []string{"small_a", "small_b", "large"} {
		blocks, _ := pl.Tracker().Projected(name)
		total += blocks
	}
	return total
}

func TestGate_Saturation(t *testing.T) {
	t.Parallel()
	pl := placement.New("placement", testPolicy())
	gate := NewGate("gate", pl)

	assert.InDelta(t, 1.0, gate.Saturation(context.Background(), nil), 1e-9, "no capacity metrics reads as saturated")
	assert.InDelta(t, 0.0, gate.Saturation(context.Background(), mixedLayout(smallTotal, largeTotal)), 1e-9)
	pl.Tracker().Charge("r", "large", 141)
	assert.InDelta(t, 141.0/float64(2*smallTotal+largeTotal),
		gate.Saturation(context.Background(), mixedLayout(smallTotal, largeTotal)), 1e-9)
}

func TestGate_BoundRequestSchedulesOnItsEndpoint(t *testing.T) {
	t.Parallel()
	pl := placement.New("placement", testPolicy())
	endpoints := mixedLayout(smallTotal, largeTotal)
	pass := NewGate("gate", pl).BeginDispatchPass(context.Background(), endpoints, true)
	require.NotNil(t, pass)
	item := queueItem("bound", 500, time.Now())
	require.Equal(t, flowcontrol.DispatchAdmit, pass.Admit(context.Background(), item))
	request := item.OriginalRequest().InferenceRequest()
	decision, ok := fwksched.ReadRequestAttribute[placement.Decision](request, placement.BoundDecisionKey)
	require.True(t, ok)
	blocksBefore, _ := pl.Tracker().Projected(decision.Endpoint)

	candidates := make([]fwksched.Endpoint, len(endpoints))
	for i, ep := range endpoints {
		candidates[i] = fwksched.NewEndpoint(ep.GetMetadata(), ep.GetMetrics(), ep.GetAttributes())
	}
	picked := pl.Filter(context.Background(), request, candidates)

	require.Len(t, picked, 1)
	assert.Equal(t, decision.Endpoint, capacity.EndpointName(picked[0].GetMetadata()))
	blocksAfter, _ := pl.Tracker().Projected(decision.Endpoint)
	assert.Equal(t, blocksBefore, blocksAfter, "scheduling a bound request does not charge it again")
}
