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

package internal

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
)

// fakeGate is a SaturationDetector and DispatchGate whose pass admits items by request ID.
type fakeGate struct {
	saturation float64
	verdicts   map[string]flowcontrol.DispatchVerdict
	onAdmit    func(item flowcontrol.QueueItemAccessor)
	noPass     bool

	begins   []bool
	visits   []string
	reverted []string
}

var (
	_ flowcontrol.DispatchGate = &fakeGate{}
	_ flowcontrol.DispatchPass = &fakeGate{}
)

func (g *fakeGate) TypedName() plugin.TypedName { return plugin.TypedName{Type: "fake-gate", Name: "fake-gate"} }
func (g *fakeGate) Saturation(context.Context, []fwkdl.Endpoint) float64 { return g.saturation }

func (g *fakeGate) BeginDispatchPass(_ context.Context, _ []fwkdl.Endpoint, queueChanged bool) flowcontrol.DispatchPass {
	g.begins = append(g.begins, queueChanged)
	if g.noPass {
		return nil
	}
	return g
}

func (g *fakeGate) Admit(_ context.Context, item flowcontrol.QueueItemAccessor) flowcontrol.DispatchVerdict {
	id := item.OriginalRequest().ID()
	g.visits = append(g.visits, id)
	if g.onAdmit != nil {
		g.onAdmit(item)
	}
	return g.verdicts[id]
}

func (g *fakeGate) Revert(item flowcontrol.QueueItemAccessor) {
	g.reverted = append(g.reverted, item.OriginalRequest().ID())
}

// newGatedHarness returns a harness whose processor uses gate as its saturation detector, with three fcfs-ordered
// items a, b, c (in arrival order) queued on one flow.
func newGatedHarness(t *testing.T, gate *fakeGate) (*testHarness, map[string]*FlowItem) {
	t.Helper()
	h := newTestHarness(t, time.Hour)
	h.processor = NewProcessor(h.ctx, "test-pool", h, nil, gate, h.endpointCandidates, usagelimits.DefaultPolicy(),
		h.clock, testNoEndpointTTL, time.Hour, 100, h.logger, nil)
	require.Same(t, gate, h.processor.dispatchGate, "a detector implementing DispatchGate should enable gated dispatch")

	key := flowcontrol.FlowKey{ID: "flow", Priority: 0}
	q := h.addQueue(key)
	ordering, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, nil)
	require.NoError(t, err)
	q.OrderingPolicyFunc = func() flowcontrol.OrderingPolicy { return ordering.(flowcontrol.OrderingPolicy) }

	items := map[string]*FlowItem{}
	for _, id := range []string{"a", "b", "c"} {
		item := h.newTestItem(id, key, time.Hour)
		h.processor.enqueue(item)
		require.Nil(t, item.FinalState(), "item %s should be queued", id)
		items[id] = item
		h.clock.Step(time.Millisecond)
	}
	return h, items
}

func outcomes(items map[string]*FlowItem) map[string]string {
	out := map[string]string{}
	for id, item := range items {
		if fs := item.FinalState(); fs != nil {
			out[id] = fs.Outcome.String()
		} else {
			out[id] = "queued"
		}
	}
	return out
}

func TestProcessor_GatedDispatch(t *testing.T) {
	t.Parallel()
	dispatched := types.QueueOutcomeDispatched.String()

	t.Run("a skipped item does not block later items", func(t *testing.T) {
		t.Parallel()
		gate := &fakeGate{verdicts: map[string]flowcontrol.DispatchVerdict{
			"b": flowcontrol.DispatchAdmit, "c": flowcontrol.DispatchAdmit,
		}}
		h, items := newGatedHarness(t, gate)

		assert.True(t, h.processor.dispatchCycle(context.Background()))
		assert.Equal(t, []string{"a", "b", "c"}, gate.visits, "items are visited in ordering-policy order")
		assert.Equal(t, map[string]string{"a": "queued", "b": dispatched, "c": dispatched}, outcomes(items))
	})

	t.Run("stop ends the pass with the item queued", func(t *testing.T) {
		t.Parallel()
		gate := &fakeGate{verdicts: map[string]flowcontrol.DispatchVerdict{
			"a": flowcontrol.DispatchAdmit, "b": flowcontrol.DispatchStop, "c": flowcontrol.DispatchAdmit,
		}}
		h, items := newGatedHarness(t, gate)

		assert.True(t, h.processor.dispatchCycle(context.Background()))
		assert.Equal(t, []string{"a", "b"}, gate.visits)
		assert.Equal(t, map[string]string{"a": dispatched, "b": "queued", "c": "queued"}, outcomes(items))
	})

	t.Run("an admitted item finalized concurrently is reverted", func(t *testing.T) {
		t.Parallel()
		gate := &fakeGate{verdicts: map[string]flowcontrol.DispatchVerdict{
			"a": flowcontrol.DispatchAdmit, "b": flowcontrol.DispatchAdmit,
		}}
		gate.onAdmit = func(item flowcontrol.QueueItemAccessor) {
			if item.OriginalRequest().ID() == "a" {
				item.(*FlowItem).FinalizeWithError(fmt.Errorf("%w: %w", types.ErrEvicted, types.ErrContextCancelled))
			}
		}
		h, items := newGatedHarness(t, gate)

		assert.True(t, h.processor.dispatchCycle(context.Background()))
		assert.Equal(t, []string{"a"}, gate.reverted, "only the item that failed to dispatch is reverted")
		assert.Equal(t, dispatched, outcomes(items)["b"])
		assert.NotEqual(t, dispatched, outcomes(items)["a"])
	})

	t.Run("queueChanged reports additions and removals between passes", func(t *testing.T) {
		t.Parallel()
		gate := &fakeGate{verdicts: map[string]flowcontrol.DispatchVerdict{}}
		h, items := newGatedHarness(t, gate)
		ctx := context.Background()

		assert.False(t, h.processor.dispatchCycle(ctx))
		assert.False(t, h.processor.dispatchCycle(ctx))
		gate.verdicts["a"] = flowcontrol.DispatchAdmit
		assert.True(t, h.processor.dispatchCycle(ctx))
		assert.False(t, h.processor.dispatchCycle(ctx))
		assert.Equal(t, []bool{true, false, false, true}, gate.begins,
			"enqueues are reported to the first pass and the dispatch of a to the pass after it")
		assert.Equal(t, dispatched, outcomes(items)["a"])
	})

	t.Run("a nil pass visits no items", func(t *testing.T) {
		t.Parallel()
		gate := &fakeGate{noPass: true, verdicts: map[string]flowcontrol.DispatchVerdict{"a": flowcontrol.DispatchAdmit}}
		h, items := newGatedHarness(t, gate)

		assert.False(t, h.processor.dispatchCycle(context.Background()))
		assert.Empty(t, gate.visits)
		assert.Equal(t, "queued", outcomes(items)["a"])
	})

	t.Run("a saturated pool opens no pass and keeps queueChanged", func(t *testing.T) {
		t.Parallel()
		gate := &fakeGate{saturation: 1.0, verdicts: map[string]flowcontrol.DispatchVerdict{"a": flowcontrol.DispatchAdmit}}
		h, items := newGatedHarness(t, gate)
		ctx := context.Background()

		assert.False(t, h.processor.dispatchCycle(ctx))
		assert.Empty(t, gate.begins)
		gate.saturation = 0
		assert.True(t, h.processor.dispatchCycle(ctx))
		assert.Equal(t, []bool{true}, gate.begins, "the enqueues before the saturated cycle are still reported")
		assert.Equal(t, dispatched, outcomes(items)["a"])
	})
}
