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

package capacity

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

func endpoint(name string, role Role, free, total, block, running, waiting int) fwksched.Endpoint {
	attrs := fwkdl.NewAttributes()
	attrs.Put(FreeBlocksKey, attrmetrics.ScalarMetricValue(free))
	attrs.Put(TotalBlocksKey, attrmetrics.ScalarMetricValue(total))
	attrs.Put(BlockSizeKey, attrmetrics.ScalarMetricValue(block))
	meta := &fwkdl.EndpointMetadata{
		ID:     types.NamespacedName{Namespace: "ns", Name: name + "-pod"},
		Labels: map[string]string{InstanceIDLabel: name, RoleLabel: string(role), TPLabel: "16", MaxModelLenLabel: "36864"},
	}
	return fwksched.NewEndpoint(meta, &fwkdl.Metrics{RunningRequestsSize: running, WaitingQueueSize: waiting}, attrs)
}

func TestObserve(t *testing.T) {
	tests := []struct {
		name      string
		free      int
		running   int
		waiting   int
		projected []int
		want      EndpointCapacity
	}{
		{
			name: "one growth block per running request is not admissible",
			free: 200, running: 5, waiting: 2,
			want: EndpointCapacity{FreeBlocks: 200, AdmissibleBlocks: 195, Running: 5, Waiting: 2},
		},
		{
			name: "growth allowance is capped by free blocks",
			free: 3, running: 5,
			want: EndpointCapacity{FreeBlocks: 3, AdmissibleBlocks: 0, Running: 5},
		},
		{
			name: "projection larger than observed use lowers free blocks",
			free: 280, running: 1, projected: []int{100, 50},
			// total 282 - max(2, 150); waiting counts projected requests not yet running
			want: EndpointCapacity{FreeBlocks: 132, AdmissibleBlocks: 132, Running: 1, Waiting: 1},
		},
		{
			name: "projection smaller than observed use is ignored",
			free: 100, projected: []int{50},
			want: EndpointCapacity{FreeBlocks: 100, AdmissibleBlocks: 100, Waiting: 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tracker := NewTracker()
			for i, blocks := range tt.projected {
				tracker.Charge(string(rune('a'+i)), "large", blocks)
			}
			got, ok := tracker.Observe(endpoint("large", RoleLarge, tt.free, 282, 256, tt.running, tt.waiting))
			require.True(t, ok)
			want := tt.want
			want.Name, want.Role, want.TP, want.MaxModelLen, want.BlockSize, want.TotalBlocks = "large", RoleLarge, 16, 36864, 256, 282
			assert.Equal(t, want, got)
		})
	}
}

func TestObserveMissingMetrics(t *testing.T) {
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "ns", Name: "pod"}}
	tracker := NewTracker()

	_, ok := tracker.Observe(fwksched.NewEndpoint(meta, &fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 412}, nil))
	assert.False(t, ok, "free blocks have no fallback")

	attrs := fwkdl.NewAttributes()
	attrs.Put(FreeBlocksKey, attrmetrics.ScalarMetricValue(10))
	got, ok := tracker.Observe(fwksched.NewEndpoint(meta, &fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 412}, attrs))
	require.True(t, ok, "block size and total fall back to core metrics")
	assert.Equal(t, "ns/pod", got.Name)
	assert.Equal(t, 64, got.BlockSize)
	assert.Equal(t, 412, got.TotalBlocks)
}

func TestTrackerChargeRelease(t *testing.T) {
	tracker := NewTracker()
	tracker.Charge("r1", "small_a", 10)
	tracker.Charge("r2", "small_a", 5)
	blocks, requests := tracker.Projected("small_a")
	assert.Equal(t, [2]int{15, 2}, [2]int{blocks, requests})

	// Charging the same request again moves it instead of adding.
	v := tracker.Version()
	tracker.Charge("r1", "large", 4)
	assert.Greater(t, tracker.Version(), v)
	blocks, requests = tracker.Projected("small_a")
	assert.Equal(t, [2]int{5, 1}, [2]int{blocks, requests})
	blocks, requests = tracker.Projected("large")
	assert.Equal(t, [2]int{4, 1}, [2]int{blocks, requests})

	assert.True(t, tracker.Release("r1"))
	v = tracker.Version()
	assert.False(t, tracker.Release("r1"), "release is idempotent")
	assert.Equal(t, v, tracker.Version())
	blocks, requests = tracker.Projected("large")
	assert.Equal(t, [2]int{0, 0}, [2]int{blocks, requests})
	_, _, ok := tracker.Charged("r1")
	assert.False(t, ok)
}

func TestHasRoom(t *testing.T) {
	large := EndpointCapacity{Name: "large", Role: RoleLarge, BlockSize: 256, TotalBlocks: 282, AdmissibleBlocks: 150}
	small := EndpointCapacity{Name: "small", Role: RoleSmall, BlockSize: 64, TotalBlocks: 412, AdmissibleBlocks: 3}
	policy := DefaultPolicy()
	policy.ReserveLargeHeadroomBlocks = 16
	policy.LongReserveTokens = 36864
	dynamic := policy
	dynamic.DynamicLongReserve = true

	tests := []struct {
		name     string
		ep       EndpointCapacity
		context  int
		long     bool
		policy   Policy
		extra    int
		wantRoom bool
	}{
		{name: "large keeps headroom and the long reserve from shorts", ep: large, context: 756, policy: policy},
		{name: "long request may use the reserve", ep: large, context: 30256, long: true, policy: policy, wantRoom: true},
		{name: "long request still keeps the headroom", ep: large, context: 134*256 + 1, long: true, policy: policy},
		{name: "long request fits exactly above the headroom", ep: large, context: 134 * 256, long: true, policy: policy, wantRoom: true},
		{name: "dynamic reserve drops the static reserve", ep: large, context: 756, policy: dynamic, wantRoom: true},
		{name: "extra held blocks apply on top of headroom", ep: large, context: 756, policy: dynamic, extra: 132},
		{name: "small endpoint keeps no headroom", ep: small, context: 192, policy: policy, wantRoom: true},
		{name: "small endpoint without room", ep: small, context: 193, policy: policy},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.wantRoom, HasRoom(tt.ep, tt.context, tt.long, tt.policy, tt.extra))
		})
	}
}
