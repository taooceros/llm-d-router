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
	"time"

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
			want.Name, want.ID, want.Role, want.TP, want.MaxModelLen, want.BlockSize, want.TotalBlocks = "large", "ns/large-pod", RoleLarge, 16, 36864, 256, 282
			assert.Equal(t, want, got)
		})
	}
}

func TestObserveMissingMetrics(t *testing.T) {
	meta := &fwkdl.EndpointMetadata{
		ID:     types.NamespacedName{Namespace: "ns", Name: "pod"},
		Labels: map[string]string{RoleLabel: "large", MaxModelLenLabel: "36864", GenerationLabel: "3"},
	}
	tracker := NewTracker()

	got, ok := tracker.Observe(fwksched.NewEndpoint(meta, &fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 412}, nil))
	assert.False(t, ok, "free blocks have no fallback")
	assert.Equal(t, EndpointCapacity{Name: "ns/pod", ID: "ns/pod", Generation: "3", Role: RoleLarge, MaxModelLen: 36864, MissingMetrics: true}, got,
		"label facts are reported without metrics")
	assert.False(t, HasRoom(got, 1, true, DefaultPolicy(), 0))

	attrs := fwkdl.NewAttributes()
	attrs.Put(FreeBlocksKey, attrmetrics.ScalarMetricValue(10))
	got, ok = tracker.Observe(fwksched.NewEndpoint(meta, &fwkdl.Metrics{CacheBlockSize: 64, CacheNumBlocks: 412}, attrs))
	require.True(t, ok, "block size and total fall back to core metrics")
	assert.False(t, got.MissingMetrics)
	assert.Equal(t, 64, got.BlockSize)
	assert.Equal(t, 412, got.TotalBlocks)
}

func TestTrackerChargeRelease(t *testing.T) {
	tracker := NewTracker()
	tracker.Charge("r1", "small_a", 10)
	tracker.Charge("r2", "small_a", 5)
	blocks, requests := tracker.Projected("small_a")
	assert.Equal(t, [2]int{15, 2}, [2]int{blocks, requests})

	// Charging the same token again moves it instead of adding.
	v := tracker.Version()
	tracker.Charge("r1", "large", 4)
	assert.Greater(t, tracker.Version(), v)
	blocks, requests = tracker.Projected("small_a")
	assert.Equal(t, [2]int{5, 1}, [2]int{blocks, requests})
	blocks, requests = tracker.Projected("large")
	assert.Equal(t, [2]int{4, 1}, [2]int{blocks, requests})

	_, released := tracker.Release("r1")
	assert.True(t, released)
	v = tracker.Version()
	_, released = tracker.Release("r1")
	assert.False(t, released, "release is idempotent")
	assert.Equal(t, v, tracker.Version())
	blocks, requests = tracker.Projected("large")
	assert.Equal(t, [2]int{0, 0}, [2]int{blocks, requests})
	_, _, ok := tracker.Charged("r1")
	assert.False(t, ok)
}

func TestTrackerLease(t *testing.T) {
	now := time.Unix(1000, 0)
	tracker := NewTracker()
	tracker.SetClock(func() time.Time { return now })
	var expired []string
	tracker.OnExpire(func(token string, r Reservation) {
		assert.True(t, r.Large)
		expired = append(expired, token)
	})

	tracker.Lease("unconfirmed", "large", 40, true, 10*time.Second)
	tracker.Lease("confirmed", "large", 30, true, 10*time.Second)
	tracker.Lease("later", "large", 20, true, 20*time.Second)
	require.True(t, tracker.Confirm("confirmed"))
	assert.False(t, tracker.Confirm("absent"))
	r, _ := tracker.Reservation("confirmed")
	assert.False(t, r.Pending())

	now = now.Add(10*time.Second - time.Nanosecond)
	blocks, requests := tracker.Projected("large")
	assert.Equal(t, [2]int{90, 3}, [2]int{blocks, requests}, "leases hold until the deadline")

	now = now.Add(time.Nanosecond)
	blocks, requests = tracker.Projected("large")
	assert.Equal(t, [2]int{50, 2}, [2]int{blocks, requests}, "an unconfirmed lease lapses at its deadline")
	assert.Equal(t, []string{"unconfirmed"}, expired)

	now = now.Add(time.Hour)
	assert.Equal(t, 1, tracker.Expire())
	assert.Equal(t, []string{"unconfirmed", "later"}, expired)
	blocks, requests = tracker.Projected("large")
	assert.Equal(t, [2]int{30, 1}, [2]int{blocks, requests}, "a confirmed reservation has no deadline")

	// A lease replaced by a charge no longer expires.
	tracker.Lease("moved", "small_a", 5, false, time.Second)
	tracker.Charge("moved", "small_b", 6)
	now = now.Add(time.Hour)
	assert.Zero(t, tracker.Expire())
	blocks, _ = tracker.Projected("small_b")
	assert.Equal(t, 6, blocks)

	perEndpoint, leases := tracker.State()
	assert.Zero(t, leases)
	assert.Equal(t, map[string]ProjectionState{"large": {Blocks: 30, Requests: 1}, "small_b": {Blocks: 6, Requests: 1}}, perEndpoint)
	assert.Equal(t, 2, tracker.Reset())
	perEndpoint, _ = tracker.State()
	assert.Empty(t, perEndpoint)
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
