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
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
)

type endpointSpec struct {
	name, generation string
	role             capacity.Role
	maxModelLen      string
	free, total      int
	block            int
	noMetrics        bool
}

func (s endpointSpec) build() fwksched.Endpoint {
	attrs := fwkdl.NewAttributes()
	if !s.noMetrics {
		attrs.Put(capacity.FreeBlocksKey, attrmetrics.ScalarMetricValue(s.free))
		attrs.Put(capacity.TotalBlocksKey, attrmetrics.ScalarMetricValue(s.total))
		attrs.Put(capacity.BlockSizeKey, attrmetrics.ScalarMetricValue(s.block))
	}
	maxLen := s.maxModelLen
	if maxLen == "" {
		maxLen = "16384"
	}
	meta := &fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Namespace: "ns", Name: s.name + "-" + s.generation},
		Labels: map[string]string{
			capacity.InstanceIDLabel:  s.name,
			capacity.RoleLabel:        string(s.role),
			capacity.MaxModelLenLabel: maxLen,
			capacity.GenerationLabel:  s.generation,
		},
	}
	return fwksched.NewEndpoint(meta, fwkdl.NewMetrics(), attrs)
}

func schedEndpoint(name string, role capacity.Role, free, total, block int) fwksched.Endpoint {
	return endpointSpec{name: name, generation: "1", role: role, free: free, total: total, block: block}.build()
}

func completion(id string, promptTokens int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: id,
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{TokenIDs: [][]uint32{make([]uint32, promptTokens)}}},
		},
	}
}

func blindPlugin(edit func(*capacity.Policy)) *Plugin {
	p := capacity.DefaultPolicy()
	p.LargeFreshTrafficShare = 0
	p.DecodeLengthBlind = true
	if edit != nil {
		edit(&p)
	}
	return New("placement", p)
}

func runProfile(t *testing.T, p *Plugin, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.SchedulingResult, error) {
	t.Helper()
	kept := p.Filter(context.Background(), request, endpoints)
	results := map[string]*fwksched.ProfileRunResult{"default": nil}
	if len(kept) > 0 {
		results["default"] = &fwksched.ProfileRunResult{TargetEndpoints: kept}
	}
	return p.ProcessResults(context.Background(), request, results)
}

// dispatch schedules request and runs PreRequest, as the director does.
func dispatch(t *testing.T, p *Plugin, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (string, error) {
	t.Helper()
	result, err := runProfile(t, p, request, endpoints)
	if err != nil {
		return "", err
	}
	require.NoError(t, p.PreRequest(context.Background(), request, result))
	return capacity.EndpointName(result.ProfileResults[result.PrimaryProfileName].TargetEndpoints[0].GetMetadata()), nil
}

func finish(p *Plugin, request *fwksched.InferenceRequest) {
	p.ResponseBody(context.Background(), request, &fwkrc.Response{EndOfStream: true}, nil)
}

func refusal(t *testing.T, err error) (errcommon.Error, Trace) {
	t.Helper()
	var typed errcommon.Error
	require.True(t, errors.As(err, &typed), "error %v", err)
	var trace Trace
	require.NoError(t, json.Unmarshal([]byte(typed.Headers[TraceHeader]), &trace))
	return typed, trace
}

// TestProjectionSpreadsBurst places a burst against one stale metrics
// snapshot: only the projection keeps requests from landing on the same free
// blocks, and a completion returns them.
func TestProjectionSpreadsBurst(t *testing.T) {
	p := blindPlugin(nil)
	endpoints := []fwksched.Endpoint{
		schedEndpoint("small_a", capacity.RoleSmall, 100, 100, 64),
		schedEndpoint("small_b", capacity.RoleSmall, 100, 100, 64),
	}
	// 3000 prompt tokens: need ceil(3064/64) = 48 blocks, charge ceil(3000/64) + 1 = 48.
	var requests []*fwksched.InferenceRequest
	var got []string
	for range 4 {
		request := completion("same-client-id", 3000)
		requests = append(requests, request)
		target, err := dispatch(t, p, request, endpoints)
		require.NoError(t, err)
		got = append(got, target)
	}
	assert.Equal(t, []string{"small_b", "small_a", "small_b", "small_a"}, got,
		"requests sharing a client request ID keep separate reservations")
	blocks, reserved := p.Tracker().Projected("small_b")
	assert.Equal(t, [2]int{96, 2}, [2]int{blocks, reserved})

	_, err := dispatch(t, p, completion("r4", 3000), endpoints)
	typed, trace := refusal(t, err)
	assert.Equal(t, errcommon.ResourceExhausted, typed.Code)
	assert.Equal(t, string(errcommon.RequestDroppedReasonSaturated), typed.Headers[errcommon.RequestDroppedReasonHeaderKey])
	assert.Equal(t, RefusalNoCapacity, trace.Refusal)
	assert.Equal(t, "r4", trace.RequestID)

	// A streamed chunk keeps the reservation; the end of the response releases it.
	p.ResponseBody(context.Background(), requests[0], &fwkrc.Response{}, nil)
	blocks, _ = p.Tracker().Projected("small_b")
	assert.Equal(t, 96, blocks)
	p.ResponseBody(context.Background(), requests[0], &fwkrc.Response{EndOfStream: true, TerminationCause: fwkrc.TerminationCauseClientDisconnect}, nil)
	blocks, _ = p.Tracker().Projected("small_b")
	assert.Equal(t, 48, blocks, "only the finished request's reservation is released")

	target, err := dispatch(t, p, completion("r5", 3000), endpoints)
	require.NoError(t, err)
	assert.Equal(t, "small_b", target)
}

func TestRefusalCodes(t *testing.T) {
	p := blindPlugin(nil)
	stringPrompt := &fwksched.InferenceRequest{RequestID: "s", Body: &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}},
	}}
	scraped := endpointSpec{name: "small_a", generation: "1", role: capacity.RoleSmall, free: 100, total: 100, block: 64}
	unscrapedLarge := endpointSpec{name: "large", generation: "1", role: capacity.RoleLarge, maxModelLen: "36864", noMetrics: true}
	tests := []struct {
		name        string
		request     *fwksched.InferenceRequest
		endpoints   []endpointSpec
		wantCode    string
		wantRefusal string
	}{
		{
			name:        "context longer than every endpoint supports",
			request:     completion("long", 20000),
			endpoints:   []endpointSpec{scraped},
			wantCode:    errcommon.BadRequest,
			wantRefusal: RefusalNoContextSupport,
		},
		{
			name:        "context supported only by an endpoint awaiting metrics is retriable",
			request:     completion("long", 20000),
			endpoints:   []endpointSpec{scraped, unscrapedLarge},
			wantCode:    errcommon.ResourceExhausted,
			wantRefusal: RefusalNoCapacity,
		},
		{
			name:        "context above the limit of an endpoint awaiting metrics",
			request:     completion("long", 40000),
			endpoints:   []endpointSpec{scraped, unscrapedLarge},
			wantCode:    errcommon.BadRequest,
			wantRefusal: RefusalNoContextSupport,
		},
		{
			name:        "prompt without token count",
			request:     stringPrompt,
			endpoints:   []endpointSpec{scraped},
			wantCode:    errcommon.BadRequest,
			wantRefusal: RefusalUnknownPrompt,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var endpoints []fwksched.Endpoint
			for _, spec := range tt.endpoints {
				endpoints = append(endpoints, spec.build())
			}
			_, err := runProfile(t, p, tt.request, endpoints)
			typed, trace := refusal(t, err)
			assert.Equal(t, tt.wantCode, typed.Code)
			assert.Equal(t, tt.wantRefusal, trace.Refusal)
		})
	}
}

// bindAhead binds request as the late-binding gate does and returns the decision.
func bindAhead(t *testing.T, p *Plugin, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint, opts Options) Decision {
	t.Helper()
	sources := make([]capacity.Source, len(endpoints))
	for i, ep := range endpoints {
		sources[i] = ep
	}
	promptTokens, _ := capacity.PromptTokens(request)
	decision, ok := p.Bind(sources, Request{ID: request.RequestID, Token: ReservationToken(request), PromptTokens: promptTokens}, opts)
	require.True(t, ok)
	request.PutAttribute(BoundDecisionKey, decision)
	return decision
}

func TestBoundDecision(t *testing.T) {
	smalls := func(generationB string) []fwksched.Endpoint {
		return []fwksched.Endpoint{
			schedEndpoint("small_a", capacity.RoleSmall, 100, 100, 64),
			endpointSpec{name: "small_b", generation: generationB, role: capacity.RoleSmall, free: 100, total: 100, block: 64}.build(),
		}
	}

	t.Run("filter keeps the bound endpoint and PreRequest confirms it", func(t *testing.T) {
		p := blindPlugin(nil)
		request := completion("r", 3000)
		decision := bindAhead(t, p, request, smalls("1"), Options{LateBinding: true})
		require.Equal(t, "small_b", decision.Endpoint)
		version := p.Tracker().Version()
		kept := p.Filter(context.Background(), request, smalls("1"))
		require.Len(t, kept, 1)
		assert.Equal(t, "small_b", capacity.EndpointName(kept[0].GetMetadata()))
		assert.Equal(t, version, p.Tracker().Version(), "the bound reservation is not charged again")

		r, _ := p.Tracker().Reservation(ReservationToken(request))
		assert.True(t, r.Pending())
		result := &fwksched.SchedulingResult{PrimaryProfileName: "default",
			ProfileResults: map[string]*fwksched.ProfileRunResult{"default": {TargetEndpoints: kept}}}
		require.NoError(t, p.PreRequest(context.Background(), request, result))
		r, _ = p.Tracker().Reservation(ReservationToken(request))
		assert.False(t, r.Pending())

		response := &fwkrc.Response{Headers: map[string]string{}}
		p.ResponseHeader(context.Background(), request, response, nil)
		var trace Trace
		require.NoError(t, json.Unmarshal([]byte(response.Headers[TraceHeader]), &trace))
		assert.True(t, trace.Bound)
		assert.Equal(t, "small_b", trace.Selected)
	})

	tests := []struct {
		name       string
		endpoints  []fwksched.Endpoint
		expire     bool
		wantReason string
	}{
		{name: "bound endpoint gone", endpoints: smalls("1")[:1], wantReason: RebindEndpointGone},
		{name: "bound endpoint replaced by a new generation", endpoints: smalls("2"), wantReason: RebindEndpointReplaced},
		{name: "bound reservation lapsed", endpoints: smalls("1"), expire: true, wantReason: RebindReservationLapsed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			now := time.Unix(1000, 0)
			p := blindPlugin(nil)
			p.Tracker().SetClock(func() time.Time { return now })
			request := completion("r", 3000)
			bindAhead(t, p, request, smalls("1"), Options{LateBinding: true})
			if tt.expire {
				now = now.Add(DefaultBindLease)
			}
			kept := p.Filter(context.Background(), request, tt.endpoints)
			require.Len(t, kept, 1)
			name := capacity.EndpointName(kept[0].GetMetadata())
			endpoint, _, ok := p.Tracker().Charged(ReservationToken(request))
			require.True(t, ok)
			assert.Equal(t, name, endpoint, "only the new selection is reserved")
			blocks, _ := p.Tracker().Projected("small_b")
			if name != "small_b" {
				assert.Zero(t, blocks)
			} else {
				assert.Equal(t, 48, blocks)
			}
			dispatched, _ := p.Selector().Counters()
			assert.Equal(t, 1, dispatched, "the abandoned bind is rolled back")
			trace, _ := fwksched.ReadRequestAttribute[*Trace](request, traceKey)
			assert.Equal(t, tt.wantReason, trace.RebindReason)
			assert.True(t, trace.LateBinding)
		})
	}

	t.Run("reselection keeps the dynamic reserve of the bind", func(t *testing.T) {
		p := blindPlugin(func(p *capacity.Policy) {
			p.ReserveLargeHeadroomBlocks = 16
			p.DynamicLongReserve = true
			p.WorkConservingFallback = false
		})
		large := schedEndpoint("large", capacity.RoleLarge, 150, 282, 256)
		request := completion("short", 500)
		bindAhead(t, p, request, []fwksched.Endpoint{large, schedEndpoint("small_a", capacity.RoleSmall, 100, 100, 64)},
			Options{LateBinding: true, ExtraHeld: map[string]int{"large": 132}})
		_, err := runProfile(t, p, request, []fwksched.Endpoint{large})
		_, trace := refusal(t, err)
		assert.Equal(t, RefusalNoCapacity, trace.Refusal, "150 - 16 - 132 < 3 blocks")
		assert.Equal(t, map[string]int{"large": 132}, trace.ExtraHeld)
		_, _, charged := p.Tracker().Charged(ReservationToken(request))
		assert.False(t, charged)
	})
}

// TestLeaseLifecycle covers requests that fail before the director can report
// their end: the lease lapses and the quota is rolled back.
func TestLeaseLifecycle(t *testing.T) {
	now := time.Unix(1000, 0)
	p := blindPlugin(func(p *capacity.Policy) { p.LargeFreshTrafficShare = 0.5 })
	p.Tracker().SetClock(func() time.Time { return now })
	endpoints := []fwksched.Endpoint{
		schedEndpoint("small_a", capacity.RoleSmall, 100, 100, 64),
		schedEndpoint("large", capacity.RoleLarge, 282, 282, 256),
	}

	abandoned := completion("abandoned", 3000)
	decision := bindAhead(t, p, abandoned, endpoints, Options{LateBinding: true})
	require.Equal(t, "large", decision.Endpoint, "first request is owed to large")
	dispatched, toLarge := p.Selector().Counters()
	assert.Equal(t, [2]int{1, 1}, [2]int{dispatched, toLarge})

	now = now.Add(DefaultBindLease - time.Millisecond)
	blocks, _ := p.Tracker().Projected("large")
	assert.Equal(t, 13, blocks)
	now = now.Add(time.Millisecond)
	blocks, _ = p.Tracker().Projected("large")
	assert.Zero(t, blocks, "an unconfirmed reservation lapses")
	dispatched, toLarge = p.Selector().Counters()
	assert.Equal(t, [2]int{0, 0}, [2]int{dispatched, toLarge}, "lapsing rolls back the quota")

	reverted := completion("reverted", 3000)
	bindAhead(t, p, reverted, endpoints, Options{LateBinding: true})
	p.Unbind(reverted)
	dispatched, toLarge = p.Selector().Counters()
	assert.Equal(t, [2]int{0, 0}, [2]int{dispatched, toLarge}, "unbinding a pending reservation rolls back the quota")

	served := completion("served", 3000)
	target, err := dispatch(t, p, served, endpoints)
	require.NoError(t, err)
	require.Equal(t, "large", target)
	now = now.Add(time.Hour)
	blocks, _ = p.Tracker().Projected("large")
	assert.Equal(t, 13, blocks, "a confirmed reservation does not lapse")
	finish(p, served)
	dispatched, toLarge = p.Selector().Counters()
	assert.Equal(t, [2]int{1, 1}, [2]int{dispatched, toLarge}, "completion keeps the dispatch counted")
}

func TestPreRequestChargesDispatchedEndpoint(t *testing.T) {
	p := blindPlugin(nil)
	large := schedEndpoint("large", capacity.RoleLarge, 282, 282, 256)
	request := completion("r", 3000)
	p.Tracker().Lease(ReservationToken(request), "small_a", 48, false, time.Minute)
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults:     map[string]*fwksched.ProfileRunResult{"default": {TargetEndpoints: []fwksched.Endpoint{large}}},
	}
	require.NoError(t, p.PreRequest(context.Background(), request, result))
	r, ok := p.Tracker().Reservation(ReservationToken(request))
	require.True(t, ok)
	assert.Equal(t, "large", r.Endpoint)
	// ceil(3000/256) + 1 growth block
	assert.Equal(t, 13, r.Blocks)
	assert.False(t, r.Pending())
}

func TestReset(t *testing.T) {
	p := blindPlugin(nil)
	endpoints := []fwksched.Endpoint{schedEndpoint("small_a", capacity.RoleSmall, 100, 100, 64)}
	request := completion("r", 3000)
	_, err := dispatch(t, p, request, endpoints)
	require.NoError(t, err)
	server := httptest.NewServer(p.adminHandler())
	defer server.Close()

	post := func(query string) (int, resetResponse) {
		resp, err := http.Post(server.URL+AdminResetPath+query, "", nil)
		require.NoError(t, err)
		defer resp.Body.Close()
		var body resetResponse
		require.NoError(t, json.NewDecoder(resp.Body).Decode(&body))
		return resp.StatusCode, body
	}

	status, body := post("")
	assert.Equal(t, http.StatusConflict, status, "reset refuses while a request holds a reservation")
	assert.Equal(t, 1, body.Before.Reservations)
	assert.Equal(t, 1, p.State().FreshDispatched)

	finish(p, request)
	status, body = post("")
	assert.Equal(t, http.StatusOK, status)
	assert.Equal(t, State{FreshDispatched: 1, Projection: map[string]capacity.ProjectionState{}}, body.Before)
	assert.Equal(t, State{Projection: map[string]capacity.ProjectionState{}}, p.State())

	_, err = dispatch(t, p, completion("r2", 3000), endpoints)
	require.NoError(t, err)
	status, _ = post("?force=true")
	assert.Equal(t, http.StatusOK, status)
	assert.Zero(t, p.State().Reservations)

	resp, err := http.Get(server.URL + AdminStatePath)
	require.NoError(t, err)
	defer resp.Body.Close()
	var state State
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&state))
	assert.Equal(t, State{Projection: map[string]capacity.ProjectionState{}}, state)
}

func TestFactory(t *testing.T) {
	tests := []struct {
		name    string
		params  string
		wantErr string
	}{
		{name: "defaults", params: `{}`},
		{name: "full policy", params: `{"smallPromptTokenLimit":16384,"largeFreshTrafficShare":0.5,"reserveLargeHeadroomBlocks":16,
			"longReserveTokens":36864,"workConservingFallback":true,"decodeLengthBlind":true,"admissionGrowthBlocks":1,
			"dynamicLongReserve":true,"bindLeaseSeconds":2.5}`},
		{name: "share above one", params: `{"largeFreshTrafficShare":1.5}`, wantErr: "largeFreshTrafficShare"},
		{name: "dispatch order belongs to the flow-control band", params: `{"dispatchOrder":"arrival"}`, wantErr: "unknown field"},
		{name: "negative reserve", params: `{"longReserveTokens":-1}`, wantErr: "non-negative"},
		{name: "non-positive lease", params: `{"bindLeaseSeconds":0}`, wantErr: "bindLeaseSeconds"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Factory("placement", fwkplugin.StrictDecoder(json.RawMessage(tt.params)), nil)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorContains(t, err, tt.wantErr)
		})
	}
}
