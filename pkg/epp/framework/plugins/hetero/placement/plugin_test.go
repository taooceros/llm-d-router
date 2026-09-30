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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
)

func schedEndpoint(name string, role capacity.Role, free, total, block int) fwksched.Endpoint {
	attrs := fwkdl.NewAttributes()
	attrs.Put(capacity.FreeBlocksKey, attrmetrics.ScalarMetricValue(free))
	attrs.Put(capacity.TotalBlocksKey, attrmetrics.ScalarMetricValue(total))
	attrs.Put(capacity.BlockSizeKey, attrmetrics.ScalarMetricValue(block))
	meta := &fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Namespace: "ns", Name: name},
		Labels: map[string]string{
			capacity.InstanceIDLabel:  name,
			capacity.RoleLabel:        string(role),
			capacity.MaxModelLenLabel: "16384",
		},
	}
	return fwksched.NewEndpoint(meta, fwkdl.NewMetrics(), attrs)
}

func completion(id string, promptTokens int) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: id,
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{TokenIDs: [][]uint32{make([]uint32, promptTokens)}}},
		},
	}
}

func blindPlugin() *Plugin {
	p := capacity.DefaultPolicy()
	p.LargeFreshTrafficShare = 0
	p.DecodeLengthBlind = true
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

func target(result *fwksched.SchedulingResult) string {
	return capacity.EndpointName(result.ProfileResults[result.PrimaryProfileName].TargetEndpoints[0].GetMetadata())
}

// TestProjectionSpreadsBurst places a burst against one stale metrics
// snapshot: only the projection keeps requests from landing on the same free
// blocks, and a completion returns them.
func TestProjectionSpreadsBurst(t *testing.T) {
	p := blindPlugin()
	endpoints := []fwksched.Endpoint{
		schedEndpoint("small_a", capacity.RoleSmall, 100, 100, 64),
		schedEndpoint("small_b", capacity.RoleSmall, 100, 100, 64),
	}
	// 3000 prompt tokens: need ceil(3064/64) = 48 blocks, charge ceil(3000/64) + 1 = 48.
	var got []string
	for i, id := range []string{"r0", "r1", "r2", "r3"} {
		result, err := runProfile(t, p, completion(id, 3000), endpoints)
		require.NoError(t, err, "request %d", i)
		got = append(got, target(result))
		require.NoError(t, p.PreRequest(context.Background(), completion(id, 3000), result))
	}
	assert.Equal(t, []string{"small_b", "small_a", "small_b", "small_a"}, got)
	blocks, requests := p.Tracker().Projected("small_b")
	assert.Equal(t, [2]int{96, 2}, [2]int{blocks, requests}, "PreRequest on the charged endpoint does not charge again")

	_, err := runProfile(t, p, completion("r4", 3000), endpoints)
	var typed errcommon.Error
	require.True(t, errors.As(err, &typed))
	assert.Equal(t, errcommon.ResourceExhausted, typed.Code)
	assert.Equal(t, string(errcommon.RequestDroppedReasonSaturated), typed.Headers[errcommon.RequestDroppedReasonHeaderKey])
	var trace Trace
	require.NoError(t, json.Unmarshal([]byte(typed.Headers[TraceHeader]), &trace))
	assert.Equal(t, RefusalNoCapacity, trace.Refusal)
	assert.Equal(t, "r4", trace.RequestID)

	// A streamed chunk keeps the charge; the end of the response releases it.
	p.ResponseBody(context.Background(), completion("r0", 3000), &fwkrc.Response{}, nil)
	_, _, charged := p.Tracker().Charged("r0")
	assert.True(t, charged)
	p.ResponseBody(context.Background(), completion("r0", 3000), &fwkrc.Response{EndOfStream: true, TerminationCause: fwkrc.TerminationCauseClientDisconnect}, nil)
	_, _, charged = p.Tracker().Charged("r0")
	assert.False(t, charged)

	result, err := runProfile(t, p, completion("r5", 3000), endpoints)
	require.NoError(t, err)
	assert.Equal(t, "small_b", target(result))
}

func TestRefusalCodes(t *testing.T) {
	p := blindPlugin()
	endpoints := []fwksched.Endpoint{schedEndpoint("small_a", capacity.RoleSmall, 100, 100, 64)}
	stringPrompt := &fwksched.InferenceRequest{RequestID: "s", Body: &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}},
	}}
	tests := []struct {
		name        string
		request     *fwksched.InferenceRequest
		wantRefusal string
	}{
		{name: "context longer than every endpoint supports", request: completion("long", 20000), wantRefusal: RefusalNoContextSupport},
		{name: "prompt without token count", request: stringPrompt, wantRefusal: RefusalUnknownPrompt},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := runProfile(t, p, tt.request, endpoints)
			var typed errcommon.Error
			require.True(t, errors.As(err, &typed))
			assert.Equal(t, errcommon.BadRequest, typed.Code)
			assert.True(t, strings.HasSuffix(typed.Msg, tt.wantRefusal), typed.Msg)
		})
	}
}

func TestBoundDecision(t *testing.T) {
	p := blindPlugin()
	endpoints := []fwksched.Endpoint{
		schedEndpoint("small_a", capacity.RoleSmall, 100, 100, 64),
		schedEndpoint("small_b", capacity.RoleSmall, 100, 100, 64),
	}
	sources := []capacity.Source{endpoints[0], endpoints[1]}
	decision, ok := p.Bind(sources, Request{ID: "r", PromptTokens: 3000}, Options{LateBinding: true})
	require.True(t, ok)
	require.Equal(t, "small_b", decision.Endpoint)

	t.Run("filter keeps the bound endpoint without charging again", func(t *testing.T) {
		request := completion("r", 3000)
		request.PutAttribute(BoundDecisionKey, decision)
		version := p.Tracker().Version()
		kept := p.Filter(context.Background(), request, endpoints)
		require.Len(t, kept, 1)
		assert.Equal(t, "small_b", capacity.EndpointName(kept[0].GetMetadata()))
		assert.Equal(t, version, p.Tracker().Version())

		response := &fwkrc.Response{Headers: map[string]string{}}
		p.ResponseHeader(context.Background(), request, response, nil)
		var trace Trace
		require.NoError(t, json.Unmarshal([]byte(response.Headers[TraceHeader]), &trace))
		assert.True(t, trace.Bound)
		assert.Equal(t, "small_b", trace.Selected)
	})

	t.Run("a vanished bound endpoint is released and placement selects again", func(t *testing.T) {
		request := completion("r", 3000)
		request.PutAttribute(BoundDecisionKey, decision)
		kept := p.Filter(context.Background(), request, endpoints[:1])
		require.Len(t, kept, 1)
		assert.Equal(t, "small_a", capacity.EndpointName(kept[0].GetMetadata()))
		charged, _, ok := p.Tracker().Charged("r")
		require.True(t, ok)
		assert.Equal(t, "small_a", charged)
		blocks, _ := p.Tracker().Projected("small_b")
		assert.Zero(t, blocks)
	})
}

func TestPreRequestChargesDispatchedEndpoint(t *testing.T) {
	p := blindPlugin()
	large := schedEndpoint("large", capacity.RoleLarge, 282, 282, 256)
	request := completion("r", 3000)
	p.Tracker().Charge("r", "small_a", 48)
	result := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults:     map[string]*fwksched.ProfileRunResult{"default": {TargetEndpoints: []fwksched.Endpoint{large}}},
	}
	require.NoError(t, p.PreRequest(context.Background(), request, result))
	endpoint, blocks, ok := p.Tracker().Charged("r")
	require.True(t, ok)
	assert.Equal(t, "large", endpoint)
	// ceil(3000/256) + 1 growth block
	assert.Equal(t, 13, blocks)
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
			"dispatchOrder":"longest_prompt_first","dynamicLongReserve":true}`},
		{name: "share above one", params: `{"largeFreshTrafficShare":1.5}`, wantErr: "largeFreshTrafficShare"},
		{name: "unknown dispatch order", params: `{"dispatchOrder":"random"}`, wantErr: "dispatchOrder"},
		{name: "negative reserve", params: `{"longReserveTokens":-1}`, wantErr: "non-negative"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Factory("placement", json.NewDecoder(strings.NewReader(tt.params)), nil)
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorContains(t, err, tt.wantErr)
		})
	}
}
