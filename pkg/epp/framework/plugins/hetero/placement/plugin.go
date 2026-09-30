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

// Package placement implements initial placement for heterogeneous TPU
// serving as one plugin: a filter that selects and charges the endpoint, the
// profile handler that reports refusals, and the request-control hooks that
// keep the capacity projection in step with the request lifecycle.
package placement

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
)

const (
	// PlacementType is the plugin type of Plugin.
	PlacementType = "hetero-placement"
	// TraceHeader carries the JSON decision trace on responses and refusals.
	TraceHeader = "x-omp-route-trace"
)

var (
	// BoundDecisionKey is the request attribute holding a Decision made before
	// scheduling. The filter keeps the bound endpoint instead of selecting.
	BoundDecisionKey = plugin.NewDataKey("hetero/bound-decision", PlacementType)
	traceKey         = plugin.NewDataKey("hetero/route-trace", PlacementType)
)

var (
	_ fwksched.Filter               = &Plugin{}
	_ fwksched.ProfileHandler       = &Plugin{}
	_ fwkrc.PreRequest              = &Plugin{}
	_ fwkrc.ResponseHeaderProcessor = &Plugin{}
	_ fwkrc.ResponseBodyProcessor   = &Plugin{}
	_ plugin.ConsumerPlugin         = &Plugin{}
)

// Factory builds a Plugin from Policy parameters.
func Factory(name string, decoder *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	policy := capacity.DefaultPolicy()
	if decoder != nil {
		if err := decoder.Decode(&policy); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PlacementType, err)
		}
	}
	if err := validate(policy); err != nil {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: %w", PlacementType, err)
	}
	return New(name, policy), nil
}

func validate(p capacity.Policy) error {
	switch {
	case p.LargeFreshTrafficShare < 0 || p.LargeFreshTrafficShare > 1:
		return errors.New("'largeFreshTrafficShare' must be within [0, 1]")
	case p.SmallPromptTokenLimit < 0, p.ReserveLargeHeadroomBlocks < 0, p.LongReserveTokens < 0, p.AdmissionGrowthBlocks < 0:
		return errors.New("token and block parameters must be non-negative")
	case p.DispatchOrder != capacity.DispatchArrival && p.DispatchOrder != capacity.DispatchLongestPromptFirst:
		return fmt.Errorf("'dispatchOrder' must be %q or %q", capacity.DispatchArrival, capacity.DispatchLongestPromptFirst)
	}
	return nil
}

// New returns a Plugin with its own capacity tracker and selector.
func New(name string, policy capacity.Policy) *Plugin {
	return &Plugin{
		typedName: plugin.TypedName{Type: PlacementType, Name: name},
		policy:    policy,
		tracker:   capacity.NewTracker(),
		selector:  NewSelector(policy),
	}
}

// Plugin places fresh requests per the heterogeneous routing policy. It must
// be the last filter of its profile, since it charges the endpoint it keeps.
type Plugin struct {
	typedName plugin.TypedName
	policy    capacity.Policy
	tracker   *capacity.Tracker
	selector  *Selector
	// bindMu makes observation, selection and charge one step, so concurrent
	// requests do not select against the same free blocks.
	bindMu sync.Mutex
}

func (p *Plugin) TypedName() plugin.TypedName { return p.typedName }

// Policy returns the routing policy.
func (p *Plugin) Policy() capacity.Policy { return p.policy }

// Tracker returns the capacity projection.
func (p *Plugin) Tracker() *capacity.Tracker { return p.tracker }

// Selector returns the selector holding the quota counters.
func (p *Plugin) Selector() *Selector { return p.selector }

// Consumes declares the custom capacity metrics. They are optional so a
// missing metric removes the endpoint from placement instead of failing startup.
func (p *Plugin) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{Optional: capacity.ConsumedKeys()}
}

// Bind observes endpoints, selects one and charges it to req.ID.
func (p *Plugin) Bind(endpoints []capacity.Source, req Request, opts Options) (Decision, bool) {
	p.bindMu.Lock()
	defer p.bindMu.Unlock()
	snap := make([]capacity.EndpointCapacity, 0, len(endpoints))
	for _, ep := range endpoints {
		if c, ok := p.tracker.Observe(ep); ok {
			snap = append(snap, c)
		}
	}
	decision, ok := p.selector.Select(snap, req, opts)
	if ok {
		p.tracker.Charge(req.ID, decision.Endpoint, decision.ChargeBlocks)
	}
	return decision, ok
}

// Filter keeps the single endpoint chosen for the request, or none.
func (p *Plugin) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)
	if bound, ok := fwksched.ReadRequestAttribute[Decision](request, BoundDecisionKey); ok {
		if ep := find(endpoints, bound.Endpoint); ep != nil {
			if charged, _, ok := p.tracker.Charged(request.RequestID); ok && charged == bound.Endpoint {
				trace := bound.Trace
				trace.Bound = true
				request.PutAttribute(traceKey, &trace)
				logger.V(logging.DEBUG).Info("Hetero placement kept bound endpoint", "endpoint", bound.Endpoint)
				return []fwksched.Endpoint{ep}
			}
		}
		p.tracker.Release(request.RequestID)
		logger.Info("Hetero placement bound endpoint is gone, selecting again", "endpoint", bound.Endpoint)
	}

	promptTokens, ok := capacity.PromptTokens(request)
	if !ok {
		trace := &Trace{Phase: phaseInitial, RequestID: request.RequestID, Refusal: RefusalUnknownPrompt}
		request.PutAttribute(traceKey, trace)
		logger.Info("Hetero placement refused", "refusal", trace.Refusal)
		return nil
	}
	sources := make([]capacity.Source, len(endpoints))
	for i, ep := range endpoints {
		sources[i] = ep
	}
	decision, ok := p.Bind(sources, Request{
		ID:           request.RequestID,
		PromptTokens: promptTokens,
		MaxTokens:    capacity.MaxTokens(request),
	}, Options{})
	trace := decision.Trace
	request.PutAttribute(traceKey, &trace)
	logger.Info("Hetero placement decision", "promptTokens", promptTokens, "context", trace.Context,
		"reason", trace.Reason, "pools", trace.Pools, "fallback", trace.Fallback,
		"selected", trace.Selected, "refusal", trace.Refusal, "scores", trace.Scores)
	if !ok {
		return nil
	}
	return []fwksched.Endpoint{find(endpoints, decision.Endpoint)}
}

// Pick runs every profile once.
func (p *Plugin) Pick(_ context.Context, _ *fwksched.InferenceRequest, profiles map[string]fwksched.SchedulerProfile,
	profileResults map[string]*fwksched.ProfileRunResult) map[string]fwksched.SchedulerProfile {
	if len(profiles) == len(profileResults) {
		return map[string]fwksched.SchedulerProfile{}
	}
	return profiles
}

// ProcessResults returns the single profile result, or the placement refusal
// as a typed error: context that no endpoint supports is a bad request, and
// missing capacity is resource exhaustion.
func (p *Plugin) ProcessResults(_ context.Context, request *fwksched.InferenceRequest,
	profileResults map[string]*fwksched.ProfileRunResult) (*fwksched.SchedulingResult, error) {
	if len(profileResults) != 1 {
		return nil, fmt.Errorf("'%s' supports a single scheduling profile, got %d", PlacementType, len(profileResults))
	}
	for name, result := range profileResults {
		if result != nil {
			return &fwksched.SchedulingResult{ProfileResults: profileResults, PrimaryProfileName: name}, nil
		}
		// A filter after this plugin may have removed the charged endpoint.
		p.tracker.Release(request.RequestID)
		trace, ok := fwksched.ReadRequestAttribute[*Trace](request, traceKey)
		if !ok || trace.Refusal == "" {
			return nil, fmt.Errorf("failed to run scheduler profile '%s'", name)
		}
		headers := map[string]string{TraceHeader: encode(trace)}
		code := errcommon.BadRequest
		if trace.Refusal == RefusalNoCapacity {
			code = errcommon.ResourceExhausted
			headers[errcommon.RequestDroppedReasonHeaderKey] = string(errcommon.RequestDroppedReasonSaturated)
		}
		return nil, errcommon.Error{Code: code, Msg: "hetero placement refused: " + trace.Refusal, Headers: headers}
	}
	return nil, errors.New("unreachable")
}

// PreRequest charges the endpoint the request is sent to when it differs from
// the one charged during filtering.
func (p *Plugin) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	if request == nil || result == nil {
		return nil
	}
	primary := result.ProfileResults[result.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return nil
	}
	target := primary.TargetEndpoints[0]
	name := capacity.EndpointName(target.GetMetadata())
	if charged, _, ok := p.tracker.Charged(request.RequestID); ok && charged == name {
		return nil
	}
	observed, ok := p.tracker.Observe(target)
	if !ok {
		return nil
	}
	promptTokens, _ := capacity.PromptTokens(request)
	blocks := p.policy.ChargeBlocks(promptTokens, capacity.MaxTokens(request), observed.BlockSize)
	p.tracker.Charge(request.RequestID, name, blocks)
	log.FromContext(ctx).V(logging.DEBUG).Info("Hetero placement charged the dispatched endpoint", "endpoint", name, "blocks", blocks)
	return nil
}

// ResponseHeader adds the decision trace to the response.
func (p *Plugin) ResponseHeader(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || response.Headers == nil {
		return
	}
	if trace, ok := fwksched.ReadRequestAttribute[*Trace](request, traceKey); ok {
		response.Headers[TraceHeader] = encode(trace)
	}
}

// ResponseBody releases the charge when the response ends for any reason.
func (p *Plugin) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	p.tracker.Release(request.RequestID)
}

func find(endpoints []fwksched.Endpoint, name string) fwksched.Endpoint {
	for _, ep := range endpoints {
		if capacity.EndpointName(ep.GetMetadata()) == name {
			return ep
		}
	}
	return nil
}

func encode(trace *Trace) string {
	body, err := json.Marshal(trace)
	if err != nil {
		return ""
	}
	return string(body)
}
