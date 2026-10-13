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
// serving as one plugin: a filter that selects and reserves the endpoint, the
// profile handler that reports refusals, and the request-control hooks that
// keep the capacity projection in step with the request lifecycle.
package placement

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

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
	// DefaultBindLease is how long a reservation made by Bind survives without
	// the request reaching PreRequest.
	DefaultBindLease = 10 * time.Second
)

// Reasons a bound decision is not kept, recorded as Trace.RebindReason.
const (
	RebindEndpointGone      = "endpoint_gone"
	RebindEndpointReplaced  = "endpoint_replaced"
	RebindReservationLapsed = "reservation_lapsed"
)

var (
	// BoundDecisionKey is the request attribute holding a Decision made before
	// scheduling. The filter keeps the bound endpoint instead of selecting.
	BoundDecisionKey = plugin.NewDataKey("hetero/bound-decision", PlacementType)
	traceKey         = plugin.NewDataKey("hetero/route-trace", PlacementType)
	tokenKey         = plugin.NewDataKey("hetero/reservation-token", PlacementType)
	tokenSeq         atomic.Uint64
)

var (
	_ fwksched.Filter               = &Plugin{}
	_ fwksched.ProfileHandler       = &Plugin{}
	_ fwkrc.PreRequest              = &Plugin{}
	_ fwkrc.ResponseHeaderProcessor = &Plugin{}
	_ fwkrc.ResponseBodyProcessor   = &Plugin{}
	_ plugin.ConsumerPlugin         = &Plugin{}
	_ plugin.StateDumper            = &Plugin{}
)

// ReservationToken returns the token keying request's reservation, creating it
// on first use. The token is unique per InferenceRequest, so clients reusing a
// request ID across live requests do not share a reservation.
func ReservationToken(request *fwksched.InferenceRequest) string {
	if token, ok := fwksched.ReadRequestAttribute[string](request, tokenKey); ok {
		return token
	}
	token := "rsv-" + strconv.FormatUint(tokenSeq.Add(1), 10)
	request.PutAttribute(tokenKey, token)
	return token
}

type parameters struct {
	capacity.Policy
	// BindLeaseSeconds bounds how long a reservation is held for a request that
	// has not reached PreRequest.
	BindLeaseSeconds float64 `json:"bindLeaseSeconds"`
	// AdminAddress, when set, serves GET /hetero/state and POST /hetero/reset.
	AdminAddress string `json:"adminAddress"`
}

// Factory builds a Plugin from Policy parameters.
func Factory(name string, decoder *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	params := parameters{Policy: capacity.DefaultPolicy(), BindLeaseSeconds: DefaultBindLease.Seconds()}
	if decoder != nil {
		if err := decoder.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PlacementType, err)
		}
	}
	if err := validate(params); err != nil {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: %w", PlacementType, err)
	}
	p := New(name, params.Policy)
	p.bindLease = time.Duration(params.BindLeaseSeconds * float64(time.Second))
	if params.AdminAddress != "" {
		ctx := context.Background()
		if handle != nil {
			ctx = handle.Context()
		}
		if err := p.serveAdmin(ctx, params.AdminAddress); err != nil {
			return nil, fmt.Errorf("'%s' plugin admin server: %w", PlacementType, err)
		}
	}
	return p, nil
}

func validate(p parameters) error {
	switch {
	case p.LargeFreshTrafficShare < 0 || p.LargeFreshTrafficShare > 1:
		return errors.New("'largeFreshTrafficShare' must be within [0, 1]")
	case p.SmallPromptTokenLimit < 0, p.ReserveLargeHeadroomBlocks < 0, p.LongReserveTokens < 0, p.AdmissionGrowthBlocks < 0:
		return errors.New("token and block parameters must be non-negative")
	case p.BindLeaseSeconds <= 0:
		return errors.New("'bindLeaseSeconds' must be positive")
	}
	return nil
}

// New returns a Plugin with its own capacity tracker and selector.
func New(name string, policy capacity.Policy) *Plugin {
	p := &Plugin{
		typedName: plugin.TypedName{Type: PlacementType, Name: name},
		policy:    policy,
		tracker:   capacity.NewTracker(),
		selector:  NewSelector(policy),
		bindLease: DefaultBindLease,
	}
	p.tracker.OnExpire(func(_ string, r capacity.Reservation) { p.selector.Rollback(r.Large) })
	return p
}

// Plugin places fresh requests per the heterogeneous routing policy. It must
// be the last filter of its profile, since it reserves the endpoint it keeps.
//
// A reservation starts as a lease taken by Bind and becomes permanent in
// PreRequest, after which the director reports the end of the request to
// ResponseBody. A request that fails between the two never reaches either, so
// its lease lapses after the bind lease and rolls back the quota.
type Plugin struct {
	typedName plugin.TypedName
	policy    capacity.Policy
	tracker   *capacity.Tracker
	selector  *Selector
	bindLease time.Duration
	// bindMu makes observation, selection and reservation one step, so
	// concurrent requests do not select against the same free blocks.
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
// missing metric makes the endpoint unavailable instead of failing startup.
func (p *Plugin) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{Optional: capacity.ConsumedKeys()}
}

// Bind observes endpoints, selects one and leases its blocks to req.Token.
func (p *Plugin) Bind(endpoints []capacity.Source, req Request, opts Options) (Decision, bool) {
	p.bindMu.Lock()
	defer p.bindMu.Unlock()
	snap := make([]capacity.EndpointCapacity, 0, len(endpoints))
	for _, ep := range endpoints {
		c, _ := p.tracker.Observe(ep)
		snap = append(snap, c)
	}
	decision, ok := p.selector.Select(snap, req, opts)
	if ok {
		large := false
		for _, c := range snap {
			if c.Name == decision.Endpoint {
				large = c.Role == capacity.RoleLarge
				break
			}
		}
		p.tracker.Lease(req.Token, decision.Endpoint, decision.ChargeBlocks, large, p.bindLease)
	}
	return decision, ok
}

// Unbind drops request's reservation. A reservation that was never confirmed
// also rolls back its quota advance.
func (p *Plugin) Unbind(request *fwksched.InferenceRequest) {
	if token, ok := fwksched.ReadRequestAttribute[string](request, tokenKey); ok {
		p.release(token)
	}
}

func (p *Plugin) release(token string) {
	if r, ok := p.tracker.Release(token); ok && r.Pending() {
		p.selector.Rollback(r.Large)
	}
}

// Filter keeps the single endpoint chosen for the request, or none.
func (p *Plugin) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)
	token := ReservationToken(request)
	var opts Options
	rebind := ""
	if bound, ok := fwksched.ReadRequestAttribute[Decision](request, BoundDecisionKey); ok {
		ep, reason := p.boundEndpoint(endpoints, bound, token)
		if ep != nil {
			trace := bound.Trace
			trace.Bound = true
			request.PutAttribute(traceKey, &trace)
			logger.V(logging.DEBUG).Info("Hetero placement kept bound endpoint", "endpoint", bound.Endpoint)
			return []fwksched.Endpoint{ep}
		}
		p.release(token)
		opts, rebind = bound.Options, reason
		logger.Info("Hetero placement selecting again for a bound request", "endpoint", bound.Endpoint, "reason", reason)
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
		Token:        token,
		PromptTokens: promptTokens,
		MaxTokens:    capacity.MaxTokens(request),
	}, opts)
	trace := decision.Trace
	trace.RebindReason = rebind
	request.PutAttribute(traceKey, &trace)
	logger.Info("Hetero placement decision", "promptTokens", promptTokens, "context", trace.Context,
		"reason", trace.Reason, "pools", trace.Pools, "fallback", trace.Fallback,
		"selected", trace.Selected, "refusal", trace.Refusal, "scores", trace.Scores, "rebindReason", rebind)
	if !ok {
		return nil
	}
	return []fwksched.Endpoint{find(endpoints, decision.Endpoint)}
}

// boundEndpoint returns the candidate a bound decision named when it is the
// same engine and still holds the reservation, or the reason it cannot be kept.
func (p *Plugin) boundEndpoint(endpoints []fwksched.Endpoint, bound Decision, token string) (fwksched.Endpoint, string) {
	ep := find(endpoints, bound.Endpoint)
	if ep == nil {
		return nil, RebindEndpointGone
	}
	meta := ep.GetMetadata()
	if meta.ID.String() != bound.EndpointID || meta.Labels[capacity.GenerationLabel] != bound.Generation {
		return nil, RebindEndpointReplaced
	}
	if r, ok := p.tracker.Reservation(token); !ok || r.Endpoint != bound.Endpoint {
		return nil, RebindReservationLapsed
	}
	return ep, ""
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
		p.Unbind(request)
		return nil, fmt.Errorf("'%s' supports a single scheduling profile, got %d", PlacementType, len(profileResults))
	}
	for name, result := range profileResults {
		if result != nil {
			return &fwksched.SchedulingResult{ProfileResults: profileResults, PrimaryProfileName: name}, nil
		}
		// A filter after this plugin may have removed the reserved endpoint.
		p.Unbind(request)
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

// PreRequest confirms the reservation of the endpoint the request is sent to,
// reserving that endpoint when it differs from the one selected.
func (p *Plugin) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	if request == nil || result == nil {
		return nil
	}
	primary := result.ProfileResults[result.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return nil
	}
	token := ReservationToken(request)
	target := primary.TargetEndpoints[0]
	name := capacity.EndpointName(target.GetMetadata())
	if r, ok := p.tracker.Reservation(token); ok && r.Endpoint == name {
		p.tracker.Confirm(token)
		return nil
	}
	observed, ok := p.tracker.Observe(target)
	if !ok {
		return nil
	}
	promptTokens, _ := capacity.PromptTokens(request)
	blocks := p.policy.ChargeBlocks(promptTokens, capacity.MaxTokens(request), observed.BlockSize)
	p.tracker.Charge(token, name, blocks)
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

// ResponseBody releases the reservation when the response ends for any reason.
func (p *Plugin) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	p.Unbind(request)
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
