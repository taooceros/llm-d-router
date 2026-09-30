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

// Package latebinding holds the flow-control plugins that keep requests in the EPP queue until an endpoint has room
// for them: a dispatch gate that binds each queued request to an endpoint at dispatch time, and a
// longest-prompt-first ordering policy.
package latebinding

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/placement"
)

// GateType is the registration type of the late-binding dispatch gate.
//
// Configured as the flow-control saturation detector, the gate dispatches a queued request only when the referenced
// placement plugin can bind it to an endpoint with room, charges the endpoint at that moment, and hands the decision
// to placement through placement.BoundDecisionKey so scheduling routes to the same endpoint. A request that fits
// nowhere stays queued without blocking later requests. With the placement policy's dynamicLongReserve, the first
// long request in dispatch order that fits nowhere holds its block need on every eligible large endpoint against
// short requests for the rest of the pass.
const GateType = "hetero-late-binding-gate"

type gateParameters struct {
	// PlacementPluginRef names the placement plugin whose policy, selector and charge tracker the gate uses. The
	// pluginRef tag makes the config loader instantiate that plugin first.
	PlacementPluginRef string `json:"placementPluginRef" pluginRef:""`
}

// GateConfigParser parses the gate parameters.
func GateConfigParser(raw *json.Decoder, _ plugin.Handle) (any, error) {
	var params gateParameters
	if raw != nil {
		if err := raw.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse %s parameters: %w", GateType, err)
		}
	}
	if params.PlacementPluginRef == "" {
		return nil, fmt.Errorf("%s requires placementPluginRef", GateType)
	}
	return params, nil
}

// GateFactory creates a late-binding gate.
func GateFactory(name string, raw *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	parsed, err := GateConfigParser(raw, handle)
	if err != nil {
		return nil, err
	}
	params := parsed.(gateParameters)
	ref := handle.Plugin(params.PlacementPluginRef)
	if ref == nil {
		return nil, fmt.Errorf("placement plugin not found: %s", params.PlacementPluginRef)
	}
	binder, ok := ref.(*placement.Plugin)
	if !ok {
		return nil, fmt.Errorf("plugin %s is not a %T", params.PlacementPluginRef, binder)
	}
	if name == "" {
		name = GateType
	}
	return NewGate(name, binder), nil
}

// Binder is the placement surface the gate uses. *placement.Plugin implements it.
type Binder interface {
	Policy() capacity.Policy
	Tracker() *capacity.Tracker
	Bind(endpoints []capacity.Source, req placement.Request, opts placement.Options) (placement.Decision, bool)
}

// Gate is the late-binding dispatch gate. BeginDispatchPass and the pass methods run on the flow-control processor
// goroutine; Saturation may be called from any goroutine.
type Gate struct {
	typedName plugin.TypedName
	binder    Binder

	// Pass state, owned by the processor goroutine.
	sources  []capacity.Source
	snap     []capacity.EndpointCapacity
	lastSnap []capacity.EndpointCapacity
	havePass bool
}

var (
	_ flowcontrol.SaturationDetector = &Gate{}
	_ flowcontrol.DispatchGate       = &Gate{}
)

// NewGate returns a gate that binds through binder.
func NewGate(name string, binder Binder) *Gate {
	return &Gate{typedName: plugin.TypedName{Type: GateType, Name: name}, binder: binder}
}

func (g *Gate) TypedName() plugin.TypedName { return g.typedName }

// Saturation reports the fraction of the pool's KV blocks that are not admissible, after the placement projection.
// It is 1.0 when no endpoint reports capacity metrics, which stops dispatch before any pass.
func (g *Gate) Saturation(_ context.Context, endpoints []datalayer.Endpoint) float64 {
	tracker := g.binder.Tracker()
	var admissible, total int
	for _, ep := range endpoints {
		if c, ok := tracker.Observe(capacity.FromDatalayer(ep)); ok {
			admissible += c.AdmissibleBlocks
			total += c.TotalBlocks
		}
	}
	if total <= 0 {
		return 1.0
	}
	return 1.0 - float64(admissible)/float64(total)
}

// BeginDispatchPass returns nil when neither the queues nor any endpoint's projected capacity changed since the
// previous pass, or when no endpoint has room for even the smallest request.
func (g *Gate) BeginDispatchPass(ctx context.Context, endpoints []datalayer.Endpoint, queueChanged bool) flowcontrol.DispatchPass {
	g.sources = g.sources[:0]
	for _, ep := range endpoints {
		g.sources = append(g.sources, capacity.FromDatalayer(ep))
	}
	g.observe()
	unchanged := g.havePass && slices.Equal(g.snap, g.lastSnap)
	g.lastSnap = append(g.lastSnap[:0], g.snap...)
	g.havePass = true
	if unchanged && !queueChanged {
		return nil
	}
	policy := g.binder.Policy()
	if !anyRoom(g.snap, policy) {
		return nil
	}
	return &pass{gate: g, policy: policy, logger: log.FromContext(ctx)}
}

// observe fills g.snap with the projected capacity of g.sources, sorted by endpoint name so that snapshots compare
// independently of endpoint listing order.
func (g *Gate) observe() {
	tracker := g.binder.Tracker()
	g.snap = g.snap[:0]
	for _, src := range g.sources {
		if c, ok := tracker.Observe(src); ok {
			g.snap = append(g.snap, c)
		}
	}
	slices.SortFunc(g.snap, func(a, b capacity.EndpointCapacity) int { return cmp.Compare(a.Name, b.Name) })
}

// anyRoom reports whether some endpoint could admit the smallest possible request.
func anyRoom(snap []capacity.EndpointCapacity, policy capacity.Policy) bool {
	minContext := policy.ContextTokens(0, 0, snap)
	for _, ep := range snap {
		if capacity.HasRoom(ep, minContext, true, policy, 0) {
			return true
		}
	}
	return false
}

type pass struct {
	gate   *Gate
	policy capacity.Policy
	logger logr.Logger
	// held is the dynamic long reserve by large endpoint name, applied to short requests. It is set once per pass,
	// by the first long request that fits nowhere; nil means no long request is waiting unplaced.
	held map[string]int
	full bool
}

// Admit binds item through placement. A request whose prompt length is unknown is admitted unbound, so placement
// refuses it at scheduling instead of the request waiting in the queue indefinitely.
func (p *pass) Admit(_ context.Context, item flowcontrol.QueueItemAccessor) flowcontrol.DispatchVerdict {
	if p.full {
		return flowcontrol.DispatchStop
	}
	fcReq := item.OriginalRequest()
	request := fcReq.InferenceRequest()
	promptTokens, known := capacity.PromptTokens(request)
	if !known {
		return flowcontrol.DispatchAdmit
	}
	req := placement.Request{ID: fcReq.ID(), PromptTokens: promptTokens, MaxTokens: capacity.MaxTokens(request)}
	long := p.policy.IsLong(promptTokens)

	opts := placement.Options{LateBinding: true}
	if !long {
		opts.ExtraHeld = p.held
	}
	decision, ok := p.gate.binder.Bind(p.gate.sources, req, opts)
	if !ok {
		if long && p.policy.DynamicLongReserve && p.held == nil {
			p.held = p.gate.reserveFor(decision.ContextTokens)
			p.logger.V(logutil.DEBUG).Info("Holding dynamic long reserve", "requestID", req.ID,
				"promptTokens", promptTokens, "reserveBlocks", p.held)
		}
		return flowcontrol.DispatchSkip
	}
	request.PutAttribute(placement.BoundDecisionKey, decision)
	p.gate.observe()
	p.full = !anyRoom(p.gate.snap, p.policy)
	return flowcontrol.DispatchAdmit
}

// Revert releases the charge Bind made for item. Releasing an unbound request is a no-op.
func (p *pass) Revert(item flowcontrol.QueueItemAccessor) {
	p.gate.binder.Tracker().Release(item.OriginalRequest().ID())
}

// reserveFor returns the blocks a context of contextTokens needs on every large endpoint that could ever hold it:
// its max-model-len admits the context and the need fits within the endpoint's total blocks after headroom. The map
// is non-nil even when empty, so the first unplaced long request of a pass fixes the reserve.
func (g *Gate) reserveFor(contextTokens int) map[string]int {
	policy := g.binder.Policy()
	held := make(map[string]int, len(g.snap))
	for _, ep := range g.snap {
		if ep.Role != capacity.RoleLarge || ep.MaxModelLen < contextTokens {
			continue
		}
		need := capacity.NeedBlocks(ep, contextTokens)
		if need+capacity.HeadroomBlocks(ep, true, policy) > ep.TotalBlocks {
			continue
		}
		held[ep.Name] = need
	}
	return held
}
