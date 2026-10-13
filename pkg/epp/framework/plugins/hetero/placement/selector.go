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
	"math"
	"sort"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
)

// Scorer weights of the queue-depth and KV-utilization signals.
const (
	queueScorerWeight = 2.0
	kvScorerWeight    = 2.0
)

// Refusal and reason values recorded in Trace.
const (
	RefusalNoContextSupport  = "no_instance_supports_context"
	RefusalNoCapacity        = "no_destination_has_admissible_capacity"
	RefusalUnknownPrompt     = "prompt_tokens_unknown"
	ReasonLongPrompt         = "prompt_does_not_fit_a_small_instance"
	ReasonLargeFreshTraffic  = "large_fresh_traffic_share"
	ReasonPreferSmall        = "prefer_small"
	phaseInitial             = "initial"
	scoreRoundingDenominator = 1e5
)

// Request is the part of a request initial placement reads.
type Request struct {
	// ID identifies the request in traces.
	ID string
	// Token keys the request's reservation; see ReservationToken.
	Token        string
	PromptTokens int
	// MaxTokens is ignored when the policy is decode-length blind.
	MaxTokens int
}

// Options adjusts one selection.
type Options struct {
	// LateBinding makes the fresh-traffic quota a preference among endpoints
	// with room: the second pool is always tried.
	LateBinding bool
	// ExtraHeld holds blocks back per endpoint name on top of the headroom.
	ExtraHeld map[string]int
}

// Trace records one placement decision.
type Trace struct {
	Phase               string             `json:"phase"`
	RequestID           string             `json:"request_id,omitempty"`
	PromptTokens        int                `json:"prompt_tokens"`
	MaxTokens           *int               `json:"max_tokens"`
	DecodeLengthBlind   bool               `json:"decode_length_blind"`
	Context             int                `json:"context"`
	Considered          []string           `json:"considered"`
	MissingMetrics      []string           `json:"missing_metrics,omitempty"`
	Eligible            []string           `json:"eligible"`
	FitsSmall           []string           `json:"fits_small,omitempty"`
	WorkConservingLarge bool               `json:"work_conserving_large"`
	Reason              string             `json:"reason,omitempty"`
	Pools               [][]string         `json:"pools,omitempty"`
	Fallback            bool               `json:"fallback,omitempty"`
	Refusal             string             `json:"refusal,omitempty"`
	Scores              map[string]float64 `json:"scores,omitempty"`
	Selected            string             `json:"selected,omitempty"`
	SelectedFreeBlocks  int                `json:"selected_free_blocks,omitempty"`
	LateBinding         bool               `json:"late_binding,omitempty"`
	ExtraHeld           map[string]int     `json:"dynamic_reserve_blocks,omitempty"`
	// Bound marks a request placed ahead of scheduling by late binding.
	Bound bool `json:"bound,omitempty"`
	// RebindReason is set when a bound decision could not be kept and the
	// request was placed again.
	RebindReason string `json:"rebind_reason,omitempty"`
}

// Decision is the outcome of a selection.
type Decision struct {
	Endpoint string
	// EndpointID and Generation identify the engine Endpoint named when the
	// decision was made.
	EndpointID    string
	Generation    string
	ContextTokens int
	// NeedBlocks is the admission need on Endpoint.
	NeedBlocks int
	// ChargeBlocks is the projection charged to Endpoint.
	ChargeBlocks int
	Long         bool
	// Options are the options the decision was made with.
	Options Options
	Trace   Trace
}

// Selector implements initial placement and owns the fresh-traffic quota
// counters. It is safe for concurrent use.
type Selector struct {
	policy capacity.Policy

	mu              sync.Mutex
	freshDispatched int
	freshToLarge    int
}

// NewSelector returns a Selector for policy.
func NewSelector(policy capacity.Policy) *Selector {
	return &Selector{policy: policy}
}

// Select chooses the endpoint for a fresh request from snap. Entries with
// MissingMetrics only decide whether some endpoint supports the context. The
// quota counters advance only when it returns true.
func (s *Selector) Select(snap []capacity.EndpointCapacity, req Request, opts Options) (Decision, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	policy := s.policy
	observed := make([]capacity.EndpointCapacity, 0, len(snap))
	var missing []capacity.EndpointCapacity
	for _, ep := range snap {
		if ep.MissingMetrics {
			missing = append(missing, ep)
		} else {
			observed = append(observed, ep)
		}
	}
	context := policy.ContextTokens(req.PromptTokens, req.MaxTokens, observed)
	trace := Trace{
		Phase:             phaseInitial,
		RequestID:         req.ID,
		PromptTokens:      req.PromptTokens,
		DecodeLengthBlind: policy.DecodeLengthBlind,
		Context:           context,
		Considered:        names(observed),
		LateBinding:       opts.LateBinding,
		ExtraHeld:         opts.ExtraHeld,
	}
	if len(missing) > 0 {
		trace.MissingMetrics = names(missing)
	}
	if !policy.DecodeLengthBlind {
		maxTokens := req.MaxTokens
		trace.MaxTokens = &maxTokens
	}
	long := policy.IsLong(req.PromptTokens)
	decision := Decision{ContextTokens: context, Long: long, Options: opts}

	var eligible, small, large, fitsSmall []capacity.EndpointCapacity
	for _, ep := range observed {
		if ep.MaxModelLen < context {
			continue
		}
		eligible = append(eligible, ep)
		switch ep.Role {
		case capacity.RoleSmall:
			small = append(small, ep)
			if capacity.NeedBlocks(ep, context) <= max(1, ep.TotalBlocks-1) {
				fitsSmall = append(fitsSmall, ep)
			}
		case capacity.RoleLarge:
			large = append(large, ep)
		}
	}
	trace.Eligible = names(eligible)
	if len(eligible) == 0 {
		// An endpoint that supports the context but has no metrics yet can
		// serve the request once it is scraped.
		trace.Refusal = RefusalNoContextSupport
		for _, ep := range missing {
			if ep.MaxModelLen >= context {
				trace.Refusal = RefusalNoCapacity
				break
			}
		}
		decision.Trace = trace
		return decision, false
	}
	trace.FitsSmall = names(fitsSmall)

	wantLarge := false
	if len(large) > 0 && policy.LargeFreshTrafficShare > 0 {
		target := policy.LargeFreshTrafficShare * float64(s.freshDispatched+1)
		wantLarge = float64(s.freshToLarge) < target
	}
	trace.WorkConservingLarge = wantLarge

	var pools [][]capacity.EndpointCapacity
	switch {
	case len(fitsSmall) == 0:
		pools = [][]capacity.EndpointCapacity{orElse(large, eligible)}
		trace.Reason = ReasonLongPrompt
	case wantLarge:
		pools = [][]capacity.EndpointCapacity{orElse(large, fitsSmall), fitsSmall}
		trace.Reason = ReasonLargeFreshTraffic
	default:
		pools = [][]capacity.EndpointCapacity{fitsSmall, large}
		trace.Reason = ReasonPreferSmall
	}
	if !policy.WorkConservingFallback && !opts.LateBinding {
		pools = pools[:1]
	}
	for _, pool := range pools {
		trace.Pools = append(trace.Pools, names(pool))
	}

	var pool []capacity.EndpointCapacity
	for index, option := range pools {
		pool = pool[:0]
		for _, ep := range option {
			if capacity.HasRoom(ep, context, long, policy, opts.ExtraHeld[ep.Name]) {
				pool = append(pool, ep)
			}
		}
		if len(pool) > 0 {
			trace.Fallback = index > 0
			break
		}
	}
	if len(pool) == 0 {
		trace.Refusal = RefusalNoCapacity
		decision.Trace = trace
		return decision, false
	}

	worstQueue := 1
	for _, ep := range snap {
		worstQueue = max(worstQueue, ep.Running+ep.Waiting)
	}
	type scored struct {
		score float64
		ep    capacity.EndpointCapacity
	}
	ranked := make([]scored, len(pool))
	for i, ep := range pool {
		ranked[i] = scored{score: score(ep, worstQueue), ep: ep}
	}
	// Highest score first; ties go to the lexically greatest name.
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].score != ranked[j].score {
			return ranked[i].score > ranked[j].score
		}
		return ranked[i].ep.Name > ranked[j].ep.Name
	})
	chosen := ranked[0].ep
	s.freshDispatched++
	if chosen.Role == capacity.RoleLarge {
		s.freshToLarge++
	}

	trace.Scores = make(map[string]float64, len(ranked))
	for _, r := range ranked {
		trace.Scores[r.ep.Name] = math.Round(r.score*scoreRoundingDenominator) / scoreRoundingDenominator
	}
	trace.Selected = chosen.Name
	trace.SelectedFreeBlocks = chosen.FreeBlocks

	decision.Endpoint = chosen.Name
	decision.EndpointID = chosen.ID
	decision.Generation = chosen.Generation
	decision.NeedBlocks = capacity.NeedBlocks(chosen, context)
	decision.ChargeBlocks = policy.ChargeBlocks(req.PromptTokens, req.MaxTokens, chosen.BlockSize)
	decision.Trace = trace
	return decision, true
}

// Rollback undoes the quota advance of one selection that was not dispatched.
func (s *Selector) Rollback(large bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.freshDispatched = max(0, s.freshDispatched-1)
	if large {
		s.freshToLarge = max(0, s.freshToLarge-1)
	}
}

// Reset zeroes the quota counters.
func (s *Selector) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.freshDispatched, s.freshToLarge = 0, 0
}

// Counters returns the fresh requests dispatched and those sent to large endpoints.
func (s *Selector) Counters() (dispatched, toLarge int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.freshDispatched, s.freshToLarge
}

func score(ep capacity.EndpointCapacity, worstQueue int) float64 {
	queue := 1.0
	if worstQueue > 0 {
		queue = max(0, 1-float64(ep.Waiting+ep.Running)/float64(worstQueue))
	}
	kv := 0.0
	if ep.TotalBlocks > 0 {
		kv = max(0, min(1, float64(ep.FreeBlocks)/float64(ep.TotalBlocks)))
	}
	return (queueScorerWeight*queue + kvScorerWeight*kv) / (queueScorerWeight + kvScorerWeight)
}

func orElse(a, b []capacity.EndpointCapacity) []capacity.EndpointCapacity {
	if len(a) > 0 {
		return a
	}
	return b
}

func names(eps []capacity.EndpointCapacity) []string {
	out := make([]string, len(eps))
	for i, ep := range eps {
		out[i] = ep.Name
	}
	return out
}
