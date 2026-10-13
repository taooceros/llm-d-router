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

package flowcontrol

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// DispatchGate is an optional interface of a SaturationDetector that decides dispatch per request.
//
// When the configured SaturationDetector implements DispatchGate, a dispatch cycle opens one DispatchPass and, for
// every priority band the UsageLimitPolicy leaves open (highest priority first), visits the queued items of all flows
// in the band in the band's OrderingPolicy order. Every item the pass admits is dispatched in the same cycle. An item
// the pass skips stays queued and the items behind it are still visited, so a request that fits no endpoint does not
// block requests that fit. FairnessPolicy is not consulted in this mode. Dispatch cycles run on the dispatch ticker
// only; an arrival does not start a cycle of its own.
//
// Conformance: BeginDispatchPass and every DispatchPass method are called from the single flow-control processor
// goroutine, one pass at a time.
type DispatchGate interface {
	// BeginDispatchPass starts a pass over the queued items against the given endpoint pool. queueChanged reports
	// whether items were added to or removed from the queues since the previous call. It returns nil when no queued
	// item can have become dispatchable since the previous pass, in which case the cycle visits no items.
	BeginDispatchPass(ctx context.Context, endpoints []datalayer.Endpoint, queueChanged bool) DispatchPass
}

// DispatchVerdict is a DispatchPass decision for one queued item.
type DispatchVerdict int

const (
	// DispatchSkip leaves the item queued; the pass continues with the next item.
	DispatchSkip DispatchVerdict = iota
	// DispatchAdmit dispatches the item. The pass has committed whatever resources the item needs.
	DispatchAdmit
	// DispatchStop leaves the item queued and ends the pass.
	DispatchStop
)

// DispatchPass is one ordered visit of the queued items.
type DispatchPass interface {
	// Admit decides whether item dispatches now.
	Admit(ctx context.Context, item QueueItemAccessor) DispatchVerdict
	// Revert releases what Admit committed for an item that was admitted but could not be dispatched, for example
	// because it was cancelled or expired concurrently.
	Revert(item QueueItemAccessor)
}
