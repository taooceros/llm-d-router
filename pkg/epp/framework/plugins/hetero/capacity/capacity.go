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

// Package capacity derives per-endpoint KV block capacity for heterogeneous
// TPU serving from endpoint labels and scraped engine metrics, and keeps the
// router-side projection of blocks charged to requests that have been placed
// but not yet completed.
package capacity

import (
	"strconv"
	"sync"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

// Endpoint labels.
const (
	InstanceIDLabel  = "omp.llm-d.ai/instance-id"
	RoleLabel        = "omp.llm-d.ai/role"
	TPLabel          = "omp.llm-d.ai/tp"
	MaxModelLenLabel = "omp.llm-d.ai/max-model-len"
)

// Custom metric attribute keys configured on the core metrics extractor.
const (
	FreeBlocksAttribute  = "omp/kv-free-blocks"
	TotalBlocksAttribute = "omp/kv-total-blocks"
	BlockSizeAttribute   = "omp/kv-block-size"
)

var (
	FreeBlocksKey  = attrmetrics.ScalarMetricDataKey(FreeBlocksAttribute)
	TotalBlocksKey = attrmetrics.ScalarMetricDataKey(TotalBlocksAttribute)
	BlockSizeKey   = attrmetrics.ScalarMetricDataKey(BlockSizeAttribute)
)

// ConsumedKeys lists the endpoint attributes Observe reads, for plugins to
// declare in Consumes.
func ConsumedKeys() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{
		FreeBlocksKey:  attrmetrics.ScalarMetricValue(0),
		TotalBlocksKey: attrmetrics.ScalarMetricValue(0),
		BlockSizeKey:   attrmetrics.ScalarMetricValue(0),
	}
}

// Role is the engine size class read from RoleLabel.
type Role string

const (
	RoleSmall Role = "small"
	RoleLarge Role = "large"
)

// DefaultBlockSize is the block size used for the length-blind growth
// allowance when no endpoint reports one.
const DefaultBlockSize = 256

// Source is the endpoint view Observe reads. fwksched.Endpoint satisfies it;
// wrap a fwkdl.Endpoint with FromDatalayer.
type Source interface {
	GetMetadata() *fwkdl.EndpointMetadata
	GetMetrics() *fwkdl.Metrics
	Get(plugin.DataKey) (fwkdl.Cloneable, bool)
}

type datalayerSource struct{ fwkdl.Endpoint }

func (s datalayerSource) Get(key plugin.DataKey) (fwkdl.Cloneable, bool) {
	return s.GetAttributes().Get(key)
}

// FromDatalayer adapts a datalayer endpoint to Source.
func FromDatalayer(ep fwkdl.Endpoint) Source { return datalayerSource{ep} }

// EndpointName identifies an endpoint in charges, decisions and traces: the
// instance-id label when set, otherwise the endpoint ID.
func EndpointName(meta *fwkdl.EndpointMetadata) string {
	if meta == nil {
		return ""
	}
	if id := meta.Labels[InstanceIDLabel]; id != "" {
		return id
	}
	return meta.ID.String()
}

// EndpointCapacity is one endpoint's capacity after applying the projection.
type EndpointCapacity struct {
	Name        string
	Role        Role
	TP          int
	MaxModelLen int
	BlockSize   int
	TotalBlocks int
	// FreeBlocks is total - max(observed used, projected used), floored at 0.
	FreeBlocks int
	// AdmissibleBlocks is min(engine free - min(engine free, running), FreeBlocks).
	AdmissibleBlocks int
	Running          int
	// Waiting is the engine queue plus projected requests the engine does not
	// yet report as running.
	Waiting int
}

// Policy holds the routing parameters shared by placement and late binding.
// JSON names are the RoutingPolicy field names in camelCase.
type Policy struct {
	SmallPromptTokenLimit      int     `json:"smallPromptTokenLimit"`
	LargeFreshTrafficShare     float64 `json:"largeFreshTrafficShare"`
	ReserveLargeHeadroomBlocks int     `json:"reserveLargeHeadroomBlocks"`
	LongReserveTokens          int     `json:"longReserveTokens"`
	WorkConservingFallback     bool    `json:"workConservingFallback"`
	DecodeLengthBlind          bool    `json:"decodeLengthBlind"`
	AdmissionGrowthBlocks      int     `json:"admissionGrowthBlocks"`
	DispatchOrder              string  `json:"dispatchOrder"`
	DynamicLongReserve         bool    `json:"dynamicLongReserve"`
}

// Dispatch orders accepted in Policy.DispatchOrder.
const (
	DispatchArrival            = "arrival"
	DispatchLongestPromptFirst = "longest_prompt_first"
)

// DefaultPolicy returns the RoutingPolicy defaults.
func DefaultPolicy() Policy {
	return Policy{
		SmallPromptTokenLimit:      4096,
		LargeFreshTrafficShare:     0.25,
		ReserveLargeHeadroomBlocks: 64,
		AdmissionGrowthBlocks:      1,
		DispatchOrder:              DispatchArrival,
	}
}

// IsLong reports whether a prompt counts as a long request.
func (p Policy) IsLong(promptTokens int) bool {
	return promptTokens > p.SmallPromptTokenLimit
}

// ContextTokens is the context admission checks for. Length-blind policies
// add AdmissionGrowthBlocks of the largest block size in snap instead of
// maxTokens.
func (p Policy) ContextTokens(promptTokens, maxTokens int, snap []EndpointCapacity) int {
	if !p.DecodeLengthBlind {
		return promptTokens + maxTokens
	}
	blockSize := 0
	for _, ep := range snap {
		blockSize = max(blockSize, ep.BlockSize)
	}
	if blockSize == 0 {
		blockSize = DefaultBlockSize
	}
	return promptTokens + p.AdmissionGrowthBlocks*blockSize
}

// ChargeBlocks is the projection charged to the endpoint a request is placed on.
func (p Policy) ChargeBlocks(promptTokens, maxTokens, blockSize int) int {
	if p.DecodeLengthBlind {
		return BlocksFor(promptTokens, blockSize) + p.AdmissionGrowthBlocks
	}
	return BlocksFor(promptTokens+maxTokens, blockSize)
}

// BlocksFor is ceil(tokens / blockSize), or 0 for a non-positive block size.
func BlocksFor(tokens, blockSize int) int {
	if blockSize <= 0 {
		return 0
	}
	return (tokens + blockSize - 1) / blockSize
}

// NeedBlocks is the block count contextTokens occupies on ep.
func NeedBlocks(ep EndpointCapacity, contextTokens int) int {
	return BlocksFor(contextTokens, ep.BlockSize)
}

// HeadroomBlocks is the admissible capacity ep keeps back from a request.
// Large endpoints keep ReserveLargeHeadroomBlocks, plus LongReserveTokens for
// short requests unless DynamicLongReserve replaces the static reserve.
func HeadroomBlocks(ep EndpointCapacity, long bool, p Policy) int {
	if ep.Role != RoleLarge {
		return 0
	}
	headroom := p.ReserveLargeHeadroomBlocks
	if !long && !p.DynamicLongReserve {
		headroom += BlocksFor(p.LongReserveTokens, ep.BlockSize)
	}
	return headroom
}

// HasRoom reports whether ep can admit a request of contextTokens.
// extraHeldBlocks is held back on top of the headroom; late binding uses it for
// the dynamic long-request reserve.
func HasRoom(ep EndpointCapacity, contextTokens int, long bool, p Policy, extraHeldBlocks int) bool {
	return ep.AdmissibleBlocks-HeadroomBlocks(ep, long, p)-extraHeldBlocks >= NeedBlocks(ep, contextTokens)
}

// PromptTokens returns the prompt length of request from its tokenized form,
// or from a token-ID completions prompt when it has not been tokenized.
func PromptTokens(request *fwksched.InferenceRequest) (int, bool) {
	if request == nil || request.Body == nil {
		return 0, false
	}
	if n := request.Body.TokenizedRequest.TokenCount(); n > 0 {
		return n, true
	}
	if c := request.Body.Completions; c != nil {
		if n := c.Prompt.TokenCountHint(); n >= 0 {
			return n, true
		}
	}
	return 0, false
}

// MaxTokens returns the request's output token cap, 0 when absent.
func MaxTokens(request *fwksched.InferenceRequest) int {
	if request == nil || request.Body == nil || request.Body.MaxOutputTokens == nil {
		return 0
	}
	return int(*request.Body.MaxOutputTokens)
}

type charge struct {
	endpoint string
	blocks   int
}

type projection struct {
	blocks   int
	requests int
}

// Tracker holds the blocks charged to placed requests, keyed by request ID.
// It is safe for concurrent use.
type Tracker struct {
	mu          sync.Mutex
	charges     map[string]charge
	perEndpoint map[string]projection
	version     uint64
}

// NewTracker returns an empty Tracker.
func NewTracker() *Tracker {
	return &Tracker{
		charges:     map[string]charge{},
		perEndpoint: map[string]projection{},
	}
}

// Charge charges blocks on endpoint for requestID. Charging a requestID again
// replaces its previous charge.
func (t *Tracker) Charge(requestID, endpoint string, blocks int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if old, ok := t.charges[requestID]; ok {
		t.removeLocked(old)
	}
	c := charge{endpoint: endpoint, blocks: blocks}
	t.charges[requestID] = c
	row := t.perEndpoint[endpoint]
	row.blocks += blocks
	row.requests++
	t.perEndpoint[endpoint] = row
	t.version++
}

// Release drops the charge of requestID and reports whether one existed.
func (t *Tracker) Release(requestID string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	c, ok := t.charges[requestID]
	if !ok {
		return false
	}
	delete(t.charges, requestID)
	t.removeLocked(c)
	t.version++
	return true
}

func (t *Tracker) removeLocked(c charge) {
	row := t.perEndpoint[c.endpoint]
	row.blocks = max(0, row.blocks-c.blocks)
	row.requests = max(0, row.requests-1)
	if row.requests == 0 {
		delete(t.perEndpoint, c.endpoint)
		return
	}
	t.perEndpoint[c.endpoint] = row
}

// Charged returns the endpoint and blocks charged to requestID.
func (t *Tracker) Charged(requestID string) (endpoint string, blocks int, ok bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	c, ok := t.charges[requestID]
	return c.endpoint, c.blocks, ok
}

// Projected returns the blocks and requests charged to endpoint.
func (t *Tracker) Projected(endpoint string) (blocks, requests int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	row := t.perEndpoint[endpoint]
	return row.blocks, row.requests
}

// Version increases on every Charge and Release.
func (t *Tracker) Version() uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.version
}

// Observe computes the capacity of src. ok is false when src does not report
// free blocks, total blocks and block size.
func (t *Tracker) Observe(src Source) (EndpointCapacity, bool) {
	meta := src.GetMetadata()
	if meta == nil {
		return EndpointCapacity{}, false
	}
	metrics := src.GetMetrics()
	if metrics == nil {
		metrics = fwkdl.NewMetrics()
	}
	free, okFree := scalar(src, FreeBlocksKey)
	total, okTotal := scalar(src, TotalBlocksKey)
	if !okTotal {
		total, okTotal = metrics.CacheNumBlocks, metrics.CacheNumBlocks > 0
	}
	blockSize, okBlock := scalar(src, BlockSizeKey)
	if !okBlock {
		blockSize, okBlock = metrics.CacheBlockSize, metrics.CacheBlockSize > 0
	}
	if !okFree || !okTotal || !okBlock || total <= 0 || blockSize <= 0 {
		return EndpointCapacity{}, false
	}
	name := EndpointName(meta)
	projectedBlocks, projectedRequests := t.Projected(name)
	running := metrics.RunningRequestsSize
	engineAdmissible := max(0, free-min(free, running))
	effectiveFree := max(0, total-max(total-free, projectedBlocks))
	return EndpointCapacity{
		Name:             name,
		Role:             Role(meta.Labels[RoleLabel]),
		TP:               atoi(meta.Labels[TPLabel]),
		MaxModelLen:      atoi(meta.Labels[MaxModelLenLabel]),
		BlockSize:        blockSize,
		TotalBlocks:      total,
		FreeBlocks:       effectiveFree,
		AdmissibleBlocks: min(engineAdmissible, effectiveFree),
		Running:          running,
		Waiting:          metrics.WaitingQueueSize + max(0, projectedRequests-running),
	}, true
}

// Snapshot observes endpoints, skipping those without capacity metrics.
func (t *Tracker) Snapshot(endpoints []fwksched.Endpoint) []EndpointCapacity {
	snap := make([]EndpointCapacity, 0, len(endpoints))
	for _, ep := range endpoints {
		if c, ok := t.Observe(ep); ok {
			snap = append(snap, c)
		}
	}
	return snap
}

func scalar(src Source, key plugin.DataKey) (int, bool) {
	raw, ok := src.Get(key)
	if !ok {
		return 0, false
	}
	v, ok := raw.(attrmetrics.ScalarMetricValue)
	if !ok {
		return 0, false
	}
	return int(v), true
}

func atoi(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return n
}
