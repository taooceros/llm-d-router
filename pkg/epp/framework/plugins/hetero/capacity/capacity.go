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
	"time"

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
	GenerationLabel  = "omp.llm-d.ai/generation"
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
	Name string
	// ID is the endpoint's datastore identity and Generation its generation
	// label; together they tell a replacement engine from the one a decision
	// was made against.
	ID          string
	Generation  string
	Role        Role
	TP          int
	MaxModelLen int
	// MissingMetrics marks an endpoint whose capacity metrics are not yet
	// scraped. Only its label facts are set; it has no room.
	MissingMetrics bool
	BlockSize      int
	TotalBlocks    int
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
	DynamicLongReserve         bool    `json:"dynamicLongReserve"`
}

// DefaultPolicy returns the RoutingPolicy defaults.
func DefaultPolicy() Policy {
	return Policy{
		SmallPromptTokenLimit:      4096,
		LargeFreshTrafficShare:     0.25,
		ReserveLargeHeadroomBlocks: 64,
		AdmissionGrowthBlocks:      1,
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

// Reservation is the projection held for one placed request.
type Reservation struct {
	Endpoint string
	Blocks   int
	// Large marks a reservation counted toward the large fresh-traffic quota.
	Large bool
	// Expires is the lease deadline of an unconfirmed reservation. It is zero
	// once the reservation is confirmed.
	Expires time.Time
}

// Pending reports whether the reservation is an unconfirmed lease.
func (r Reservation) Pending() bool { return !r.Expires.IsZero() }

type projection struct {
	blocks   int
	requests int
}

// Tracker holds the blocks reserved for placed requests, keyed by a
// reservation token unique to each live request. A reservation starts as a
// lease that is dropped at its deadline unless confirmed, so a request that
// fails before the router can observe its end does not hold blocks forever.
// It is safe for concurrent use.
type Tracker struct {
	mu           sync.Mutex
	now          func() time.Time
	onExpire     func(token string, r Reservation)
	reservations map[string]Reservation
	leases       map[string]time.Time
	nextExpiry   time.Time
	perEndpoint  map[string]projection
	version      uint64
}

// NewTracker returns an empty Tracker.
func NewTracker() *Tracker {
	return &Tracker{
		now:          time.Now,
		reservations: map[string]Reservation{},
		leases:       map[string]time.Time{},
		perEndpoint:  map[string]projection{},
	}
}

// SetClock replaces the clock used for lease deadlines.
func (t *Tracker) SetClock(now func() time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.now = now
}

// OnExpire sets the function called, outside the tracker lock, for every lease
// dropped at its deadline.
func (t *Tracker) OnExpire(fn func(token string, r Reservation)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.onExpire = fn
}

// Lease reserves blocks on endpoint for token until ttl passes without Confirm.
// Reserving a token again replaces its previous reservation.
func (t *Tracker) Lease(token, endpoint string, blocks int, large bool, ttl time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	expires := t.now().Add(ttl)
	t.putLocked(token, Reservation{Endpoint: endpoint, Blocks: blocks, Large: large, Expires: expires})
	t.leases[token] = expires
	if t.nextExpiry.IsZero() || expires.Before(t.nextExpiry) {
		t.nextExpiry = expires
	}
}

// Charge reserves blocks on endpoint for token without a deadline. Charging a
// token again replaces its previous reservation.
func (t *Tracker) Charge(token, endpoint string, blocks int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.putLocked(token, Reservation{Endpoint: endpoint, Blocks: blocks})
}

// Confirm removes the deadline of token's lease and reports whether token
// holds a reservation.
func (t *Tracker) Confirm(token string) bool {
	t.Expire()
	t.mu.Lock()
	defer t.mu.Unlock()
	r, ok := t.reservations[token]
	if !ok {
		return false
	}
	if r.Pending() {
		r.Expires = time.Time{}
		t.reservations[token] = r
		delete(t.leases, token)
	}
	return true
}

// Release drops the reservation of token and returns it.
func (t *Tracker) Release(token string) (Reservation, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	r, ok := t.reservations[token]
	if ok {
		t.dropLocked(token, r)
	}
	return r, ok
}

// Reservation returns the reservation held by token.
func (t *Tracker) Reservation(token string) (Reservation, bool) {
	t.Expire()
	t.mu.Lock()
	defer t.mu.Unlock()
	r, ok := t.reservations[token]
	return r, ok
}

// Charged returns the endpoint and blocks reserved for token.
func (t *Tracker) Charged(token string) (endpoint string, blocks int, ok bool) {
	r, ok := t.Reservation(token)
	return r.Endpoint, r.Blocks, ok
}

// Projected returns the blocks and requests reserved on endpoint, after
// dropping expired leases.
func (t *Tracker) Projected(endpoint string) (blocks, requests int) {
	t.Expire()
	t.mu.Lock()
	defer t.mu.Unlock()
	row := t.perEndpoint[endpoint]
	return row.blocks, row.requests
}

// Expire drops the leases whose deadline has passed and returns how many.
func (t *Tracker) Expire() int {
	t.mu.Lock()
	now := t.now()
	if len(t.leases) == 0 || now.Before(t.nextExpiry) {
		t.mu.Unlock()
		return 0
	}
	type expired struct {
		token string
		r     Reservation
	}
	var dropped []expired
	t.nextExpiry = time.Time{}
	for token, deadline := range t.leases {
		if now.Before(deadline) {
			if t.nextExpiry.IsZero() || deadline.Before(t.nextExpiry) {
				t.nextExpiry = deadline
			}
			continue
		}
		r := t.reservations[token]
		t.dropLocked(token, r)
		dropped = append(dropped, expired{token, r})
	}
	hook := t.onExpire
	t.mu.Unlock()
	if hook != nil {
		for _, e := range dropped {
			hook(e.token, e.r)
		}
	}
	return len(dropped)
}

// Reset drops every reservation and returns how many there were.
func (t *Tracker) Reset() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	n := len(t.reservations)
	t.reservations = map[string]Reservation{}
	t.leases = map[string]time.Time{}
	t.nextExpiry = time.Time{}
	t.perEndpoint = map[string]projection{}
	t.version++
	return n
}

// ProjectionState is the reservation total on one endpoint.
type ProjectionState struct {
	Blocks   int `json:"blocks"`
	Requests int `json:"requests"`
}

// State returns the reservation totals by endpoint and the number of
// unconfirmed leases.
func (t *Tracker) State() (perEndpoint map[string]ProjectionState, leases int) {
	t.Expire()
	t.mu.Lock()
	defer t.mu.Unlock()
	perEndpoint = make(map[string]ProjectionState, len(t.perEndpoint))
	for name, row := range t.perEndpoint {
		perEndpoint[name] = ProjectionState{Blocks: row.blocks, Requests: row.requests}
	}
	return perEndpoint, len(t.leases)
}

func (t *Tracker) putLocked(token string, r Reservation) {
	if old, ok := t.reservations[token]; ok {
		t.dropLocked(token, old)
	}
	t.reservations[token] = r
	row := t.perEndpoint[r.Endpoint]
	row.blocks += r.Blocks
	row.requests++
	t.perEndpoint[r.Endpoint] = row
	t.version++
}

func (t *Tracker) dropLocked(token string, r Reservation) {
	delete(t.reservations, token)
	delete(t.leases, token)
	row := t.perEndpoint[r.Endpoint]
	row.blocks = max(0, row.blocks-r.Blocks)
	row.requests = max(0, row.requests-1)
	if row.requests == 0 {
		delete(t.perEndpoint, r.Endpoint)
	} else {
		t.perEndpoint[r.Endpoint] = row
	}
	t.version++
}

// Version increases on every reservation change.
func (t *Tracker) Version() uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.version
}

// Observe computes the capacity of src. ok is false when src does not report
// free blocks, total blocks and block size; the result then carries only the
// label facts and MissingMetrics.
func (t *Tracker) Observe(src Source) (EndpointCapacity, bool) {
	meta := src.GetMetadata()
	if meta == nil {
		return EndpointCapacity{MissingMetrics: true}, false
	}
	labels := EndpointCapacity{
		Name:        EndpointName(meta),
		ID:          meta.ID.String(),
		Generation:  meta.Labels[GenerationLabel],
		Role:        Role(meta.Labels[RoleLabel]),
		TP:          atoi(meta.Labels[TPLabel]),
		MaxModelLen: atoi(meta.Labels[MaxModelLenLabel]),
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
		labels.MissingMetrics = true
		return labels, false
	}
	projectedBlocks, projectedRequests := t.Projected(labels.Name)
	running := metrics.RunningRequestsSize
	engineAdmissible := max(0, free-min(free, running))
	effectiveFree := max(0, total-max(total-free, projectedBlocks))
	c := labels
	c.BlockSize = blockSize
	c.TotalBlocks = total
	c.FreeBlocks = effectiveFree
	c.AdmissibleBlocks = min(engineAdmissible, effectiveFree)
	c.Running = running
	c.Waiting = metrics.WaitingQueueSize + max(0, projectedRequests-running)
	return c, true
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
