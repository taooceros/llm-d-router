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

package latebinding

import (
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
)

// The deployed engines: TP8 serves 412 blocks of 64 tokens at max-model-len 16384, TP16 serves 282 blocks of 256
// tokens at max-model-len 36864.
const (
	smallTotal  = 412
	smallBlock  = 64
	smallMaxLen = 16384
	largeTotal  = 282
	largeBlock  = 256
	largeMaxLen = 36864
)

var testFlow = flowcontrol.FlowKey{ID: "flow", Priority: 0}

func endpoint(name string, role capacity.Role, maxLen, block, total, free int) *datalayer.ModelServer {
	tp := "8"
	if role == capacity.RoleLarge {
		tp = "16"
	}
	ep := datalayer.NewEndpoint(&datalayer.EndpointMetadata{
		ID: types.NamespacedName{Name: name, Namespace: "default"},
		Labels: map[string]string{
			capacity.InstanceIDLabel:  name,
			capacity.RoleLabel:        string(role),
			capacity.TPLabel:          tp,
			capacity.MaxModelLenLabel: strconv.Itoa(maxLen),
		},
	}, datalayer.NewMetrics())
	ep.GetAttributes().Put(capacity.TotalBlocksKey, attrmetrics.ScalarMetricValue(total))
	ep.GetAttributes().Put(capacity.BlockSizeKey, attrmetrics.ScalarMetricValue(block))
	setFree(ep, free)
	return ep
}

func setFree(ep datalayer.Endpoint, free int) {
	ep.GetAttributes().Put(capacity.FreeBlocksKey, attrmetrics.ScalarMetricValue(free))
}

func small(name string, free int) *datalayer.ModelServer {
	return endpoint(name, capacity.RoleSmall, smallMaxLen, smallBlock, smallTotal, free)
}

func large(name string, free int) *datalayer.ModelServer {
	return endpoint(name, capacity.RoleLarge, largeMaxLen, largeBlock, largeTotal, free)
}

// testPolicy mirrors the policy of tests/test_mixed_length_policy.py in hetero-research.
func testPolicy() capacity.Policy {
	p := capacity.DefaultPolicy()
	p.SmallPromptTokenLimit = 16384
	p.LargeFreshTrafficShare = 0.5
	p.ReserveLargeHeadroomBlocks = 16
	p.DecodeLengthBlind = true
	return p
}

// inferenceRequest returns a completions request whose prompt is promptTokens token IDs.
func inferenceRequest(id string, promptTokens int) *fwksched.InferenceRequest {
	maxTokens := int64(256)
	return &fwksched.InferenceRequest{
		RequestID: id,
		Body: &fwkrh.InferenceRequestBody{
			Completions:     &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{TokenIDs: [][]uint32{make([]uint32, promptTokens)}}},
			MaxOutputTokens: &maxTokens,
		},
	}
}

func flowRequest(id string, promptTokens int) *mocks.MockFlowControlRequest {
	req := mocks.NewMockFlowControlRequest(100, id, testFlow)
	req.InferenceRequestV = inferenceRequest(id, promptTokens)
	req.ReceivedTimestampV = time.Now()
	return req
}

func queueItem(id string, promptTokens int, enqueued time.Time) *mocks.MockQueueItemAccessor {
	return &mocks.MockQueueItemAccessor{
		EnqueueTimeV:     enqueued,
		OriginalRequestV: flowRequest(id, promptTokens),
		HandleV:          &mocks.MockQueueItemHandle{},
	}
}
