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
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
)

// LongestPromptFirstOrderingPolicyType is the registration type of the longest-prompt-first ordering policy. It
// dispatches the request with the most prompt tokens first and breaks ties by enqueue time. A request whose prompt
// length is unknown (not tokenized and not a token-ID prompt) counts as zero tokens.
const LongestPromptFirstOrderingPolicyType = "hetero-longest-prompt-first-ordering-policy"

// LongestPromptFirstFactory creates a longest-prompt-first ordering policy. It takes no parameters.
func LongestPromptFirstFactory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	if name == "" {
		name = LongestPromptFirstOrderingPolicyType
	}
	return &longestPromptFirst{typedName: plugin.TypedName{Type: LongestPromptFirstOrderingPolicyType, Name: name}}, nil
}

type longestPromptFirst struct {
	typedName plugin.TypedName
}

var _ flowcontrol.OrderingPolicy = &longestPromptFirst{}

func (p *longestPromptFirst) TypedName() plugin.TypedName { return p.typedName }

// Less reports whether a dispatches before b.
func (p *longestPromptFirst) Less(a, b flowcontrol.QueueItemAccessor) bool {
	if a == nil || b == nil {
		return a != nil
	}
	pa, pb := itemPromptTokens(a), itemPromptTokens(b)
	if pa != pb {
		return pa > pb
	}
	return a.EnqueueTime().Before(b.EnqueueTime())
}

func itemPromptTokens(item flowcontrol.QueueItemAccessor) int {
	req := item.OriginalRequest()
	if req == nil {
		return 0
	}
	n, _ := capacity.PromptTokens(req.InferenceRequest())
	return n
}
