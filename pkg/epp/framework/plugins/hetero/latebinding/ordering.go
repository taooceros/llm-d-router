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
	"fmt"
	"math"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/capacity"
)

// LongestPromptFirstOrderingPolicyType is the registration type of the longest-prompt-first ordering policy. It
// dispatches the request with the highest priority key first and breaks ties by enqueue time. The key is the prompt
// token count plus agingTokensPerSecond times the seconds the request has been queued; with the default aging of 0 the
// order is pure longest prompt first. A request whose prompt length is unknown (not tokenized and not a token-ID
// prompt) counts as zero tokens.
//
// Aging bounds how long newer, longer prompts can overtake a request: once it has waited P / agingTokensPerSecond
// seconds, no request arriving later with a prompt of P tokens or fewer dispatches before it. The key difference
// between two queued requests does not change while both wait, so the queue order stays valid over time.
const LongestPromptFirstOrderingPolicyType = "hetero-longest-prompt-first-ordering-policy"

type longestPromptFirstParameters struct {
	AgingTokensPerSecond float64 `json:"agingTokensPerSecond"`
}

// LongestPromptFirstFactory creates a longest-prompt-first ordering policy.
func LongestPromptFirstFactory(name string, raw *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var params longestPromptFirstParameters
	if raw != nil {
		if err := raw.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse %s parameters: %w", LongestPromptFirstOrderingPolicyType, err)
		}
	}
	if params.AgingTokensPerSecond < 0 || math.IsInf(params.AgingTokensPerSecond, 0) ||
		math.IsNaN(params.AgingTokensPerSecond) {
		return nil, fmt.Errorf("%s agingTokensPerSecond must be a finite non-negative number, got %v",
			LongestPromptFirstOrderingPolicyType, params.AgingTokensPerSecond)
	}
	if name == "" {
		name = LongestPromptFirstOrderingPolicyType
	}
	return &longestPromptFirst{
		typedName: plugin.TypedName{Type: LongestPromptFirstOrderingPolicyType, Name: name},
		aging:     params.AgingTokensPerSecond,
	}, nil
}

type longestPromptFirst struct {
	typedName plugin.TypedName
	aging     float64
}

var _ flowcontrol.OrderingPolicy = &longestPromptFirst{}

func (p *longestPromptFirst) TypedName() plugin.TypedName { return p.typedName }

// Less reports whether a dispatches before b. The aged keys compare as
// (prompt_a - prompt_b) - aging * (enqueue_a - enqueue_b), which does not depend on the current time.
func (p *longestPromptFirst) Less(a, b flowcontrol.QueueItemAccessor) bool {
	if a == nil || b == nil {
		return a != nil
	}
	ea, eb := a.EnqueueTime(), b.EnqueueTime()
	diff := float64(itemPromptTokens(a) - itemPromptTokens(b))
	if p.aging > 0 {
		diff -= p.aging * ea.Sub(eb).Seconds()
	}
	if diff != 0 {
		return diff > 0
	}
	return ea.Before(eb)
}

func itemPromptTokens(item flowcontrol.QueueItemAccessor) int {
	req := item.OriginalRequest()
	if req == nil {
		return 0
	}
	n, _ := capacity.PromptTokens(req.InferenceRequest())
	return n
}
