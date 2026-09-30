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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestLongestPromptFirst_Less(t *testing.T) {
	t.Parallel()
	p, err := LongestPromptFirstFactory("", nil, nil)
	require.NoError(t, err)
	policy := p.(flowcontrol.OrderingPolicy)

	now := time.Now()
	long := queueItem("long", 30000, now.Add(time.Second))
	short := queueItem("short", 500, now)
	shortLater := queueItem("short-later", 500, now.Add(2*time.Second))
	textPrompt := &mocks.MockQueueItemAccessor{
		EnqueueTimeV: now.Add(-time.Second),
		OriginalRequestV: &mocks.MockFlowControlRequest{IDV: "text", InferenceRequestV: &fwksched.InferenceRequest{
			Body: &fwkrh.InferenceRequestBody{Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: "hello"}}},
		}},
	}

	cases := []struct {
		name string
		a, b flowcontrol.QueueItemAccessor
		want bool
	}{
		{"a longer prompt dispatches first even when it arrived later", long, short, true},
		{"a shorter prompt dispatches after a longer one", short, long, false},
		{"equal prompts dispatch in arrival order", short, shortLater, true},
		{"equal prompts arriving later dispatch later", shortLater, short, false},
		{"an item is not before itself", short, short, false},
		{"a prompt of unknown length counts as zero tokens", textPrompt, short, false},
		{"a known prompt dispatches before a prompt of unknown length", short, textPrompt, true},
		{"a non-nil item dispatches before nil", short, nil, true},
		{"nil dispatches after a non-nil item", nil, short, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.want, policy.Less(tc.a, tc.b))
		})
	}
}
