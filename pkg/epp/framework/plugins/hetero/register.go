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

// Package hetero registers the heterogeneous TPU serving plugins.
package hetero

import (
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/latebinding"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/hetero/placement"
)

// Register registers the factories of all heterogeneous serving plugins.
func Register() {
	// Alpha
	fwkplugin.Register(placement.PlacementType, fwkplugin.StabilityAlpha, placement.Factory)
	fwkplugin.Register(latebinding.LongestPromptFirstOrderingPolicyType, fwkplugin.StabilityAlpha, latebinding.LongestPromptFirstFactory)
	fwkplugin.RegisterWithPluginDependencies(latebinding.GateType, fwkplugin.StabilityAlpha, latebinding.GateFactory, latebinding.GateConfigParser)
}
