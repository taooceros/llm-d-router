[![Go Report Card](https://goreportcard.com/badge/github.com/llm-d/llm-d-router)](https://goreportcard.com/report/github.com/llm-d/llm-d-router)
[![Go Reference](https://pkg.go.dev/badge/github.com/llm-d/llm-d-router.svg)](https://pkg.go.dev/github.com/llm-d/llm-d-router)
[![License](https://img.shields.io/github/license/llm-d/llm-d-router)](./LICENSE)
[![Join Slack](https://img.shields.io/badge/Join_Slack-blue?logo=slack)](https://llm-d.slack.com/archives/C08SBNRRSBD)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fllm-d%2Fllm-d-router.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fllm-d%2Fllm-d-router?ref=badge_shield)

# llm-d Router

> [!IMPORTANT]
> **Terminology Change**: The *Inference Scheduler* has been renamed to **llm-d Router**; see [Terminology](README.md#terminology).

> [!IMPORTANT]
> **API & Code Consolidation**: Core Endpoint Picker (EPP) code and the `InferenceObjective` and `InferenceModelRewrite` APIs have been merged into this repository from [Gateway API Inference Extension (GIE)]. The GIE repository now exclusively hosts the `InferencePool` API—an extension of the [Kubernetes Gateway API]—and defines the Endpoint Picker Protocol.

The **llm-d Router** is the intelligent entry point for inference traffic, delivering LLM load and prefix-cache aware routing, request prioritization, and advanced flow control across diverse request formats to fulfill complex serving objectives. It supports a flexible deployment model: it can run in **Standalone Mode**, where a self-managed Envoy proxy runs either alongside the EPP or as a separate service, or integrate with L7 load balancers—including self-managed instances (e.g., Istio, AgentGateway) and cloud-managed services (e.g., Google Cloud's Application Load Balancer)—via the Kubernetes Gateway API.

The router achieves its intelligence through an **Endpoint Picker (EPP)** that integrates with production-grade proxies (such as [Envoy]) via the [ext-proc] protocol, injecting real-time signals into the data plane to optimize request placement.

<p align="center">
  <img src="docs/images/llm-d-router.svg" width="800" alt="llm-d Router Architecture">
</p>

## Core Components and APIs

This repository hosts the following core components:

- **Endpoint Picker (EPP)**: The intelligent routing engine that serves as the "brain" of the router. It evaluates incoming requests against the current state of the [InferencePool], considering factors like KV-cache locality, current load, and priority to make optimal placement decisions. It integrates with L7 proxies via the `ext-proc` protocol.
- **Request Management APIs**: These resources directly influence the EPP's request handling behavior:
    - **InferenceObjective**: Configures the EPP's scheduling goals for specific requests, including priority levels and performance targets.
    - **InferenceModelRewrite**: Directs the EPP to perform model name rewriting, enabling flexible traffic management for A/B testing and canary rollouts.
- **Disaggregation Sidecar**: A coordination component deployed alongside model servers (typically as a sidecar to the decode worker). It orchestrates complex multi-stage inference lifecycles, such as **P/D (Prefill/Decode)** and **E/P/D (Encode/Prefill/Decode)**, by communicating with specialized encode and prefill workers to manage KV-cache and embedding transfers. For more details, see the [Disaggregation Documentation].

## Modes of Operation

The llm-d Router supports two primary deployment modes as specified in the [Kubernetes Gateway API Inference Extensions]:

### 1. Standalone Mode
A deployment with a self-managed Envoy proxy that does not require Gateway API infrastructure. The standalone Helm chart supports two Envoy proxy topologies:

- **Sidecar mode** (default): The proxy runs in the EPP pod. This topology is suited to basic testing and local evaluations.
- **Service mode**: The proxy runs as a separate, horizontally scalable Deployment and Service and reaches EPP through the EPP Service. Set `router.proxy.mode=service` to scale the proxy independently from EPP.

See the [Helm chart documentation] for configuration examples.

### 2. Gateway Mode (Inference Gateway)
The recommended mode for production environments, leveraging the official [Gateway API]. In this mode, the EPP acts as a backend for an `InferencePool`, which is referenced by an `HTTPRoute` on a shared `Gateway`. This enables advanced traffic management, multi-cluster load balancing, and shared infrastructure for both inference and traditional workloads.

For more details on the router architecture, routing logic, and different plugins (filters and scorers), see the [Architecture Documentation]. For resource provisioning and container sizing recommendations under heavy or long-context workloads, see the [EPP Container Sizing Guide]. The [OpenTelemetry JSON stdout logs] document describes the log record format used by the EPP and routing sidecar.

---

> [!NOTE]
> The project provides tools for automatic Envoy installation. However, if you install or
> configure it yourself, please note that the only supported [request_body_mode and response_body_mode](https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto)
> is `FULL_DUPLEX_STREAMED`

## Terminology

To ensure clarity across the project, we use the following standard terminology:

- **llm-d Router**: The complete intelligent entry point, comprising both the **Proxy** (e.g., Envoy) and the **Endpoint Picker (EPP)**. This term replaces "Inference Scheduler" in all contexts.
- **llm-d Endpoint Picker (EPP)**: The specific component that implements the routing intelligence and scoring logic. Use this term when referring to capabilities or configurations specific to the EPP itself, rather than the request routing system as a whole.
- **Inference Gateway**: A synonym for the **llm-d Router** when operating in **Gateway Mode**.
- **Request Scheduler**: A sub-component within the EPP responsible for the queuing and dispatching of requests.

[Kubernetes]:https://kubernetes.io
[Kubernetes Gateway API]:https://gateway-api.sigs.k8s.io/
[Architecture Documentation]:docs/architecture.md
[Disaggregation Documentation]:docs/disaggregation.md
[EPP Container Sizing Guide]:docs/operations.md
[OpenTelemetry JSON stdout logs]:docs/otel-json-stdout.md
[InferencePool]:https://github.com/kubernetes-sigs/gateway-api-inference-extension
[Gateway API Inference Extension (GIE)]:https://github.com/kubernetes-sigs/gateway-api-inference-extension
[Kubernetes Gateway API Inference Extensions]:https://github.com/kubernetes-sigs/gateway-api-inference-extension
[Gateway API]:https://github.com/kubernetes-sigs/gateway-api
[Envoy]:https://github.com/envoyproxy/envoy
[ext-proc]:https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter
[Helm chart documentation]:config/charts/README.md

## Contributing

Start with the [llm-d organization contributing guide][org-contributing] for project-wide guidelines, code of conduct, and community resources, then see [CONTRIBUTING.md](CONTRIBUTING.md) for what is specific to this repository, including how to claim an issue.

Our community meeting is bi-weekly at Wednesday 10AM PDT ([Google Meet], [Meeting Notes]).

We currently utilize the [#sig-router] channel in llm-d Slack workspace for communications.

For large changes please [create an issue] first describing the change so the
maintainers can do an assessment, and work on the details with you. See
[DEVELOPMENT.md](DEVELOPMENT.md) for details on how to work with the codebase.

Contributions are welcome!

[org-contributing]:https://github.com/llm-d/llm-d/blob/main/CONTRIBUTING.md
[create an issue]:https://github.com/llm-d/llm-d-router/issues/new
[discussion]:https://github.com/llm-d/llm-d-router/discussions/new?category=q-a
[Slack]:https://llm-d.slack.com/
[Google Meet]:https://meet.google.com/ozx-goao-cxh
[Meeting Notes]:https://docs.google.com/document/d/1Pf3x7ZM8nNpU56nt6CzePAOmFZ24NXDeXyaYb565Wq4
[#sig-router]:https://llm-d.slack.com/?redir=%2Fmessages%2Fsig-router

## Security
See [SECURITY.md](SECURITY.md) for vulnerability reporting. Published container images carry a signed provenance attestation and an SBOM (software bill of materials). See [Verifying Published Artifacts](docs/verifying-releases.md) for how to check them.

## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fllm-d%2Fllm-d-router.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fllm-d%2Fllm-d-router?ref=badge_large)

## Experimental heterogeneous-serving research

The [experimental subtree](experimental/heterogeneous-serving/) preserves the
Python Envoy ExtProc gateway from the heterogeneous TPU serving research
snapshot. **It is separate from the upstream Go router/EPP, not a native Go
plugin or an upstream integration.** The copied gateway implements fixed-layout
admission policy, ownership epochs, discovery and request handoff. Its original
Python imports are retained; this subtree alone is not a standalone package.

The research compares fixed layouts on equal 32-chip TPU v5e resources: four
TP8 instances, two TP16 instances, and a TP8 + TP8 + TP16 mixture. It does not
implement online topology resizing or establish heterogeneous accelerator-type
support. Historical FP32-state results belong to the original frozen runtime,
not a physically requalified native upstream port. In particular, the 24
migration-enabled timed runs made **zero migration proposals**: the observed
active-source free-block minimum, 13/102, stayed above the frozen 0.12 pressure
threshold. They cannot establish an end-to-end migration or KV-transport
benefit, nor show that executing migration cannot help. See the copied research
README and compact historical evidence in the sibling llm-d and llm-d-async
experimental subtrees; policy thresholds were not retuned for migration.

### Assembly and local verification

Check out all five forks (`vllm`, `tpu-inference`, `llm-d-router`, `llm-d`, and
`llm-d-async`) as sibling directories on
`research/heterogeneous-tpu-20260918`. From this repository root, use the
cross-fork assembler owned by the llm-d experimental subtree:

```sh
python3 ../llm-d/experimental/heterogeneous-serving/assemble.py --forks-root .. --check
python3 ../llm-d/experimental/heterogeneous-serving/assemble.py --forks-root .. --output /tmp/hetero-workspace
```

The output must be a new directory outside the fork checkout root. Assembly
verifies each manifest-listed source hash, preserves the flat research
workspace, and records `workspace-provenance.json`; it does not fall back to
the original research checkout. [migration.json](experimental/heterogeneous-serving/migration.json)
records the exact copied gateway and test hashes.

The [migration verification receipt](experimental/heterogeneous-serving/verification.json)
records 25 passing existing checks in three isolated processes: discovery (5),
handoff/ownership/policy (12), and lifecycle (8). The latter exercises a real
localhost gRPC ExtProc wire with injected control-plane peers, not accelerator
execution or a live cluster. All loaded project modules were checked to come
from the assembled copies. Reproduce these scoped checks in an isolated Python
environment with `xds-protos==1.84.0`, `ray`, `fastapi`, and `httpx` installed
(the receipt records tested versions):

```sh
cd /tmp/hetero-workspace
for module in test_gateway_discovery test_gateway_handoff test_gateway_lifecycle; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/tests" python3 -m unittest -v "$module" || exit 1
done
```

### Scoped journal fault envelope

The ownership journal supports coordinator process loss only with the same
journal/lock path and unchanged live engine generations, plus delayed or
duplicate RPCs and lost acknowledgements. Complete records are fsynced before
recoverability-barrier RPCs; torn final appends are discarded, malformed complete
records fail closed, and write/fsync errors latch uncertainty rather than
authorize rollback. The file lock is a local single-writer guard, **not a
distributed lease**. Missing journals beside lock markers, corruption,
unavailable receipts, and replaced engines refuse new traffic. Loss of the
entire directory or node requires operator retirement/drain of the old fabric
before creating a new ledger; the default `/tmp` path does not survive machine
replacement. Interrupted HTTP responses are not reconstructed: cancellation or
an explicitly idle-proven reset is required. Native tensors never enter the
journal. These are scoped protocol guarantees, not general service HA claims.

