"""Bridge the Ray serving-instance registry into the installed llm-d discovery.

The installed `InferencePool` CRD (`inference.networking.k8s.io/v1`, bundle
v1.4.0) supports exactly one discovery contract: `spec.selector.matchLabels`
over Pods in the same namespace, each endpoint addressed as
``podIP:targetPortNumber``.  There is no external-endpoint provider field, so a
Ray actor cannot be discovered directly.

Ray TPU worker pods run with ``hostNetwork: true``, so a pod's IP *is* its host
IP.  Each logical instance therefore exposes its single serving endpoint on its
coordinator host, and only that coordinator pod carries the pool label.  That
keeps the gateway-facing representation a plain Kubernetes label set while the
Ray registry supplies the actual engine identity, generation and lifecycle.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
DEFAULT_POOL_NAME = "omp-hetero"
INSTANCE_LABEL = "omp.llm-d.ai/instance-id"
GENERATION_LABEL = "omp.llm-d.ai/generation"
LAYOUT_LABEL = "omp.llm-d.ai/layout"


class InClusterKube:
    """Minimal read/patch client for the in-cluster Kubernetes API."""

    def __init__(self, namespace: Optional[str] = None) -> None:
        self.host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        self.port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        token_path = os.path.join(SA_DIR, "token")
        ns_path = os.path.join(SA_DIR, "namespace")
        ca_path = os.path.join(SA_DIR, "ca.crt")
        if not os.path.exists(token_path):
            raise RuntimeError(
                "no in-cluster service account token; the gateway must run inside the cluster"
            )
        with open(token_path, "r", encoding="utf-8") as handle:
            self.token = handle.read().strip()
        if namespace:
            self.namespace = namespace
        else:
            with open(ns_path, "r", encoding="utf-8") as handle:
                self.namespace = handle.read().strip()
        self.context = ssl.create_default_context(cafile=ca_path)

    def _request(self, method: str, path: str, body: Optional[Any] = None,
                 content_type: str = "application/json") -> Any:
        url = f"https://{self.host}:{self.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", content_type)
        with urllib.request.urlopen(request, context=self.context, timeout=20) as response:
            payload = response.read()
        return json.loads(payload) if payload else {}

    def get_inference_pool(self, name: str) -> Dict[str, Any]:
        return self._request(
            "GET",
            f"/apis/inference.networking.k8s.io/v1/namespaces/{self.namespace}"
            f"/inferencepools/{name}",
        )

    def list_pods(self, label_selector: str) -> List[Dict[str, Any]]:
        from urllib.parse import quote

        body = self._request(
            "GET",
            f"/api/v1/namespaces/{self.namespace}/pods?labelSelector={quote(label_selector)}",
        )
        return body.get("items", [])

    def label_pod(self, pod_name: str, labels: Dict[str, Optional[str]]) -> Dict[str, Any]:
        patch = {"metadata": {"labels": labels}}
        return self._request(
            "PATCH",
            f"/api/v1/namespaces/{self.namespace}/pods/{pod_name}",
            body=patch,
            content_type="application/merge-patch+json",
        )

    def pod_for_ip(self, ip: str) -> Optional[Dict[str, Any]]:
        # hostNetwork means multiple system Pods can share this IP. Never
        # label the first arbitrary match as a logical serving coordinator.
        matches = [pod for pod in self.list_pods("ray.io/node-type=worker")
                   if (pod.get("status", {}) or {}).get("podIP") == ip
                   and pod.get("spec", {}).get("hostNetwork") is True
                   and not pod.get("metadata", {}).get("deletionTimestamp")]
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous Ray coordinator pods for {ip}")
        return matches[0] if matches else None


def selector_string(match_labels: Dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(match_labels.items()))


class DiscoveryBridge:
    """Maps gateway-visible pod endpoints back to stable Ray instance ids."""

    def __init__(
        self,
        *,
        pool_name: str = DEFAULT_POOL_NAME,
        namespace: Optional[str] = None,
        kube: Optional[InClusterKube] = None,
    ) -> None:
        self.pool_name = pool_name
        self.kube = kube or InClusterKube(namespace)
        self._pool: Optional[Dict[str, Any]] = None

    def pool(self, refresh: bool = False) -> Dict[str, Any]:
        if self._pool is None or refresh:
            pool = self.kube.get_inference_pool(self.pool_name)
            spec = pool.get("spec", {})
            picker = spec.get("endpointPickerRef", {})
            if (picker.get("name") != "omp-gateway"
                    or picker.get("kind", "Service") != "Service"
                    or picker.get("group", "") != ""
                    or picker.get("port", {}).get("number") != 9002
                    or picker.get("failureMode", "FailClose") != "FailClose"):
                raise RuntimeError("InferencePool must reference omp-gateway gRPC :9002 FailClose")
            if not spec.get("selector", {}).get("matchLabels"):
                raise RuntimeError("InferencePool must have a nonempty logical-instance selector")
            if len(spec.get("targetPorts", [])) != 1:
                raise RuntimeError("logical instances require exactly one InferencePool target port")
            self._pool = pool
        return self._pool

    def target_port(self) -> int:
        return int(self.pool()["spec"]["targetPorts"][0]["number"])

    def endpoints(self) -> List[Dict[str, Any]]:
        """Kubernetes-visible coordinator endpoints; this makes no placement decision."""
        spec = self.pool()["spec"]
        selector = selector_string(spec["selector"]["matchLabels"])
        port = self.target_port()
        rows = []
        for pod in self.kube.list_pods(selector):
            status = pod.get("status", {}) or {}
            metadata = pod["metadata"]
            ip = status.get("podIP")
            if not ip:
                continue
            ready = not metadata.get("deletionTimestamp") and any(
                c.get("type") == "Ready" and c.get("status") == "True"
                for c in status.get("conditions", []) or []
            )
            rows.append({
                "pod": metadata["name"], "pod_ip": ip, "port": port,
                "endpoint": f"{ip}:{port}", "pod_ready": bool(ready),
                "labels": metadata.get("labels", {}),
                "coordinator_pod": pod.get("spec", {}).get("hostNetwork") is True
                    and metadata.get("labels", {}).get("ray.io/node-type") == "worker",
            })
        return sorted(rows, key=lambda row: row["pod"])

    def _records(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        identities, endpoints = set(), set()
        for row in rows:
            instance_id, endpoint = row["instance_id"], row.get("endpoint")
            if instance_id in identities or endpoint in endpoints:
                raise RuntimeError("duplicate logical instance or serving endpoint in registry")
            identities.add(instance_id)
            endpoints.add(endpoint)
            if (not endpoint or not row.get("endpoint_host")
                    or row.get("endpoint_port") != self.target_port()
                    or endpoint != f"{row['endpoint_host']}:{self.target_port()}"):
                raise RuntimeError(f"registry endpoint does not match InferencePool: {instance_id}")
            coordinators = [host for host in row.get("host_chip_rank_map", [])
                            if host.get("process_rank") == 0]
            if len(coordinators) != 1 or coordinators[0].get("node_ip") != row["endpoint_host"]:
                raise RuntimeError(f"endpoint is not the rank-zero coordinator: {instance_id}")
            if not row.get("layout_id") or not isinstance(row.get("generation"), int):
                raise RuntimeError(f"missing logical generation identity: {instance_id}")
        return rows

    @staticmethod
    def _identity(record: Dict[str, Any]) -> Dict[str, str]:
        return {INSTANCE_LABEL: record["instance_id"],
                GENERATION_LABEL: str(record["generation"]), LAYOUT_LABEL: record["layout_id"]}

    def resolve(
        self, registry_rows: List[Dict[str, Any]], *,
        maintenance_instances: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Require every non-maintenance identity and no extra labelled endpoint.

        Maintenance is an explicit gateway admission fence, not permission to
        ignore unhealthy peers, ambiguous identities, or stale target labels.
        Validate the complete registry before excluding declared targets.
        """
        excluded = set(maintenance_instances)
        records = [row for row in self._records(registry_rows)
                   if row["instance_id"] not in excluded]
        endpoints = self.endpoints()
        by_endpoint: Dict[str, List[Dict[str, Any]]] = {}
        for row in endpoints:
            by_endpoint.setdefault(row["endpoint"], []).append(row)
        resolved, unmatched_registry, mismatches = [], [], []
        for record in records:
            matches = by_endpoint.pop(record["endpoint"], [])
            if len(matches) != 1:
                unmatched_registry.append(record["instance_id"])
                if matches:
                    mismatches.append({"instance_id": record["instance_id"],
                                       "reason": "ambiguous_pod_endpoint"})
                continue
            match = matches[0]
            identity_matches = match["coordinator_pod"] and all(
                match["labels"].get(key) == value for key, value in self._identity(record).items()
            )
            if not identity_matches:
                mismatches.append({"instance_id": record["instance_id"],
                                   "reason": "pod_logical_identity_mismatch", "pod": match["pod"]})
            resolved.append({
                "instance_id": record["instance_id"], "generation": record["generation"],
                "layout_id": record["layout_id"], "endpoint": record["endpoint"],
                "pod": match["pod"], "pod_ready": match["pod_ready"],
                "identity_matches": identity_matches, "registry_state": record["state"],
                "routable": bool(identity_matches and match["pod_ready"]
                                 and record.get("routable") and record["state"] == "ready"),
                "tp_size": record.get("tp_size"), "kv_usable_blocks": record.get("kv_usable_blocks"),
            })
        ready = (bool(resolved) and len(resolved) == len(records) and not mismatches
                 and not by_endpoint and all(row["routable"] for row in resolved))
        return {
            "pool": self.pool_name, "target_port": self.target_port(), "ready": ready,
            "selector": selector_string(self.pool()["spec"]["selector"]["matchLabels"]),
            "endpoint_picker": self.pool()["spec"]["endpointPickerRef"],
            "protocol": "envoy-ext-proc-v3", "response_mode": "ImmediateResponse",
            "resolved": sorted(resolved, key=lambda row: row["instance_id"]),
            "registry_without_pod_endpoint": sorted(unmatched_registry),
            "pod_endpoints_without_instance": sorted(by_endpoint), "identity_mismatches": mismatches,
            "logical_instance_count": sum(row["identity_matches"] for row in resolved),
            "pod_endpoint_count": len(endpoints),
            "maintenance_instances": sorted(excluded),
        }

    def bind_labels(
        self, registry_rows: List[Dict[str, Any]], *,
        maintenance_instances: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Publish ready coordinators, withdrawing declared maintenance targets."""
        self.pool(refresh=True)
        records = self._records(registry_rows)
        match_labels = self.pool()["spec"]["selector"]["matchLabels"]
        current = self.endpoints()
        wanted, applied = set(), []
        for record in records:
            # Draining/expired engines are not advertised as serving backends.
            if (record["instance_id"] in maintenance_instances
                    or record["state"] != "ready" or not record.get("routable")):
                continue
            pod = self.kube.pod_for_ip(record["endpoint_host"])
            if pod is None:
                applied.append({"instance_id": record["instance_id"], "labelled": False,
                                "reason": "no_ray_coordinator_pod"})
                continue
            name = pod["metadata"]["name"]
            wanted.add(name)
            labels = {**match_labels, **self._identity(record)}
            if any(pod["metadata"].get("labels", {}).get(key) != value
                   for key, value in labels.items()):
                self.kube.label_pod(name, labels)
                applied.append({"instance_id": record["instance_id"], "pod": name,
                                "generation": record["generation"], "labelled": True})
        removed = []
        for row in current:
            if row["pod"] not in wanted:
                self.kube.label_pod(row["pod"], {key: None for key in (
                    *match_labels, INSTANCE_LABEL, GENERATION_LABEL, LAYOUT_LABEL)})
                removed.append({"pod": row["pod"], "pod_ip": row["pod_ip"], "unlabelled": True})
        return {"applied": applied, "removed_stale": removed}
