"""Logical-instance identity regressions; no Kubernetes mutation or TPU work."""

import copy
import unittest

from hetero.gateway.discovery import (
    DiscoveryBridge, GENERATION_LABEL, INSTANCE_LABEL, LAYOUT_LABEL, InClusterKube,
)


class KubeFixture(InClusterKube):
    def __init__(self):
        self.pool_body = {
            "spec": {
                "selector": {"matchLabels": {"omp.llm-d.ai/instance": "true"}},
                "targetPorts": [{"number": 8000}],
                "endpointPickerRef": {"name": "omp-gateway", "port": {"number": 9002},
                                      "failureMode": "FailClose"},
            },
        }
        self.pods = [self.pod("system-daemon", "10.0.0.1", worker=False),
                     self.pod("coordinator", "10.0.0.1"),
                     self.pod("noncoordinator-worker", "10.0.0.2")]
        self.patches = []

    @staticmethod
    def pod(name, ip, *, worker=True):
        return {"metadata": {"name": name, "labels": {"ray.io/node-type": "worker"} if worker else {}},
                "spec": {"hostNetwork": True},
                "status": {"podIP": ip, "conditions": [{"type": "Ready", "status": "True"}]}}

    def get_inference_pool(self, name):
        return copy.deepcopy(self.pool_body)

    def list_pods(self, selector):
        labels = dict(part.split("=", 1) for part in selector.split(","))
        return [pod for pod in self.pods if all(pod["metadata"]["labels"].get(k) == v
                                               for k, v in labels.items())]

    def label_pod(self, name, labels):
        self.patches.append((name, labels))
        pod = next(pod for pod in self.pods if pod["metadata"]["name"] == name)
        for key, value in labels.items():
            if value is None:
                pod["metadata"]["labels"].pop(key, None)
            else:
                pod["metadata"]["labels"][key] = value
        return pod


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.kube = KubeFixture()
        self.bridge = DiscoveryBridge(kube=self.kube)
        self.row = {"instance_id": "small_a", "generation": 1, "layout_id": "M",
                    "endpoint": "10.0.0.1:8000", "endpoint_host": "10.0.0.1", "endpoint_port": 8000,
                    "host_chip_rank_map": [{"process_rank": 0, "node_ip": "10.0.0.1"},
                                           {"process_rank": 1, "node_ip": "10.0.0.2"}],
                    "state": "ready", "routable": True, "tp_size": 8}

    def test_coordinator_generation_changes_without_changing_endpoint(self):
        self.bridge.bind_labels([self.row])
        view = self.bridge.resolve([self.row])
        self.assertTrue(view["ready"])
        self.assertEqual([row["pod"] for row in view["resolved"]], ["coordinator"])
        self.assertNotIn(INSTANCE_LABEL, self.kube.pods[0]["metadata"]["labels"])
        self.assertNotIn(INSTANCE_LABEL, self.kube.pods[2]["metadata"]["labels"])
        self.row["generation"] = 2
        self.assertFalse(self.bridge.resolve([self.row])["ready"])
        self.bridge.bind_labels([self.row])
        refreshed = self.bridge.resolve([self.row])
        self.assertTrue(refreshed["ready"])
        self.assertEqual(refreshed["resolved"][0]["generation"], 2)
        self.assertEqual(self.kube.pods[1]["metadata"]["labels"][GENERATION_LABEL], "2")
        self.assertEqual(self.kube.pods[1]["metadata"]["labels"][LAYOUT_LABEL], "M")

    def test_drained_and_stale_worker_endpoints_are_withdrawn(self):
        self.bridge.bind_labels([self.row])
        self.kube.pods[2]["metadata"]["labels"].update({
            "omp.llm-d.ai/instance": "true", INSTANCE_LABEL: "obsolete", GENERATION_LABEL: "0",
        })
        self.assertFalse(self.bridge.resolve([self.row])["ready"])
        self.bridge.bind_labels([self.row])
        self.assertTrue(self.bridge.resolve([self.row])["ready"])
        self.row.update(state="draining", routable=False)
        self.bridge.bind_labels([self.row])
        self.assertFalse(self.bridge.resolve([self.row])["ready"])
        self.assertEqual(self.bridge.endpoints(), [])

    def test_maintenance_removes_target_labels_but_never_ignores_stale_addresses(self):
        peer = {**self.row, "instance_id": "small_b", "endpoint_host": "10.0.0.2",
                "endpoint": "10.0.0.2:8000",
                "host_chip_rank_map": [{"process_rank": 0, "node_ip": "10.0.0.2"}]}
        rows = [self.row, peer]
        self.bridge.bind_labels(rows)
        excluded = ["small_a"]
        # Declaring maintenance alone is not evidence of label withdrawal.
        self.assertFalse(self.bridge.resolve(rows, maintenance_instances=excluded)["ready"])
        self.bridge.bind_labels(rows, maintenance_instances=excluded)
        view = self.bridge.resolve(rows, maintenance_instances=excluded)
        self.assertTrue(view["ready"])
        self.assertEqual([row["instance_id"] for row in view["resolved"]], ["small_b"])
        self.row.update(state="draining", routable=False)
        self.assertFalse(self.bridge.resolve(rows)["ready"])
        self.assertTrue(self.bridge.resolve(rows, maintenance_instances=excluded)["ready"])
        self.kube.pods.append(self.kube.pod("replacement-coordinator", "10.0.0.3"))
        self.row.update(generation=2, state="ready", routable=True,
                        endpoint_host="10.0.0.3", endpoint="10.0.0.3:8000",
                        host_chip_rank_map=[{"process_rank": 0, "node_ip": "10.0.0.3"}])
        self.bridge.bind_labels(rows)
        restored = self.bridge.resolve(rows)
        self.assertTrue(restored["ready"])
        self.assertEqual({row["endpoint"] for row in restored["resolved"]},
                         {"10.0.0.2:8000", "10.0.0.3:8000"})
        self.assertNotIn(INSTANCE_LABEL, self.kube.pods[1]["metadata"]["labels"])

    def test_endpoint_and_pod_ambiguity_never_silently_selects_an_owner(self):
        self.bridge.bind_labels([self.row])
        duplicate = copy.deepcopy(self.kube.pods[1])
        duplicate["metadata"]["name"] = "second-coordinator-pod"
        self.kube.pods.append(duplicate)
        self.assertFalse(self.bridge.resolve([self.row])["ready"])
        with self.assertRaises(RuntimeError):
            self.bridge.bind_labels([self.row])
        with self.assertRaises(RuntimeError):
            self.bridge.resolve([self.row, {**self.row, "instance_id": "other"}])

    def test_http_picker_and_noncoordinator_registry_binding_are_rejected(self):
        self.kube.pool_body["spec"]["endpointPickerRef"]["port"]["number"] = 80
        with self.assertRaises(RuntimeError):
            self.bridge.bind_labels([self.row])
        self.kube.pool_body["spec"]["endpointPickerRef"]["port"]["number"] = 9002
        self.row.update(endpoint_host="10.0.0.2", endpoint="10.0.0.2:8000")
        with self.assertRaises(RuntimeError):
            self.bridge.bind_labels([self.row])
        self.assertEqual(self.kube.patches, [])


if __name__ == "__main__":
    unittest.main()
