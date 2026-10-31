"""Verify the saved model through a local Kubernetes Service without fitting or test access."""

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8005")
    parser.add_argument("--kubeconfig", type=Path, default=ROOT / "tmp/kubernetes/kubeconfig")
    parser.add_argument("--context", default="kind-hydroguard-midterm")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/kubernetes/verification.json")
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.url)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.username
        or url.password
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        parser.error("Use the localhost HTTP URL of your kubectl port-forward")
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        parser.error("Install kubectl and make it available on PATH")
    kubeconfig = args.kubeconfig.resolve(strict=True)
    command = [kubectl, "--kubeconfig", str(kubeconfig), "--context", args.context]

    def kubernetes(*arguments):
        result = subprocess.run(
            [*command, *arguments, "-o", "json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    def request(path, body=None):
        raw = None if body is None else json.dumps(body, allow_nan=False).encode()
        query = urllib.request.Request(
            args.url.rstrip("/") + path,
            data=raw,
            headers={"Content-Type": "application/json"} if raw is not None else {},
        )
        try:
            response = urllib.request.urlopen(query, timeout=15)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)

    history = json.loads((ROOT / "config/review-integrity-v1.json").read_text("utf-8"))
    if any(sha256(ROOT / name) != digest for name, digest in history["files"].items()):
        raise RuntimeError("Protected historical files changed")
    deployment = kubernetes("-n", "hydroguard", "get", "deployment", "hydroguard")
    pods = kubernetes("-n", "hydroguard", "get", "pods", "-l", "app=hydroguard")["items"]
    service = kubernetes("-n", "hydroguard", "get", "service", "hydroguard")
    endpoints = kubernetes(
        "-n", "hydroguard", "get", "endpointslices", "-l", "kubernetes.io/service-name=hydroguard"
    )["items"]
    assert deployment["status"].get("readyReplicas", 0) == 1
    assert len(pods) == 1 and pods[0]["status"]["phase"] == "Running"
    container = pods[0]["status"]["containerStatuses"][0]
    assert container["ready"] is True
    assert service["spec"]["type"] == "ClusterIP"
    assert any(
        endpoint["conditions"].get("ready") is True
        for item in endpoints
        for endpoint in item.get("endpoints", [])
    )
    health_status, health = request("/health")
    assert health_status == 200 and health["ready"] is True
    payload_file = ROOT / "serving/examples/development-cycle.json"
    provenance = json.loads((ROOT / "serving/examples/provenance.json").read_text("utf-8"))
    assert sha256(payload_file) == provenance["request_sha256"]
    assert provenance["partition"] == "development" and provenance["cycle_id"] == 1
    payload = json.loads(payload_file.read_text("utf-8"))
    prediction_status, prediction = request("/predict", payload)
    expected = provenance["direct_historical_prediction"]
    assert prediction_status == 200
    assert all(prediction[k] == expected[k] for k in ["class", "label", "classes"])
    assert prediction["model_version"] == health["model_version"]
    assert prediction["model_version"].endswith(provenance["model_sha256"])
    probabilities = prediction["probabilities"]
    assert len(probabilities) == 3 and all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
    assert math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12)
    difference = max(
        abs(actual - reference)
        for actual, reference in zip(probabilities, expected["probabilities"], strict=True)
    )
    assert difference <= 1e-12
    invalid = {"missing_sensors": {}, "extra_metadata": {**payload, "cycle_id": 1}}
    invalid["mixed_boolean"] = copy.deepcopy(payload)
    invalid["mixed_boolean"]["sensors"]["PS1"][0] = True
    invalid["wrong_length"] = copy.deepcopy(payload)
    invalid["wrong_length"]["sensors"]["PS1"].pop()
    rejections = {}
    for name, body in invalid.items():
        code, response = request("/predict", body)
        assert code == 422 and len(json.dumps(response)) < 4096
        rejections[name] = {"http_status": code, "response": response}
    version = kubernetes("version")
    result = {
        "status": "passed",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "context": args.context,
        "url": args.url,
        "kubernetes_server_version": version["serverVersion"]["gitVersion"],
        "kubectl_version": version["clientVersion"]["gitVersion"],
        "deployment": deployment["metadata"]["name"],
        "ready_replicas": deployment["status"]["readyReplicas"],
        "pod": pods[0]["metadata"]["name"],
        "pod_uid": pods[0]["metadata"]["uid"],
        "container_image": container["image"],
        "container_image_id": container["imageID"],
        "container_restart_count": container["restartCount"],
        "probes": deployment["spec"]["template"]["spec"]["containers"][0],
        "service": {"name": service["metadata"]["name"], "type": service["spec"]["type"]},
        "health": {"http_status": health_status, "response": health},
        "prediction": {"http_status": prediction_status, "response": prediction},
        "request_file": "serving/examples/development-cycle.json",
        "request_sha256": provenance["request_sha256"],
        "development_cycle_id": 1,
        "max_probability_difference_from_recorded_direct_prediction": difference,
        "invalid_requests": rejections,
        "historical_files_verified": len(history["files"]),
        "new_training": False,
        "new_final_test_evaluation": False,
        "public_cloud_deployment": False,
    }
    output = args.output.resolve()
    if output in {(ROOT / name).resolve() for name in history["files"]}:
        parser.error("Verification output must not replace a protected historical file")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps({"status": "passed", "prediction": prediction, "report": str(output)}, indent=2)
    )


if __name__ == "__main__":
    main()
