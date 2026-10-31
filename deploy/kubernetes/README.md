# HydroGuard on local Kubernetes (kind)

This deploys the existing official model API to a single-node local Kubernetes
cluster. The model, data split, feature extractor and final-test results do not
change. The local browser demonstration is a separate optional client.

The course rubric accepts a **local or remote Kubernetes cluster** for its
cloud/Kubernetes criterion when deployment code and testing evidence are supplied.
See the [actual verification and screenshot](../../docs/kubernetes-verification.md).
This is local deployment, with no public endpoint or cloud account.

Start with [Get a fresh copy](../../README.md#get-a-fresh-copy) in the main README.
All commands below run from that clone's root. You can follow this deployment
without downloading the ML dataset or running the training/notebook workflow.

## Prerequisites

- Docker Desktop running Linux containers.
- `kubectl` available on PATH. The checked run used v1.36.1 with Kubernetes v1.36.4.
- kind v0.33.0. The configuration pins the release's v1.36.4 node image by digest.
- Internet access for the initial kind/node/runtime image downloads and enough
  Docker resources for a Kubernetes control plane (allow several GB).
- `uv` for the optional verification/browser client commands, or Python 3.11+.

The commands below were tested in **Windows PowerShell**, from the repository root.
For another OS, install kind/kubectl from their official instructions and use the
[Bash commands below](#linuxmacos-bash) with the same pinned configuration.
Windows execution was verified; the Bash variant is provided for other shells.

## 1. Download the verified kind binary on Windows

No system PATH changes or global Kubernetes settings are required:

```powershell
New-Item -ItemType Directory -Path tmp/kubernetes-tools,tmp/kubernetes -Force | Out-Null
Invoke-WebRequest -Uri "https://github.com/kubernetes-sigs/kind/releases/download/v0.33.0/kind-windows-amd64" -OutFile "tmp/kubernetes-tools/kind.exe"
$hydroguardKindHash = (Get-FileHash "tmp/kubernetes-tools/kind.exe" -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hydroguardKindHash -ne "4b22adaa135368c5a465d56bbd8e520cbea87272a06ca00b6078e7b81515c9fc") { throw "kind checksum mismatch" }
$hydroguardKind = (Resolve-Path "tmp/kubernetes-tools/kind.exe").Path
& $hydroguardKind version
kubectl version --client
docker info
```

The binary and kubeconfig stay in Git-ignored `tmp/`. The binary checksum is from
the official kind v0.33.0 release. Users who already have that version installed
can set `$hydroguardKind = "kind"` instead.

## 2. Build the API image and create the dedicated cluster

```powershell
docker build -t hydroguard:kubernetes-v1 .
& $hydroguardKind create cluster --name hydroguard-midterm --config deploy/kubernetes/kind.yaml --kubeconfig tmp/kubernetes/kubeconfig --wait 120s
& $hydroguardKind load docker-image hydroguard:kubernetes-v1 --name hydroguard-midterm
```

If this exact cluster already exists, reuse it and skip creation, or explicitly
remove it using the cleanup command below before starting again. The separate
kubeconfig prevents overwriting or selecting another user cluster. Treat that
file as a local credential and keep it out of Git.

The application tag is deliberately not `latest`. `imagePullPolicy: Never` requires
the preceding local image load and prevents an accidental registry pull of a
similarly named image. After an intentional image rebuild, load it again and use
`kubectl rollout restart deployment/hydroguard` with the same explicit context.

## 3. Apply and check the deployment

```powershell
$hydroguardKube = @("--kubeconfig", "tmp/kubernetes/kubeconfig", "--context", "kind-hydroguard-midterm")
kubectl @hydroguardKube apply -f deploy/kubernetes/hydroguard.yaml
kubectl @hydroguardKube -n hydroguard rollout status deployment/hydroguard --timeout=120s
kubectl @hydroguardKube -n hydroguard get deployment,pods,service
```

Expected: one ready Deployment replica, a Running/Ready Pod and a ClusterIP Service
on port 8000. Kubernetes startup/readiness/liveness probes call `/health`. The Pod
runs as the image's non-root user with a read-only filesystem, dropped capabilities
and CPU/memory requests and limits. The saved bundle is inside the image; there
are no raw-data mounts or runtime training. A single replica is a teaching setup,
not a claim of high availability.

## 4. Forward the Service to localhost

In the same terminal:

```powershell
kubectl @hydroguardKube -n hydroguard port-forward service/hydroguard 8005:8000 --address 127.0.0.1
```

This command stays running and does not return a prompt. Seeing
`Forwarding from 127.0.0.1:8005 -> 8000` means the forward is active.

Keep **terminal 1** open. Open **terminal 2**, change into the same cloned
repository root and check your location with `Get-Location`. If VS Code has
activated another project's environment, run `deactivate` first. Then run:

```powershell
curl.exe --fail http://127.0.0.1:8005/health
curl.exe --fail -H "Content-Type: application/json" --data-binary "@serving/examples/development-cycle.json" http://127.0.0.1:8005/predict
uv run --locked python scripts/verify_kubernetes.py
```

The example is development cycle 1, expected class 0 (`no leakage`). Its probabilities
are uncalibrated. The verifier checks the current Deployment/Pod/Service/endpoints,
model version, recorded direct prediction agreement, and HTTP 422 for missing
sensors, boolean mixtures, wrong vector length and extra metadata. It verifies all
72 protected historical files. Its report is `tmp/kubernetes/verification.json`;
it does not alter the published evidence or run training/final-test evaluation.

## 5. Optional browser demonstration

The verifier should print `status: passed`, with class 0 and label `no leakage`.
After it finishes, use the **same terminal 2** to start the browser client:

```powershell
uv run --locked python scripts/kubernetes_demo.py
```

Open http://127.0.0.1:8006 and click **Send example to Kubernetes**. This sends the
actual checked-in request through the forwarded Service and displays the real
response. The page runs on localhost; the model API runs in the Kubernetes Pod.
Deployment details show the timestamp of the preceding verification, not continuous
monitoring. Keep port-forward running. The page needs no raw dataset and makes no
request to a public cloud service. The saved screenshot shows this exact interaction.

## Cleanup and troubleshooting

Press **Ctrl+C in terminal 2** to stop the browser client. Press **Ctrl+C in
terminal 1** to stop port-forward. Then, in terminal 1 (where
`$hydroguardKind` is still defined), run:

```powershell
& $hydroguardKind delete cluster --name hydroguard-midterm --kubeconfig tmp/kubernetes/kubeconfig
```

This removes only the dedicated HydroGuard cluster. It does not delete project
files, datasets or Docker images. To use it again, repeat creation/load/apply.

- Docker unavailable: start Docker Desktop and wait for `docker info` to succeed.
- `ErrImageNeverPull`: load the exact application image into this named cluster.
- Pod not ready: inspect `kubectl @hydroguardKube -n hydroguard describe pod -l app=hydroguard`
  and `kubectl @hydroguardKube -n hydroguard logs deployment/hydroguard`.
- Port 8005 occupied: choose another local port in port-forward, and pass the same
  URL with `verify_kubernetes.py --url ...` and `kubernetes_demo.py --api-url ...`.
- Connection ends after Pod replacement: restart port-forward and verification.

## Linux/macOS (Bash)

Install kind v0.33.0 and a compatible kubectl using the official links below, and
start a Docker-compatible Linux container runtime. From the cloned root, in
**terminal 1**:

```bash
mkdir -p tmp/kubernetes
docker build -t hydroguard:kubernetes-v1 .
kind create cluster --name hydroguard-midterm --config deploy/kubernetes/kind.yaml --kubeconfig tmp/kubernetes/kubeconfig --wait 120s
kind load docker-image hydroguard:kubernetes-v1 --name hydroguard-midterm
kubectl --kubeconfig tmp/kubernetes/kubeconfig --context kind-hydroguard-midterm apply -f deploy/kubernetes/hydroguard.yaml
kubectl --kubeconfig tmp/kubernetes/kubeconfig --context kind-hydroguard-midterm -n hydroguard rollout status deployment/hydroguard --timeout=120s
kubectl --kubeconfig tmp/kubernetes/kubeconfig --context kind-hydroguard-midterm -n hydroguard get deployment,pods,service
kubectl --kubeconfig tmp/kubernetes/kubeconfig --context kind-hydroguard-midterm -n hydroguard port-forward service/hydroguard 8005:8000 --address 127.0.0.1
```

Leave port-forward running. In **terminal 2**, change into the same cloned root:

```bash
curl --fail http://127.0.0.1:8005/health
curl --fail -H "Content-Type: application/json" --data-binary "@serving/examples/development-cycle.json" http://127.0.0.1:8005/predict
uv run --locked python scripts/verify_kubernetes.py
uv run --locked python scripts/kubernetes_demo.py
```

Open http://127.0.0.1:8006 and click **Send example to Kubernetes**. Expected: HTTP
200, class 0, no leakage. Afterward stop the browser client and port-forward with
Ctrl+C in their respective terminals, then remove the dedicated cluster:

```bash
kind delete cluster --name hydroguard-midterm --kubeconfig tmp/kubernetes/kubeconfig
```

Official references: [kind quick start](https://kind.sigs.k8s.io/docs/user/quick-start/),
[kind v0.33.0 images](https://github.com/kubernetes-sigs/kind/releases/tag/v0.33.0),
[kubectl installation](https://kubernetes.io/docs/tasks/tools/),
[Kubernetes probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/),
[course rubric](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/projects/README.md).
