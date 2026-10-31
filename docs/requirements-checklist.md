# Peer-review requirements checklist

Checked against the current official [ML Zoomcamp project instructions and rubric](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/cad7ce2070d7d115917aa59349b0858d7d1b460b/projects/README.md) on 2026-10-31 (Europe/Belgrade). `course-criteria-source.json` records the live download URL, retrieval time and content hash. The repository's shared project rubric is used; no separate 2026-specific scoring rubric was found. The final grade belongs to peer reviewers, not this self-check.

| Official criterion (maximum) | Repository evidence | Actual status |
| --- | --- | --- |
| Problem and intended use (2) | README opening: maintenance-engineering use, academic boundary, weak-leakage failure and limitations | Present |
| Exploratory analysis and feature interpretation (2) | `notebooks/01_eda.ipynb`, `docs/eda-findings.md`, `notebooks/02_modeling.ipynb`; loader/audit and standardized classwise LR coefficients | Historical clean run: 9 and 7 code cells; current modeling notebook: 9 executed code cells, including development reproduction ([verification](submission-verification.md)) |
| Multiple models and parameter search (3) | Dummy, LR and RF in `docs/baseline-results.*`; 23 candidates/69 fits, fixed search and selection rule in `config/tuning-v1.json`, `docs/tuning-results.*` | Historical execution preserved; no new tuning |
| Separate training script (1) | `scripts/reproduce_training.py`, shared extractor/model construction, independent output bundle | Present; only 1,328 locked development cycles |
| Re-executable notebook and training workflow (1) | README fresh-acquisition commands, `scripts/run_notebooks.py`, `docs/release-readiness.md` | Fresh download and one development-only fit passed; reproduced pickle SHA equals official; original provenance checks still reject the new timestamp |
| Model served through a web framework (1) | FastAPI app, health/predict, actual `docs/images/local-api.png`, `docs/prediction-demo.*`, serving/release verification | Local service only; no public URL |
| Managed dependencies and isolated environment (2) | Root and serving pyproject/uv.lock, Python version files, uv environment instructions | Both environments created from absent .venv directories; 304 project and 117 API tests passed; locks unchanged |
| Documented running container (2) | Dockerfile, default-deny .dockerignore, README build/run/request/stop, actual clean-copy Docker checks | Clean-copy build, healthy container, development prediction agreement and shutdown passed |
| Cloud or Kubernetes deployment (2) | `deploy/kubernetes/`, `scripts/verify_kubernetes.py`, actual `docs/images/kubernetes-demo.jpg` and `docs/kubernetes-verification.*` | Local kind v0.33.0 / Kubernetes v1.36.4 deployed and tested; code, instructions and actual browser prediction screenshot present; eligible evidence for 2 points, subject to peer review |

The current modeling notebook combines saved model-selection evidence, one actual separate development training reproduction and descriptive coefficient analysis; it does not rerun tuning or the final test. The new training script uses an identical pinned dataset while recording the actual fresh acquisition timestamp separately. Data download instructions are public and need no credentials. UCI 447 is not one of the course's prohibited example datasets; the attribution and CC BY 4.0 terms are in `docs/dataset.md` and the example provenance.

Historical deployment evidence includes the container health screenshot. The current [prediction demonstration](prediction-demo.md) shows an actual POST /predict interaction with the same development example, class interpretation and limitations. Localhost is not a cloud deployment.

The official instructions also require a reviewable repository snapshot and three peer reviews. The public GitHub repository and initial commit have been published. The final improvement commit must be published before submitting its exact SHA. Course submission and the three required peer reviews remain pending; see [2026 submission guide](submission-guide.md). Names such as train.py/predict.py are suggestions in the instructions; this project documents its equivalent scripts and FastAPI module.

Publication checks: include both notebooks, the official model bundle, serving lock, example request, source, locks, manifests and reports. Exclude raw/processed datasets, environments, caches and temporary runs. Existing CRLF checksum files use the owner-accepted byte-preserving Git exceptions; the release-readiness report retains the original explanation. No other students' code was copied.

## Kubernetes extension

The current revision adds the optional Kubernetes criterion using a local kind
cluster. All nine rubric categories now have concrete evidence. This is not a
guarantee of 16/16. The API returned the saved development prediction through
the Service; four invalid request categories returned 422. All 421 tests,
both lock checks and Ruff passed again. The existing 72 protected historical
files, official model and final-test metrics stayed unchanged. Earlier release
and submission-verification reports remain evidence for their original
revisions; [Kubernetes verification](kubernetes-verification.md) is the current
deployment record. Public cloud hosting is not claimed.
