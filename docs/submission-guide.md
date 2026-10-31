# ML Zoomcamp 2026 submission and review guide

## Review this project

1. Read the README problem statement, metrics and limitations.
2. Follow its Docker quick start; send the checked-in development-cycle request to
   POST /predict. See [actual prediction evidence](prediction-demo.md).
3. Open [01_eda](../notebooks/01_eda.ipynb) for data validation and exploration,
   then [02_modeling](../notebooks/02_modeling.ipynb) for model comparison, tuning,
   the fixed winner, one actual development fit and coefficient interpretation.
4. Follow the README fresh-acquisition commands to execute notebooks and the
   standalone training script. Both training entry points write separate bundles;
   they do not replace the official serving model or reevaluate the test.
5. Consult [criteria mapping](requirements-checklist.md),
   [project explanation](project-walkthrough.md) and the historical reports for details.

## Cohort and deadlines

The owner confirmed the 2026 cohort. Its [official schedule](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/README.md)
states submission by **9 November 2026, 23:00 UTC** and peer review by
**16 November 2026, 23:00 UTC**. In Europe/Belgrade these are 10 November and
17 November at 00:00 respectively; plan to finish before the named calendar dates.
The anonymous submission page displayed 16 November during this check, which
conflicts with the schedule's submission date. Use the earlier 9 November deadline
and verify the signed-in course page for any official update.

## Publication and owner actions

The public repository is [Stefancina9888/hydroguard](https://github.com/Stefancina9888/hydroguard).
The initial published commit is e35f530a025fb43f15af51695a1453c356d8b8e6.
These final review improvements require a newer commit; submit its full SHA after
publication and verify that GitHub displays that exact version. Historical Phase 5
reports describe the earlier run and must not be read as the current publication status.
The owner accepted the documented byte-preserving CRLF exceptions before publication.

Use the [2026 midterm submission form](https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm).
Provide the public repository URL and the exact full commit SHA, your actual time
spent and certificate name. Learning-in-public and FAQ contribution links are
optional; include only work actually done. Confirm that the platform saved the
submission. A GitHub push by itself is not a course submission.

Complete reviews of **three assigned student projects** through the course platform
before the peer-review deadline. Use the official rubric, inspect their evidence
and give project-specific feedback. Missing peer reviews fails the project under
the [shared instructions](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/projects/README.md).
No peer reviews or course submission are claimed by this repository.

Cloud/Kubernetes is optional bonus work and is not implemented or claimed here.
All final metrics remain those of the fixed historical model. Neither a self-check
nor a passing test suite guarantees the peer reviewers' score.
