# Final submission revision verification

This revision makes the existing midterm easier to review. The README is shortened
and links to the retained historical command reference, architecture diagram,
project explanation, actual prediction evidence and 2026 submission guide.
The modeling notebook now actually executes one separate development reproduction.
Official experiment source, splits, parameters, results, locks and model stay fixed.

## Actual checks

- 304 project tests and 117 API tests passed (421 total, no failures).
- Ruff lint/format and both uv lock checks passed. The initial format check found
  formatting in two added notebook cells; those cells were formatted and the check passed.
- Modeling notebook executed all nine code cells in a fresh kernel without errors.
- Its single Pipeline fit used exactly 1,328 locked development cycles. Saved model
  bytes equaled the official model in this run. No test features, predictions or
  metrics were produced. The official serving bundle was not replaced.
- All 72 protected historical files stayed byte-identical, including the two locks,
  original EDA notebook, source, model, splits and test reports.
- Docker build passed using cached runtime layers. The read-only container reached
  healthy status. Real development POST prediction returned HTTP 200 with exactly
  the recorded direct prediction; mixed boolean input returned HTTP 422.
- Relative links in the updated review documents resolve locally.

Full run facts and new document hashes are in [submission-verification.json](submission-verification.json).
The current API exchange is in [prediction-demo.json](prediction-demo.json).
The earlier [Phase 5 clean-copy report](release-readiness.md) remains historical
evidence for fresh acquisition and standalone training; it describes the earlier
seven-cell modeling notebook, not this nine-cell revision.

## Warnings and remaining owner work

The local notebook kernel printed its existing TCP transport encryption warning.
API tests printed the existing Starlette/httpx TestClient deprecation warning;
real HTTP requests and the container checks succeeded. No dependencies were changed.
Automatic new browser capture stopped because the Windows tool could not reliably
determine the current browser URL. Existing actual health imagery is retained;
the new prediction evidence is the actual HTTP response, not a fabricated screenshot.

Publish this reviewed revision and use its exact full commit SHA for the course
submission. Then confirm the course form saved it and complete three assigned peer
reviews. The 2026 schedule lists 9 November for submission and 16 November for peer
review, both 23:00 UTC; see [submission guide](submission-guide.md).
There is no cloud deployment or claim of bonus points. The historical weak-leakage
recall remains 18/103; no practical reliability or guaranteed course grade is claimed.
