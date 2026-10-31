"""Local browser demonstration that sends the checked-in example through Kubernetes."""

import argparse
import html
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8005")
    parser.add_argument("--port", type=int, default=8006)
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.api_url)
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
    request_bytes = (ROOT / "serving/examples/development-cycle.json").read_bytes()
    verification = ROOT / "tmp/kubernetes/verification.json"
    if not verification.exists():
        parser.error("Run scripts/verify_kubernetes.py first")
    record = json.loads(verification.read_text("utf-8"))
    if record["url"].rstrip("/") != args.api_url.rstrip("/") or record["status"] != "passed":
        parser.error("Run verification against this API URL before starting the demo")

    def page(status=None, prediction=None):
        output = "<p>Send the example to see the actual HTTP response.</p>"
        if prediction is not None:
            output = (
                f'<p class="success">POST /predict &middot; HTTP {status}</p>'
                '<pre class="response">'
                + html.escape(json.dumps(prediction, indent=2))
                + "</pre><p class='small'>Response received at "
                + datetime.now(UTC).isoformat()
                + "</p>"
            )
        return (
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>HydroGuard Kubernetes demo</title><style>"
            "*{box-sizing:border-box}body{margin:0;background:#edf2f6;color:#182b3b;"
            "font:16px system-ui,sans-serif}main{max-width:1140px;margin:24px auto;"
            "padding:26px 32px;background:white;border-radius:12px}"
            "h1{font-size:30px;margin:8px 0}h2{font-size:21px}p{line-height:1.5}"
            ".small{font-size:13px;color:#506476}.grid{display:grid;"
            "grid-template-columns:0.9fr 1.2fr;gap:24px}.card{background:#f3f7fa;"
            "padding:18px;border:1px solid #d7e1e8;border-radius:8px}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}"
            ".response{background:#142b3d;color:#f1f7fc;padding:16px;border-radius:6px}"
            "button{background:#1460a5;color:white;padding:12px 18px;border:0;"
            "border-radius:6px;font-size:16px;cursor:pointer}.success{color:#16643c;"
            "font-weight:700}.note{margin-top:20px;padding:12px 16px;background:#fff2d9;"
            "border-left:4px solid #bd801d;font-size:14px}"
            "</style><main><p class='small'>ML ZOOMCAMP 2026 &middot; LOCAL KIND CLUSTER</p>"
            "<h1>HydroGuard: prediction through Kubernetes</h1>"
            "<p>Browser &rarr; local demo &rarr; kubectl port-forward &rarr; "
            "Kubernetes Service &rarr; FastAPI Pod</p><div class='grid'>"
            "<section class='card'><h2>Verified deployment</h2><p class='success'>"
            "1 / 1 replica ready &middot; Service: ClusterIP</p><p>Kubernetes "
            + html.escape(record["kubernetes_server_version"])
            + "<br>Namespace: hydroguard<br>Pod: "
            + html.escape(record["pod"])
            + "</p><p class='small'>Deployment state verified at "
            + html.escape(record["checked_at_utc"])
            + "</p><h2>Development example</h2><p>Cycle 1 &middot; 60 seconds<br>"
            "14 physical sensors &middot; 70 computed attributes<br>"
            "No target or metadata enters the model.</p><p class='small'>"
            "serving/examples/development-cycle.json<br>UCI 447, Helwig et al. (2015), "
            "CC BY 4.0</p><form method='post' action='/predict-example'>"
            "<button>Send example to Kubernetes</button></form><p class='small'>API: "
            + html.escape(args.api_url.rstrip("/") + "/predict")
            + "</p></section><section><h2>Actual API response</h2>"
            + output
            + "</section></div><div class='note'><b>Academic demonstrator.</b> "
            "Historical test macro-F1: 0.675047; weak-leakage recall: 18/103. "
            "Probabilities are uncalibrated; 1.0 is not a reliability guarantee. "
            "This development example is not another final-test evaluation.</div>"
            "<p class='small'>Class 0: no leakage &middot; Class 1: weak leakage &middot; "
            "Class 2: severe leakage. The demo page runs locally; the model API runs "
            "inside Kubernetes. This is not a public cloud endpoint.</p></main></html>"
        ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path not in {"/", "/predict-example"}:
                self.respond(b"Not found", 404)
                return
            self.respond(page())

        def do_POST(self):
            if self.path != "/predict-example":
                self.respond(b"Not found", 404)
                return
            request = urllib.request.Request(
                args.api_url.rstrip("/") + "/predict",
                data=request_bytes,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    code, prediction = response.status, json.load(response)
            except urllib.error.HTTPError as error:
                code, prediction = error.code, json.load(error)
            except OSError:
                self.respond(b"Kubernetes API unavailable; check the port-forward terminal.", 502)
                return
            self.respond(page(code, prediction))

    print(f"Open http://127.0.0.1:{args.port}; stop with Ctrl+C", flush=True)
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
