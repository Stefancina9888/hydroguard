"""Publish or verify the approved Phase 2B split lock before model training."""

from pathlib import Path

from hydroguard.baselines import LOCK_RELATIVE_PATH, build_payload, verified_context
from hydroguard.splitting import lock_or_verify

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metadata, fingerprints, config = verified_context(ROOT)
    payload = build_payload(metadata, fingerprints, config)
    lock = lock_or_verify(ROOT / LOCK_RELATIVE_PATH, payload)
    print(f"Split lock verified: {ROOT / LOCK_RELATIVE_PATH}")
    print(f"Payload SHA-256: {lock['payload_sha256']}")
    print("Outer: development=1328, test=339 (133/103/103), excluded=538")
    for fold in payload["memberships"]["inner"]:
        print(f"Inner fold {fold['fold']}: {fold['diagnostics']}")


if __name__ == "__main__":
    main()
