#!/usr/bin/env python3
"""
Checks whether key grading artifacts carry explicit provenance metadata.

Exit codes:
  0 -> all checked artifacts have provenance
  1 -> one or more artifacts missing provenance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    REPO_ROOT / "LayoutLM" / "domain_generalization_artifacts" / "domain_transfer_report.json",
    REPO_ROOT / "LayoutLM" / "onnx_artifacts" / "benchmark_report.json",
    REPO_ROOT / "LayoutLM" / "onnx_artifacts" / "onnx_coreml_run_all_report.json",
]


def _has_provenance(payload: dict) -> bool:
    prov = payload.get("provenance")
    if not isinstance(prov, dict):
        return False
    required = ("generated_at_utc", "command", "script_sha256")
    return all(k in prov for k in required)


def main() -> int:
    missing = []
    for p in TARGETS:
        if not p.exists():
            missing.append(f"{p} (missing file)")
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            missing.append(f"{p} (invalid json: {exc})")
            continue
        if not _has_provenance(payload):
            missing.append(f"{p} (missing provenance)")

    if missing:
        print("Artifact provenance check failed:")
        for item in missing:
            print(f"- {item}")
        return 1

    print("Artifact provenance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

