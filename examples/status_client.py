"""Minimal client for checking a running LEGALWORLD backend."""

from __future__ import annotations

import argparse
import json
from urllib.parse import urljoin

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running backend.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status_url = urljoin(args.base_url.rstrip("/") + "/", "api/status")
    response = requests.get(status_url, timeout=args.timeout)
    response.raise_for_status()
    payload = response.json()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload.get("status") == "running" else 1


if __name__ == "__main__":
    raise SystemExit(main())
