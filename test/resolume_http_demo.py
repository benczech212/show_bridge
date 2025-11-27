#!/usr/bin/env python3
"""
Simple demo to connect to Resolume Arena HTTP API using connections.yaml
and perform a GET on the composition endpoint.

Layout expected in connections.yaml:

outputs:
  http:
    resolume_arena:
      - name: "arena_http_out_main"
        host: "127.0.0.1"
        port: 8080
        use_https: false
        api_base: "/api/v1"
        username: "admin"
        password: "secret"
        timeout: 2.0
        verify_ssl: true
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml


CONNECTIONS_FILE = "settings/connections.yaml"


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------

@dataclass
class HttpEndpoint:
    name: str
    host: str
    port: int
    use_https: bool = False
    api_base: str = "/api/v1"
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: float = 2.0
    verify_ssl: bool = True

    def base_url(self) -> str:
        scheme = "https" if self.use_https else "http"
        # Normalize api_base to have a leading slash and no trailing slash
        base = self.api_base or "/"
        if not base.startswith("/"):
            base = "/" + base
        base = base.rstrip("/")
        return f"{scheme}://{self.host}:{self.port}{base}"

    def make_session(self) -> requests.Session:
        s = requests.Session()

        if self.username or self.password:
            # Basic auth
            s.auth = (self.username or "", self.password or "")

        s.verify = self.verify_ssl
        return s


# -------------------------------------------------------------------
# Loading connections.yaml
# -------------------------------------------------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"connections file not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def list_resolume_http_outputs(cfg: Dict[str, Any]) -> List[HttpEndpoint]:
    """
    Read outputs.http.resolume_arena list and build HttpEndpoint objects.
    """
    outputs = cfg.get("outputs", {})
    http_out = outputs.get("http", {})
    arena_list = http_out.get("resolume_arena", []) or []

    endpoints: List[HttpEndpoint] = []
    for entry in arena_list:
        try:
            ep = HttpEndpoint(
                name=entry.get("name", "unnamed"),
                host=entry["host"],
                port=int(entry.get("port", 8080)),
                use_https=bool(entry.get("use_https", False)),
                api_base=entry.get("api_base", "/api/v1"),
                username=entry.get("username"),
                password=entry.get("password"),
                timeout=float(entry.get("timeout", 2.0)),
                verify_ssl=bool(entry.get("verify_ssl", True)),
            )
            endpoints.append(ep)
        except KeyError as e:
            print(f"Skipping invalid resolume_arena HTTP output entry {entry}: missing {e}")

    return endpoints


def choose_endpoint(endpoints: List[HttpEndpoint]) -> HttpEndpoint:
    """
    If there is exactly one endpoint, use it; otherwise prompt.
    """
    if not endpoints:
        raise RuntimeError("No outputs.http.resolume_arena entries found in connections.yaml")

    if len(endpoints) == 1:
        ep = endpoints[0]
        print(f"Using only available Resolume HTTP endpoint: {ep.name} ({ep.base_url()})")
        return ep

    print("Available Resolume Arena HTTP outputs:")
    for i, ep in enumerate(endpoints):
        print(f"  [{i}] {ep.name} -> {ep.base_url()}")

    while True:
        choice = input("Select endpoint index: ").strip()
        try:
            idx = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue

        if 0 <= idx < len(endpoints):
            return endpoints[idx]

        print(f"Index out of range. Choose between 0 and {len(endpoints) - 1}.")


# -------------------------------------------------------------------
# API calls
# -------------------------------------------------------------------

def get_composition(endpoint: HttpEndpoint) -> requests.Response:
    """
    Perform GET /composition on the given Resolume HTTP endpoint.
    """
    session = endpoint.make_session()
    url = endpoint.base_url() + "/composition"

    print(f"[HTTP] GET {url}  (timeout={endpoint.timeout}, verify={endpoint.verify_ssl})")
    resp = session.get(url, timeout=endpoint.timeout)
    return resp


def pretty_print_response(resp: requests.Response) -> None:
    """
    Try to pretty-print JSON; fall back to raw text.
    """
    print(f"[HTTP] Status: {resp.status_code}")
    print(f"[HTTP] Content-Type: {resp.headers.get('Content-Type', '')}")

    text = resp.text
    if not text:
        print("[HTTP] (empty body)")
        return

    # Try JSON first
    try:
        data = resp.json()
    except ValueError:
        print("\n--- Response body (text) ---")
        print(text)
        print("--- end ---")
    else:
        print("\n--- Response body (JSON) ---")
        print(json.dumps(data, indent=2, sort_keys=True))
        print("--- end ---")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    cfg = load_yaml(CONNECTIONS_FILE)
    endpoints = list_resolume_http_outputs(cfg)
    endpoint = choose_endpoint(endpoints)

    print(f"\nSelected Resolume endpoint: {endpoint.name}")
    print(f"Base URL: {endpoint.base_url()}\n")

    try:
        resp = get_composition(endpoint)
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] HTTP request failed: {exc}")
        return

    pretty_print_response(resp)
    # save response to a file for inspection
    out_path = Path("resolume_composition_response.json")
    out_path.write_text(resp.text, encoding="utf-8")

if __name__ == "__main__":
    main()
