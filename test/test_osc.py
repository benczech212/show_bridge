#!/usr/bin/env python3
"""
Quick OSC connectivity tester for Resolume.

Usage examples:

  # Default: send 0.5 to /composition/master using first OSC (or fallback)
  python osc_test.py

  # Explicit address and value
  python osc_test.py --address /composition/master --value 0.75

  # Send a pulse (1 then 0) to tempo tap
  python osc_test.py --address /composition/tempocontroller/tempotap --pulse

This script:

  - Loads settings/connections.yaml
  - Looks for outputs.osc.resolume_arena (or outputs.osc.arena)
  - If not found, falls back to outputs.http.resolume_arena host with port 7000
  - Sends one OSC message (or a pulse) to that host/port
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from pythonosc.udp_client import SimpleUDPClient

DEFAULT_CONNECTIONS_PATH = "settings/connections.yaml"
DEFAULT_OSC_PORT = 7000


def load_connections(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"connections.yaml not found at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_resolume_osc_connection(
    connections_cfg: Dict[str, Any],
    name: Optional[str] = None,
    io_section: str = "outputs",
) -> Tuple[str, int]:
    """
    Try to get an OSC connection config from connections.yaml.

    Expected shape:

      outputs:
        osc:
          resolume_arena:
            - name: "arena_osc_out_main"
              host: "127.0.0.1"
              port: 7000

    If not present, fall back to HTTP host and DEFAULT_OSC_PORT:

      outputs:
        http:
          resolume_arena:
            - host: "127.0.0.1"
              port: 8080
    """
    osc_cfg = connections_cfg.get(io_section, {}).get("osc", {})
    arena_list = osc_cfg.get("resolume_arena", []) or osc_cfg.get("arena", [])

    if arena_list:
        if name is None:
            entry = arena_list[0]
        else:
            entry = None
            for e in arena_list:
                if e.get("name") == name:
                    entry = e
                    break
            if entry is None:
                entry = arena_list[0]

        host = entry.get("host", "127.0.0.1")
        port = int(entry.get("port", DEFAULT_OSC_PORT))
        return host, port

    # Fallback: piggyback off HTTP host
    http_cfg = connections_cfg.get(io_section, {}).get("http", {})
    http_list = http_cfg.get("resolume_arena", []) or http_cfg.get("arena", [])
    if not http_list:
        raise RuntimeError(
            "No OSC or HTTP Resolume connections found under "
            f"{io_section}.osc / {io_section}.http"
        )

    http_entry = http_list[0]
    host = http_entry.get("host", "127.0.0.1")
    port = DEFAULT_OSC_PORT
    return host, port


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Send a test OSC message to Resolume.")
    parser.add_argument(
        "--connections",
        type=str,
        default=DEFAULT_CONNECTIONS_PATH,
        help="Path to settings/connections.yaml (default: settings/connections.yaml)",
    )
    parser.add_argument(
        "--conn-name",
        type=str,
        default=None,
        help="Optional connection name to select from outputs.osc.resolume_arena[*].name",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="/composition/master",
        help="OSC address to send to (default: /composition/master)",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=0.5,
        help="Float value to send (default: 0.5)",
    )
    parser.add_argument(
        "--pulse",
        action="store_true",
        help="Send a pulse (value then 0.0) instead of a single value.",
    )

    args = parser.parse_args(argv)

    connections_path = Path(args.connections)
    try:
        connections_cfg = load_connections(connections_path)
    except Exception as e:
        print(f"[ERROR] Failed to load connections.yaml: {e}")
        return

    try:
        host, port = get_resolume_osc_connection(
            connections_cfg,
            name=args.conn_name,
            io_section="outputs",
        )
    except Exception as e:
        print(f"[ERROR] Failed to resolve OSC connection from connections.yaml: {e}")
        return

    print(f"[OSC] Using host={host}, port={port}")
    print(f"[OSC] Address={args.address}, value={args.value}, pulse={args.pulse}")

    client = SimpleUDPClient(host, port)

    if args.pulse:
        # Send value then zero
        print(f"[OSC] Sending pulse: {args.address} {args.value} -> 0.0")
        client.send_message(args.address, float(args.value))
        client.send_message(args.address, 0.0)
    else:
        print(f"[OSC] Sending single message: {args.address} {args.value}")
        client.send_message(args.address, float(args.value))

    print("[OSC] Done. Check Resolume OSC Monitor for incoming messages.")


if __name__ == "__main__":
    main()
