#!/usr/bin/env python3
"""
Diagnostic tool: fetch the live Resolume composition and verify how it maps
against the YAML files in composition_mappings/.

Run from the project root:
    python check_composition.py [--host 127.0.0.1] [--port 8080]

Exit codes:
    0  mapping found, all apc_groups resolved
    1  mapping found but some apc_groups had no match
    2  no mapping file matched the composition name
    3  could not reach Resolume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
import yaml


MAPPING_DIR = Path("composition_mappings")
API_BASE = "/api/v1"


# ---------------------------------------------------------------------------
# Resolume helpers (inline so this script has no run_v3 dependency)
# ---------------------------------------------------------------------------

def _name_from_field(val, default: str) -> str:
    if isinstance(val, dict):
        return str(val.get("value", default))
    if val is not None:
        return str(val)
    return default


def _guess_composition_name(comp_json: dict) -> str:
    raw = comp_json.get("name") or comp_json.get("compositionName")
    return _name_from_field(raw, "Unnamed Composition")


def _iter_layergroups(comp_json: dict) -> list:
    return (
        comp_json.get("layergroups")
        or comp_json.get("groups")
        or comp_json.get("layerGroups")
        or []
    )


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run(host: str, port: int) -> int:
    base_url = f"http://{host}:{port}{API_BASE}"

    # ── Fetch composition ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Resolume HTTP  {base_url}/composition")
    print(f"{'='*60}")
    try:
        resp = requests.get(f"{base_url}/composition", timeout=5)
        resp.raise_for_status()
        comp_json = resp.json()
    except Exception as exc:
        print(f"\n  ERROR: could not reach Resolume: {exc}")
        return 3

    comp_name = _guess_composition_name(comp_json)
    print(f"\n  Composition name (exact): {comp_name!r}")

    # ── List every YAML in composition_mappings/ ─────────────────────────────
    if not MAPPING_DIR.exists():
        print(f"\n  ERROR: mapping directory not found: {MAPPING_DIR.resolve()}")
        return 2

    yaml_files = sorted(MAPPING_DIR.glob("*.y*ml"))
    if not yaml_files:
        print(f"\n  ERROR: no YAML files in {MAPPING_DIR}/")
        return 2

    print(f"\n{'='*60}")
    print("  Mapping files")
    print(f"{'='*60}")

    matched_path = None
    matched_raw = None

    for path in yaml_files:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            print(f"\n  {path.name}: PARSE ERROR — {exc}")
            continue

        yaml_comp_name = raw.get("composition_name", "<not set>")
        match = yaml_comp_name == comp_name
        marker = "  <<<  MATCHED" if match else ""
        print(f"\n  {path.name}")
        print(f"    composition_name: {yaml_comp_name!r}{marker}")
        if not match:
            # show a diff hint if the names are close
            if isinstance(yaml_comp_name, str) and yaml_comp_name.lower() == comp_name.lower():
                print(f"    ^ case differs from Resolume name {comp_name!r}")
            elif isinstance(yaml_comp_name, str) and yaml_comp_name.strip() == comp_name.strip():
                print(f"    ^ whitespace differs from Resolume name {comp_name!r}")

        if match and matched_path is None:
            matched_path = path
            matched_raw = raw

    # ── Did we find a mapping? ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    if matched_path is None:
        print("  RESULT: NO mapping matched the composition name.")
        print(f"\n  Resolume reports:  {comp_name!r}")
        print("  None of the YAML files above have a matching composition_name.")
        print("\n  Fix: either rename the composition in Resolume to match a YAML,")
        print("       or create a new YAML with:")
        print(f'       composition_name: "{comp_name}"')
        return 2

    print(f"  RESULT: Matched  {matched_path.name}")
    print(f"{'='*60}")

    apc_groups: dict[int, str] = {
        int(k): v for k, v in (matched_raw.get("apc_groups") or {}).items()
    }

    # Build lookup: lower-cased resolume group name -> APC index
    name_to_apc: dict[str, int] = {v.lower(): k for k, v in apc_groups.items()}

    print("\n  apc_groups from YAML:")
    for apc_idx in sorted(apc_groups):
        print(f"    APC {apc_idx} -> {apc_groups[apc_idx]!r}")

    # ── Walk groups from Resolume ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Groups in Resolume composition")
    print(f"{'='*60}")

    lg_list = _iter_layergroups(comp_json)
    unresolved = []

    for g_idx, g_json in enumerate(lg_list):
        g_name = _name_from_field(g_json.get("name"), f"group-{g_json.get('id', '?')}")
        apc_idx = name_to_apc.get(g_name.strip().lower())
        is_output = "output -" in g_name.lower()

        if apc_idx is not None:
            status = f"-> APC {apc_idx}"
        elif is_output:
            status = "output group (auto-detected, right-side APC)"
        else:
            status = "NOT MAPPED"
            unresolved.append(g_name)

        print(f"  [{g_idx}] {g_name!r:40s}  {status}")

        # Extra diagnostics: show near-misses from the YAML
        if apc_idx is None and not is_output:
            for yaml_idx, yaml_name in apc_groups.items():
                if g_name.strip().lower() == yaml_name.strip().lower():
                    print(f"        ^ case/whitespace mismatch with YAML entry {yaml_name!r} (APC {yaml_idx})")
                elif g_name.lower() in yaml_name.lower() or yaml_name.lower() in g_name.lower():
                    print(f"        ^ partial match with YAML entry {yaml_name!r} (APC {yaml_idx})")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if not unresolved:
        print("  All groups resolved. Mapping looks correct.")
        return 0
    else:
        print(f"  {len(unresolved)} group(s) not mapped:")
        for n in unresolved:
            print(f"    - {n!r}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify show_bridge composition mapping")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    sys.exit(run(args.host, args.port))


if __name__ == "__main__":
    main()
