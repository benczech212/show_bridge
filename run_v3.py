#!/usr/bin/env python3
"""
Generic APC40-style state-machine demo driven by mapping profiles,
with Resolume composition + composition_mappings integration and OSC control.

Each mapping profile is a single YAML file in ./mappings, e.g.:

  mappings/apc40.yaml

with structure:

  controller_name: "Akai APC40"
  description: "..."
  author: "..."
  version: 2.0

  input_mappings:
    groups: ...
    global: ...
    velocity_mappings: ...

  state_mappings:
    groups: ...
    global: ...

This script:

  - Scans ./mappings for *.yml / *.yaml
  - Treats each as a "device profile"
  - Lets you select one (or auto-selects if it matches MIDI ports)
  - Builds a state machine for 8 "output groups":

        playing: bool
        effects: bool
        transforms: bool
        fft_mask: bool
        color: bool
        opacity: float (0–1)
        intensity: float (0–1)

  - Supports short / long / double press semantics
  - Supports per-note LED velocities via velocity_mappings

  - Loads Resolume HTTP connection from settings/connections.yaml
  - Fetches /composition from Resolume HTTP API
  - Loads composition mappings from ./composition_mappings/*.yaml
    and uses `composition_name` + `layer_roles` to:

      * Map APC group indices (1–8) to Resolume groups
      * Map each layer to a high-level role (colors, effects, etc.)
      * Map APC group actions (playing/effects/...) to those roles

  - Uses OSC to control Resolume:

      Play clip:
        /composition/layers/{layer_index}/clips/{clip_index}/connect

      Group opacity:
        /composition/groups/{group_index}/master

      Master opacity:
        /composition/master

      Tap tempo (pulse 1 then 0):
        /composition/tempocontroller/tempotap

      Resync (pulse 1 then 0):
        /composition/tempocontroller/resync

      Nudge - (pulse 1 then 0):
        /composition/tempocontroller/tempopull

      Nudge + (pulse 1 then 0):
        /composition/tempocontroller/tempopush

      Scroll clips horizontally:
        /application/ui/clipsscrollhorizontal
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
import json
import threading
import http.server
import socketserver
import os
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import mido
import re
from lib.show_bridge_logging import setup_logging, sb_log
import yaml
import requests
from pythonosc.udp_client import SimpleUDPClient

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

# Try to import psutil for resource metrics; fall back gracefully
try:
    import psutil  # type: ignore
except Exception:
    psutil = None
import logging
from logging.handlers import RotatingFileHandler
# -------------------------------------------------------------------
# Paths / defaults
# -------------------------------------------------------------------

DEFAULT_MAPPINGS_DIR = "mappings"
DEFAULT_CONNECTIONS_PATH = "settings/connections.yaml"
DEFAULT_COMPOSITION_MAPPING_DIR = "composition_mappings"
LOG_DIR = Path("logs")
OSC_LOG_DIR = LOG_DIR / "osc"

DEFAULT_OSC_PORT = 7000  # fallback if not defined in connections.yaml

MidiKey = Tuple[str, int, int]  # ("note"|"cc", channel, note_or_cc)

# APC40 color velocities – fallback defaults (used when no map present)
RED_VELOCITY = 3     # typical: red
ORANGE_VELOCITY = 5  # typical: amber/orange


# In Resolume:
#   column 1 = OFF
#   column 2 = Passthrough
#   column 3+ = actual clip content
FIRST_CONTENT_COLUMN = 3

# ------------------------------------------------------------
# OSC server registry (avoid double-binding same host/port)
# ------------------------------------------------------------
_transport_osc_servers = {}
_transport_osc_servers_lock = threading.Lock()
# General OSC server registry (for non-transport servers)
_osc_servers: Dict[Tuple[str, int], ThreadingOSCUDPServer] = {}
_osc_servers_lock = threading.Lock()

# OSC client registry to avoid creating multiple UDP clients to same host/port
_osc_clients: Dict[Tuple[str, int], SimpleUDPClient] = {}
_osc_clients_lock = threading.Lock()
def get_osc_client(host: str, port: int) -> SimpleUDPClient:
    """Return a shared SimpleUDPClient for (host,port), creating it if needed."""
    key = (host, int(port))
    with _osc_clients_lock:
        client = _osc_clients.get(key)
        if client is None:
            client = SimpleUDPClient(host, int(port))
            _osc_clients[key] = client
    return client
"""
Logging implementation moved to `show_bridge_logging.py`.
Import `setup_logging` and `sb_log` from that module instead of using local definitions.
"""

# =========================================================
# Resolume composition + mapping model
# =========================================================

@dataclass
class ClipInfo:
    id: str
    name: str
    column_index: int   # 1-based column index in the Resolume grid
    connected: bool = False  # True when Resolume reports Connected / Connected & previewing


@dataclass
class LayerInfo:
    id: str
    name: str
    index_in_group: int      # 0-based index inside the layergroup
    global_index: int        # 0-based index across ALL layers in the composition
    role: Optional[str] = None  # e.g. "colors", "effects", etc.
    clips: List[ClipInfo] = field(default_factory=list)
    active_column_index: Optional[int] = None  # 1-based column index currently active (if known)


@dataclass
class GroupInfo:
    id: str
    name: str
    index_in_composition: int  # 0-based index inside composition.layergroups
    apc_group_index: Optional[int] = None  # 1..8 if mapped, else None
    layers: List[LayerInfo] = field(default_factory=list)


@dataclass
class CompositionInfo:
    name: str
    id: Optional[str] = None
    groups: List[GroupInfo] = field(default_factory=list)

    def layers_for_apc_group(self, apc_group_index: int) -> List[LayerInfo]:
        layers: List[LayerInfo] = []
        for g in self.groups:
            if g.apc_group_index == apc_group_index:
                layers.extend(g.layers)
        return layers

    def group_for_apc(self, apc_group_index: int) -> Optional[GroupInfo]:
        for g in self.groups:
            if g.apc_group_index == apc_group_index:
                return g
        return None


@dataclass
class CompositionMapping:
    composition_name: str
    apc_groups: Dict[int, str]          # APC index -> Resolume group name
    layer_roles: Dict[str, List[str]] = field(default_factory=dict)  # role -> list of name patterns

    @classmethod
    def from_yaml_dir(
        cls,
        composition_name: str,
        mapping_dir: Path = Path(DEFAULT_COMPOSITION_MAPPING_DIR),
    ) -> "CompositionMapping":
        """
        Look for a mapping file where composition_name matches, or
        fallback to <composition_name>.yaml / .yml.
        """
        if not mapping_dir.exists():
            raise RuntimeError(f"Mapping directory does not exist: {mapping_dir}")

        # 1) Scan all YAML files for explicit composition_name match
        for path in sorted(mapping_dir.glob("*.y*ml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if raw.get("composition_name") == composition_name:
                return cls(
                    composition_name=composition_name,
                    apc_groups={int(k): v for k, v in (raw.get("apc_groups") or {}).items()},
                    layer_roles=raw.get("layer_roles") or {},
                )

        # 2) Filename-based: <composition_name>.yaml/.yml
        for suffix in (".yaml", ".yml"):
            candidate = mapping_dir / f"{composition_name}{suffix}"
            if candidate.exists():
                raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                return cls(
                    composition_name=composition_name,
                    apc_groups={int(k): v for k, v in (raw.get("apc_groups") or {}).items()},
                    layer_roles=raw.get("layer_roles") or {},
                )

        raise RuntimeError(
            f"No composition mapping found for composition '{composition_name}' in {mapping_dir}"
        )


# ---- Resolume HTTP helpers ----

def _name_from_field(val: Any, default: str) -> str:
    """
    Resolume sometimes returns names as { "value": "Comp 1" }.
    Handle that and plain strings.
    """
    if isinstance(val, dict) and "value" in val:
        inner = val["value"]
        if isinstance(inner, str):
            return inner
    if isinstance(val, str):
        return val
    return default


def _guess_composition_name(comp_json: Dict[str, Any]) -> str:
    raw = comp_json.get("name") or comp_json.get("compositionName")
    return _name_from_field(raw, "Unnamed Composition")


def _iter_layergroups(comp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "layergroups" in comp_json:
        return comp_json["layergroups"]
    if "groups" in comp_json:
        return comp_json["groups"]
    return comp_json.get("layerGroups", [])


def _iter_layers(group_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "layers" in group_json:
        return group_json["layers"]
    return group_json.get("Layers", [])


def _iter_clips(layer_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "clips" in layer_json:
        return layer_json["clips"]
    return layer_json.get("Clips", [])


def _clip_column_index(clip_json: Dict[str, Any], fallback_index: int) -> int:
    """
    Determine the column index for a clip.

    Priority:
      1) Explicit numeric fields in the clip JSON (column, columnIndex, col, etc.)
      2) 'index' if present
      3) Fallback to the clip's position in the layer's clips list
         (fallback_index, 0-based) + 1 so OSC uses 1-based indexing:

            1 = OFF, 2 = Passthrough, 3+ = content
    """
    # Try obvious fields first (as-is, assuming they already match OSC semantics)
    for key in ("column", "columnIndex", "col", "colIndex"):
        if key in clip_json:
            try:
                return int(clip_json[key])
            except (TypeError, ValueError):
                pass

    # Try 'index' field – often 0-based, so we convert to 1-based
    if "index" in clip_json:
        try:
            return int(clip_json["index"]) + 1
        except (TypeError, ValueError):
            pass

    # Fallback: use the position in the clips list (0-based) + 1
    col = int(fallback_index) + 1
    return col



def _clip_name(clip_json: Dict[str, Any]) -> str:
    """
    Return the clip's *actual* name, or "" if there isn't one.
    We no longer fabricate a fallback name, so we can drop
    nameless clips from the model.
    """
    raw = clip_json.get("name") or clip_json.get("displayName")
    # If both name/displayName are missing or empty, this will become ""
    name = _name_from_field(raw, "")
    return name.strip()



def _layer_name(layer_json: Dict[str, Any]) -> str:
    raw = layer_json.get("name") or layer_json.get("displayName")
    return _name_from_field(raw, f"layer-{layer_json.get('id', '?')}")


def _match_layer_role(layer_name: str, mapping: CompositionMapping) -> Optional[str]:
    """
    Use mapping.layer_roles (role -> list of substrings) to assign a role
    based on layer name.
    """
    lname = layer_name.lower()
    for role, patterns in mapping.layer_roles.items():
        for p in patterns:
            if p.lower() in lname:
                return role
    return None

def debug_dump_composition_columns(comp: CompositionInfo, logger) -> None:
    sb_log(logger,logging.DEBUG,"ARENA","INIT",f"Composition '{comp.name}' column layout:")
    for g in comp.groups:
        apc_str = f"APC={g.apc_group_index}" if g.apc_group_index is not None else "APC=-"
        logger.info (f"  Group {g.index_in_composition} '{g.name}' ({apc_str})")
        for layer in g.layers:
            cols = ", ".join(f"{c.column_index}:{c.name}" for c in layer.clips)
            role = layer.role or "-"
            logger.info(
                f"    Layer {layer.index_in_group} (global {layer.global_index}) "
                f"'{layer.name}' role={role} -> {cols}"
            )


def build_composition_model(
    comp_json: Dict[str, Any],
    mapping: CompositionMapping,
) -> CompositionInfo:
    """
    Build CompositionInfo using group / layer indices and a CompositionMapping.

    - Group index: position in layergroups list
    - Layer indices:
        * index_in_group: position in group's layers list
        * global_index: sequential across ALL groups (0-based)
    - Layer roles: derived from mapping.layer_roles patterns
    - APC group mapping: from mapping.apc_groups (APC idx -> Resolume group name)
    """
    comp_name = _guess_composition_name(comp_json)
    comp_id = comp_json.get("id")
    comp = CompositionInfo(name=comp_name, id=comp_id, groups=[])

    lg_list = _iter_layergroups(comp_json)

    # Resolume group name -> APC index (strip+lowercase both sides for robustness)
    resolume_group_to_apc: Dict[str, int] = {
        v.strip().lower(): k for k, v in mapping.apc_groups.items()
    }
    next_layer_index = 0  # fallback global layer index across all groups

    for g_idx, g_json in enumerate(lg_list):
        g_id = str(g_json.get("id"))
        g_name = _name_from_field(g_json.get("name"), f"group-{g_id}")

        apc_idx: Optional[int] = None
        key = g_name.strip().lower()
        if key in resolume_group_to_apc:
            apc_idx = resolume_group_to_apc[key]

        group_info = GroupInfo(
            id=g_id,
            name=g_name,
            index_in_composition=g_idx,
            apc_group_index=apc_idx,
        )

        layers_json = _iter_layers(g_json)
        for l_idx, layer_json in enumerate(layers_json):
            l_id = str(layer_json.get("id"))
            l_name = _layer_name(layer_json)
            role = _match_layer_role(l_name, mapping)

            # --- IMPORTANT: use Resolume's own global layer index if present ---
            if "index" in layer_json:
                global_index = int(layer_json["index"])
                # keep our fallback counter in sync / ahead
                next_layer_index = max(next_layer_index, global_index + 1)
            else:
                global_index = next_layer_index
                next_layer_index += 1

            layer_info = LayerInfo(
                id=l_id,
                name=l_name,
                index_in_group=l_idx,
                global_index=global_index,
                role=role,
                clips=[],
            )

            clips_json = _iter_clips(layer_json)
            active_column_index = None
            for col0, clip_json in enumerate(clips_json):
                c_id = str(clip_json.get("id"))
                c_name = _clip_name(clip_json)

                # Drop nameless clips so they don't confuse autoplay
                if not c_name:
                    # print(f"[DEBUG] Skipping nameless clip id={c_id} on layer '{l_name}', local col={col0}")
                    continue

                col_idx = _clip_column_index(clip_json, col0)
                conn_raw = clip_json.get("connected")
                c_connected = (
                    isinstance(conn_raw, dict) and conn_raw.get("index", 0) >= 3
                )  # index 3=Connected, 4=Connected & previewing
                layer_info.clips.append(
                    ClipInfo(
                        id=c_id,
                        name=c_name,
                        column_index=col_idx,
                        connected=c_connected,
                    )
                )

            group_info.layers.append(layer_info)

        comp.groups.append(group_info)

    return comp

def fetch_active_clip_column_for_layer(conn: Dict[str, Any], layer_index: int) -> Optional[int]:
    """
    Determine which column index is currently 'active' on a given layer by
    querying the Resolume HTTP API:

      GET /composition/layers/{layer_index}

    Returns:
      - int column index (1-based) if we can infer one
      - 1 if nothing looks connected (treated as OFF)

    We look for any clip with a 'connected' or 'state' / 'connectionState'
    that indicates it's connected/playing.
    """
    base_url = make_resolume_base_url(conn)
    url = f"{base_url}/composition/layers/{layer_index}"

    auth = None
    if conn.get("username") and conn.get("password"):
        auth = (conn["username"], conn["password"])

    timeout = conn.get("timeout", 2.0)
    verify = bool(conn.get("verify_ssl", True))

    try:
        print(f"[HTTP] GET {url} (layer active clip)")
        resp = requests.get(url, auth=auth, timeout=timeout, verify=verify)
        resp.raise_for_status()
        layer_json = resp.json()
    except Exception as e:
        print(f"[WARN] Could not fetch layer {layer_index} details: {e}")
        # Fall back to 'OFF'
        return 1

    clips = list(_iter_clips(layer_json))


    print(f"[DEBUG] Layer {layer_index} JSON keys: {list(layer_json.keys())}")
    print(f"[DEBUG] Layer {layer_index} has {len(clips)} clips in HTTP response")

    active_col: Optional[int] = None

    for idx, clip_json in enumerate(clips):
        # Try to extract some identifier for debug
        clip_name = clip_json.get("name") or clip_json.get("Name") or clip_json.get("id") or f"#{idx+1}"

        # Various possible places "connected/playing" could hide
        connected_raw = clip_json.get("connected")
        if isinstance(connected_raw, dict):
            connected_flag = connected_raw.get('value')
            clip_index = connected_raw.get('index')
        else:
            connected_flag = None
            clip_index = None
        state_field = clip_json.get("state") or clip_json.get("connectionState")
        transport = clip_json.get("transport") or {}
        transport_state = transport.get("state")

        # Normalize to strings where needed
        def _to_str(v):
            # parameters sometimes wrap value in dicts like {"value": "..."}
            if isinstance(v, dict) and "value" in v:
                return str(v["value"])
            return str(v)

        state_str = _to_str(state_field) if state_field is not None else ""
        transport_str = _to_str(transport_state) if transport_state is not None else ""

        
        print(
            f"  [DEBUG] clip {idx+1}: name={clip_name!r}, "
            f"connected={connected_flag!r}, state={state_str!r}, "
            f"transport.state={transport_str!r}"
        )

        connected = False

        
        if isinstance(connected_flag, str):
            if connected_flag.lower().startswith("connected"):
                connected = True
                active_col = clip_index + 1  # convert 0-based to 1-based
        


    return active_col



# ---- Connections.yaml helpers ----

def load_connections(path: Path = Path(DEFAULT_CONNECTIONS_PATH)) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"connections.yaml not found at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
def get_resolume_http_connection(
    connections_cfg: Dict[str, Any],
    name: Optional[str] = None,
    io_section: str = "outputs",
) -> Dict[str, Any]:
    """
    Grab one HTTP connection config for Resolume from connections.yaml.

    New expected shape (method -> software -> from/to):

      http:
        resolume_arena:
          from:
            - name: "arena_http_in_local"
              host: "127.0.0.1"
              port: 8080
              use_https: false
              api_base: "/api/v1"
              timeout: 2.0

          to:
            - name: "arena_http_out_main"
              host: "127.0.0.1"
              port: 8080
              use_https: false
              api_base: "/api/v1"
              timeout: 2.0

    io_section:
      - "outputs"/"output"/"to" -> uses 'to' list
      - "inputs"/"input"/"from" -> uses 'from' list
    """
    http_cfg = connections_cfg.get("http", {})
    software_cfg = http_cfg.get("resolume_arena", {}) or http_cfg.get("arena", {})

    if not software_cfg:
        raise RuntimeError("No 'resolume_arena' section found under http in connections.yaml")

    # Map old io_section idea to new from/to
    sec_key = "to" if io_section.lower() in ("outputs", "output", "to") else "from"
    arena_list = software_cfg.get(sec_key, [])

    if not arena_list:
        raise RuntimeError(
            f"No resolume_arena HTTP entries found under http.resolume_arena.{sec_key}"
        )

    if name is None:
        return arena_list[0]

    for entry in arena_list:
        if entry.get("name") == name:
            return entry

    raise RuntimeError(
        f"HTTP Resolume connection named '{name}' not found under http.resolume_arena.{sec_key}"
    )
def get_show_bridge_osc_connection(
    connections_cfg: Dict[str, Any],
    name: Optional[str] = None,
    io_section: str = "outputs",
) -> Tuple[str, int]:
    """
    Get an OSC connection config for this program (show_bridge) from connections.yaml.

    Expected shape:

      osc:
        show_bridge:
          from:
            - name: "sb_osc_in_local"
              host: "127.0.0.1"
              port: 13001  # we LISTEN on this

          to:
            - name: "sb_osc_out_local"
              host: "127.0.0.1"
              port: 13000  # we SEND on this

    io_section:
      - "outputs"/"to"   -> use 'to' (where we send state)
      - "inputs"/"from"  -> use 'from' (where we listen)
    """
    osc_cfg = connections_cfg.get("osc", {})
    software_cfg = osc_cfg.get("show_bridge", {})

    if not software_cfg:
        raise RuntimeError("No 'show_bridge' section found under osc in connections.yaml")

    sec_key = "to" if io_section.lower() in ("outputs", "output", "to") else "from"
    sb_list = software_cfg.get(sec_key, [])

    if not sb_list:
        raise RuntimeError(
            f"No 'show_bridge' OSC entries found under osc.show_bridge.{sec_key}"
        )

    if name is None:
        entry = sb_list[0]
    else:
        entry = None
        for e in sb_list:
            if e.get("name") == name:
                entry = e
                break
        if entry is None:
            entry = sb_list[0]

    host = entry.get("host", "127.0.0.1")
    port = int(entry.get("port", 13000 if sec_key == "to" else 13001))
    return host, port

def get_resolume_osc_connection(
    connections_cfg: Dict[str, Any],
    name: Optional[str] = None,
    io_section: str = "outputs",
) -> Tuple[str, int]:
    """
    Try to get an OSC connection config for Resolume from connections.yaml.

    New expected shape:

      osc:
        resolume_arena:
          from:
            - name: "arena_osc_in_local"
              host: "127.0.0.1"
              port: 7001   # port we LISTEN on for incoming OSC

          to:
            - name: "arena_osc_out_local"
              host: "127.0.0.1"
              port: 7000   # port ARENA listens on

    If no OSC entry is present, we fall back to the HTTP host and DEFAULT_OSC_PORT.
    """
    osc_cfg = connections_cfg.get("osc", {})
    software_cfg = osc_cfg.get("resolume_arena", {}) or osc_cfg.get("arena", {})

    # Map old io_section idea to new from/to
    sec_key = "to" if io_section.lower() in ("outputs", "output", "to") else "from"
    arena_list = software_cfg.get(sec_key, [])

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

    # Fallback: piggyback off HTTP host if OSC isn't configured
    http_conn = get_resolume_http_connection(connections_cfg, name=None, io_section=io_section)
    host = http_conn.get("host", "127.0.0.1")
    port = DEFAULT_OSC_PORT
    return host, port
def get_synesthesia_osc_connection(
    connections_cfg: Dict[str, Any],
    name: Optional[str] = None,
    io_section: str = "outputs",
) -> Tuple[str, int]:
    """
    Get an OSC connection config for Synesthesia from connections.yaml.

    New expected shape:

      osc:
        synesthesia:
          from:
            - name: "syn_osc_in"
              host: "127.0.0.1"
              port: 12000

          to:
            - name: "syn_osc_out"
              host: "127.0.0.1"
              port: 12001

    io_section:
      - "outputs" -> use 'to' (we SEND to Synesthesia)
      - "inputs"  -> use 'from' (we LISTEN from Synesthesia)
    """
    osc_cfg = connections_cfg.get("osc", {})
    software_cfg = osc_cfg.get("synesthesia", {})

    if not software_cfg:
        raise RuntimeError("No 'synesthesia' section found under osc in connections.yaml")

    sec_key = "to" if io_section.lower() in ("outputs", "output", "to") else "from"
    syn_list = software_cfg.get(sec_key, [])

    if not syn_list:
        raise RuntimeError(
            f"No 'synesthesia' OSC entries found under osc.synesthesia.{sec_key}"
        )

    if name is None:
        entry = syn_list[0]
    else:
        entry = None
        for e in syn_list:
            if e.get("name") == name:
                entry = e
                break
        if entry is None:
            entry = syn_list[0]

    host = entry.get("host", "127.0.0.1")
    port = int(entry.get("port", 12001 if sec_key == "to" else 12000))
    return host, port

def make_resolume_base_url(conn: Dict[str, Any]) -> str:
    host = conn.get("host", "127.0.0.1")
    port = conn.get("port", 8080)
    use_https = bool(conn.get("use_https", False))
    api_base = conn.get("api_base", "/api/v1").rstrip("/")

    scheme = "https" if use_https else "http"
    return f"{scheme}://{host}:{port}{api_base}"


def fetch_composition_json(conn: Dict[str, Any]) -> Dict[str, Any]:
    base_url = make_resolume_base_url(conn)
    url = f"{base_url}/composition"

    auth = None
    if conn.get("username") and conn.get("password"):
        auth = (conn["username"], conn["password"])

    timeout = conn.get("timeout", 2.0)
    verify = bool(conn.get("verify_ssl", True))

    print(f"[HTTP] GET {url}")
    resp = requests.get(url, auth=auth, timeout=timeout, verify=verify)
    resp.raise_for_status()
    data = resp.json()
    print("[HTTP] /composition OK")
    return data


# ---------------------------------------------------------
# State (APC40 groups)
# ---------------------------------------------------------

@dataclass
class OutputGroupState:
    playing: bool = False
    playing_autopilot: bool = True

    effects: bool = False
    effects_autopilot: bool = True

    transforms: bool = False
    transforms_autopilot: bool = True

    # Dynamic mask layers (formerly fft_masks)
    masks: bool = False
    masks_autopilot: bool = True

    color: bool = False
    color_autopilot: bool = True

    # Per-group opacity and intensity
    opacity: float = 0.0
    intensity: float = 0.0

@dataclass
class LayerRuntimeState:
    """
    Per-layer runtime state used for OFF/PASSTHROUGH/CONTENT logic.

    Keyed by the 0-based global layer index in the composition.
    """
    playing: bool = False
    autopilot: bool = True
    current_clip_index: int | None = None
    last_clip_index: int | None = None


@dataclass
class GroupControlAddresses:
    """
    Where to send LED feedback for each boolean property and intensity presets.
    """
    channel: int
    prop_notes: Dict[str, int]          # property_name -> note number
    reset_note: int
    slider_cc: int
    intensity_preset_notes: list[int] = field(default_factory=list)


@dataclass
class ActionSpec:
    """
    Represents a bound action for a particular MIDI key.
    """
    action: str
    scope: str                   # "group" or "global"
    property_name: str | None = None
    group_index: int | None = None
    intensity_value: float | None = None  # for intensity presets
    scene_index: int | None = None        # for scene buttons


@dataclass
class VelocityProfile:
    """
    Describes how to light a note.
    """
    off: int = 0
    on: int = 127
    colors: list[int] = field(default_factory=list)

    def resolved_on(self) -> int:
        """
        Return the velocity we should use for a simple "on" state.
        Priority:
          1) First non-off color in colors[]
          2) Explicit on value
        """
        for c in self.colors:
            if c != self.off:
                return c
        return self.on


@dataclass
class MappingRuntime:
    action_map: Dict[MidiKey, ActionSpec] = field(default_factory=dict)
    group_addrs: Dict[int, GroupControlAddresses] = field(default_factory=dict)
    note_velocity: Dict[int, VelocityProfile] = field(default_factory=dict)
    scene_buttons: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # scene_idx -> (channel, note)


@dataclass
class ButtonPressState:
    is_down: bool = False
    last_down: float = 0.0
    pending_click: bool = False
    pending_click_time: float = 0.0


@dataclass
class MappingProfile:
    """
    Represents a single controller mapping profile loaded from ./mappings.
    One YAML file = one profile.
    """
    file_path: Path
    name: str            # human-friendly name (e.g. controller_name or file stem)
    controller_name: str
    input_cfg: Dict[str, Any]
    state_cfg: Dict[str, Any]


# ---------------------------------------------------------
# Press manager (short/long/double)
# ---------------------------------------------------------

class PressManager:
    """
    Detect short / long / double presses for note buttons.

    - Short: quick tap; only fired if no second tap within DOUBLE_WINDOW
    - Long: duration >= LONG_THRESHOLD (fires immediately on release)
    - Double: two short taps within DOUBLE_WINDOW
    """

    def __init__(self, double_window: float = 0.35, long_threshold: float = 0.6):
        self.double_window = double_window
        self.long_threshold = long_threshold
        self.states: Dict[MidiKey, ButtonPressState] = {}

    def handle_note_message(self, key: MidiKey, msg: mido.Message, now: float):
        """
        Process a note_on/note_off and return a list of (key, press_type)
        events that should fire immediately (long, double).
        Short presses are deferred and returned from poll().
        """
        events: list[Tuple[MidiKey, str]] = []
        st = self.states.setdefault(key, ButtonPressState())

        if msg.type == "note_on" and msg.velocity > 0:
            if not st.is_down:
                st.is_down = True
                st.last_down = now

        elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            if not st.is_down:
                return events
            st.is_down = False
            duration = now - st.last_down

            if duration >= self.long_threshold:
                st.pending_click = False
                events.append((key, "long"))
            else:
                if st.pending_click and (now - st.pending_click_time) <= self.double_window:
                    st.pending_click = False
                    events.append((key, "double"))
                else:
                    st.pending_click = True
                    st.pending_click_time = now

        return events

    def poll(self, now: float):
        """
        Check for pending single clicks whose double window has expired.
        Returns a list of (key, "short") events.
        """
        events: list[Tuple[MidiKey, str]] = []
        for key, st in self.states.items():
            if st.pending_click and (now - st.pending_click_time) > self.double_window:
                st.pending_click = False
                events.append((key, "short"))
        return events


# ---------------------------------------------------------
# Config loading (APC mappings)
# ---------------------------------------------------------

def load_yaml_file(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_mapping_profiles(mappings_dir: Path) -> List[MappingProfile]:
    """
    Load all *.yml / *.yaml in mappings_dir and return a list of MappingProfile.
    Each file is expected to contain:
      - controller_name
      - input_mappings
      - state_mappings
    """
    if not mappings_dir.exists():
        raise RuntimeError(f"Mappings directory does not exist: {mappings_dir}")

    profiles: List[MappingProfile] = []

    for path in sorted(mappings_dir.glob("*.y*ml")):
        raw = load_yaml_file(path)

        controller_name = raw.get("controller_name", path.stem)
        input_cfg = raw.get("input_mappings") or {}
        state_cfg = raw.get("state_mappings") or {}

        if not input_cfg or not state_cfg:
            print(f"WARNING: Skipping {path} (missing input_mappings or state_mappings)")
            continue

        profiles.append(
            MappingProfile(
                file_path=path,
                name=controller_name,
                controller_name=controller_name,
                input_cfg=input_cfg,
                state_cfg=state_cfg,
            )
        )

    if not profiles:
        raise RuntimeError(f"No valid mapping profiles found in {mappings_dir}")

    return profiles


def choose_mapping_profile(
    profiles: List[MappingProfile],
    midi_input_names: List[str],
    desired_name: Optional[str] = None,
) -> MappingProfile:
    """
    Select a mapping profile.

    Priority:
      1) If desired_name is provided, match by profile.name (case-insensitive) or file stem.
      2) Try to auto-match controller_name to a MIDI input port name.
      3) If only one profile exists, use it.
      4) Otherwise, prompt user to pick.
    """
    # 1) Explicit selection by name
    if desired_name:
        matches = [
            p for p in profiles
            if p.name.lower() == desired_name.lower()
            or p.file_path.stem.lower() == desired_name.lower()
        ]
        if matches:
            p = matches[0]
            print(f"Selected mapping profile by name: {p.name} ({p.file_path.name})")
            return p
        else:
            print(f"WARNING: No mapping profile named '{desired_name}' found. Ignoring.")

    # 2) Auto-match by controller_name inside MIDI input port name
    lower_ports = [n.lower() for n in midi_input_names]
    auto_candidates: List[MappingProfile] = []
    for p in profiles:
        cname = p.controller_name.lower()
        if any(cname in port for port in lower_ports):
            auto_candidates.append(p)

    if len(auto_candidates) == 1:
        p = auto_candidates[0]
        print(f"Automatically selected mapping profile '{p.name}' "
              f"based on MIDI ports and controller_name='{p.controller_name}'.")
        return p

    # 3) Only one profile total
    if len(profiles) == 1:
        p = profiles[0]
        print(f"Using only available mapping profile: {p.name} ({p.file_path.name})")
        return p

    # 4) Prompt user
    print("Available controller mapping profiles:")
    for idx, p in enumerate(profiles):
        print(f"  [{idx}] {p.name}  (file: {p.file_path.name})")

    while True:
        choice = input("Select mapping profile index: ").strip()
        try:
            idx = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue

        if 0 <= idx < len(profiles):
            return profiles[idx]

        print(f"Index out of range. Choose between 0 and {len(profiles) - 1}.")


def build_note_velocity_map(input_cfg: Dict[str, Any]) -> Dict[int, VelocityProfile]:
    """
    Build a map: note_number -> VelocityProfile from config like:

      velocity_mappings:
        maps:
          toggle_green:
            off: 0
            on: 1
          toggle_red: 3          # shorthand: off=0, on=3
          multi:
            off: 0
            colors: [1, 3, 5]

        ranges:
          - { start: 48, end: 56, map: toggle_green }

        notes:
          "57": toggle_orange
    """
    vel_cfg = input_cfg.get("velocity_mappings")
    if not vel_cfg:
        print("[VEL] No velocity_mappings section found.")
        return {}

    maps_cfg = vel_cfg.get("maps", {})
    ranges_cfg = vel_cfg.get("ranges", [])
    notes_cfg = vel_cfg.get("notes", {})

    named_profiles: Dict[str, VelocityProfile] = {}

    print("[VEL] maps:")
    for name, m in maps_cfg.items():
        if isinstance(m, int):
            prof = VelocityProfile(off=0, on=int(m), colors=[])
            print(f"  map '{name}' (int): off={prof.off} on={prof.on}")

        elif isinstance(m, list):
            colors = [int(v) for v in m]
            prof = VelocityProfile(off=0, on=127, colors=colors)
            print(f"  map '{name}' (list): off={prof.off} colors={prof.colors}")

        elif isinstance(m, dict):
            off_val = int(m.get("off", 0))

            colors_raw = m.get("colors", [])
            colors = [int(v) for v in colors_raw] if isinstance(colors_raw, list) else []

            on_raw = m.get("on")
            if on_raw is None:
                derived = next((c for c in colors if c != off_val), None)
                if derived is not None:
                    on_val = derived
                    print(f"  map '{name}': derived on={on_val} from colors={colors}")
                else:
                    on_val = 127
                    print(f"  WARNING: map '{name}' has no 'on' and no usable colors; "
                          f"defaulting on={on_val}")
            else:
                on_val = int(on_raw)

            prof = VelocityProfile(off=off_val, on=on_val, colors=colors)
            print(f"  map '{name}' (dict): off={prof.off} on={prof.on} colors={prof.colors}")

        else:
            print(f"  WARNING: map '{name}' has unsupported type {type(m)}; "
                  f"defaulting to off=0 on=127")
            prof = VelocityProfile(off=0, on=127, colors=[])

        named_profiles[name] = prof

    note_map: Dict[int, VelocityProfile] = {}

    print("[VEL] ranges:")
    for r in ranges_cfg:
        try:
            start = int(r["start"])
            end = int(r["end"])
            map_name = r["map"]
        except KeyError as e:
            print(f"  WARNING: invalid range entry {r}: missing {e}")
            continue

        profile = named_profiles.get(map_name)
        if not profile:
            print(f"  WARNING: unknown map '{map_name}' in range {r}")
            continue

        a, b = sorted((start, end))
        for note in range(a, b + 1):
            note_map[note] = profile
        print(f"  notes {a}–{b} -> map '{map_name}'")

    print("[VEL] notes:")
    for note_str, map_name in notes_cfg.items():
        try:
            note = int(note_str)
        except ValueError:
            print(f"  WARNING: invalid note key '{note_str}'")
            continue

        profile = named_profiles.get(map_name)
        if not profile:
            print(f"  WARNING: unknown map '{map_name}' for note {note}")
            continue

        note_map[note] = profile
        print(f"  note {note} -> map '{map_name}' (off={profile.off} on={profile.on} "
              f"colors={profile.colors})")

    if 57 in note_map:
        prof = note_map[57]
        print(f"[VEL] FINAL: note 57 mapped to off={prof.off} on={prof.on} "
              f"colors={prof.colors}")
    else:
        print("[VEL] FINAL: note 57 has NO velocity profile")

    return note_map


def build_runtime_mapping(input_cfg: Dict[str, Any],
                          state_cfg: Dict[str, Any]) -> MappingRuntime:
    """
    Combine input_mappings + state_mappings into a mapping from
    raw MIDI events -> ActionSpec, plus LED address information.
    """
    runtime = MappingRuntime()

    runtime.note_velocity = build_note_velocity_map(input_cfg)

    group_state_cfg = state_cfg.get("groups", {})
    group_props_cfg = group_state_cfg.get("properties", {})
    group_reset_cfg = group_state_cfg.get("reset", {})
    group_opacity_cfg = group_state_cfg.get("opacity", {})
    group_intensity_presets_cfg = group_state_cfg.get("intensity_presets", [])

    # --- Per-group controls ---
    for group_id_str, g in input_cfg["groups"].items():
        group_idx = int(group_id_str) - 1  # 1-8 -> 0-7
        channel = g["channel"]

        control_note_map: Dict[str, int] = {}
        for btn in g["clip_buttons"]:
            control_note_map[btn["name"]] = btn["note"]
        for name in ("clip_stop", "track_select", "activator", "solo", "arm"):
            if name in g:
                control_note_map[name] = g[name]["note"]

        slider_cc = g["slider"]["cc"]

        prop_notes: Dict[str, int] = {}
        preset_notes: list[int] = []

        # 1) boolean properties
        for prop_name, cfg_prop in group_props_cfg.items():
            control_name = cfg_prop["control"]
            action_name = cfg_prop["action"]

            if control_name not in control_note_map:
                continue

            note = control_note_map[control_name]
            key = ("note", channel, note)

            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="group",
                property_name=prop_name,
                group_index=group_idx,
            )
            prop_notes[prop_name] = note

        # 2) reset (if present)
        reset_note = -1
        reset_control_name = group_reset_cfg.get("control")
        reset_action_name = group_reset_cfg.get("action")
        if reset_control_name and reset_control_name in control_note_map:
            reset_note = control_note_map[reset_control_name]
            key = ("note", channel, reset_note)
            runtime.action_map[key] = ActionSpec(
                action=reset_action_name,
                scope="group",
                property_name=None,
                group_index=group_idx,
            )

        # 3) opacity slider
        if group_opacity_cfg.get("control") == "slider":
            action_name = group_opacity_cfg["action"]
            key = ("cc", channel, slider_cc)
            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="group",
                property_name="opacity",
                group_index=group_idx,
            )

        # 4) intensity presets
        for preset in group_intensity_presets_cfg:
            control_name = preset["control"]
            value = float(preset["value"])
            if control_name not in control_note_map:
                continue
            note = control_note_map[control_name]
            preset_notes.append(note)

            key = ("note", channel, note)
            runtime.action_map[key] = ActionSpec(
                action="set_intensity_preset",
                scope="group",
                property_name="intensity",
                group_index=group_idx,
                intensity_value=value,
            )

        runtime.group_addrs[group_idx] = GroupControlAddresses(
            channel=channel,
            prop_notes=prop_notes,
            reset_note=reset_note,
            slider_cc=slider_cc,
            intensity_preset_notes=preset_notes,
        )

    # --- Global controls ---
    global_state_cfg = state_cfg.get("global", {})
    input_global_cfg = input_cfg.get("global", {})

    timing_cfg = input_global_cfg.get("timing_controls", {})

    for name, entry in global_state_cfg.items():
        control_name = entry["control"]
        action_name = entry["action"]

        if control_name.startswith("scene_launch"):
            scene_list = input_global_cfg.get("scene_launch", [])
            target = None
            for idx, s_entry in enumerate(scene_list):
                if s_entry.get("name") == control_name:
                    target = (idx, s_entry)
                    break

            if not target:
                print(f"WARNING: state_mappings.global '{name}' refers to "
                      f"control '{control_name}' but no matching entry in "
                      f"input_mappings.global.scene_launch.")
                continue

            scene_idx, s_entry = target
            key = ("note", s_entry.get("channel", 0), s_entry["note"])

            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="global",
                property_name=None,
                group_index=None,
                scene_index=scene_idx,
            )

            runtime.scene_buttons[scene_idx] = (
                s_entry.get("channel", 0),
                s_entry["note"],
            )

        elif control_name == "global_slider" and "global_slider" in input_global_cfg:
            g_slider = input_global_cfg["global_slider"]
            if "cc" in g_slider:
                key = ("cc", g_slider.get("channel", 0), g_slider["cc"])
            elif "note" in g_slider:
                key = ("note", g_slider.get("channel", 0), g_slider["note"])
            else:
                continue

            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="global",
                property_name=None,
                group_index=None,
            )

        elif control_name == "queue_level" and "queue_level" in input_global_cfg:
            q = input_global_cfg["queue_level"]
            if "note" in q:
                key = ("note", q.get("channel", 0), q["note"])
            elif "cc" in q:
                key = ("cc", q.get("channel", 0), q["cc"])
            else:
                continue

            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="global",
                property_name=None,
                group_index=None,
            )

        elif control_name in ("play", "stop") and "transport" in input_global_cfg:
            t = input_global_cfg["transport"].get(control_name)
            if not t:
                continue
            key = ("note", t.get("channel", 0), t["note"])
            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="global",
                property_name=None,
                group_index=None,
            )

        elif control_name == "stop_all_clips" and "stop_all_clips" in input_global_cfg:
            s = input_global_cfg["stop_all_clips"]
            key = ("note", s.get("channel", 0), s["note"])
            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="global",
                property_name=None,
                group_index=None,
            )

        elif control_name in ("nudge_minus", "nudge_plus", "tap_tempo", "shift_resync"):
            src = timing_cfg.get(control_name)
            if not src:
                continue
            key = ("note", src.get("channel", 0), src["note"])
            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="global",
                property_name=None,
                group_index=None,
            )

    return runtime


# ---------------------------------------------------------
# MIDI port selection
# ---------------------------------------------------------

def auto_select_port(port_names, controller_name: str, kind: str) -> str:
    if not port_names:
        raise RuntimeError(f"No MIDI {kind} ports found.")

    name_lower = controller_name.lower()
    matches = [n for n in port_names if name_lower in n.lower()]

    if len(matches) == 1:
        print(f"Automatically selected {kind} '{matches[0]}' "
              f"for controller_name='{controller_name}'.")
        return matches[0]
    elif len(matches) > 1:
        print(f"Multiple {kind} ports match controller_name='{controller_name}':")
        for i, n in enumerate(matches):
            print(f"  [{i}] {n}")
        print(f"Choosing first match: {matches[0]}")
        return matches[0]
    else:
        raise RuntimeError(
            f"No MIDI {kind} port matched '{controller_name}'. "
            f"Available: {port_names}"
        )


# ---------------------------------------------------------
# State machine
# ---------------------------------------------------------

class Apc40StateMachine:
    """
    Manages 8 output groups and pushes LED updates out via MIDI.
    Also has a reference to a Resolume CompositionInfo and its
    CompositionMapping, an OSC client, and precomputes which
    layers belong to which high-level roles per APC group.
    """

    # Boolean properties that can be toggled via the controller.  These correspond
    # to high-level layer roles in the composition mapping.  Added 'masks'
    # to support per-group dynamic mask toggling.
    BOOL_PROPS = ("playing", "effects", "transforms", "masks", "color")


    # Map state-machine boolean properties to layer roles in the
    # composition mapping. You can tweak this to your naming scheme.
    PROP_TO_ROLE = {
        # Group 'playing' property drives the fill layers.
        "playing": "fills",
        "effects": "effects",
        "transforms": "transforms",
        # Dynamic mask layers (formerly fft_masks) use the 'masks' role
        "masks": "masks",
        # Color button targets layers tagged with the 'colors' role
        "color": "colors",
    }

    def __init__(
        self,
        runtime: MappingRuntime,
        midi_out: mido.ports.BaseOutput,
        osc_client: Optional[SimpleUDPClient] = None,          # Resolume control
        resolume_conn: Optional[Dict[str, Any]] = None,
        mapping_dir: str | os.PathLike[str] = DEFAULT_COMPOSITION_MAPPING_DIR,
        syn_osc_client: Optional[SimpleUDPClient] = None,
        broadcast_osc_client: Optional[SimpleUDPClient] = None,  # show_bridge bus
        state_output_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.runtime = runtime
        self.midi_out = midi_out
        self.osc_client = osc_client                    # Resolume
        self.resolume_conn = resolume_conn
        self.mapping_dir = str(mapping_dir)
        self.syn_osc_client = syn_osc_client
        self.broadcast_osc_client = broadcast_osc_client  # show_bridge
        self.state_output_cfg = state_output_cfg or {}

         # --- loggers ---
        self.logger = logging.getLogger("show_bridge.state")
        self.osc_resolume_logger = logging.getLogger("osc.resolume.out")
        self.osc_syn_logger = logging.getLogger("osc.synesthesia.out")
        self.osc_bus_logger = logging.getLogger("osc.show_bridge.out")
        self.osc_bus_in_logger = logging.getLogger("osc.show_bridge.in")


        """
        Initialize the state machine with runtime configuration, MIDI output and OSC client.
        Optionally accept a Resolume HTTP connection configuration and a mapping
        directory so that the composition can be re-fetched on demand via
        `resync_with_resolume`.
        """
        self.runtime = runtime
        self.midi_out = midi_out
        self.osc_client = osc_client

        osc_client: Optional[SimpleUDPClient] = None
        syn_osc_client: Optional[SimpleUDPClient] = None
        sb_osc_client: Optional[SimpleUDPClient] = None

        try:
            connections_cfg = load_connections(Path(DEFAULT_CONNECTIONS_PATH))

            # HTTP for composition JSON (Resolume)
            conn_http = get_resolume_http_connection(connections_cfg, name=None, io_section="outputs")
            comp_json = fetch_composition_json(conn_http)
            comp_name = _guess_composition_name(comp_json)
            comp_mapping = CompositionMapping.from_yaml_dir(
                composition_name=comp_name,
                mapping_dir=Path(DEFAULT_COMPOSITION_MAPPING_DIR),
            )
            comp_model = build_composition_model(comp_json, comp_mapping)
            self.logger.info(f"[RESOLUME] Loaded composition '{comp_model.name}' with {len(comp_model.groups)} groups.")
            debug_dump_composition_columns(comp_model, self.logger)

            # OSC client (Resolume control)
            osc_host, osc_port = get_resolume_osc_connection(connections_cfg, name=None, io_section="outputs")
            osc_client = get_osc_client(osc_host, osc_port)
            self.logger.info(f"[OSC] Using Resolume OSC at {osc_host}:{osc_port}")

            # OSC client (show_bridge telemetry / state broadcast)
            try:
                sb_host, sb_port = get_show_bridge_osc_connection(connections_cfg, name=None, io_section="outputs")
                sb_osc_client = get_osc_client(sb_host, sb_port)
                self.logger.info(f"[OSC] Using show_bridge OSC at {sb_host}:{sb_port} for state broadcasts")
            except Exception as sb_e:
                self.logger.warning(f"[OSC] WARNING: could not initialize show_bridge OSC: {sb_e}")
                sb_osc_client = None

            # OSC client (Synesthesia) - optional
            try:
                syn_host, syn_port = get_synesthesia_osc_connection(connections_cfg, name=None, io_section="outputs")
                syn_osc_client = get_osc_client(syn_host, syn_port)
                self.logger.info(f"[SYN] Using Synesthesia OSC at {syn_host}:{syn_port}")
            except Exception as syn_e:
                self.logger.warning(f"[SYN] WARNING: could not initialize Synesthesia OSC: {syn_e}")
                syn_osc_client = None

        except Exception as e:
            sb_log(self.logger, logging.WARNING, "BRIDGE", "RESOLUME", f"Could not initialize Resolume composition/OSC: {e}")
            conn_http = None
            osc_client = None
            sb_osc_client = None
            syn_osc_client = None

        # Store Resolume connection details for resync operations
        self.resolume_conn: Optional[Dict[str, Any]] = resolume_conn
        # Path to directory containing composition mapping files
        self.mapping_dir = mapping_dir

        # State-output configuration.  If provided at construction time,
        # use it directly; otherwise attempt to load from a default file.
        # This allows the user to configure how state-machine updates
        # are forwarded via OSC without recompiling Python code.
        if state_output_cfg is not None:
            self.state_output_cfg = state_output_cfg
        else:
            try:
                # Load from settings/state_machine_mappings.yaml if present
                cfg_path = Path("settings") / "state_machine_mappings.yaml"
                if cfg_path.exists():
                    self.state_output_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                else:
                    self.state_output_cfg = {}
            except Exception as e:
                sb_log(self.logger, logging.WARNING, "BRIDGE", "STATE-OUTPUT", f"Failed to load state-machine output config: {e}")
                self.state_output_cfg = {}

        self.groups = [OutputGroupState() for _ in range(8)]

        # Resolume pieces
        self.composition: Optional[CompositionInfo] = None
        self.composition_mapping: Optional[CompositionMapping] = None
        # APC index (1–8) -> role name -> list[LayerInfo]
        self.group_role_layers: Dict[int, Dict[str, List[LayerInfo]]] = {}

        # Per-note velocity profiles
        self.note_velocity: Dict[int, VelocityProfile] = runtime.note_velocity

        self.global_autopilot: bool = True
        self.global_intensity_scale = 1.0
        self.global_opacity = 1.0
        self.global_queue_level: int = 0

        # Per-layer runtime state keyed by global layer index
        self.layer_states: dict[int, LayerRuntimeState] = {}

        # Routing layers: APC group index -> Router LayerInfo
        self.routing_layers_by_group: dict[int, LayerInfo] = {}
        self.routing_refresh_period = 2.0  # seconds
        self.last_routing_refresh = time.monotonic()

        # Beat-based autopilot: number of beat4 pulses until next roll
        self.autopilot_beats_remaining = 64

        # Blink engine
        self.fast_blink_on = True
        self.slow_blink_on = True
        self.fast_period = 0.25
        self.slow_period = 0.75
        now = time.monotonic()
        self.last_fast_toggle = now
        self.last_slow_toggle = now

        # Scene snapshots
        self.scene_snapshots: Dict[int, list[OutputGroupState]] = {}
        self.active_scene: int | None = None
        self.active_scene_pristine: bool = False

        # Initialize intensity LEDs
        self._init_intensity_preset_leds()

    def on_external_state_osc(self, address: str, *args) -> None:
        """
        Handle inbound OSC from 'to show bridge' (TouchDesigner or other controllers).
        
        When user modifies state in TD UI, it sends OSC messages. We parse and apply
        those updates to the state machine, keeping local and remote UIs in sync.
        """
        if not args:
            return
        
        value = args[0]
        self._apply_external_state_update(address, value)

    def _apply_external_state_update(self, address: str, value: Any) -> None:
        """
        Apply an external state update (from TouchDesigner or another controller).
        
        Parses addresses like:
          /state/group/{group}/playing/enabled
          /state/group/{group}/opacity
          /state/group/{group}/playing/autopilot
          /state/group/{group}/playing/name
        """
        try:
            # Try to parse /state/group/{group}/{prop_or_role}/{subprop}
            # Examples:
            #   /state/group/1/opacity -> group=1, role_or_prop=opacity
            #   /state/group/1/playing/enabled -> group=1, role=playing, subprop=enabled
            #   /state/group/1/playing/autopilot -> group=1, role=playing, subprop=autopilot
            parts = address.split("/")
            if len(parts) < 5 or parts[1] != "state" or parts[2] != "group":
                return

            try:
                group_num = int(parts[3])  # 1-based APC group
            except (ValueError, IndexError):
                return

            group_idx = group_num - 1  # Convert to 0-based internal index
            if not (0 <= group_idx < len(self.groups)):
                sb_log(self.logger, logging.DEBUG, "BRIDGE", "STATE-INPUT", f"Group index out of range: {group_num}")
                return

            g = self.groups[group_idx]

            # Handle /state/group/{group}/opacity or /state/group/{group}/intensity
            if len(parts) == 5:
                prop_name = parts[4]
                if prop_name == "opacity":
                    try:
                        g.opacity = max(0.0, min(1.0, float(value)))
                        sb_log(self.logger, logging.INFO, "BRIDGE", "STATE-INPUT", f"Updated group {group_num} opacity -> {g.opacity}")
                        self._broadcast_state_update(group_idx, "opacity", g.opacity)
                    except (ValueError, TypeError):
                        pass
                elif prop_name == "intensity":
                    try:
                        g.intensity = max(0.0, min(1.0, float(value)))
                        sb_log(self.logger, logging.INFO, "BRIDGE", "STATE-INPUT", f"Updated group {group_num} intensity -> {g.intensity}")
                        self._broadcast_state_update(group_idx, "intensity", g.intensity)
                    except (ValueError, TypeError):
                        pass
                elif prop_name == "name":
                    # Group name (read-only feedback, typically ignored on inbound)
                    pass
                return

            # Handle /state/group/{group}/{role}/{subprop}
            if len(parts) >= 6:
                role_or_prop = parts[4]
                subprop = parts[5]

                # Boolean properties: /state/group/{group}/{prop}/enabled or /{prop}/autopilot
                if subprop == "enabled":
                    # playing/enabled, effects/enabled, etc.
                    prop_name = role_or_prop
                    if prop_name in self.BOOL_PROPS:
                        try:
                            bool_value = bool(value) if not isinstance(value, (int, float)) else (value != 0.0)
                            setattr(g, prop_name, bool_value)
                            sb_log(self.logger, logging.INFO, "BRIDGE", "STATE-INPUT", f"Updated group {group_num} {prop_name} -> {bool_value}")
                            self._broadcast_state_update(group_idx, prop_name, bool_value)
                        except Exception as e:
                            sb_log(self.logger, logging.DEBUG, "BRIDGE", "STATE-INPUT", f"Failed to set {prop_name}: {e}")

                elif subprop == "autopilot":
                    # playing/autopilot, effects/autopilot, etc.
                    prop_name = role_or_prop
                    auto_attr = f"{prop_name}_autopilot"
                    if prop_name in self.BOOL_PROPS and hasattr(g, auto_attr):
                        try:
                            bool_value = bool(value) if not isinstance(value, (int, float)) else (value != 0.0)
                            setattr(g, auto_attr, bool_value)
                            sb_log(self.logger, logging.INFO, "BRIDGE", "STATE-INPUT", f"Updated group {group_num} {auto_attr} -> {bool_value}")
                            self._broadcast_state_update(group_idx, auto_attr, bool_value)
                        except Exception as e:
                            sb_log(self.logger, logging.DEBUG, "BRIDGE", "STATE-INPUT", f"Failed to set {auto_attr}: {e}")

                elif subprop == "name":
                    # e.g., playing/name, effects/name (read-only feedback, typically ignored)
                    pass
        except Exception as e:
            sb_log(self.logger, logging.DEBUG, "BRIDGE", "STATE-INPUT", f"Error parsing external state update {address}: {e}")

    def on_clip_connect(self, address: str, *args):
        """
        Resolume callback when a clip is connected / starts playing.

        address: /composition/layers/{layer}/clips/{column}/connect
        args[0]: usually clip name (string) or 1.0
        """
        m = re.match(r"^/composition/layers/(\d+)/clips/(\d+)/connect$", address)
        if not m:
            return

        osc_layer_index = int(m.group(1))       # 1-based
        column_index = int(m.group(2))          # 1-based
        global_layer_index = osc_layer_index - 1

        clip_name = ""
        if args:
            # Resolume tends to send the clip name as a string arg
            if isinstance(args[0], str):
                clip_name = args[0]
            else:
                clip_name = str(args[0])

        layer_info = self.layer_index_map.get(global_layer_index)
        if not layer_info:
            sb_log(self.logger, logging.WARNING, "BRIDGE", "ARENA-CLIP", f"Unknown layer index {global_layer_index}")
            return

        layer_state = self.layer_states.setdefault(
            global_layer_index, LayerRuntimeState()
        )
        layer_state.current_clip_index = column_index
        layer_state.playing = bool(column_index and column_index >= 2)

        sb_log(
            self.logger,
            logging.INFO,
            "BRIDGE",
            "ARENA-CLIP",
            f"Layer {global_layer_index} '{layer_info.name}' -> column={column_index} name={clip_name!r} playing={layer_state.playing}",
        )

        # --- NEW: map this layer back to APC group + role and broadcast clip name ---
        group_idx_for_state: int | None = None
        role_for_state: str | None = None

        for apc_index, role_layers in self.group_role_layers.items():
            for role, layers in role_layers.items():
                for li in layers:
                    if li.global_index == global_layer_index:
                        group_idx_for_state = apc_index - 1   # APC 1-8 -> 0-7 internal
                        role_for_state = role
                        break
                if role_for_state is not None:
                    break
            if role_for_state is not None:
                break

        if (
            group_idx_for_state is not None
            and role_for_state is not None
            and clip_name
        ):
            try:
                # This ends up at /state/group/<idx>/<role>/name
                self._broadcast_state_update(
                    group_idx_for_state,
                    f"{role_for_state}_name",
                    clip_name,
                )
            except Exception:
                pass

        # (rest of your existing on_clip_connect logic continues here)
        # e.g. self._refresh_group_booleans()

    def _refresh_group_booleans_from_layer_states(self) -> None:
        """
        Recompute each APC group's boolean properties (playing, effects, transforms,
        masks, color) based on the current per-layer 'playing' state and the
        composition mapping in group_role_layers.
        """
        if self.composition is None:
            return

        for apc_index, role_layers in self.group_role_layers.items():
            group_idx = apc_index - 1
            if not (0 <= group_idx < len(self.groups)):
                continue

            g_state = self.groups[group_idx]

            for prop, role in self.PROP_TO_ROLE.items():
                layers_for_role = role_layers.get(role, [])
                any_on = False
                for layer in layers_for_role:
                    st = self.layer_states.get(layer.global_index)
                    if st and st.playing:
                        any_on = True
                        break
                setattr(g_state, prop, any_on)

    def broadcast_full_state(self) -> None:
        """
        Broadcast the *entire* state machine snapshot over OSC.

        - Called on startup (after composition + mappings are initialized).
        - Called on scene launches.
        """
        if not self.state_output_cfg:
            sb_log(self.logger, logging.INFO, "BRIDGE", "STATE-OSC", "No state output config; skipping full-state broadcast.")
            return

        # 1) Per-group properties (playing/effects/transforms/masks/color, opacity, intensity)
        for group_idx, g in enumerate(self.groups):
            # Boolean props and their autopilots
            for prop in self.BOOL_PROPS:
                try:
                    val = getattr(g, prop)
                    self._broadcast_state_update(group_idx, prop, val)
                except Exception as e:
                    sb_log(self.logger, logging.WARNING, "BRIDGE", "STATE-OSC", f"Failed bool broadcast for group={group_idx+1} prop={prop}: {e}")

                auto_name = f"{prop}_autopilot"
                if hasattr(g, auto_name):
                    try:
                        auto_val = getattr(g, auto_name)
                        self._broadcast_state_update(group_idx, auto_name, auto_val)
                    except Exception as e:
                        sb_log(self.logger, logging.WARNING, "BRIDGE", "STATE-OSC", f"Failed autopilot broadcast for group={group_idx+1} prop={auto_name}: {e}")

            # Scalar props
            for scalar_prop in ("opacity", "intensity"):
                try:
                    val = getattr(g, scalar_prop)
                    self._broadcast_state_update(group_idx, scalar_prop, float(val))
                except Exception as e:
                    sb_log(self.logger, logging.WARNING, "BRIDGE", "STATE-OSC", f"Failed scalar broadcast for group={group_idx+1} prop={scalar_prop}: {e}")

            # NEW: broadcast the user-facing group name for each APC row.
            # Prefer the mapping from composition_mapping.apc_groups (YAML). If
            # that's not present, fall back to the composition model's group
            # name. If neither is available, broadcast an empty string.
            try:
                apc_index = group_idx + 1
                group_name = ""
                if getattr(self, "composition_mapping", None) is not None:
                    group_name = self.composition_mapping.apc_groups.get(apc_index, "") or ""
                if not group_name and getattr(self, "composition", None) is not None:
                    grp = self.composition.group_for_apc(apc_index)
                    if grp is not None:
                        group_name = grp.name or ""
                self._broadcast_state_update(group_idx, "name", group_name)
            except Exception as e:
                sb_log(self.logger, logging.WARNING, "BRIDGE", "STATE-OSC", f"Failed name broadcast for group={group_idx+1}: {e}")
                

        # 2) Global props: global_autopilot, global_intensity_scale, global_opacity
        try:
            self._broadcast_state_update(None, "autopilot", bool(self.global_autopilot))
        except Exception:
            pass

        try:
            self._broadcast_state_update(None, "global_intensity_scale", float(self.global_intensity_scale))
        except Exception:
            pass

        try:
            self._broadcast_state_update(None, "global_opacity", float(self.global_opacity))
        except Exception:
            pass
        sb_log(self.logger, logging.INFO, "BRIDGE", "STATE-OSC", "Full state broadcast completed.")

    def on_transport_beat(self, beat_kind: str) -> None:
        """
        Called from the OSC server thread when a beat pulse is received.
        We use beat4 as the master clock for the autopilot.
        """
        if beat_kind != "beat4":
            return

        if self.autopilot_beats_remaining > 0:
            self.autopilot_beats_remaining -= 1

        if self.autopilot_beats_remaining <= 0:
            self._run_autopilot_cycle()
            self.autopilot_beats_remaining = 64
    def _run_autopilot_cycle(self) -> None:
        """
        Run one autopilot cycle over all groups and properties:

        For each group and each boolean property where:
          - property is True
          - its *_autopilot flag is True
          - global_autopilot is True

        trigger the "next" behavior for that property.
        """
        if not self.global_autopilot:
            return

        self.logger.info("[AUTOPILOT] Running beat-based autopilot cycle")

        for group_idx, g in enumerate(self.groups):
            for prop_name in self.BOOL_PROPS:
                value = getattr(g, prop_name)
                autopilot_attr = f"{prop_name}_autopilot"
                autopilot = getattr(g, autopilot_attr, False)

                if value and autopilot:
                    self.logger.info(
                        f"[AUTOPILOT] Group {group_idx + 1}, prop={prop_name} "
                        f"-> NEXT (autopilot cycle)"
                    )
                    # Use your existing method:
                    self.next_property(group_idx, prop_name)


    def _choose_clip_for_layer(
        self,
        layer: LayerInfo,
        effective_intensity: float,
        rng: random.Random,
    ) -> int:
        """
        Decide whether to use PASSTHROUGH (2) or CONTENT (>=3) based on intensity,
        choose a random valid CONTENT clip if needed, and update per-layer state.

        Returns the chosen column index (2 or >=3).
        """
        layer_state = self.layer_states.setdefault(
            layer.global_index, LayerRuntimeState()
        )

        # Collect valid content clips (>= 3, non-empty)
        content_clips = [c for c in layer.clips if c.column_index >= 3]
        if not content_clips:
            # No content, fallback to passthrough (2)
            chosen_index = 2
        else:
            roll = rng.random()
            if roll > effective_intensity:
                # Choose PASSTHROUGH (2)
                chosen_index = 2
            else:
                chosen_clip = rng.choice(content_clips)
                chosen_index = chosen_clip.column_index

        layer_state.last_clip_index = layer_state.current_clip_index
        layer_state.current_clip_index = chosen_index
        layer_state.playing = True

        return chosen_index

    def _set_layer_to_clip(
        self,
        layer: LayerInfo,
        column_index: int,
        save_current: bool,
    ) -> None:
        """
        Send /connect to a specific column for a layer.

        If save_current is True and column_index >= 2, update current_clip_index.
        """
        osc_layer_index = layer.global_index + 1
        osc_path = f"/composition/layers/{osc_layer_index}/clips/{column_index}/connect"
        self.send_osc(osc_path, 1.0)

        layer_state = self.layer_states.setdefault(
            layer.global_index, LayerRuntimeState()
        )
        if column_index == 1:
            layer_state.playing = False
            # Don't change current_clip_index
        elif save_current and column_index >= 2:
            layer_state.last_clip_index = layer_state.current_clip_index
            layer_state.current_clip_index = column_index
            layer_state.playing = True

    # ---- Synesthesia helpers ----

    def send_synesthesia_playlist_next(self) -> None:
        """
        Send /playlist/next to Synesthesia (if configured).
        """
        addr = "/playlist/next"
        val = 1.0

        if self.syn_osc_client is None:
            msg = f"(no client) WOULD SEND {addr} {val!r}"
            sb_log(self.logger, logging.WARNING, "BRIDGE", "SYN", msg)
            sb_log(self.osc_syn_logger, logging.WARNING, "SYN", "OSC", msg)
            return

        self.osc_syn_logger.info("SEND %s %r", addr, val)
        sb_log(self.osc_syn_logger, logging.INFO, "SYN", "OSC", f"SEND {addr} {val!r}")
        self.syn_osc_client.send_message(addr, val)

    # ---- OSC helpers ----

    def send_osc(self, address: str, value: Any | None = None) -> None:
        """
        Send a single OSC message to Resolume if osc_client is available.
        Logs via the osc.resolume.out logger and still prints to console.
        """
        if self.osc_client is None:
            msg = f"(no client) WOULD SEND {address} {value!r}"
            sb_log(self.osc_resolume_logger, logging.WARNING, "ARENA", "OSC", msg)
            # self.osc_resolume_logger.warning(msg)
            return

        # Normal path: log + send
        self.osc_resolume_logger.info("SEND %s %r", address, value)
        sb_log(self.osc_resolume_logger, logging.INFO, "ARENA", "OSC", "SEND %s %r", address, value)

        try:
            if value is None:
                self.osc_client.send_message(address, 0.0)
            else:
                self.osc_client.send_message(address, value)

        except OSError as e:
            sb_log(self.osc_resolume_logger, logging.ERROR, "ARENA", "OSC",
                   "Send failed %s %r: %s", address, value, e)
        finally:
            # If we're sending a clip connect command ourselves, proactively
            # broadcast the clip name to the state bus so consumers see the
            # change immediately (rather than waiting for Resolume to echo).
            try:
                m = re.match(r"^/composition/layers/(\d+)/clips/(\d+)/connect$", address)
                if m and self.composition is not None:
                    osc_layer_index = int(m.group(1))
                    column_index = int(m.group(2))
                    global_layer_index = osc_layer_index - 1

                    # Resolve layer info -> clip name (if available)
                    layer_info = self.layer_index_map.get(global_layer_index)
                    clip_name = ""
                    if layer_info:
                        for c in getattr(layer_info, 'clips', []) or []:
                            if getattr(c, 'column_index', None) == column_index:
                                clip_name = getattr(c, 'name', '') or ''
                                break

                    # Map layer -> apc group + role
                    group_idx_for_state = None
                    role_for_state = None
                    for apc_index, role_layers in self.group_role_layers.items():
                        for role, layers in role_layers.items():
                            for li in layers:
                                if li.global_index == global_layer_index:
                                    group_idx_for_state = apc_index - 1
                                    role_for_state = role
                                    break
                            if role_for_state is not None:
                                break
                        if role_for_state is not None:
                            break

                    if group_idx_for_state is not None and role_for_state is not None:
                        # Use <role>_name mapping (e.g. playing_name)
                        prop = f"{role_for_state}_name"
                        try:
                            self._broadcast_state_update(group_idx_for_state, prop, clip_name)
                        except Exception:
                            pass
            except Exception:
                pass
                   
    def _osc_pulse(self, address: str, value: float = 1.0, off_value: float = 0.0) -> None:
        """
        Send a quick on/off pulse as two OSC messages.
        """
        self.logger.info(f"[OSC] PULSE {address} {value} -> {off_value}")
        self.send_osc(address, float(value))
        self.send_osc(address, float(off_value))

    # ---- Resolume hook ----
    def initialize_state_from_composition(self) -> None:
        if self.composition is None:
            return
        self.logger.info("[INIT] Syncing with composition...")
        

        # Pass 1 — apply per-layer ON/OFF based on active column
        for group in self.composition.groups:
            for layer in group.layers:
                col = layer.active_column_index or 1
                state = self.layer_states.setdefault(layer.global_index, LayerRuntimeState())
                state.current_clip_index = col
                state.playing = bool(col >= 2)

                self.logger.info(f"[INIT] Layer {layer.global_index} '{layer.name}' col={col} -> ON={state.playing}")

        # Pass 2 — aggregate to group state
        for apc_group_index, roles in self.group_role_layers.items():
            g = self.groups[apc_group_index - 1]

            for prop, role in self.PROP_TO_ROLE.items():
                layers_for_role = roles.get(role, [])
                any_on = any(
                    self.layer_states.get(layer.global_index, LayerRuntimeState()).playing
                    for layer in layers_for_role
                )
                setattr(g, prop, any_on)

            self.logger.info(
                f"[INIT] Group {apc_group_index} -> "
                f"playing={g.playing}, effects={g.effects}, transforms={g.transforms}, "
                f"masks={g.masks}, color={g.color}"
            )

        self._update_all_leds()
    # ---- STATE OUTPUTS (OSC to show_bridge) ----

    def _broadcast_state_update(
        self,
        group_idx: int | None,
        prop: str,
        value: Any,
    ) -> None:
        """
        Send state-machine changes out to OSC for TouchDesigner / show_bridge.

        Mapping is driven by settings/state_machine_mappings.yaml, section:
          outputs:
            global:
              boolean: ...
              float: ...
            group:
              boolean: ...
              float: ...
              string: ...   # <-- clip names
        """
        if not self.state_output_cfg:
            # State output mapping not configured — helpful debug log so
            # users know why no OSC state messages are emitted.
            self.logger.debug("[STATE-OUTPUT] State output mapping disabled (no state_output_cfg).")
            return

        # Support both:
        # - {"outputs": {...}}   (your YAML)
        # - {"global": ..., "group": ...} (older style)
        cfg = self.state_output_cfg.get("outputs", self.state_output_cfg)

        # ---------- classify value ----------
        if isinstance(value, bool):
            kind = "boolean"
            send_value = value
        elif isinstance(value, (int, float)):
            kind = "float"
            send_value = float(value)
        else:
            kind = "string"
            send_value = str(value)

        # ---------- choose mapping section ----------
        if group_idx is None:
            # GLOBAL section (no group index)
            global_cfg = cfg.get("global", {})
            role_map = global_cfg.get(kind, {})
            addr_tpl = role_map.get(prop)
            if not addr_tpl:
                # Mapping missing for this kind/prop — warn at debug level so
                # configuration problems are visible without spamming INFO.
                self.logger.debug(
                    f"[STATE-OUTPUT] No mapping for group {group_idx + 1} prop={prop} kind={kind} in state_output_cfg"
                )
                return

            try:
                addr = addr_tpl.format()
            except Exception:
                addr = addr_tpl
        else:
            # GROUP section (per-APC group)
            group_cfg = cfg.get("group", {})
            role_map = group_cfg.get(kind, {})
            addr_tpl = role_map.get(prop)
            if not addr_tpl:
                self.logger.debug(
                    f"[STATE-OUTPUT] No mapping for global prop={prop} kind={kind} in state_output_cfg"
                )
                return

            # internal group_idx is 0-based; mapping uses 1-based {group}
            try:
                addr = addr_tpl.format(group=group_idx + 1)
            except Exception:
                addr = addr_tpl

        # ---------- actually send ----------
        self._send_state_osc(addr, send_value)
        self.logger.info(f"[STATE-OUTPUT] {kind} {prop} -> {addr} {send_value!r}")

        
    def _send_state_osc(self, address: str, value: Any | None = None) -> None:
        """
        Send OSC on the show_bridge state bus (broadcast_osc_client), with its own logger.
        """
        if self.broadcast_osc_client is None:
            msg = f"(no broadcast client) WOULD SEND {address} {value!r}"
            sb_log(self.osc_bus_logger, logging.WARNING, "BRIDGE", "OSC", msg)

            # self.osc_bus_logger.warning(msg)
            return

        self.osc_bus_logger.info("SEND %s %r", address, value)
        sb_log(self.osc_bus_logger, logging.INFO, "BRIDGE", "OSC", "SEND %s %r", address, value)


        if value is None:
            self.broadcast_osc_client.send_message(address, 0.0)
        else:
            self.broadcast_osc_client.send_message(address, value)


    def _osc_pulse(self, address: str, value: float = 1.0, off_value: float = 0.0) -> None:
        """
        Send a quick on/off pulse as two OSC messages.
        """
        self.logger.info(f"[OSC] PULSE {address} {value} -> {off_value}")
        self.send_osc(address, float(value))
        self.send_osc(address, float(off_value))

    # ---- Resolume hook ----

    def attach_composition(self, comp: CompositionInfo, mapping: CompositionMapping) -> None:
        """
        Attach a Resolume composition model and its mapping, and build
        high-level role lookups:

            group_role_layers[apc_group_index][role] = [LayerInfo,...]
        """
        self.composition = comp
        self.composition_mapping = mapping
        self.group_role_layers.clear()

        # Reset Synesthesia group index until we find it
        self.synesthesia_group_idx = None

        # First, derive which APC group (1..8) is mapped to the Synesthesia group
        # based on the composition mapping's apc_groups (APC idx -> group name).
        for apc_idx, group_name in mapping.apc_groups.items():
            name_norm = group_name.strip().lower()
            # Handle both 'synesthesia' and misspelled 'synestesia'
            if name_norm in ("synesthesia", "synestesia"):
                # Store 0-based APC index so it matches our internal group_idx
                self.synesthesia_group_idx = apc_idx - 1
                self.logger.info(f"[SYN] Synesthesia mapped to APC group {apc_idx} (internal group_idx={self.synesthesia_group_idx})")
                break

        for g in comp.groups:
            if g.apc_group_index is None:
                continue

            role_map: Dict[str, List[LayerInfo]] = {}
            for layer in g.layers:
                role = layer.role or "unassigned"
                role_map.setdefault(role, []).append(layer)

            self.group_role_layers[g.apc_group_index] = role_map

        # Build routing map once after role maps are ready
        self._build_routing_layer_map()

        self.logger.info(f"[RESOLUME] Attached composition '{comp.name}' with {len(comp.groups)} groups.")
        for apc_idx, roles in self.group_role_layers.items():
            self.logger.info(
                f"  APC group {apc_idx}: roles -> "
                + ", ".join(f"{r}({len(layers)})" for r, layers in roles.items())
            )

    def _build_routing_layer_map(self) -> None:
        """
        Build a map: APC group index -> Router layer.

        Router layers live in a dedicated 'Routing Group' and are named
        '<GroupName> Router', where <GroupName> matches the Resolume group name.
        """
        self.routing_layers_by_group.clear()

        if self.composition is None:
            return

        # Map group name -> APC index (for content groups)
        name_to_apc: dict[str, int] = {}
        for g in self.composition.groups:
            if g.apc_group_index is not None:
                name_to_apc[g.name.strip().lower()] = g.apc_group_index

        # Find router layers by role or by name
        for g in self.composition.groups:
            for layer in g.layers:
                lname = layer.name.strip()
                if lname.lower().endswith(" router"):
                    base = lname[: -len(" router")].strip().lower()
                else:
                    continue

                apc_index = name_to_apc.get(base)
                if apc_index is None:
                    continue

                self.routing_layers_by_group[apc_index] = layer

        self.logger.info(
            "[ROUTING] Mapped routing layers:",
            {k: v.name for k, v in self.routing_layers_by_group.items()},
        )
    def initialize_state_from_resolume(self) -> None:
        """Synchronize the state machine with Resolume's current clip selection.

        For each layer in the attached CompositionInfo, we query Resolume over HTTP
        to find which column (clip index) is currently connected/playing.

        Rules:
          - If the active clip index is 1, the layer is treated as OFF.
          - If the active clip index is 2 or any other valid index >= 2,
            the layer is treated as ON.

        This method then:
          - Updates LayerRuntimeState.playing/current_clip_index for every layer.
          - Sets each APC group's boolean properties (playing/effects/...)
            based on whether any mapped layer for that role is ON.
          - Refreshes all LEDs to match the resolved state.

        Safe to call at startup after attach_composition(), and again later
        if you ever want to re-sync with Resolume.
        """
        if self.composition is None:
            sb_log(self.logger,logging.INFO,"ARENA","INIT","No composition attached; skipping initial state sync.")
            
            return
        if not self.resolume_conn:
            sb_log(self.logger,logging.INFO,'ARENA','INIT','No Resolume HTTP config; skipping initial state sync.')
            
            return
                    
        sb_log(self.logger,logging.INFO,'ARENA','INIT','Syncing state from Resolume composition...')

        

        # --- Pass 1: per-layer state from HTTP ---
        for group in self.composition.groups:
            for layer in group.layers:
                osc_layer_index = layer.global_index + 1  # Resolume layers are 1-based
                try:
                    active_col = fetch_active_clip_column_for_layer(
                        self.resolume_conn, osc_layer_index
                    )
                except Exception as e:
                    sb_log(self.logger,logging.INFO,'ARENA','INIT',f"[INIT] Error while fetching active clip for layer {osc_layer_index}: {e}")
                    active_col = 1

                layer_state = self.layer_states.setdefault(
                    layer.global_index, LayerRuntimeState()
                )
                layer_state.current_clip_index = active_col
                # OFF if column 1, ON otherwise
                layer_state.playing = bool(active_col and active_col >= 2)

                msg=f"Layer {layer.global_index} '{layer.name}' "+f"(group {group.index_in_composition}, osc={osc_layer_index}) "+f"active clip={active_col} -> playing={layer_state.playing}"
                sb_log(self.logger,logging.INFO,'ARENA','INIT',msg)
                

        # --- Pass 2: aggregate into APC group booleans ---
        for apc_index, role_layers in self.group_role_layers.items():
            gi = apc_index - 1
            if not (0 <= gi < len(self.groups)):
                continue
            g_state = self.groups[gi]

            for prop, role in self.PROP_TO_ROLE.items():
                layers = role_layers.get(role, [])
                any_on = False
                for layer in layers:
                    st = self.layer_states.get(layer.global_index)
                    if st and st.playing:
                        any_on = True
                        break
                setattr(g_state, prop, any_on)

            sb_log(self.logger,logging.INFO,'ARENA','INIT',
                f"Group {apc_index} -> "
                f"playing={g_state.playing}, "
                f"effects={g_state.effects}, "
                f"transforms={g_state.transforms}, "
                f"masks={g_state.masks}, "
                f"color={g_state.color}"
            )

        # Finally, push LED state to the controller
        self._update_all_leds()
        sb_log(self.logger,logging.INFO,'ARENA','INIT',"State sync from Resolume complete.")
    def _autoplay_fill_layers_for_group(self, group_idx: int) -> None:
        """
        Autopilot for 'fill' layers in a given APC group.

        Trigger logic:
          - Uses group's intensity * global_intensity_scale as a probability [0,1].
          - For each fill layer, roll a random number in [0,1).
              - If roll <= effective_intensity: pick a random clip on that layer and trigger it.
              - Otherwise: skip that layer.
        """
        if self.composition is None:
            sb_log(self.logger,logging.INFO,'BRIDGE','AUTOPILOT',"(no composition attached) would autoplay fill layers here.")
            return

        apc_index = group_idx + 1
        group = self.composition.group_for_apc(apc_index)
        if group is None:
            sb_log(self.logger,logging.INFO,'BRIDGE','AUTOPILOT',f"No Resolume group mapped for APC group {apc_index}")
            return

        # Effective intensity (local * global), clamped 0–1
        raw_intensity = self.groups[group_idx].intensity
        effective_intensity = max(0.0, min(1.0, raw_intensity * self.global_intensity_scale))

        if effective_intensity <= 0.0:
            sb_log(self.logger,logging.INFO,'BRIDGE','AUTOPILOT',f"Group {apc_index} intensity={effective_intensity:.2f} -> skipping fills.")
            return

        # Which layers are "fill"?
        role_layers_map = self.group_role_layers.get(apc_index, {})
        fill_roles = ("fill", "fills", "background")
        fill_layers: List[LayerInfo] = []
        for r in fill_roles:
            fill_layers.extend(role_layers_map.get(r, []))

        # Fallback: if nothing explicitly marked as fill, use all layers
        if not fill_layers:
            fill_layers = group.layers

        self.logger.info(
            f"[AUTOPILOT] Group {apc_index} '{group.name}' "
            f"fill autoplay @ intensity={effective_intensity:.2f} "
            f"on {len(fill_layers)} layer(s)"
        )

        rng = random.Random()
        """
        Autopilot for 'fill' layers in a given APC group.

        Trigger logic:
          - Uses group's intensity * global_intensity_scale as a probability [0,1].
          - For each fill layer, roll a random number in [0,1).
              - If roll <= effective_intensity: pick a random *content* clip
                (column >= FIRST_CONTENT_COLUMN) and trigger it.
              - Otherwise: skip that layer.
        """
        if self.composition is None:
            self.logger.info("[AUTOPILOT] (no composition attached) would autoplay fill layers here.")
            return

        apc_index = group_idx + 1
        group = self.composition.group_for_apc(apc_index)
        if group is None:
            self.logger.info(f"[AUTOPILOT] No Resolume group mapped for APC group {apc_index}")
            return

        raw_intensity = self.groups[group_idx].intensity
        effective_intensity = max(0.0, min(1.0, raw_intensity * self.global_intensity_scale))

        if effective_intensity <= 0.0:
            self.logger.info(f"[AUTOPILOT] Group {apc_index} intensity={effective_intensity:.2f} -> skipping fills.")
            return

        role_layers_map = self.group_role_layers.get(apc_index, {})
        fill_roles = ("fill", "fills", "background")
        fill_layers: List[LayerInfo] = []
        for r in fill_roles:
            fill_layers.extend(role_layers_map.get(r, []))

        if not fill_layers:
            fill_layers = group.layers

        self.logger.info(
            f"[AUTOPILOT] Group {apc_index} '{group.name}' "
            f"fill autoplay @ intensity={effective_intensity:.2f} "
            f"on {len(fill_layers)} layer(s)"
        )

        rng = random.Random()
        for layer in fill_layers:
            # Only consider content clips (3+), skip OFF (1) and Passthrough (2)
            content_clips = [c for c in layer.clips if c.column_index >= FIRST_CONTENT_COLUMN]
            if not content_clips:
                self.logger.info(
                    f"[AUTOPILOT]  Layer '{layer.name}' has no content clips "
                    f"(col >= {FIRST_CONTENT_COLUMN}), skipping."
                )
                continue

            roll = rng.random()
            if roll > effective_intensity:
                self.logger.info(
                    f"[AUTOPILOT]  Layer '{layer.name}' roll={roll:.2f} "
                    f"> {effective_intensity:.2f}, no trigger."
                )
                column_index = 2  # PASSTHROUGH
                osc_layer_index = layer.global_index + 1
                osc_path = f"/composition/layers/{osc_layer_index}/clips/{column_index}/connect"
                self.logger.info(
                    f"[AUTOPILOT]  Layer '{layer.name}' "
                    f"(global {layer.global_index} -> osc {osc_layer_index}) "
                    f"setting to OFF col {column_index} via {osc_path}"
                )
                self.send_osc(osc_path, 1.0)
                continue

            clip = rng.choice(content_clips)
            layer_index = layer.global_index + 1   # 1-based
            column_index = clip.column_index   # 1-based

            osc_path = f"/composition/layers/{layer_index}/clips/{column_index}/connect"
            self.logger.info(
                f"[AUTOPILOT]  Layer '{layer.name}' "
                f"(global {layer.global_index} -> osc {layer_index}) "
                f"roll={roll:.2f} <= {effective_intensity:.2f} -> "
                f"trigger clip col {column_index} ('{clip.name}') via {osc_path}"
            )
            self.send_osc(osc_path, 1.0)


    def _set_fill_layers_off(self, group_idx: int) -> None:
        """
        When 'playing' is turned OFF, set all fill layers to column 1 (Off).
        """
        if self.composition is None:
            self.logger.info(f"[AUTOPILOT] (no composition) would clear fills for group {group_idx + 1}.")
            return

        apc_index = group_idx + 1
        role_layers_map = self.group_role_layers.get(apc_index)
        if not role_layers_map:
            self.logger.info(f"[AUTOPILOT] Group {apc_index} has no role mapping; cannot clear fills.")
            return

        fill_roles = ("fill", "fills", "background")
        fill_layers: list[LayerInfo] = []
        for r in fill_roles:
            fill_layers.extend(role_layers_map.get(r, []))

        if not fill_layers:
            self.logger.info(f"[AUTOPILOT] Group {apc_index} has no fill/background layers to clear.")
            return

        self.logger.info(f"[AUTOPILOT] Group {apc_index} clearing fills -> column 1 (Off) on {len(fill_layers)} layer(s):")
        for layer in fill_layers:
            osc_layer_index = layer.global_index + 1
            column_index = 1  # OFF

            osc_path = f"/composition/layers/{osc_layer_index}/clips/{column_index}/connect"
            self.logger.info(
                f"  -> Layer '{layer.name}' "
                f"(global {layer.global_index} -> osc {osc_layer_index}) "
                f"col {column_index} via {osc_path}"
            )
            self.send_osc(osc_path, 1.0)



    def _set_role_layers_passthrough(self, group_idx: int, role: str) -> None:
        """
        When a role like 'color', 'effects', or 'transforms' is turned OFF,
        set all layers for that role to the Passthrough column (2).

        Uses:
          /composition/layers/{layer_index}/clips/{column_index}/connect
        """
        if self.composition is None:
            self.logger.info(f"[AUTOPILOT] (no composition) would set role '{role}' passthrough for group {group_idx + 1}.")
            return

        apc_index = group_idx + 1
        role_layers_map = self.group_role_layers.get(apc_index, {})
        layers = role_layers_map.get(role, [])

        if not layers:
            self.logger.info(f"[AUTOPILOT] Group {apc_index} has no layers for role '{role}' to set passthrough.")
            return

        self.logger.info(
            f"[AUTOPILOT] Group {apc_index} setting role '{role}' "
            f"to Passthrough (col 2) on {len(layers)} layer(s)"
        )

        for layer in layers:
            # Try to use the actual col=2 clip if it exists; otherwise just target column 2.
            passthrough_clip = next((c for c in layer.clips if c.column_index == 2), None)
            column_index = passthrough_clip.column_index if passthrough_clip else 2

            osc_layer_index = layer.global_index + 1  # 1-based for OSC
            osc_path = f"/composition/layers/{osc_layer_index}/clips/{column_index}/connect"
            self.logger.info(
                f"  -> Layer '{layer.name}' "
                f"(global {layer.global_index} -> osc {osc_layer_index}) "
                f"col {column_index} via {osc_path}"
            )
            self.send_osc(osc_path, 1.0)

    def _set_role_layers_off(self, group_idx: int, role: str) -> None:
        """
        When a role like 'color', 'effects', 'transforms', or 'masks' is turned OFF,
        set all layers for that role to OFF (column 1) via /connect, and do not change
        their current_clip_index.
        """
        if self.composition is None:
            return

        apc_index = group_idx + 1
        role_layers_map = self.group_role_layers.get(apc_index, {})
        layers = role_layers_map.get(role, [])

        if not layers:
            self.logger.info(f"[AUTOPILOT] Group {apc_index} has no layers for role '{role}' to turn OFF.")
            return

        for layer in layers:
            self._set_layer_to_clip(layer, column_index=1, save_current=False)

    def _autoplay_role_single_layer(self, group_idx: int, role: str) -> None:
        """
        Autopilot for a single role ('color', 'effects', 'transforms') in a group.

        Behavior:
          - Uses group's intensity * global_intensity_scale -> effective_intensity in [0,1].
          - For each layer with that role:
              * Identify:
                  - passthrough clips: column == 2
                  - content clips:     column >= FIRST_CONTENT_COLUMN (3+)
              * With probability = effective_intensity, choose a content clip.
              * With probability = 1 - effective_intensity, choose passthrough.
              * If one category is missing, fall back to the other.
        """
        if self.composition is None:
            self.logger.info(f"[AUTOPILOT] (no composition attached) would autoplay role '{role}' for group {group_idx + 1}.")
            return

        apc_index = group_idx + 1
        group = self.composition.group_for_apc(apc_index)
        if group is None:
            self.logger.info(f"[AUTOPILOT] No Resolume group mapped for APC group {apc_index} (role '{role}')")
            return

        raw_intensity = self.groups[group_idx].intensity
        effective_intensity = max(0.0, min(1.0, raw_intensity * self.global_intensity_scale))

        role_layers_map = self.group_role_layers.get(apc_index, {})
        layers = role_layers_map.get(role, [])

        if not layers:
            self.logger.info(f"[AUTOPILOT] Group {apc_index} '{group.name}' has no layers for role '{role}'.")
            return

        self.logger.info(
            f"[AUTOPILOT] Group {apc_index} '{group.name}' role '{role}' "
            f"autoplay @ intensity={effective_intensity:.2f} on {len(layers)} layer(s)"
        )

        rng = random.Random()
        for layer in layers:
            passthrough_clips = [c for c in layer.clips if c.column_index == 2]
            content_clips = [c for c in layer.clips if c.column_index >= FIRST_CONTENT_COLUMN]

            if not passthrough_clips and not content_clips:
                self.logger.info(
                    f"[AUTOPILOT]  Layer '{layer.name}' has no passthrough (col=2) or "
                    f"content (col>={FIRST_CONTENT_COLUMN}) clips, skipping."
                )
                continue

            # Effective intensity controls "how often we pick content".
            roll = rng.random()

            pick_content = False
            if content_clips and passthrough_clips:
                pick_content = roll <= effective_intensity
            elif content_clips:
                # Only content exists
                pick_content = roll <= effective_intensity
            else:
                # Only passthrough exists
                pick_content = False

            if pick_content and content_clips:
                clip = rng.choice(content_clips)
                choice_kind = "CONTENT"
            elif passthrough_clips:
                clip = rng.choice(passthrough_clips)
                choice_kind = "PASSTHROUGH"
            else:
                self.logger.info(f"[AUTOPILOT]  Layer '{layer.name}': no usable clip after selection, skipping.")
                continue

            # Resolume layers are 1-based for OSC
            osc_layer_index = layer.global_index + 1
            osc_clip_index = clip.column_index  # 1-based

            osc_path = f"/composition/layers/{osc_layer_index}/clips/{osc_clip_index}/connect"
            self.logger.info(
                f"  [{choice_kind}] layer {layer.index_in_group} "
                f"(global {layer.global_index} -> osc {osc_layer_index}) "
                f"clip {osc_clip_index} ('{clip.name}') via {osc_path}"
            )
            self.send_osc(osc_path, 1.0)

    def _trigger_clips_for_group(
        self,
        group_idx: int,
        column_index: int,
        role: Optional[str] = None,
    ) -> None:
        """
        Trigger clips in Resolume via OSC for a given APC group.

        - group_idx: 0-based APC index (0..7), APC index = group_idx + 1
        - column_index: 1-based column index in the Resolume grid
        - role: optional high-level role; if provided and we have role mapping
          for that APC group, restrict to those layers; otherwise use all layers.

        OSC:
          /composition/layers/{layer_index}/clips/{column_index}/connect

        where layer_index is the global_index (0-based) we computed
        across all groups.
        """
        if self.composition is None:
            self.logger.info("[RESOLUME] (no composition attached) would set clips here.")
            return

        apc_index = group_idx + 1
        group = self.composition.group_for_apc(apc_index)
        if group is None:
            self.logger.info(f"[RESOLUME] No Resolume group mapped for APC group {apc_index}")
            return

        role_layers_map = self.group_role_layers.get(apc_index)
        if role and role_layers_map and role in role_layers_map:
            layers = role_layers_map[role]
        else:
            layers = group.layers

        role_info = f"role '{role}'" if role else "all layers"
        self.logger.info(f"[RESOLUME] Group {apc_index} '{group.name}' {role_info}:")

        for layer in layers:
            clip = next((c for c in layer.clips if c.column_index == column_index), None)
            if not clip:
                continue

            osc_layer_index = layer.global_index + 1
            osc_clip_index = clip.column_index

            osc_path = f"/composition/layers/{osc_layer_index}/clips/{osc_clip_index}/connect"
            self.logger.info(
                f"  Setting layer {layer.index_in_group} "
                f"(global {layer.global_index} -> osc {osc_layer_index}) "
                f"to clip {osc_clip_index} ('{clip.name}') via {osc_path}"
            )
            self.send_osc(osc_path, 1.0)

    # ---- Internal helpers ----

    def _mark_state_changed(self):
        if self.active_scene is not None and self.active_scene_pristine:
            self.active_scene_pristine = False
            self._update_scene_leds()

        # Whenever the state changes, persist the current state to JSON so that
        # external applications can monitor or react to updates.  This file is
        # updated in real time as properties change.
        try:
            self.save_state_to_json()
        except Exception as e:
            # Don't let file I/O errors disrupt the state machine; just log.
            self.logger.warning(f"[STATE] WARNING: failed to save current state to JSON: {e}")


    def _init_intensity_preset_leds(self):
        for gi in range(len(self.groups)):
            self._update_intensity_leds(gi)

    # ---- Gesture handlers ----

    def handle_property_press(self, group_idx: int, prop: str, press_type: str):
        """
        Implement:
          - Short press:
              * if False -> True, then trigger appropriate autoplay for that prop
              * if True  -> next_<prop>()
          - Long press:  if True  -> False
          - Double press: toggle <prop>_autopilot
        """
        g = self.groups[group_idx]
        value = getattr(g, prop)

        if press_type == "short":
            if not value:
                # Turn it on
                self.set_property(group_idx, prop, True)
                # After toggling ON, run any prop-specific autoplay
                self._handle_after_toggle_on(group_idx, prop)
            else:
                # Already on -> NEXT
                self.next_property(group_idx, prop)

        elif press_type == "long":
            if value:
                self.set_property(group_idx, prop, False)

        elif press_type == "double":
            self.toggle_autopilot(group_idx, prop)
    def _handle_after_toggle_on(self, group_idx: int, prop: str) -> None:
        """
        Called after a property is turned ON (False -> True).
        Used to kick off appropriate autoplay for that group.
        """
        if prop == "playing":
            # Fill layers autoplay when playing is turned on
            self._autoplay_fill_layers_for_group(group_idx)
        elif prop in ("color", "effects", "transforms", "masks"):
            # For color, effects, transforms and masks, trigger
            # autoplay for a single layer corresponding to the role.  For
            # masks this will choose a content clip or passthrough for
            # each dynamic mask layer based on intensity.
            role = self.PROP_TO_ROLE.get(prop)
            if role:
                self._autoplay_role_single_layer(group_idx, role)

    def handle_intensity_preset_press(self, group_idx: int, preset_value: float, press_type: str):
        if press_type == "short":
            self.set_intensity_from_preset(group_idx, preset_value)

        elif press_type == "long":
            self.set_intensity_from_preset(group_idx, 0.0)

        elif press_type == "double":
            possible = [0.0, 0.25, 0.5, 0.75, 1.0]
            current = self.groups[group_idx].intensity
            candidates = [v for v in possible if abs(v - current) > 1e-6]
            if not candidates:
                candidates = possible
            rand_value = random.choice(candidates)
            self.logger.info(f"[STATE] Group {group_idx + 1} intensity RANDOM -> {rand_value:.2f}")
            self.set_intensity_from_preset(group_idx, rand_value)

    def set_property(self, group_idx: int, prop: str, value: bool):
        self._mark_state_changed()
        g = self.groups[group_idx]
        old_value = getattr(g, prop)
        setattr(g, prop, value)
        self.logger.info(f"[STATE] Group {group_idx + 1} {prop} -> {value}")
        self._update_led_for_property(group_idx, prop)

        # Broadcast the boolean state update via OSC
        try:
            self._broadcast_state_update(group_idx, prop, value)
        except Exception:
            pass

        # Extra behavior for 'playing':
        if prop == "playing":
            if value:
                # When playing turns ON, autoplay fills
                if hasattr(self, "_autoplay_fill_layers_for_group"):
                    self._autoplay_fill_layers_for_group(group_idx)
            else:
                # When playing turns OFF, force fills to column 1 (Off)
                self._set_fill_layers_off(group_idx)

        # Extra behavior for color/effects/transforms/masks when turning OFF:
        elif not value and prop in ("color", "effects", "transforms", "masks"):
            # When turning off masks or other roles, set their layers to
            # Passthrough (col=2) or OFF (col=1) depending on the role type.  Use
            # _set_role_layers_off to send OSC messages to Resolume.
            role = self.PROP_TO_ROLE.get(prop)
            if role:
                self._set_role_layers_off(group_idx, role)





    def toggle_autopilot(self, group_idx: int, prop: str):
        g = self.groups[group_idx]
        attr = f"{prop}_autopilot"
        current = getattr(g, attr, True)
        new_val = not current
        setattr(g, attr, new_val)
        self.logger.info(f"[STATE] Group {group_idx + 1} {prop}_autopilot -> {new_val}")
        self._update_led_for_property(group_idx, prop)

        # Broadcast /state/group/<idx>/<role>/autopilot
        try:
            self._broadcast_state_update(group_idx, f"{prop}_autopilot", new_val)
        except Exception:
            pass


    def next_property(self, group_idx: int, prop: str):
        """
        Called when short-press on a property that's already True.

        For this controller:
          - playing: run fill autoplay again
          - color/effects/transforms/masks: reroll that role's clip choice

        Additionally:
          - If this is the Synesthesia group and we're NEXT-ing its 'playing'
            property, send /playlist/next to Synesthesia.
        """
        self.logger.info(f"[STATE] Group {group_idx + 1} NEXT {prop}")

        if prop == "playing":
            self._autoplay_fill_layers_for_group(group_idx)

            # If this group is mapped as the Synesthesia group, advance Synesthesia playlist.
            if self.synesthesia_group_idx is not None and group_idx == self.synesthesia_group_idx:
                self.send_synesthesia_playlist_next()

        elif prop in ("color", "effects", "transforms", "masks"):
            role = self.PROP_TO_ROLE.get(prop)
            if role:
                self._autoplay_role_single_layer(group_idx, role)


    def reset_group(self, group_idx: int):
        self.groups[group_idx] = OutputGroupState()
        self.logger.info(f"[STATE] Group {group_idx + 1} reset")
        for prop in self.BOOL_PROPS:
            self._update_led_for_property(group_idx, prop)
        self._update_intensity_leds(group_idx)

    def set_opacity_from_cc(self, group_idx: int, value: int):
        v = max(0.0, min(1.0, value / 127.0))
        self.groups[group_idx].opacity = v
        self.logger.info(f"[STATE] Group {group_idx + 1} opacity -> {v:.3f}")

        apc_index = group_idx + 1

        # First try a mapped Router layer
        router_layer = self.routing_layers_by_group.get(apc_index)
        if router_layer is not None:
            osc_layer_index = router_layer.global_index + 1  # 1-based
            osc_path = f"/composition/layers/{osc_layer_index}/video/opacity"
            self.logger.info(
                f"[OSC] Router opacity: layer={osc_layer_index}, path={osc_path}, value={v:.3f}"
            )
            self.send_osc(osc_path, v)
        else:
            # Fallback: drive the actual Resolume group master like before
            if self.composition is not None:
                group = self.composition.group_for_apc(apc_index)
                if group is not None:
                    osc_group_index = group.index_in_composition + 1
                    osc_path = f"/composition/groups/{osc_group_index}/master"
                    self.logger.info(
                        f"[OSC] Group master opacity: group={osc_group_index}, path={osc_path}, value={v:.3f}"
                    )
                    self.send_osc(osc_path, v)

        self._update_intensity_leds(group_idx)

        # Broadcast opacity state update via OSC
        try:
            self._broadcast_state_update(group_idx, "opacity", v)
        except Exception:
            pass

    def _refresh_routing_layers(self, now: float) -> None:
        """Periodically (every routing_refresh_period) re-connect routing clips on column 2."""
        # if now - self.last_routing_refresh < self.routing_refresh_period:
            # return

        # self.last_routing_refresh = now

        # for apc_index, layer in self.routing_layers_by_group.items():
        #     osc_layer_index = layer.global_index + 1
        #     osc_path = f"/composition/layers/{osc_layer_index}/clips/2/connect"
        #     print(
        #         f"[ROUTING] Ensure routing clip playing: "
        #         f"APC group={apc_index}, layer={osc_layer_index}, path={osc_path}"
        #     )
        #     self.send_osc(osc_path, 1.0)


    def set_intensity_from_preset(self, group_idx: int, value: float):
        v = max(0.0, min(1.0, value))
        self.groups[group_idx].intensity = v
        self.logger.info(f"[STATE] Group {group_idx + 1} intensity -> {v:.3f}")
        self._update_intensity_leds(group_idx)

        # Broadcast intensity state update via OSC
        try:
            self._broadcast_state_update(group_idx, "intensity", v)
        except Exception:
            pass

    def _update_intensity_leds(self, group_idx: int):
        addrs = self.runtime.group_addrs[group_idx]
        notes = addrs.intensity_preset_notes
        if not notes:
            return

        intensity = self.groups[group_idx].intensity
        count = len(notes)
        lit_count = int(round(intensity * count))

        ch = addrs.channel
        for idx, note in enumerate(notes):
            if idx < lit_count:
                profile = self.note_velocity.get(note)
                vel = profile.resolved_on() if profile else RED_VELOCITY
            else:
                vel = 0
            msg = mido.Message("note_on", channel=ch, note=note, velocity=vel)
            self.midi_out.send(msg)

    def set_global_intensity_from_cc(self, value: int):
        v = max(0.0, min(1.0, value / 127.0))
        self.global_intensity_scale = v
        self.logger.info(f"[STATE] Global intensity -> {v:.3f}")

        # Drive Resolume master opacity via OSC:
        #   /composition/master
        self.send_osc("/composition/master", float(v))

        # Broadcast global intensity scale via OSC (if configured)
        try:
            self._broadcast_state_update(None, "global_intensity_scale", v)
        except Exception:
            pass

    def toggle_effects_all(self):
        any_on = any(g.effects for g in self.groups)
        new_val = not any_on
        for idx in range(len(self.groups)):
            self.set_property(idx, "effects", new_val)

        self.global_queue_level = 1 - self.global_queue_level
        self.logger.info(f"[STATE] All groups effects -> {new_val}, global_queue_level -> {self.global_queue_level}")

    def set_global_autopilot(self, value: bool):
        """
        Enable or disable global autopilot.

        When turning on autopilot while it is already enabled, trigger a
        global next clip so that active layers advance immediately.  This
        satisfies the user's requirement that toggling the global autopilot
        on when it is already on will advance to the next clip on all
        currently playing layers.  LEDs are always refreshed and the
        autopilot state is broadcast via OSC.
        """
        # Preserve the previous state so we can detect repeated enables
        old_state = getattr(self, "global_autopilot", False)
        self.global_autopilot = bool(value)
        self.logger.info(f"[STATE] Global autopilot -> {self.global_autopilot}")

        # Always refresh LEDs to reflect any change
        for gi in range(len(self.groups)):
            for prop in self.BOOL_PROPS:
                try:
                    self._update_led_for_property(gi, prop)
                except Exception:
                    # In case the LED update helper is not available
                    pass

        # Broadcast the updated autopilot state
        try:
            self._broadcast_state_update(None, "autopilot", self.global_autopilot)
        except Exception:
            pass

        # If autopilot is being turned on and it was already on previously,
        # trigger an immediate global next clip.  This will run a single
        # autopilot cycle across all groups, advancing any active layers.
        if self.global_autopilot and old_state:
            try:
                self.global_next_clip()
            except Exception as e:
                self.logger.warning(f"[STATE] WARNING: failed to trigger global next clip while enabling autopilot: {e}")

    def set_global_autopilot_on(self):
        self.set_global_autopilot(True)

    def set_global_autopilot_off(self):
        self.set_global_autopilot(False)

    def start_all_clips(self):
        self.logger.info("[STATE] Start all clips")
        for gi in range(len(self.groups)):
            self.set_property(gi, "playing", True)

    def stop_all_clips(self):
        self.logger.info("[STATE] Stop all clips")
        for gi in range(len(self.groups)):
            self.set_property(gi, "playing", False)

    def global_nudge(self, direction: int):
        """
        direction: -1 for nudge-, +1 for nudge+
        """
        self.logger.info(f"[STATE] Global nudge {'+' if direction > 0 else '-'}")

        if direction < 0:
            # Nudge -
            #   /composition/tempocontroller/tempopull
            self._osc_pulse("/composition/tempocontroller/tempopull", 1.0, 0.0)
        else:
            # Nudge +
            #   /composition/tempocontroller/tempopush
            self._osc_pulse("/composition/tempocontroller/tempopush", 1.0, 0.0)

    def global_nudge_plus(self):
        self.global_nudge(+1)

    def global_nudge_minus(self):
        self.global_nudge(-1)

    def global_tap_tempo(self):
        """
        Tap-tempo pulse event.
        Drive Resolume:
          /composition/tempocontroller/tempotap (pulse 1 then 0)
        """
        self.logger.info("[STATE] Tap tempo pulse")
        self._osc_pulse("/composition/tempocontroller/tempotap", 1.0, 0.0)

    def global_tempo_sync(self):
        """
        Tempo sync / resync:
          /composition/tempocontroller/resync (pulse 1 then 0)
        """
        self.logger.info("[STATE] Global tempo sync requested")
        self._osc_pulse("/composition/tempocontroller/resync", 1.0, 0.0)

    def global_next_clip(self) -> None:
            """
            Advance to the next clip across all groups.

            This reuses the autopilot cycle logic to trigger the next roll on all
            currently active properties.  If global_autopilot is disabled, it will
            temporarily enable it for a single cycle.
            """
            self.logger.info("[STATE] Global next clip requested")
            # Temporarily force-run one autopilot cycle regardless of global_autopilot
            prev = self.global_autopilot
            self.global_autopilot = True
            try:
                self._run_autopilot_cycle()
            finally:
                # Restore previous autopilot state
                self.global_autopilot = prev

    def global_previous_clip(self) -> None:
        """
        Go back to the previous clip across all groups.

        Resolume does not natively support a "previous clip" across all layers,
        so this implementation simply triggers another autopilot cycle, which
        will select new clips.  This behaviour effectively mimics stepping
        backward by re-rolling content.  A future implementation could
        maintain a history of clip selections to truly step backwards.
        """
        self.logger.info("[STATE] Global previous clip requested")
        prev = self.global_autopilot
        self.global_autopilot = True
        try:
            self._run_autopilot_cycle()
        finally:
            self.global_autopilot = prev
    def scroll_clips_horizontal(self, step: float):
        """
        Scroll clips horizontally via:
        /application/ui/clipsscrollhorizontal

        `step` is user-defined; typical values might be -1.0, +1.0, etc.
        """
        self.logger.info(f"[STATE] Scroll clips horizontally step={step}")
        self.send_osc("/application/ui/clipsscrollhorizontal", float(step))

    def scroll_clips_vertical(self, step: float):
        """
        Scroll clips vertically via:
          /application/ui/clipsscrollvertical

        `step` is user-defined; typical values might be -1.0, +1.0, etc.
        """
        self.logger.info(f"[STATE] Scroll clips vertically step={step}")
        self.send_osc("/application/ui/clipsscrollvertical", float(step))

    def scroll_h_from_cc(self, value: int) -> None:
        """
        Convert a MIDI CC value (0–127) into a horizontal scroll step and send it.

        We map the 0..127 range to -1..+1 linearly, with 64 roughly as 0.
        Values near the centre produce no scroll.
        """
        # Normalise 0–127 to 0–1
        norm = max(0.0, min(1.0, value / 127.0))
        step = (norm * 2.0) - 1.0
        # Deadband around zero to avoid jitter when slider is near the middle
        if abs(step) < 0.1:
            step = 0.0
        self.scroll_clips_horizontal(step)

    def scroll_v_from_cc(self, value: int) -> None:
        """
        Convert a MIDI CC value (0–127) into a vertical scroll step and send it.

        We map the 0..127 range to -1..+1 linearly, with 64 roughly as 0.
        Values near the centre produce no scroll.
        """
        norm = max(0.0, min(1.0, value / 127.0))
        step = (norm * 2.0) - 1.0
        if abs(step) < 0.1:
            step = 0.0
        self.scroll_clips_vertical(step)

    # -------------------------------------------------------------------
    # Persistence helpers
    # -------------------------------------------------------------------
    def get_state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for idx, g in enumerate(self.groups):
            state[f"group_{idx + 1}"] = {
                "playing": g.playing,
                "playing_autopilot": g.playing_autopilot,
                "effects": g.effects,
                "effects_autopilot": g.effects_autopilot,
                "transforms": g.transforms,
                "transforms_autopilot": g.transforms_autopilot,
                "color": g.color,
                "color_autopilot": g.color_autopilot,
                "opacity": g.opacity,
                "intensity": g.intensity,
                "masks": g.masks,
                "masks_autopilot": g.masks_autopilot,
            }
        state["global"] = {
            "global_autopilot": self.global_autopilot,
            "global_intensity_scale": self.global_intensity_scale,
            "global_queue_level": self.global_queue_level,
        }
        return state


    def save_state_to_json(self, filepath: str = "current_state_machine.json") -> None:
        """
        Write the current state machine settings to a JSON file on disk. This
        file can be monitored by other applications for real-time updates.
        """
        state = self.get_state_dict()
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, filepath)

    def record_current_state_and_vote(self, vote_file: str = "votes.json", state_file: str = "current_state_machine.json") -> None:
        """
        Persist the current state machine and register a "vote" for the
        currently active combination. This can be used to train future
        algorithms or curate preferred clip combinations.
        """
        # Save the state snapshot
        self.save_state_to_json(state_file)
        # Build a key representing the combination of booleans and intensities
        combo: List[Any] = []
        for g in self.groups:
            combo.append((g.playing, g.effects, g.transforms, g.fft_mask, g.color, g.intensity, g.opacity, getattr(g, "masks", False)))
        key = str(combo)
        # Load existing votes if available
        votes: Dict[str, int] = {}
        if os.path.exists(vote_file):
            try:
                with open(vote_file, "r", encoding="utf-8") as vf:
                    votes = json.load(vf)
            except Exception:
                votes = {}
        # Increment the vote count
        votes[key] = votes.get(key, 0) + 1
        # Write back votes atomically
        tmp_vote = vote_file + ".tmp"
        with open(tmp_vote, "w", encoding="utf-8") as vf:
            json.dump(votes, vf, indent=2)
        os.replace(tmp_vote, vote_file)

    # -------------------------------------------------------------------
    # Resolume resync
    # -------------------------------------------------------------------
    def resync_with_resolume(self) -> None:
        """
        Re-fetch the current composition from Resolume and rebuild the
        composition model and mapping.  This is useful if the composition has
        changed on the Resolume side while this script is running.  The
        connection configuration and mapping directory must have been provided
        during initialization.
        """
        if not self.resolume_conn:
            self.logger.info("[RESYNC] No Resolume HTTP connection configured; cannot resync.")
            return
        try:
            comp_json = fetch_composition_json(self.resolume_conn)
            comp_name = _guess_composition_name(comp_json)
            comp_mapping = CompositionMapping.from_yaml_dir(
                composition_name=comp_name,
                mapping_dir=Path(self.mapping_dir),
            )
            comp_model = build_composition_model(comp_json, comp_mapping)
            self.attach_composition(comp_model, comp_mapping)
            # Note: we intentionally do not alter the group state booleans or
            # intensities here; the existing state machine values remain.  If
            # you'd like to synchronize the state machine to Resolume's
            # current clip selection, additional logic would be required.
            self.logger.info(f"[RESYNC] Re-attached composition '{comp_name}' via resync.")
        except Exception as e:
            self.logger.warning(f"[RESYNC] Failed to resync composition: {e}")

    # ---- LED feedback & blinking ----

    def _base_velocity_for_note(self, note: int, prop: str | None = None) -> int:
        profile = self.note_velocity.get(note)
        if profile:
            return profile.resolved_on()
        return 127

    def _desired_led_on(self, group_idx: int, prop: str) -> bool:
        g = self.groups[group_idx]
        value = getattr(g, prop)

        if not value:
            return False

        local_auto = getattr(g, f"{prop}_autopilot", True)

        if not self.global_autopilot:
            return self.slow_blink_on
        if not local_auto:
            return self.fast_blink_on
        return True

    def _update_led_for_property(self, group_idx: int, prop: str):
        addrs = self.runtime.group_addrs[group_idx]
        if prop not in addrs.prop_notes:
            return
        note = addrs.prop_notes[prop]
        ch = addrs.channel

        on = self._desired_led_on(group_idx, prop)
        velocity = self._base_velocity_for_note(note, prop) if on else 0

        msg = mido.Message("note_on", channel=ch, note=note, velocity=velocity)
        self.midi_out.send(msg)

    def update_blink(self, now: float):
        fast_changed = False
        slow_changed = False

        if now - self.last_fast_toggle >= self.fast_period / 2.0:
            self.fast_blink_on = not self.fast_blink_on
            self.last_fast_toggle = now
            fast_changed = True

        if now - self.last_slow_toggle >= self.slow_period / 2.0:
            self.slow_blink_on = not self.slow_blink_on
            self.last_slow_toggle = now
            slow_changed = True

        if fast_changed or slow_changed:
            for gi in range(len(self.groups)):
                for prop in self.BOOL_PROPS:
                    self._update_led_for_property(gi, prop)
            self._update_scene_leds()

    # ---- Scene snapshots ----

    def _snapshot_group_state(self, g: OutputGroupState) -> OutputGroupState:
        return OutputGroupState(
            playing=g.playing,
            playing_autopilot=g.playing_autopilot,
            effects=g.effects,
            effects_autopilot=g.effects_autopilot,
            transforms=g.transforms,
            transforms_autopilot=g.transforms_autopilot,
            color=g.color,
            color_autopilot=g.color_autopilot,
            opacity=g.opacity,
            intensity=g.intensity,
            masks=g.masks,
            masks_autopilot=g.masks_autopilot,
        )

    def _save_scene_snapshot(self, scene_index: int):
        snaps = [self._snapshot_group_state(g) for g in self.groups]
        self.scene_snapshots[scene_index] = snaps
        self.logger.info(f"[SCENE] Saved snapshot for scene {scene_index + 1}")
        self._update_scene_leds()

    def _apply_scene_snapshot(self, scene_index: int):
        snaps = self.scene_snapshots.get(scene_index)
        if not snaps:
            self.logger.info(f"[SCENE] No snapshot stored for scene {scene_index + 1}")
            return

        for i in range(min(len(self.groups), len(snaps))):
            s = snaps[i]
            g = self.groups[i]

            g.playing = s.playing
            g.playing_autopilot = s.playing_autopilot

            g.effects = s.effects
            g.effects_autopilot = s.effects_autopilot

            g.transforms = s.transforms
            g.transforms_autopilot = s.transforms_autopilot

            g.intensity = s.intensity
            g.color = s.color
            g.color_autopilot = s.color_autopilot

            # Restore dynamic mask state for the group if present
            if hasattr(s, "masks"):
                g.masks = s.masks
                g.masks_autopilot = getattr(s, "masks_autopilot", True)

        self.active_scene = scene_index
        self.active_scene_pristine = True

        self.logger.info(f"[SCENE] Applied snapshot for scene {scene_index + 1}")
        self._update_all_leds()
        try:
            self.broadcast_full_state()
        except Exception as e:
            self.logger.warning(f"[STATE-OUTPUT] Failed to broadcast full state after scene launch: {e}")

    def _clear_scene_snapshot(self, scene_index: int):
        if scene_index in self.scene_snapshots:
            del self.scene_snapshots[scene_index]
            self.logger.info(f"[SCENE] Cleared snapshot for scene {scene_index + 1}")

        if self.active_scene == scene_index:
            self.active_scene = None
            self.active_scene_pristine = False

        self._update_scene_leds()

    def handle_scene_button(self, scene_index: int, press_type: str):
        if press_type == "short":
            if scene_index in self.scene_snapshots:
                self._apply_scene_snapshot(scene_index)
            else:
                self.logger.info(f"[SCENE] Scene {scene_index + 1} has no saved snapshot")

        elif press_type == "long":
            self._save_scene_snapshot(scene_index)

        elif press_type == "double":
            self._clear_scene_snapshot(scene_index)

    def _update_all_leds(self):
        for gi in range(len(self.groups)):
            for prop in self.BOOL_PROPS:
                self._update_led_for_property(gi, prop)
            self._update_intensity_leds(gi)

        self._update_scene_leds()

    def _update_scene_leds(self):
        for scene_idx, (ch, note) in self.runtime.scene_buttons.items():
            has_snapshot = scene_idx in self.scene_snapshots

            if not has_snapshot:
                vel = 0
            else:
                base = self._base_velocity_for_note(note, prop=None)
                if self.active_scene == scene_idx and self.active_scene_pristine:
                    vel = base if self.fast_blink_on else 0
                else:
                    vel = base

            msg = mido.Message("note_on", channel=ch, note=note, velocity=vel)
            self.midi_out.send(msg)


# ---------------------------------------------------------
# MIDI message handling
# ---------------------------------------------------------

IMMEDIATE_GLOBAL_ACTIONS = {
    "nudge_minus",
    "nudge_plus",
    "tap_tempo",
    "tempo_sync",
    "set_global_autopilot_on",
    "set_global_autopilot_off",
    "global_previous_clip",
    "global_next_clip",
    # New immediate actions: resync the composition from Resolume and record the
    # current state (e.g. REC button functionality). These actions fire
    # immediately on note_on and do not depend on short/long/double press.
    "resync_composition",
    "record_current_state",
}


def handle_immediate_global_action(spec: ActionSpec, sm: Apc40StateMachine):
    action = spec.action

    if action == "nudge_minus":
        sm.global_nudge_minus()
    elif action == "nudge_plus":
        sm.global_nudge_plus()
    elif action == "tap_tempo":
        sm.global_tap_tempo()
    elif action == "tempo_sync":
        sm.global_tempo_sync()
    elif action == "set_global_autopilot_on":
        sm.set_global_autopilot_on()
    elif action == "set_global_autopilot_off":
        sm.set_global_autopilot_off()
    elif action == "resync_composition":
        # Trigger a resync of the composition from Resolume.  This fetches
        # the latest composition JSON and rebuilds the mapping.  It is safe
        # to call while performing live as it does not alter existing state.
        sm.resync_with_resolume()
    elif action == "record_current_state":
        # Persist the current state machine snapshot and register a vote for
        # this combination.  Typically mapped to the REC button.
        sm.record_current_state_and_vote()
    elif action == "global_previous_clip":
        sm.global_previous_clip()
    elif action == "global_next_clip":
        sm.global_next_clip()

def handle_midi_message(msg: mido.Message,
                        runtime: MappingRuntime,
                        sm: Apc40StateMachine,
                        press_mgr: PressManager,
                        now: float):
    if msg.type in ("note_on", "note_off"):
        key = ("note", msg.channel, msg.note)

        if msg.type == "note_on" and msg.velocity > 0:
            spec = runtime.action_map.get(key)
            if spec and spec.scope == "global" and spec.action in IMMEDIATE_GLOBAL_ACTIONS:
                handle_immediate_global_action(spec, sm)
                return

        events = press_mgr.handle_note_message(key, msg, now)
        for k, press_type in events:
            dispatch_press(k, press_type, runtime, sm)

    elif msg.type == "control_change":
        key = ("cc", msg.channel, msg.control)
        spec = runtime.action_map.get(key)
        if not spec:
            return

        if spec.scope == "group" and spec.property_name == "opacity":
            sm.set_opacity_from_cc(spec.group_index, msg.value)
        elif spec.scope == "global" and spec.action == "set_global_intensity_from_cc":
            sm.set_global_intensity_from_cc(msg.value)
        elif spec.scope == "global" and spec.action == "set_global_opacity_from_cc":
            sm.set_global_opacity_from_cc(msg.value)
        elif spec.scope == "global" and spec.action in ("scroll_clips_horizontal", "scroll_h_from_cc"):
            sm.scroll_h_from_cc(msg.value)
        elif spec.scope == "global" and spec.action in ("scroll_clips_vertical", "scroll_v_from_cc"):
            sm.scroll_v_from_cc(msg.value)
        elif spec.scope == "global" and spec.action == "global_previous_clip":
            # Trigger previous clip on value movement; we ignore the CC value
            sm.global_previous_clip()
        elif spec.scope == "global" and spec.action == "global_next_clip":
            sm.global_next_clip()


def dispatch_press(key: MidiKey,
                   press_type: str,
                   runtime: MappingRuntime,
                   sm: Apc40StateMachine):
    spec = runtime.action_map.get(key)
    if not spec:
        return

    if spec.scope == "group":
        gi = spec.group_index

        if spec.property_name in sm.BOOL_PROPS:
            sm.handle_property_press(gi, spec.property_name, press_type)

        elif spec.action == "set_intensity_preset" and spec.intensity_value is not None:
            sm.handle_intensity_preset_press(gi, spec.intensity_value, press_type)

        elif spec.action == "reset_group" and press_type == "short":
            sm.reset_group(gi)

    elif spec.scope == "global":
        if spec.action == "toggle_effects_all" and press_type == "short":
            sm.toggle_effects_all()
        elif spec.action == "set_global_autopilot_on" and press_type == "short":
            sm.set_global_autopilot_on()
        elif spec.action == "set_global_autopilot_off" and press_type == "short":
            sm.set_global_autopilot_off()
        elif spec.action == "control_all_clips":
            # Extended semantics for the Stop All Clips button.  The behavior
            # depends on whether any group is currently playing (interpreted as
            # the global ON/OFF state) and the type of press.  When OFF (no
            # groups playing), a short press starts all fills; a double press
            # randomizes intensities across groups.  When ON (one or more
            # groups playing), a short press triggers the next clip on all fill
            # layers, a long press stops all fills, and a double press
            # randomizes intensities.
            any_playing = any(g.playing for g in sm.groups)
            if not any_playing:
                # Global state is OFF
                if press_type == "short":
                    # Start all fill layers by turning on playing for each group
                    for gi in range(len(sm.groups)):
                        sm.set_property(gi, "playing", True)
                elif press_type == "double":
                    # Randomize intensities for all groups
                    for gi in range(len(sm.groups)):
                        # Choose a random intensity from a set of common presets
                        candidates = [0.25, 0.5, 0.75, 1.0]
                        rand_val = random.choice(candidates)
                        print(f"[STATE] Group {gi + 1} intensity RANDOM -> {rand_val:.2f}")
                        sm.set_intensity_from_preset(gi, rand_val)
                # long press when OFF: no action
            else:
                # Global state is ON (one or more groups playing)
                if press_type == "short":
                    # Trigger next clip on all fill layers (autoplay)
                    for gi in range(len(sm.groups)):
                        sm._autoplay_fill_layers_for_group(gi)
                elif press_type == "long":
                    # Turn off playing for all groups (stop fills)
                    for gi in range(len(sm.groups)):
                        sm.set_property(gi, "playing", False)
                elif press_type == "double":
                    # Randomize intensities for all groups while playing
                    for gi in range(len(sm.groups)):
                        candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
                        rand_val = random.choice(candidates)
                        print(f"[STATE] Group {gi + 1} intensity RANDOM -> {rand_val:.2f}")
                        sm.set_intensity_from_preset(gi, rand_val)
        elif spec.action == "nudge_minus" and press_type == "short":
            sm.global_nudge_minus()
        elif spec.action == "nudge_plus" and press_type == "short":
            sm.global_nudge_plus()
        elif spec.action == "tap_tempo" and press_type == "short":
            sm.global_tap_tempo()
        elif spec.action == "tempo_sync" and press_type == "short":
            sm.global_tempo_sync()
        elif spec.action == "scene_slot" and spec.scene_index is not None:
            sm.handle_scene_button(spec.scene_index, press_type)


# ---------------------------------------------------------------------------
# Monitoring server for real-time state inspection
# ---------------------------------------------------------------------------
class MonitoringServer(threading.Thread):
    """
    A lightweight HTTP server that exposes the current state machine state as
    JSON and provides a simple HTML interface.  The server runs in its own
    thread and does not block the main MIDI processing loop.  Metrics such as
    CPU and memory usage are also exposed if psutil is available.
    """

    def __init__(self, sm: Apc40StateMachine, host: str = "0.0.0.0", port: int = 8000) -> None:
        super().__init__(daemon=True)
        self.sm = sm
        self.host = host
        self.port = port

    def run(self) -> None:
        class Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                # Suppress default logging to keep stdout clean
                pass

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    # Serve a basic HTML page that fetches /state.json
                    html = """
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset='utf-8'>
                        <title>Resolume State Monitor</title>
                        <style>
                            body { font-family: Arial, sans-serif; background: #1e1e1e; color: #e0e0e0; }
                            pre { background: #2e2e2e; padding: 1em; overflow: auto; }
                        </style>
                    </head>
                    <body>
                        <h1>Resolume State Monitor</h1>
                        <p>This page refreshes the current state every second.</p>
                        <pre id="state"></pre>
                        <script>
                        async function refresh() {
                            const res = await fetch('/state.json');
                            const data = await res.json();
                            document.getElementById('state').textContent = JSON.stringify(data, null, 2);
                        }
                        setInterval(refresh, 1000);
                        refresh();
                        </script>
                    </body>
                    </html>
                    """
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(html.encode('utf-8'))
                elif self.path == "/state.json":
                    # Return the current state as JSON, along with optional system metrics
                    state = self.server.state_supplier()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(state, indent=2).encode('utf-8'))
                else:
                    self.send_error(404, "Not found")

        # Provide access to the state supplier via the HTTPServer instance
        def state_supplier() -> Dict[str, Any]:
            state = self.sm.get_state_dict()
            # Attach resource metrics if available
            metrics: Dict[str, Any] = {}
            if psutil is not None:
                try:
                    proc = psutil.Process(os.getpid())
                    metrics = {
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "mem_info": proc.memory_info()._asdict(),
                    }
                except Exception:
                    metrics = {}
            state["metrics"] = metrics
            return state

        # Create the HTTP server; override state_supplier
        with socketserver.TCPServer((self.host, self.port), Handler) as httpd:
            # Attach state supplier on the server instance so Handler can access it
            httpd.state_supplier = state_supplier  # type: ignore
            try:
                httpd.serve_forever()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Helper functions for mapping descriptions and state-machine visualization
# ---------------------------------------------------------------------------
def describe_mappings(profile: MappingProfile) -> str:
    """
    Generate a human-readable description of the mappings defined in a
    MappingProfile.  This summary explains which controls map to which
    state-machine properties or global actions, including the gesture
    semantics (short/long/double).  The returned string can be written
    to a file or displayed on screen.
    """
    lines: List[str] = []
    lines.append(f"Controller: {profile.controller_name}\n")
    # Per-group properties
    group_props = profile.state_cfg.get("groups", {}).get("properties", {})
    lines.append("Per-Group Property Controls:")
    for prop, cfg in group_props.items():
        ctrl = cfg.get("control")
        action = cfg.get("action")
        lines.append(f"  - {prop}: control '{ctrl}', action '{action}' (short=toggle/next, long=off, double=toggle autopilot)")
    # Group intensity presets
    presets = profile.state_cfg.get("groups", {}).get("intensity_presets", [])
    if presets:
        lines.append("  Intensity presets:")
        for preset in presets:
            ctrl = preset.get("control")
            val = preset.get("value")
            lines.append(f"    * {ctrl} -> intensity {val}")
    # Global actions
    lines.append("\nGlobal Controls:")
    global_cfg = profile.state_cfg.get("global", {})
    for name, cfg in global_cfg.items():
        ctrl = cfg.get("control")
        action = cfg.get("action")
        lines.append(f"  - {name}: control '{ctrl}', action '{action}'")
    return "\n".join(lines)

def export_state_machine_dot(sm: Apc40StateMachine, filepath: str = "state_machine.dot") -> None:
    """
    Export a Graphviz DOT file representing the state transitions for each
    boolean property in each group.  This provides a node-based view of
    how short/long/double presses affect the ON/OFF state and autopilot
    toggling.  Edges are labeled according to the gesture type.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("digraph StateMachine {\n")
        f.write("  rankdir=LR;\n")
        for gi, g in enumerate(sm.groups):
            for prop in sm.BOOL_PROPS:
                on_node = f"G{gi+1}_{prop}_ON"
                off_node = f"G{gi+1}_{prop}_OFF"
                f.write(f"  {on_node} [shape=box, label=\"G{gi+1} {prop} ON\"];\n")
                f.write(f"  {off_node} [shape=box, label=\"G{gi+1} {prop} OFF\"];\n")
                # Transition: OFF -> ON on short press
                f.write(f"  {off_node} -> {on_node} [label=\"short\"];\n")
                # Transition: ON -> OFF on long press
                f.write(f"  {on_node} -> {off_node} [label=\"long\"];\n")
                # Transition: ON -> ON on short press (next)
                f.write(f"  {on_node} -> {on_node} [label=\"short/next\"];\n")
                # Transition: OFF -> OFF on long press has no effect
                # Transition: toggle autopilot on double press
                f.write(f"  {on_node} -> {on_node} [label=\"double/autopilot\"];\n")
                f.write(f"  {off_node} -> {off_node} [label=\"double/autopilot\"];\n")
        f.write("}\n")

def start_transport_osc_listener(sm: Apc40StateMachine, host: str = "127.0.0.1", port: int = 7001):
    """
    Start a background OSC server that listens for Resolume transport beat pulses
    like /global/transport/beat4/trigger.

    Configure Resolume OSC output to send to this host/port.

    If a server is already running on (host, port) in this process,
    we reuse that one instead of binding again.
    """
    global _transport_osc_servers

    osc_in_logger = logging.getLogger("osc.resolume.in")
    dispatcher = Dispatcher()

    def make_handler(beat_kind: str):
        def handler(address: str, *args):
            # We only care about rising edges (value ~1)
            if not args:
                return
            val = args[0]
            try:
                v = float(val)
            except Exception:
                return
            if v > 0.5:
                osc_in_logger.info("RECV %s %s v=%s", beat_kind, address, v)
                sm.on_transport_beat(beat_kind)
        return handler

    for kind in ("beat1", "beat2", "beat4", "beat8", "beat16"):
        dispatcher.map(f"/global/transport/{kind}/trigger", make_handler(kind))

    key = (host, port)

    with _transport_osc_servers_lock:
        # Reuse an existing server on the same host/port
        if key in _transport_osc_servers:
            server = _transport_osc_servers[key]
            sb_log(osc_in_logger,logging.INFO,"ARENA","OSC-IN",f"Reusing existing transport listener on {host}:{port}")
            # osc_in_logger.info("Reusing existing transport listener on %s:%s", host, port)
            return server

        # Otherwise, create and register a new one
        try:
            server = ThreadingOSCUDPServer((host, port), dispatcher)
        except OSError as e:
            # Port already in use or other bind problem
            sb_log(osc_in_logger,logging.ERROR,"ARENA","OSC-IN",f"ERROR: could not bind transport listener on {host}:{port}: {e!r}")
            
            return None

        _transport_osc_servers[key] = server

    print(f"[OSC-IN] Listening for transport beats on {host}:{port}")
    osc_in_logger.info("Listening for transport beats on %s:%s", host, port)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server

def start_show_bridge_state_osc_listener(
    state_machine: "Apc40StateMachine",
    host: str,
    port: int,
) -> threading.Thread:
    """
    OSC server for inbound state updates on the show_bridge bus.

    We accept the same /state/... messages we send out. For now we just:
      - log them to osc.show_bridge.in
      - call state_machine.on_external_state_osc(address, *args)
    """
    disp = Dispatcher()

    def generic_handler(address, *args):
        try:
            state_machine.osc_bus_in_logger.info("RECV %s %s", address, args)
        except Exception:
            pass

        try:
            state_machine.on_external_state_osc(address, *args)
        except Exception as e:
            state_machine.logger.warning(
                f"[STATE-INPUT] Error handling inbound show_bridge OSC {address} {args}: {e}"
            )

    def composition_sync_handler(address, *args):
        """Handle /composition/sync <1> messages by re-fetching the composition."""
        try:
            state_machine.osc_bus_in_logger.info("RECV %s %s", address, args)
        except Exception:
            pass

        try:
            # Accept numeric or string '1' as the trigger to resync
            if args and (args[0] in (1, 1.0, '1', b'1') or str(args[0]) == '1'):
                state_machine.logger.info("[RESYNC] composition sync requested via OSC; fetching composition...")
                # Perform resync in a background thread to avoid blocking the OSC server
                threading.Thread(target=state_machine.resync_with_resolume, daemon=True).start()
            else:
                state_machine.logger.debug(f"[RESYNC] /composition/sync received but value != 1: {args}")
        except Exception as e:
            state_machine.logger.warning(f"[RESYNC] Error handling /composition/sync OSC: {e}")

    # Specific composition sync handler
    disp.map("/composition/sync", composition_sync_handler)

    # Catch all /state/... messages
    disp.map("/state/*", generic_handler)
    disp.set_default_handler(generic_handler)

    key = (host, port)
    with _osc_servers_lock:
        if key in _osc_servers:
            # Reuse existing server instead of binding again
            state_machine.logger.info(
                f"[OSC] Reusing existing show_bridge IN listener on {host}:{port}"
            )
            # We don't start a new thread here; assume the existing server is already serving
            return None

        try:
            server = ThreadingOSCUDPServer((host, port), disp)
        except OSError as e:
            # Possible race: another thread may have bound and registered a
            # server between our initial registry check and this bind attempt.
            # If so, reuse that server silently. Otherwise, log the bind error.
            if key in _osc_servers:
                state_machine.logger.info(
                    f"[OSC] Listener bind race: reusing existing show_bridge IN listener on {host}:{port}"
                )
                return None
            state_machine.logger.warning(
                f"[OSC] ERROR: could not bind show_bridge IN listener on {host}:{port}: {e!r}"
            )
            return None

        _osc_servers[key] = server

    state_machine.logger.info(
        f"[OSC] show_bridge IN listener (name='to show bridge') on {host}:{port}"
    )
    print(f"[OSC-IN] Listening for state updates on {host}:{port}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

def main(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="MIDI controller state-machine demo.")
    parser.add_argument(
        "--mappings-dir",
        type=str,
        default=DEFAULT_MAPPINGS_DIR,
        help="Directory containing *.yaml mapping profiles (default: ./mappings)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Mapping profile name to use (matches controller_name or file stem).",
    )
    args = parser.parse_args(argv)
    app_logger = setup_logging()
    sb_log(app_logger, logging.INFO, "BRIDGE", "INIT", "Starting show_bridge state machine runner")

    mappings_dir = Path(args.mappings_dir)

    profiles = load_mapping_profiles(mappings_dir)

    midi_input_names = mido.get_input_names()
    midi_output_names = mido.get_output_names()

    profile = choose_mapping_profile(
        profiles,
        midi_input_names=midi_input_names,
        desired_name=args.device,
    )

    full_cfg = load_yaml_file(profile.file_path)
    input_cfg = full_cfg.get("input_mappings", {})
    state_cfg = full_cfg.get("state_mappings", {})
    controller_name = full_cfg.get("controller_name", profile.name)

    print(f"\nUsing mapping profile: {profile.name} (file: {profile.file_path})")

    runtime = build_runtime_mapping(input_cfg, state_cfg)

    in_name = auto_select_port(midi_input_names, controller_name, "input")
    out_name = auto_select_port(midi_output_names, controller_name, "output")

    # --- Resolume / OSC wiring: load connections + composition + composition_mappings + OSC ---
    comp_model: Optional[CompositionInfo] = None
    comp_mapping: Optional[CompositionMapping] = None

    osc_client: Optional[SimpleUDPClient] = None          # Resolume control
    syn_osc_client: Optional[SimpleUDPClient] = None      # Synesthesia (optional)
    sb_osc_client: Optional[SimpleUDPClient] = None       # show_bridge broadcast bus
    conn_http: Optional[Dict[str, Any]] = None            # Resolume HTTP
    state_output_cfg: Dict[str, Any] = {}                 # YAML mapping for state broadcasts

    try:
        connections_cfg = load_connections(Path(DEFAULT_CONNECTIONS_PATH))

        # Load state broadcast mapping (optional)
        try:
            state_output_cfg = load_yaml_file(Path("settings/state_machine_mappings.yaml"))
            
            app_logger.info(f'[STATE] Loaded state_machine_mappings.yaml for OSC broadcast.')
        except FileNotFoundError:
            app_logger.info(f'[STATE] No state_machine_mappings.yaml found; state broadcast mapping disabled.')
            state_output_cfg = {}
        except Exception as cfg_e:
            app_logger.warning(f'[STATE] could not load state_machine_mappings.yaml: {cfg_e}')
            state_output_cfg = {}

        # HTTP for composition JSON (Resolume)
        conn_http = get_resolume_http_connection(connections_cfg, name=None, io_section="outputs")
        comp_json = fetch_composition_json(conn_http)
        comp_name = _guess_composition_name(comp_json)

        # Scan ./composition_mappings/*.yml for a mapping whose composition_name matches
        comp_mapping = CompositionMapping.from_yaml_dir(
            composition_name=comp_name,
            mapping_dir=Path(DEFAULT_COMPOSITION_MAPPING_DIR),
        )

        comp_model = build_composition_model(comp_json, comp_mapping)
        sb_log(app_logger, logging.INFO, "ARENA","INIT",f"Loaded composition '{comp_model.name}' with {len(comp_model.groups)} groups.")
        debug_dump_composition_columns(comp_model, app_logger)

        # OSC client (Resolume control)
        osc_host, osc_port = get_resolume_osc_connection(
            connections_cfg,
            name=None,
            io_section="outputs",
        )
        osc_client = get_osc_client(osc_host, osc_port)
        sb_log(app_logger, logging.INFO, "ARENA","OSC",f"Using Resolume OSC at {osc_host}:{osc_port}")


        # OSC client (show_bridge telemetry / state broadcast)
        # show_bridge OSC (state broadcast bus)
        try:
            sb_host, sb_port = get_show_bridge_osc_connection(
                connections_cfg,
                name="from show_bridge",   # <--- use the labeled connection
                io_section="outputs",
            )
            sb_osc_client = get_osc_client(sb_host, sb_port)
            app_logger.info(
                f"[OSC] Using show_bridge OSC OUT (name='from show_bridge') at {sb_host}:{sb_port} "
                "for state broadcasts"
            )
        except Exception as e:
            sb_osc_client = None
            app_logger.warning(
                f"[OSC] No show_bridge OSC configured for state broadcasts (from show_bridge): {e}"
            )


        # OSC client (Synesthesia) - optional
        try:
            syn_host, syn_port = get_synesthesia_osc_connection(
                connections_cfg,
                name=None,
                io_section="outputs",
            )
            syn_osc_client = get_osc_client(syn_host, syn_port)
            app_logger.info(f"[SYN] Using Synesthesia OSC at {syn_host}:{syn_port}")
        except Exception as syn_e:
            app_logger.warning(f"[SYN] WARNING: could not initialize Synesthesia OSC: {syn_e}")
            syn_osc_client = None

    except Exception as e:
        app_logger.warning(f"[RESOLUME] WARNING: could not initialize Resolume composition/OSC: {e}")
        conn_http = None
        osc_client = None
        sb_osc_client = None
        syn_osc_client = None
        state_output_cfg = {}

    app_logger.info(f"\nUsing input:  {in_name}")
    app_logger.info(f"Using output: {out_name}")
    print("Press Ctrl+C to exit.\n")

    press_mgr = PressManager()

    # ---- MIDI port opening with reconnect logic ----
    in_port = None
    out_port = None
    attempt = 0
    while True:
        try:
            midi_input_names = mido.get_input_names()
            midi_output_names = mido.get_output_names()
            in_name = auto_select_port(midi_input_names, controller_name, "input")
            out_name = auto_select_port(midi_output_names, controller_name, "output")

            in_port = mido.open_input(in_name)
            out_port = mido.open_output(out_name)
            break
        except Exception as e:
            try:
                if in_port:
                    in_port.close()
            except Exception:
                pass
            try:
                if out_port:
                    out_port.close()
            except Exception:
                pass

            attempt += 1
            delay = 5 if attempt == 1 else 10
            app_logger.warning(f"[MIDI] WARNING: could not open MIDI ports ('{in_name}', '{out_name}'): {e}")
            app_logger.warning(f"[MIDI] Will retry in {delay} seconds... (attempt {attempt})")
            time.sleep(delay)
            continue

    try:
        with in_port, out_port:
            sm = Apc40StateMachine(
                runtime,
                out_port,
                osc_client=osc_client,             # Resolume control messages
                resolume_conn=conn_http,
                mapping_dir=DEFAULT_COMPOSITION_MAPPING_DIR,
                syn_osc_client=syn_osc_client,
                broadcast_osc_client=sb_osc_client,  # show_bridge bus for state broadcasts
                state_output_cfg=state_output_cfg,
            )

            # Listen for incoming OSC (beats, clip connects) from Resolume
            start_transport_osc_listener(sm, host="127.0.0.1", port=7001)

            if comp_model is not None and comp_mapping is not None:
                sm.attach_composition(comp_model, comp_mapping)
                sm.initialize_state_from_composition()
                try:
                    sm.broadcast_full_state()
                except Exception as e:
                    sb_log(app_logger,logging.WARNING,"BRIDGE","STATE-OUTPUT",f"Failed to broadcast full state on startup: {e}")
             # -----------------------------------------------------
            # show_bridge OSC IN (accept mirrored state updates)
            # -----------------------------------------------------
            try:
                sb_in_host, sb_in_port = get_show_bridge_osc_connection(
                    connections_cfg,
                    name="to show bridge",       # <--- this is your inbound connection
                    io_section="inputs",
                )
                start_show_bridge_state_osc_listener(sm, sb_in_host, sb_in_port)
            except Exception as e:
                sb_log(app_logger,logging.WARNING,"BRIDGE","OSC",f"No show_bridge OSC listener configured for inbound state (to show bridge): {e}"
                )
            # Start the monitoring server for real-time state visualization
            try:
                monitor_port = 8765
                monitor = MonitoringServer(sm, host="0.0.0.0", port=monitor_port)
                monitor.start()
                sb_log(app_logger,logging.INFO,"BRIDGE","MONITOR",f"State monitoring HTTP server started on port {monitor_port}.")
            except Exception as e:
                sb_log(app_logger,logging.WARNING,"BRIDGE","MONITOR",f"[MONITOR] WARNING: could not start monitoring server: {e}")

            try:
                while True:
                    now = time.monotonic()

                    for msg in in_port.iter_pending():
                        handle_midi_message(msg, runtime, sm, press_mgr, now)

                    for key, pt in press_mgr.poll(now):
                        dispatch_press(key, pt, runtime, sm)

                    sm.update_blink(now)
                    sm._refresh_routing_layers(now)

                    time.sleep(0.001)
            except KeyboardInterrupt:
                print("\nExiting.")
                app_logger.info("Shutting down show_bridge state machine runner")
    finally:
        # Just in case, ensure ports are closed
        try:
            in_port.close()
        except Exception:
            pass
        try:
            out_port.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

