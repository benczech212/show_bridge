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
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import mido
import yaml
import requests
from pythonosc.udp_client import SimpleUDPClient

# -------------------------------------------------------------------
# Paths / defaults
# -------------------------------------------------------------------

DEFAULT_MAPPINGS_DIR = "mappings"
DEFAULT_CONNECTIONS_PATH = "settings/connections.yaml"
DEFAULT_COMPOSITION_MAPPING_DIR = "composition_mappings"

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

# =========================================================
# Resolume composition + mapping model
# =========================================================

@dataclass
class ClipInfo:
    id: str
    name: str
    column_index: int  # 1-based column index in the Resolume grid


@dataclass
class LayerInfo:
    id: str
    name: str
    index_in_group: int      # 0-based index inside the layergroup
    global_index: int        # 0-based index across ALL layers in the composition
    role: Optional[str] = None  # e.g. "colors", "effects", etc.
    clips: List[ClipInfo] = field(default_factory=list)


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

def debug_dump_composition_columns(comp: CompositionInfo) -> None:
    print(f"[DEBUG] Composition '{comp.name}' column layout:")
    for g in comp.groups:
        apc_str = f"APC={g.apc_group_index}" if g.apc_group_index is not None else "APC=-"
        print(f"  Group {g.index_in_composition} '{g.name}' ({apc_str})")
        for layer in g.layers:
            cols = ", ".join(f"{c.column_index}:{c.name}" for c in layer.clips)
            role = layer.role or "-"
            print(
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

    # Resolume group name -> APC index
    resolume_group_to_apc: Dict[str, int] = {
        v.lower(): k for k, v in mapping.apc_groups.items()
    }
    next_layer_index = 0  # fallback global layer index across all groups

    for g_idx, g_json in enumerate(lg_list):
        g_id = str(g_json.get("id"))
        g_name = _name_from_field(g_json.get("name"), f"group-{g_id}")

        apc_idx: Optional[int] = None
        key = g_name.lower()
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
            for col0, clip_json in enumerate(clips_json):
                c_id = str(clip_json.get("id"))
                c_name = _clip_name(clip_json)

                # Drop nameless clips so they don't confuse autoplay
                if not c_name:
                    # print(f"[DEBUG] Skipping nameless clip id={c_id} on layer '{l_name}', local col={col0}")
                    continue

                col_idx = _clip_column_index(clip_json, col0)
                layer_info.clips.append(
                    ClipInfo(
                        id=c_id,
                        name=c_name,
                        column_index=col_idx,
                    )
                )

            group_info.layers.append(layer_info)

        comp.groups.append(group_info)

    return comp


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
    Grab one HTTP connection config from connections.yaml.

    Expected shape:

      outputs:
        http:
          resolume_arena:
            - name: "arena_http_out_main"
              host: "127.0.0.1"
              port: 8080
              api_base: "/api/v1"
              username: "admin"
              password: "secret"

    io_section: "inputs" or "outputs"
    """
    http_cfg = connections_cfg.get(io_section, {}).get("http", {})
    arena_list = http_cfg.get("resolume_arena", []) or http_cfg.get("arena", [])

    if not arena_list:
        raise RuntimeError(f"No resolume_arena HTTP entries found under {io_section}.http")

    if name is None:
        return arena_list[0]

    for entry in arena_list:
        if entry.get("name") == name:
            return entry

    raise RuntimeError(f"HTTP Resolume connection named '{name}' not found in {io_section}.http")


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

    If not present, fall back to HTTP host and DEFAULT_OSC_PORT.
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
    http_conn = get_resolume_http_connection(connections_cfg, name=None, io_section=io_section)
    host = http_conn.get("host", "127.0.0.1")
    port = DEFAULT_OSC_PORT
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

    fft_mask: bool = False
    fft_mask_autopilot: bool = True

    color: bool = False
    color_autopilot: bool = True

    opacity: float = 0.0
    intensity: float = 0.0


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
        print(f"No {kind} port matched controller_name='{controller_name}'.")
        print(f"Available MIDI {kind} ports:")
        for i, n in enumerate(port_names):
            print(f"  [{i}] {n}")
        idx = int(input(f"Select {kind} port number: "))
        return port_names[idx]


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

    BOOL_PROPS = ("playing", "effects", "transforms", "fft_mask", "color")

    # Map state-machine boolean properties to layer roles in the
    # composition mapping. You can tweak this to your naming scheme.
    PROP_TO_ROLE = {
        "playing": "colors",
        "effects": "effects",
        "transforms": "transforms",
        "fft_mask": "fft_mask",
        "color": "color",
    }

    def __init__(
        self,
        runtime: MappingRuntime,
        midi_out: mido.ports.BaseOutput,
        osc_client: Optional[SimpleUDPClient] = None,
    ):
        self.runtime = runtime
        self.midi_out = midi_out
        self.osc_client = osc_client

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
        self.global_queue_level: int = 0

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

    # ---- OSC helpers ----

    def send_osc(self, address: str, value: Any | None = None) -> None:
        """
        Send a single OSC message if osc_client is available.
        Always logs what it's trying to send.
        """
        if self.osc_client is None:
            print(f"[OSC] (no client) WOULD SEND  {address} {value!r}")
            return

        # Log before sending so we always know what went out
        print(f"[OSC] SEND {address} {value!r}")
        if value is None:
            # Some OSC receivers don't love empty lists, but Resolume is fine with a dummy 0
            self.osc_client.send_message(address, 0.0)
        else:
            self.osc_client.send_message(address, value)


    def _osc_pulse(self, address: str, value: float = 1.0, off_value: float = 0.0) -> None:
        """
        Send a quick on/off pulse as two OSC messages.
        """
        print(f"[OSC] PULSE {address} {value} -> {off_value}")
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

        for g in comp.groups:
            if g.apc_group_index is None:
                continue

            role_map: Dict[str, List[LayerInfo]] = {}
            for layer in g.layers:
                role = layer.role or "unassigned"
                role_map.setdefault(role, []).append(layer)

            self.group_role_layers[g.apc_group_index] = role_map

        print(f"[RESOLUME] Attached composition '{comp.name}' with {len(comp.groups)} groups.")
        for apc_idx, roles in self.group_role_layers.items():
            print(f"  APC group {apc_idx}: roles -> "
                  + ", ".join(f"{r}({len(layers)})" for r, layers in roles.items()))
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
            print("[AUTOPILOT] (no composition attached) would autoplay fill layers here.")
            return

        apc_index = group_idx + 1
        group = self.composition.group_for_apc(apc_index)
        if group is None:
            print(f"[AUTOPILOT] No Resolume group mapped for APC group {apc_index}")
            return

        # Effective intensity (local * global), clamped 0–1
        raw_intensity = self.groups[group_idx].intensity
        effective_intensity = max(0.0, min(1.0, raw_intensity * self.global_intensity_scale))

        if effective_intensity <= 0.0:
            print(f"[AUTOPILOT] Group {apc_index} intensity={effective_intensity:.2f} -> skipping fills.")
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

        print(
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
            print("[AUTOPILOT] (no composition attached) would autoplay fill layers here.")
            return

        apc_index = group_idx + 1
        group = self.composition.group_for_apc(apc_index)
        if group is None:
            print(f"[AUTOPILOT] No Resolume group mapped for APC group {apc_index}")
            return

        raw_intensity = self.groups[group_idx].intensity
        effective_intensity = max(0.0, min(1.0, raw_intensity * self.global_intensity_scale))

        if effective_intensity <= 0.0:
            print(f"[AUTOPILOT] Group {apc_index} intensity={effective_intensity:.2f} -> skipping fills.")
            return

        role_layers_map = self.group_role_layers.get(apc_index, {})
        fill_roles = ("fill", "fills", "background")
        fill_layers: List[LayerInfo] = []
        for r in fill_roles:
            fill_layers.extend(role_layers_map.get(r, []))

        if not fill_layers:
            fill_layers = group.layers

        print(
            f"[AUTOPILOT] Group {apc_index} '{group.name}' "
            f"fill autoplay @ intensity={effective_intensity:.2f} "
            f"on {len(fill_layers)} layer(s)"
        )

        rng = random.Random()
        for layer in fill_layers:
            # Only consider content clips (3+), skip OFF (1) and Passthrough (2)
            content_clips = [c for c in layer.clips if c.column_index >= FIRST_CONTENT_COLUMN]
            if not content_clips:
                print(
                    f"[AUTOPILOT]  Layer '{layer.name}' has no content clips "
                    f"(col >= {FIRST_CONTENT_COLUMN}), skipping."
                )
                continue

            roll = rng.random()
            if roll > effective_intensity:
                print(
                    f"[AUTOPILOT]  Layer '{layer.name}' roll={roll:.2f} "
                    f"> {effective_intensity:.2f}, no trigger."
                )
                column_index = 2  # PASSTHROUGH
                osc_layer_index = layer.global_index + 1
                osc_path = f"/composition/layers/{osc_layer_index}/clips/{column_index}/connect"
                print(
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
            print(
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
            print(f"[AUTOPILOT] (no composition) would clear fills for group {group_idx + 1}.")
            return

        apc_index = group_idx + 1
        role_layers_map = self.group_role_layers.get(apc_index)
        if not role_layers_map:
            print(f"[AUTOPILOT] Group {apc_index} has no role mapping; cannot clear fills.")
            return

        fill_roles = ("fill", "fills", "background")
        fill_layers: list[LayerInfo] = []
        for r in fill_roles:
            fill_layers.extend(role_layers_map.get(r, []))

        if not fill_layers:
            print(f"[AUTOPILOT] Group {apc_index} has no fill/background layers to clear.")
            return

        print(f"[AUTOPILOT] Group {apc_index} clearing fills -> column 1 (Off) on {len(fill_layers)} layer(s):")
        for layer in fill_layers:
            osc_layer_index = layer.global_index + 1
            column_index = 1  # OFF

            osc_path = f"/composition/layers/{osc_layer_index}/clips/{column_index}/connect"
            print(
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
            print(f"[AUTOPILOT] (no composition) would set role '{role}' passthrough for group {group_idx + 1}.")
            return

        apc_index = group_idx + 1
        role_layers_map = self.group_role_layers.get(apc_index, {})
        layers = role_layers_map.get(role, [])

        if not layers:
            print(f"[AUTOPILOT] Group {apc_index} has no layers for role '{role}' to set passthrough.")
            return

        print(
            f"[AUTOPILOT] Group {apc_index} setting role '{role}' "
            f"to Passthrough (col 2) on {len(layers)} layer(s)"
        )

        for layer in layers:
            # Try to use the actual col=2 clip if it exists; otherwise just target column 2.
            passthrough_clip = next((c for c in layer.clips if c.column_index == 2), None)
            column_index = passthrough_clip.column_index if passthrough_clip else 2

            osc_layer_index = layer.global_index + 1  # 1-based for OSC
            osc_path = f"/composition/layers/{osc_layer_index}/clips/{column_index}/connect"
            print(
                f"  -> Layer '{layer.name}' "
                f"(global {layer.global_index} -> osc {osc_layer_index}) "
                f"col {column_index} via {osc_path}"
            )
            self.send_osc(osc_path, 1.0)

    def _set_role_layers_off(self, group_idx: int, role: str) -> None:
        """
        When a role like 'color', 'effects', or 'transforms' is turned OFF,
        set all layers for that role to the Passthrough column (2).

        Uses:
          /composition/layers/{layer_index}/clips/{column_index}/connect
        """
        if self.composition is None:
            print(f"[AUTOPILOT] (no composition) would set role '{role}' passthrough for group {group_idx + 1}.")
            return

        apc_index = group_idx + 1
        role_layers_map = self.group_role_layers.get(apc_index, {})
        layers = role_layers_map.get(role, [])

        if not layers:
            print(f"[AUTOPILOT] Group {apc_index} has no layers for role '{role}' to set passthrough.")
            return

        print(
            f"[AUTOPILOT] Group {apc_index} setting role '{role}' "
            f"to Passthrough (col 1) on {len(layers)} layer(s)"
        )

        for layer in layers:
            # Try to use the actual col=1 clip if it exists; otherwise just target column 1.
            passthrough_clip = next((c for c in layer.clips if c.column_index == 1), None)
            column_index = passthrough_clip.column_index if passthrough_clip else 1

            osc_layer_index = layer.global_index + 1  # 1-based for OSC
            osc_path = f"/composition/layers/{osc_layer_index}/clips/{column_index}/connect"
            print(
                f"  -> Layer '{layer.name}' "
                f"(global {layer.global_index} -> osc {osc_layer_index}) "
                f"col {column_index} via {osc_path}"
            )
            self.send_osc(osc_path, 1.0)

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
            print(f"[AUTOPILOT] (no composition attached) would autoplay role '{role}' for group {group_idx + 1}.")
            return

        apc_index = group_idx + 1
        group = self.composition.group_for_apc(apc_index)
        if group is None:
            print(f"[AUTOPILOT] No Resolume group mapped for APC group {apc_index} (role '{role}')")
            return

        raw_intensity = self.groups[group_idx].intensity
        effective_intensity = max(0.0, min(1.0, raw_intensity * self.global_intensity_scale))

        role_layers_map = self.group_role_layers.get(apc_index, {})
        layers = role_layers_map.get(role, [])

        if not layers:
            print(f"[AUTOPILOT] Group {apc_index} '{group.name}' has no layers for role '{role}'.")
            return

        print(
            f"[AUTOPILOT] Group {apc_index} '{group.name}' role '{role}' "
            f"autoplay @ intensity={effective_intensity:.2f} on {len(layers)} layer(s)"
        )

        rng = random.Random()
        for layer in layers:
            passthrough_clips = [c for c in layer.clips if c.column_index == 2]
            content_clips = [c for c in layer.clips if c.column_index >= FIRST_CONTENT_COLUMN]

            if not passthrough_clips and not content_clips:
                print(
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
                print(f"[AUTOPILOT]  Layer '{layer.name}': no usable clip after selection, skipping.")
                continue

            # Resolume layers are 1-based for OSC
            osc_layer_index = layer.global_index + 1
            osc_clip_index = clip.column_index  # 1-based

            osc_path = f"/composition/layers/{osc_layer_index}/clips/{osc_clip_index}/connect"
            print(
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
            print("[RESOLUME] (no composition attached) would set clips here.")
            return

        apc_index = group_idx + 1
        group = self.composition.group_for_apc(apc_index)
        if group is None:
            print(f"[RESOLUME] No Resolume group mapped for APC group {apc_index}")
            return

        role_layers_map = self.group_role_layers.get(apc_index)
        if role and role_layers_map and role in role_layers_map:
            layers = role_layers_map[role]
        else:
            layers = group.layers

        role_info = f"role '{role}'" if role else "all layers"
        print(f"[RESOLUME] Group {apc_index} '{group.name}' {role_info}:")

        for layer in layers:
            clip = next((c for c in layer.clips if c.column_index == column_index), None)
            if not clip:
                continue

            osc_layer_index = layer.global_index + 1
            osc_clip_index = clip.column_index

            osc_path = f"/composition/layers/{osc_layer_index}/clips/{osc_clip_index}/connect"
            print(
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
        elif prop in ("color", "effects", "transforms"):
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
            print(f"[STATE] Group {group_idx + 1} intensity RANDOM -> {rand_value:.2f}")
            self.set_intensity_from_preset(group_idx, rand_value)

    def set_property(self, group_idx: int, prop: str, value: bool):
        self._mark_state_changed()
        g = self.groups[group_idx]
        old_value = getattr(g, prop)
        setattr(g, prop, value)
        print(f"[STATE] Group {group_idx + 1} {prop} -> {value}")
        self._update_led_for_property(group_idx, prop)

        # Extra behavior for 'playing':
        if prop == "playing":
            if value:
                # When playing turns ON, autoplay fills
                if hasattr(self, "_autoplay_fill_layers_for_group"):
                    self._autoplay_fill_layers_for_group(group_idx)
            else:
                # When playing turns OFF, force fills to column 1 (Off)
                self._set_fill_layers_off(group_idx)

        # Extra behavior for color/effects/transforms when turning OFF:
        elif not value and prop in ("color", "effects", "transforms"):
            role = self.PROP_TO_ROLE.get(prop)
            if role:
                self._set_role_layers_off(group_idx, role)





    def toggle_autopilot(self, group_idx: int, prop: str):
        g = self.groups[group_idx]
        attr = f"{prop}_autopilot"
        current = getattr(g, attr, True)
        new_val = not current
        setattr(g, attr, new_val)
        print(f"[STATE] Group {group_idx + 1} {prop}_autopilot -> {new_val}")
        self._update_led_for_property(group_idx, prop)

    def next_property(self, group_idx: int, prop: str):
        """
        Called when short-press on a property that's already True.

        For this controller:
          - playing: run fill autoplay again
          - color/effects/transforms: reroll that role's clip choice
        """
        print(f"[STATE] Group {group_idx + 1} NEXT {prop}")

        if prop == "playing":
            self._autoplay_fill_layers_for_group(group_idx)
        elif prop in ("color", "effects", "transforms"):
            role = self.PROP_TO_ROLE.get(prop)
            if role:
                self._autoplay_role_single_layer(group_idx, role)

    def reset_group(self, group_idx: int):
        self.groups[group_idx] = OutputGroupState()
        print(f"[STATE] Group {group_idx + 1} reset")
        for prop in self.BOOL_PROPS:
            self._update_led_for_property(group_idx, prop)
        self._update_intensity_leds(group_idx)

    def set_opacity_from_cc(self, group_idx: int, value: int):
        v = max(0.0, min(1.0, value / 127.0))
        self.groups[group_idx].opacity = v
        print(f"[STATE] Group {group_idx + 1} opacity -> {v:.3f}")

        # Also drive Resolume group opacity via OSC:
        #   /composition/groups/{group_index}/master
        if self.composition is not None:
            apc_index = group_idx + 1
            group = self.composition.group_for_apc(apc_index)
            if group is not None:
                # Resolume OSC groups are 1-based; index_in_composition is 0-based.
                osc_group_index = group.index_in_composition + 1
                osc_path = f"/composition/groups/{osc_group_index}/master"
                print(
                    f"[OSC] Group {group.name!r} idx {group.index_in_composition} "
                    f"-> osc {osc_group_index}, opacity {v:.3f}"
                )
                self.send_osc(osc_path, float(v))


    def set_intensity_from_preset(self, group_idx: int, value: float):
        v = max(0.0, min(1.0, value))
        self.groups[group_idx].intensity = v
        print(f"[STATE] Group {group_idx + 1} intensity -> {v:.3f}")
        self._update_intensity_leds(group_idx)

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
        print(f"[STATE] Global intensity -> {v:.3f}")

        # Drive Resolume master opacity via OSC:
        #   /composition/master
        self.send_osc("/composition/master", float(v))

    def toggle_effects_all(self):
        any_on = any(g.effects for g in self.groups)
        new_val = not any_on
        for idx in range(len(self.groups)):
            self.set_property(idx, "effects", new_val)

        self.global_queue_level = 1 - self.global_queue_level
        print(f"[STATE] All groups effects -> {new_val}, global_queue_level -> {self.global_queue_level}")

    def set_global_autopilot(self, value: bool):
        self.global_autopilot = value
        print(f"[STATE] Global autopilot -> {value}")
        for gi in range(len(self.groups)):
            for prop in self.BOOL_PROPS:
                self._update_led_for_property(gi, prop)

    def set_global_autopilot_on(self):
        self.set_global_autopilot(True)

    def set_global_autopilot_off(self):
        self.set_global_autopilot(False)

    def start_all_clips(self):
        print("[STATE] Start all clips")
        for gi in range(len(self.groups)):
            self.set_property(gi, "playing", True)

    def stop_all_clips(self):
        print("[STATE] Stop all clips")
        for gi in range(len(self.groups)):
            self.set_property(gi, "playing", False)

    def global_nudge(self, direction: int):
        """
        direction: -1 for nudge-, +1 for nudge+
        """
        print(f"[STATE] Global nudge {'+' if direction > 0 else '-'}")

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
        print("[STATE] Tap tempo pulse")
        self._osc_pulse("/composition/tempocontroller/tempotap", 1.0, 0.0)

    def global_tempo_sync(self):
        """
        Tempo sync / resync:
          /composition/tempocontroller/resync (pulse 1 then 0)
        """
        print("[STATE] Global tempo sync requested")
        self._osc_pulse("/composition/tempocontroller/resync", 1.0, 0.0)

    def scroll_clips_horizontal(self, step: float):
        """
        Scroll clips horizontally via:
          /application/ui/clipsscrollhorizontal

        `step` is user-defined; typical values might be -1.0, +1.0, etc.
        """
        print(f"[STATE] Scroll clips horizontally step={step}")
        self.send_osc("/application/ui/clipsscrollhorizontal", float(step))

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
            fft_mask=g.fft_mask,
            fft_mask_autopilot=g.fft_mask_autopilot,
            color=g.color,
            color_autopilot=g.color_autopilot,
            opacity=0.0,
            intensity=0.0,
        )

    def _save_scene_snapshot(self, scene_index: int):
        snaps = [self._snapshot_group_state(g) for g in self.groups]
        self.scene_snapshots[scene_index] = snaps
        print(f"[SCENE] Saved snapshot for scene {scene_index + 1}")
        self._update_scene_leds()

    def _apply_scene_snapshot(self, scene_index: int):
        snaps = self.scene_snapshots.get(scene_index)
        if not snaps:
            print(f"[SCENE] No snapshot stored for scene {scene_index + 1}")
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

            g.fft_mask = s.fft_mask
            g.fft_mask_autopilot = s.fft_mask_autopilot

            g.color = s.color
            g.color_autopilot = s.color_autopilot

        self.active_scene = scene_index
        self.active_scene_pristine = True

        print(f"[SCENE] Applied snapshot for scene {scene_index + 1}")
        self._update_all_leds()

    def _clear_scene_snapshot(self, scene_index: int):
        if scene_index in self.scene_snapshots:
            del self.scene_snapshots[scene_index]
            print(f"[SCENE] Cleared snapshot for scene {scene_index + 1}")

        if self.active_scene == scene_index:
            self.active_scene = None
            self.active_scene_pristine = False

        self._update_scene_leds()

    def handle_scene_button(self, scene_index: int, press_type: str):
        if press_type == "short":
            if scene_index in self.scene_snapshots:
                self._apply_scene_snapshot(scene_index)
            else:
                print(f"[SCENE] Scene {scene_index + 1} has no saved snapshot")

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
            if press_type == "short":
                sm.start_all_clips()
            elif press_type == "long":
                sm.stop_all_clips()
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

    # --- Resolume wiring: load connections + composition + composition_mappings + OSC ---
    comp_model: Optional[CompositionInfo] = None
    comp_mapping: Optional[CompositionMapping] = None
    osc_client: Optional[SimpleUDPClient] = None

    try:
        connections_cfg = load_connections(Path(DEFAULT_CONNECTIONS_PATH))

        # HTTP for composition JSON
        conn_http = get_resolume_http_connection(connections_cfg, name=None, io_section="outputs")
        comp_json = fetch_composition_json(conn_http)
        comp_name = _guess_composition_name(comp_json)

        # Scan ./composition_mappings/*.yml for a mapping whose composition_name matches
        comp_mapping = CompositionMapping.from_yaml_dir(
            composition_name=comp_name,
            mapping_dir=Path(DEFAULT_COMPOSITION_MAPPING_DIR),
        )

        comp_model = build_composition_model(comp_json, comp_mapping)
        print(f"[RESOLUME] Loaded composition '{comp_model.name}' with {len(comp_model.groups)} groups.")
        debug_dump_composition_columns(comp_model)

        # OSC client
        osc_host, osc_port = get_resolume_osc_connection(connections_cfg, name=None, io_section="outputs")
        osc_client = SimpleUDPClient(osc_host, osc_port)
        print(f"[OSC] Using Resolume OSC at {osc_host}:{osc_port}")

    except Exception as e:
        print(f"[RESOLUME] WARNING: could not initialize Resolume composition/OSC: {e}")

    print(f"\nUsing input:  {in_name}")
    print(f"Using output: {out_name}")
    print("Press Ctrl+C to exit.\n")

    press_mgr = PressManager()

    with mido.open_input(in_name) as in_port, mido.open_output(out_name) as out_port:
        sm = Apc40StateMachine(runtime, out_port, osc_client=osc_client)

        # Attach composition + mapping if available
        if comp_model is not None and comp_mapping is not None:
            sm.attach_composition(comp_model, comp_mapping)

        try:
            while True:
                now = time.monotonic()

                for msg in in_port.iter_pending():
                    handle_midi_message(msg, runtime, sm, press_mgr, now)

                for key, pt in press_mgr.poll(now):
                    dispatch_press(key, pt, runtime, sm)

                sm.update_blink(now)

                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()
