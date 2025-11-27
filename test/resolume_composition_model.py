#!/usr/bin/env python3
"""
Resolume composition model + autopilot + HTTP control.

- Loads connections from settings/connections.yaml
- Fetches /composition from Resolume Arena HTTP API
- Builds a typed model of:
    CompositionInfo
      -> GroupInfo (Resolume layergroup)
         -> LayerInfo
            -> ClipInfo

- Loads a composition mapping YAML that:
    * maps APC group indices (1–8) to Resolume group names
    * optionally describes layer-role names (Colors, Effects, etc.)

- Tracks:
    * composition name
    * every layer's ID, name, group, role
    * clip IDs, names, column indexes
      - column 1 = STOP clip
      - column 2 = NONE/pass-through
      - column >=3 = CONTENT

- Provides:
    * playing_duration per clip
    * autopilot that chooses new clips based on group intensity & properties
    * ResolumeController:
        - update_from_state(apc_state_machine)
        - trigger_next_clip_for_group(apc_group_index, intensity)

  and actually fires HTTP requests to Resolume to connect clips.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import requests
import yaml

# ------------------------------------------------------------------------
# Paths / basic config
# ------------------------------------------------------------------------

DEFAULT_CONNECTIONS_PATH = "settings/connections.yaml"
DEFAULT_COMPOSITION_MAPPING_DIR = "composition_mappings"


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

def _name_from_field(val: Any, default: str) -> str:
    """
    Resolume sometimes gives { "value": "Name" } instead of a plain string.
    Handle both.
    """
    if isinstance(val, dict) and "value" in val:
        inner = val["value"]
        if isinstance(inner, str):
            return inner
    if isinstance(val, str):
        return val
    return default


# ------------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------------

@dataclass
class ClipInfo:
    id: str
    name: str
    column_index: int  # 1-based column index in the Resolume grid

    # semantic classification based on column
    is_stop: bool = False      # column 1
    is_none: bool = False      # column 2 (pass-through)
    is_content: bool = True    # columns >= 3

    @property
    def is_blank(self) -> bool:
        """For backwards compatibility: treat STOP/NONE as 'blank-ish'."""
        return self.is_stop or self.is_none

    # runtime fields (not from API)
    last_triggered_at: Optional[float] = None

    def playing_duration(self, now: Optional[float] = None) -> float:
        """Return how long this clip has been playing (seconds)."""
        if self.last_triggered_at is None:
            return 0.0
        if now is None:
            now = time.monotonic()
        return max(0.0, now - self.last_triggered_at)


@dataclass
class LayerInfo:
    id: str
    name: str
    group_name: str  # convenience: name of the Resolume layergroup
    group_id: str
    role: Optional[str] = None  # e.g. "colors", "effects", etc.

    # index of this layer inside its group (0-based)
    index_in_group: int = 0

    clips: List[ClipInfo] = field(default_factory=list)

    # runtime: currently playing clip id (if we track it)
    current_clip_id: Optional[str] = None

    def get_clip_by_id(self, clip_id: str) -> Optional[ClipInfo]:
        for c in self.clips:
            if c.id == clip_id:
                return c
        return None

    def get_stop_clips(self) -> List[ClipInfo]:
        return [c for c in self.clips if c.is_stop]

    def get_none_clips(self) -> List[ClipInfo]:
        return [c for c in self.clips if c.is_none]

    def get_content_clips(self) -> List[ClipInfo]:
        return [c for c in self.clips if c.is_content]


@dataclass
class GroupInfo:
    """Resolume layergroup, plus which APC group index (if any) maps to it."""
    id: str
    name: str

    # index of this group (0-based) in the composition's layergroups list
    index_in_composition: int = 0

    # APC mapping: which APC output group index controls this Resolume group?
    apc_group_index: Optional[int] = None

    layers: List[LayerInfo] = field(default_factory=list)


@dataclass
class CompositionInfo:
    name: str
    id: Optional[str] = None  # in case Resolume provides an ID
    groups: List[GroupInfo] = field(default_factory=list)

    # Convenient lookup tables
    groups_by_name: Dict[str, GroupInfo] = field(default_factory=dict)
    groups_by_id: Dict[str, GroupInfo] = field(default_factory=dict)
    layers_by_id: Dict[str, LayerInfo] = field(default_factory=dict)
    clips_by_id: Dict[str, ClipInfo] = field(default_factory=dict)

    def index(self):
        """Build fast lookup tables after construction."""
        self.groups_by_name = {g.name: g for g in self.groups}
        self.groups_by_id = {g.id: g for g in self.groups}
        self.layers_by_id.clear()
        self.clips_by_id.clear()

        for g in self.groups:
            for layer in g.layers:
                self.layers_by_id[layer.id] = layer
                for clip in layer.clips:
                    self.clips_by_id[clip.id] = clip

    # Example convenience: get all layers for a given APC group index
    def layers_for_apc_group(self, apc_group_index: int) -> List[LayerInfo]:
        out: List[LayerInfo] = []
        for g in self.groups:
            if g.apc_group_index == apc_group_index:
                out.extend(g.layers)
        return out


# ------------------------------------------------------------------------
# Loading connections + hitting Resolume
# ------------------------------------------------------------------------

def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_connections(path: Path = Path(DEFAULT_CONNECTIONS_PATH)) -> Dict[str, Any]:
    cfg = load_yaml(path)
    return cfg or {}


def get_resolume_http_connection(
    connections_cfg: Dict[str, Any],
    name: Optional[str] = None,
    io_section: str = "outputs",
) -> Dict[str, Any]:
    """
    Pull one HTTP connection config from settings/connections.yaml.

    io_section: "inputs" or "outputs"
    name:       if None, choose first entry under outputs.http.resolume_arena
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

    raise RuntimeError(f"HTTP Resolume connection named '{name}' not found.")


def make_resolume_base_url(conn: Dict[str, Any]) -> str:
    host = conn.get("host", "127.0.0.1")
    port = conn.get("port", 8080)
    use_https = bool(conn.get("use_https", False))
    api_base = conn.get("api_base", "/api/v1").rstrip("/")

    scheme = "https" if use_https else "http"
    return f"{scheme}://{host}:{port}{api_base}"


def fetch_composition_json(conn: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /composition from Resolume API, using basic auth if configured.
    """
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


# ------------------------------------------------------------------------
# Resolume HTTP client for triggering clips
# ------------------------------------------------------------------------

class ResolumeHTTPClient:
    """
    Small helper to call Resolume HTTP API for clip triggering.
    Uses group/layer indices & column_index -> URL.

    NOTE: The exact path may differ depending on your Resolume version.
    Adjust build_clip_connect_url() to match your setup.
    """

    def __init__(self, conn: Dict[str, Any]):
        self.conn = conn
        self.base_url = make_resolume_base_url(conn)
        self.auth = None
        if conn.get("username") and conn.get("password"):
            self.auth = (conn["username"], conn["password"])
        self.timeout = conn.get("timeout", 2.0)
        self.verify = bool(conn.get("verify_ssl", True))

    def build_clip_connect_url(self, group: GroupInfo, layer: LayerInfo, clip: ClipInfo) -> str:
        """
        Example path. You may need to tweak this to match actual Resolume API:

          /composition/layergroups/{groupIdx}/layers/{layerIdx}/clips/{column}/connect

        where indices are zero-based.
        """
        g_idx = group.index_in_composition
        l_idx = layer.index_in_group
        col = clip.column_index  # 1-based column index
        return f"{self.base_url}/composition/layergroups/{g_idx}/layers/{l_idx}/clips/{col}/connect"

    def connect_clip(self, group: GroupInfo, layer: LayerInfo, clip: ClipInfo) -> None:
        url = self.build_clip_connect_url(group, layer, clip)
        try:
            resp = requests.post(url, auth=self.auth, timeout=self.timeout, verify=self.verify)
            if 200 <= resp.status_code < 300:
                # success
                return
            print(f"[HTTP] Clip connect failed {resp.status_code}: {url}")
        except Exception as exc:
            print(f"[HTTP] ERROR {exc} when calling {url}")


# ------------------------------------------------------------------------
# Loading composition mapping (APC group → Resolume group)
# ------------------------------------------------------------------------

@dataclass
class CompositionMapping:
    composition_name: str
    apc_groups: Dict[int, str]  # APC group index -> Resolume group name
    layer_roles: Dict[str, List[str]]  # e.g. "colors" -> ["Colors", "Colour"]

    @classmethod
    def from_yaml_dir(
        cls,
        composition_name: str,
        mapping_dir: Path = Path(DEFAULT_COMPOSITION_MAPPING_DIR),
    ) -> "CompositionMapping":
        """
        Find a mapping file whose composition_name matches, else fallback to
        a file named <composition_name>.yaml/.yml, else raise.
        """
        if not mapping_dir.exists():
            raise RuntimeError(f"Mapping directory does not exist: {mapping_dir}")

        # 1) Try scanning for a file with matching composition_name
        for path in sorted(mapping_dir.glob("*.y*ml")):
            raw = load_yaml(path) or {}
            if raw.get("composition_name") == composition_name:
                return cls(
                    composition_name=composition_name,
                    apc_groups={int(k): v for k, v in (raw.get("apc_groups") or {}).items()},
                    layer_roles=raw.get("layer_roles") or {},
                )

        # 2) Try filename-based: <composition_name>.yaml / .yml
        for suffix in (".yaml", ".yml"):
            candidate = mapping_dir / f"{composition_name}{suffix}"
            if candidate.exists():
                raw = load_yaml(candidate) or {}
                return cls(
                    composition_name=composition_name,
                    apc_groups={int(k): v for k, v in (raw.get("apc_groups") or {}).items()},
                    layer_roles=raw.get("layer_roles") or {},
                )

        raise RuntimeError(
            f"No composition mapping found for composition '{composition_name}' "
            f"in {mapping_dir}"
        )


# ------------------------------------------------------------------------
# Building CompositionInfo from Resolume JSON
# ------------------------------------------------------------------------

def _guess_composition_name(comp_json: Dict[str, Any]) -> str:
    """
    Resolume usually exposes a "name" field at the top-level, sometimes as
    { "value": "Comp 1" }.
    """
    raw_name = comp_json.get("name") or comp_json.get("compositionName")
    return _name_from_field(raw_name, default="Unnamed Composition")


def _iter_layergroups(comp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return the list of "layergroups" from the JSON.

    Adjust this if the structure differs in your Resolume version.
    """
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


def _clip_column_index(clip_json: Dict[str, Any]) -> int:
    if "column" in clip_json:
        return int(clip_json["column"])
    if "columnIndex" in clip_json:
        return int(clip_json["columnIndex"])
    # Fallback: position in list; caller should treat 1 as stop, 2 as none, >=3 as content
    return int(clip_json.get("index", 0)) + 1


def _clip_name(clip_json: Dict[str, Any]) -> str:
    raw_name = clip_json.get("name") or clip_json.get("displayName")
    return _name_from_field(raw_name, default=f"clip-{clip_json.get('id', '?')}")


def _layer_name(layer_json: Dict[str, Any]) -> str:
    raw_name = layer_json.get("name") or layer_json.get("displayName")
    return _name_from_field(raw_name, default=f"layer-{layer_json.get('id', '?')}")


def _classify_clip_by_column(col: int) -> Dict[str, bool]:
    """
    Column semantics:

      - col == 1 -> STOP
      - col == 2 -> NONE (pass-through)
      - col >= 3 -> CONTENT
    """
    if col <= 0:
        col = 1

    if col == 1:
        return {"is_stop": True, "is_none": False, "is_content": False}
    elif col == 2:
        return {"is_stop": False, "is_none": True, "is_content": False}
    else:
        return {"is_stop": False, "is_none": False, "is_content": True}


def _match_layer_role(layer_name: str, mapping: CompositionMapping) -> Optional[str]:
    """
    Given a layer name and the mapping's layer_roles, return a role key like
    "colors", "effects", etc., or None if no match.
    """
    lname = layer_name.lower()
    for role, patterns in mapping.layer_roles.items():
        for p in patterns:
            if p.lower() in lname:
                return role
    return None


def build_composition_model(
    comp_json: Dict[str, Any],
    mapping: CompositionMapping
) -> CompositionInfo:
    """
    Build a CompositionInfo structure from the raw composition JSON
    and a CompositionMapping (APC -> Resolume groups).
    """
    comp_name = _guess_composition_name(comp_json)
    comp_id = comp_json.get("id")

    comp = CompositionInfo(name=comp_name, id=comp_id, groups=[])

    lg_list = _iter_layergroups(comp_json)

    # Invert APC group name mapping: Resolume group name -> APC index
    resolume_group_to_apc: Dict[str, int] = {}
    for apc_idx, grp_name in mapping.apc_groups.items():
        resolume_group_to_apc[grp_name.lower()] = apc_idx

    for g_idx, g_json in enumerate(lg_list):
        g_id = str(g_json.get("id"))
        raw_g_name = g_json.get("name")
        g_name = _name_from_field(raw_g_name, default=f"group-{g_id}")

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
            layer_info = LayerInfo(
                id=l_id,
                name=l_name,
                group_name=g_name,
                group_id=g_id,
                role=role,
                index_in_group=l_idx,
            )

            clips: List[ClipInfo] = []
            clips_json = _iter_clips(layer_json)

            for clip_json in clips_json:
                c_id = str(clip_json.get("id"))
                c_name = _clip_name(clip_json)
                col_idx = _clip_column_index(clip_json)
                klass = _classify_clip_by_column(col_idx)

                clips.append(
                    ClipInfo(
                        id=c_id,
                        name=c_name,
                        column_index=col_idx,
                        is_stop=klass["is_stop"],
                        is_none=klass["is_none"],
                        is_content=klass["is_content"],
                    )
                )

            layer_info.clips = clips
            group_info.layers.append(layer_info)

        comp.groups.append(group_info)

    comp.index()
    return comp


# ------------------------------------------------------------------------
# Autopilot logic
# ------------------------------------------------------------------------

def choose_autopilot_clip(
    layer: LayerInfo,
    intensity: float,
    rng: random.Random | None = None,
    blank_weight_at_zero: float = 0.9,
    blank_weight_at_one: float = 0.1,
    force_change: bool = False,
) -> Optional[ClipInfo]:
    """
    Pick a clip for this layer under "autopilot".

    Semantics:

      - STOP clips (is_stop) are *not* chosen by autopilot.
      - NONE clips (is_none) are "blank/pass-through".
      - CONTENT clips (is_content) are actual content.

    Rules:
      - The higher the intensity (0–1), the more likely we pick CONTENT.
      - The lower the intensity, the more likely we pick NONE.
      - We never pick STOP here.
      - We try not to pick the same CONTENT clip twice in a row.
      - If force_change == True, we avoid re-choosing the current CONTENT
        if possible (NONE is allowed to repeat).
    """
    if rng is None:
        rng = random

    blanks = layer.get_none_clips()
    content = layer.get_content_clips()

    if not blanks and not content:
        return None

    current_id = layer.current_clip_id

    # Only NONE available
    if blanks and not content:
        return rng.choice(blanks)

    # Only CONTENT available
    if content and not blanks:
        candidates = [c for c in content if not (force_change and c.id == current_id)]
        if not candidates:
            candidates = content
        return rng.choice(candidates)

    # Both available; clamp intensity
    intensity = max(0.0, min(1.0, intensity))

    # Linear interpolation of blank weight: high at intensity=0, low at intensity=1
    blank_weight = blank_weight_at_zero + (blank_weight_at_one - blank_weight_at_zero) * intensity
    content_weight = max(0.0, 1.0 - blank_weight)

    total = blank_weight + content_weight
    blank_prob = (blank_weight / total) if total > 0 else 0.5

    pick_blank = (rng.random() < blank_prob)

    if pick_blank:
        return rng.choice(blanks)

    candidates = [c for c in content if not (force_change and c.id == current_id)]
    if not candidates:
        candidates = content
    return rng.choice(candidates)


def mark_clip_triggered(layer: LayerInfo, clip: ClipInfo, now: Optional[float] = None) -> None:
    """
    Mark a clip as "triggered" now, so we can compute its playing duration later.
    """
    if now is None:
        now = time.monotonic()
    clip.last_triggered_at = now
    layer.current_clip_id = clip.id


# ------------------------------------------------------------------------
# ResolumeController – hook for APC40 state machine
# ------------------------------------------------------------------------

class ResolumeController:
    """
    Holds CompositionInfo and drives clip selection based on APC40 group state.

    Intended state-machine contract:

      sm.groups: list[OutputGroupState] with attributes:
        - playing: bool
        - playing_autopilot: bool
        - effects: bool
        - effects_autopilot: bool
        - transforms: bool
        - transforms_autopilot: bool
        - color: bool
        - color_autopilot: bool
        - fft_mask: bool
        - fft_mask_autopilot: bool
        - intensity: float (0–1)
      sm.global_autopilot: bool

    We use:
      - playing       to decide if any layers in that group should be active.
      - effects       to decide whether "effects" role layers should show content vs none.
      - intensity     to bias NONE vs CONTENT selection.
    """

    def __init__(
        self,
        composition: CompositionInfo,
        http_client: ResolumeHTTPClient,
        clip_hold_seconds: float = 30.0,
    ):
        self.composition = composition
        self.http = http_client
        self.clip_hold_seconds = clip_hold_seconds
        self.rng = random.Random()

        # Optional external callback: (layer, clip) -> None
        self.on_trigger_clip: Optional[Callable[[LayerInfo, ClipInfo], None]] = None

    # ---- public API to integrate with state machine ----

    def update_from_state(self, sm, now: Optional[float] = None) -> None:
        """
        Call this once per main loop iteration.

        Drives Resolume layers based on APC40 state:
          - playing & playing_autopilot & global_autopilot
          - effects on/off
          - intensity
        """
        if now is None:
            now = time.monotonic()

        if not getattr(sm, "global_autopilot", True):
            return

        for group_idx, g_state in enumerate(sm.groups):
            apc_group_index = group_idx + 1
            self._update_group_from_state(
                apc_group_index=apc_group_index,
                g_state=g_state,
                now=now,
            )

    def trigger_next_clip_for_group(self, apc_group_index: int, intensity: float) -> None:
        """
        Manual "play next" hook – call when state machine gets a 'next clip'
        action for a given APC group. Forces a change in CONTENT clip when possible.
        """
        layers = self.composition.layers_for_apc_group(apc_group_index)
        if not layers:
            return

        now = time.monotonic()

        for layer in layers:
            # For now, apply "next" to all roles in that group.
            new_clip = choose_autopilot_clip(
                layer,
                intensity=intensity,
                rng=self.rng,
                force_change=True,
            )
            if not new_clip:
                continue
            self._trigger_clip(layer, new_clip, now)

    # ---- internal per-group logic ----

    def _update_group_from_state(self, apc_group_index: int, g_state, now: float) -> None:
        layers = self.composition.layers_for_apc_group(apc_group_index)
        if not layers:
            return

        # If the group isn't playing, slam STOP on all its layers if we have a STOP clip.
        if not g_state.playing:
            for layer in layers:
                self._ensure_stopped(layer, now)
            return

        # Group is playing and global autopilot is on; steer per role.
        for layer in layers:
            role = (layer.role or "").lower()

            if role == "effects":
                self._update_effect_layer(layer, g_state, now)
            elif role in ("colors", "colour", "color"):
                self._update_color_layer(layer, g_state, now)
            else:
                # Fills, masks, transforms, etc. You can specialize later.
                self._update_generic_layer(layer, g_state, now)

    def _ensure_stopped(self, layer: LayerInfo, now: float) -> None:
        """If we can, send the STOP clip (col 1) for this layer."""
        stop_clips = layer.get_stop_clips()
        if not stop_clips:
            return
        stop_clip = stop_clips[0]
        if layer.current_clip_id == stop_clip.id:
            return
        self._trigger_clip(layer, stop_clip, now)

    # ---- per-role strategies ----

    def _update_effect_layer(self, layer: LayerInfo, g_state, now: float) -> None:
        """
        Effects layers obey group.effects and group.intensity.
        When an effect clip changes, log the effect name explicitly.
        """
        # If effects off, prefer NONE (pass-through). If no NONE, STOP.
        if not g_state.effects:
            none_clips = layer.get_none_clips()
            target = none_clips[0] if none_clips else None
            if target is None:
                # fallback to STOP if we have one
                stop_clips = layer.get_stop_clips()
                target = stop_clips[0] if stop_clips else None

            if target and layer.current_clip_id != target.id:
                self._trigger_clip(layer, target, now)
            return

        # Effects are ON and we have autopilot control
        current = (
            layer.get_clip_by_id(layer.current_clip_id)
            if layer.current_clip_id
            else None
        )

        # If nothing playing, start something now
        if current is None:
            new_clip = choose_autopilot_clip(
                layer,
                intensity=g_state.intensity,
                rng=self.rng,
                force_change=False,
            )
            if new_clip:
                self._trigger_clip(layer, new_clip, now)
            return

        # Otherwise check duration
        dur = current.playing_duration(now)
        if dur >= self.clip_hold_seconds:
            new_clip = choose_autopilot_clip(
                layer,
                intensity=g_state.intensity,
                rng=self.rng,
                force_change=True,
            )
            if new_clip and new_clip.id != current.id:
                self._trigger_clip(layer, new_clip, now)

    def _update_color_layer(self, layer: LayerInfo, g_state, now: float) -> None:
        """
        Color layers follow playing/intensity only (always active while group is playing).
        """
        current = (
            layer.get_clip_by_id(layer.current_clip_id)
            if layer.current_clip_id
            else None
        )

        if current is None:
            new_clip = choose_autopilot_clip(
                layer,
                intensity=g_state.intensity,
                rng=self.rng,
                force_change=False,
            )
            if new_clip:
                self._trigger_clip(layer, new_clip, now)
            return

        dur = current.playing_duration(now)
        if dur >= self.clip_hold_seconds:
            new_clip = choose_autopilot_clip(
                layer,
                intensity=g_state.intensity,
                rng=self.rng,
                force_change=True,
            )
            if new_clip and new_clip.id != current.id:
                self._trigger_clip(layer, new_clip, now)

    def _update_generic_layer(self, layer: LayerInfo, g_state, now: float) -> None:
        """
        Default for layers that aren't colors/effects (fills, masks, etc.).
        For now: same behavior as color layer.
        """
        self._update_color_layer(layer, g_state, now)

    # ---- trigger plumbing ----

    def _trigger_clip(self, layer: LayerInfo, clip: ClipInfo, now: float) -> None:
        """
        Internal trigger helper:

          - updates runtime state (last_triggered_at, current_clip_id)
          - calls HTTP client to connect the clip
          - logs effect names nicely when role is effects
          - invokes any external callback
        """
        group = self.composition.groups_by_name[layer.group_name]

        mark_clip_triggered(layer, clip, now)
        self.http.connect_clip(group, layer, clip)

        role = (layer.role or "").lower()

        if role == "effects" and clip.is_content:
            print(
                f"[EFFECT] Group '{layer.group_name}' "
                f"Layer '{layer.name}' -> Effect '{clip.name}' "
                f"(col={clip.column_index})"
            )
        else:
            print(
                f"[RESOLUME] Group '{layer.group_name}' "
                f"Layer '{layer.name}' -> Clip '{clip.name}' "
                f"(col={clip.column_index}, stop={clip.is_stop}, none={clip.is_none}, content={clip.is_content})"
            )

        if self.on_trigger_clip is not None:
            self.on_trigger_clip(layer, clip)


# ------------------------------------------------------------------------
# Demo main
# ------------------------------------------------------------------------

def demo_build_model_and_autopilot():
    """
    Small demo that:
      - loads connections.yaml
      - fetches /composition
      - loads a composition mapping
      - builds CompositionInfo
      - runs a one-shot "pick clips" demo
    """
    connections_cfg = load_connections(Path(DEFAULT_CONNECTIONS_PATH))
    conn = get_resolume_http_connection(connections_cfg, name=None, io_section="outputs")
    comp_json = fetch_composition_json(conn)

    comp_name = _guess_composition_name(comp_json)
    mapping = CompositionMapping.from_yaml_dir(
        composition_name=comp_name,
        mapping_dir=Path(DEFAULT_COMPOSITION_MAPPING_DIR),
    )

    comp_model = build_composition_model(comp_json, mapping)
    http_client = ResolumeHTTPClient(conn)
    controller = ResolumeController(comp_model, http_client, clip_hold_seconds=30.0)

    print(f"\n[MODEL] Composition: {comp_model.name!r}")
    for g in comp_model.groups:
        print(f"  Group {g.id} '{g.name}' (APC group={g.apc_group_index})")
        for layer in g.layers:
            print(f"    Layer {layer.id} '{layer.name}' role={layer.role}")
            print(
                f"      Clips: {[f'{c.column_index}:{c.name} (stop={c.is_stop}, none={c.is_none}, content={c.is_content})' for c in layer.clips]}"
            )

    # Fake a single APC group state to test autopilot logic without the real state machine
    class DummyGroupState:
        playing = True
        playing_autopilot = True
        effects = True
        effects_autopilot = True
        transforms = False
        transforms_autopilot = True
        color = True
        color_autopilot = True
        fft_mask = False
        fft_mask_autopilot = True
        intensity = 0.7

    class DummySM:
        global_autopilot = True
        groups = [DummyGroupState() for _ in range(8)]

    print("\n[AUTOPILOT DEMO] Updating from dummy state...")
    controller.update_from_state(DummySM())

if __name__ == "__main__":
    demo_build_model_and_autopilot()
