#!/usr/bin/env python3
"""
Generic APC40-style state-machine demo driven by mapping profiles.

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
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
from typing_extensions import runtime
from copy import deepcopy


import mido
import yaml

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

DEFAULT_MAPPINGS_DIR = "mappings"

MidiKey = Tuple[str, int, int]  # ("note"|"cc", channel, note_or_cc)

# APC40 color velocities – fallback defaults (used when no map present)
RED_VELOCITY = 3     # typical: red
ORANGE_VELOCITY = 5  # typical: amber/orange


# ---------------------------------------------------------
# State
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

    Supports:
      - simple toggle: off/on
      - multi-color: off + a list of colors (we pick first non-off as "on"
        for boolean usage, but keep the whole list available for future).
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
                # Long press: fire immediately, cancel pending single
                st.pending_click = False
                events.append((key, "long"))
            else:
                # Short candidate – might be first of a double
                if st.pending_click and (now - st.pending_click_time) <= self.double_window:
                    # Double press
                    st.pending_click = False
                    events.append((key, "double"))
                else:
                    # Start waiting for possible second tap
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
# Config loading
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
        # --- Different shapes allowed for m ---
        if isinstance(m, int):
            # Shorthand: toggle with off=0, on=m
            prof = VelocityProfile(off=0, on=int(m), colors=[])
            print(f"  map '{name}' (int): off={prof.off} on={prof.on}")

        elif isinstance(m, list):
            # List of colors: off=0, colors=m, derive on from first non-zero
            colors = [int(v) for v in m]
            prof = VelocityProfile(off=0, on=127, colors=colors)
            print(f"  map '{name}' (list): off={prof.off} colors={prof.colors}")

        elif isinstance(m, dict):
            # dict: off / on / colors
            off_val = int(m.get("off", 0))

            colors_raw = m.get("colors", [])
            colors = [int(v) for v in colors_raw] if isinstance(colors_raw, list) else []

            on_raw = m.get("on")
            if on_raw is None:
                # Try to derive 'on' from colors list
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
            # Completely unknown type
            print(f"  WARNING: map '{name}' has unsupported type {type(m)}; "
                  f"defaulting to off=0 on=127")
            prof = VelocityProfile(off=0, on=127, colors=[])

        named_profiles[name] = prof

    note_map: Dict[int, VelocityProfile] = {}

    # --- apply ranges ---
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

    # --- per-note overrides ---
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

    # --- final sanity for note 57 ---
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
    input_cfg is the 'input_mappings' subtree from the YAML.
    """
    runtime = MappingRuntime()

    # Per-note velocity profiles
    runtime.note_velocity = build_note_velocity_map(input_cfg)

    group_state_cfg = state_cfg.get("groups", {})
    group_props_cfg = group_state_cfg.get("properties", {})
    group_reset_cfg = group_state_cfg.get("reset", {})  # optional
    group_opacity_cfg = group_state_cfg.get("opacity", {})
    group_intensity_presets_cfg = group_state_cfg.get("intensity_presets", [])

    # --- Per-group controls ---
    for group_id_str, g in input_cfg["groups"].items():
        group_idx = int(group_id_str) - 1  # 1-8 -> 0-7
        channel = g["channel"]

        # Map control names -> note numbers for this group
        control_note_map: Dict[str, int] = {}
        for btn in g["clip_buttons"]:
            control_note_map[btn["name"]] = btn["note"]
        for name in ("clip_stop", "track_select", "activator", "solo", "arm"):
            if name in g:
                control_note_map[name] = g[name]["note"]

        slider_cc = g["slider"]["cc"]

        # Property -> note mapping for LEDs
        prop_notes: Dict[str, int] = {}
        preset_notes: list[int] = []

        # 1) boolean properties (playing/effects/transforms/fft_mask/color)
        for prop_name, cfg_prop in group_props_cfg.items():
            control_name = cfg_prop["control"]
            action_name = cfg_prop["action"]  # semantics; gesture logic is in code

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
            action_name = group_opacity_cfg["action"]  # set_opacity_from_cc
            key = ("cc", channel, slider_cc)
            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="group",
                property_name="opacity",
                group_index=group_idx,
            )

        # 4) intensity presets (clip_row_2–5 etc)
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

        # Global slider (bottom-right) – mapped as global.global_slider
        # Scene slots: map logical scene_launch_1..N to notes
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
                action=action_name,   # "scene_slot"
                scope="global",
                property_name=None,
                group_index=None,
                scene_index=scene_idx,
            )

            runtime.scene_buttons[scene_idx] = (
                s_entry.get("channel", 0),
                s_entry["note"],
            )

        # Global slider (bottom-right) – mapped as global.global_slider
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

        # Queue level (CC 47) – toggles global effects + queue-level flag
        elif control_name == "queue_level" and "queue_level" in input_global_cfg:
            q = input_global_cfg["queue_level"]
            if "note" in q:
                key = ("note", q.get("channel", 0), q["note"])
            elif "cc" in q:
                key = ("cc", q.get("channel", 0), q["cc"])
            else:
                continue

            runtime.action_map[key] = ActionSpec(
                action=action_name,   # toggle_effects_all
                scope="global",
                property_name=None,
                group_index=None,
            )

        # Play/Stop for global autopilot – from global.transport
        elif control_name in ("play", "stop") and "transport" in input_global_cfg:
            t = input_global_cfg["transport"].get(control_name)
            if not t:
                continue
            key = ("note", t.get("channel", 0), t["note"])
            runtime.action_map[key] = ActionSpec(
                action=action_name,   # set_global_autopilot_on / off
                scope="global",
                property_name=None,
                group_index=None,
            )

        # Stop All Clips (short=start all, long=stop all)
        elif control_name == "stop_all_clips" and "stop_all_clips" in input_global_cfg:
            s = input_global_cfg["stop_all_clips"]
            key = ("note", s.get("channel", 0), s["note"])
            runtime.action_map[key] = ActionSpec(
                action=action_name,   # control_all_clips
                scope="global",
                property_name=None,
                group_index=None,
            )

        # Nudge -, Nudge +, Tap Tempo, Tempo Sync – from global.timing_controls
        elif control_name in ("nudge_minus", "nudge_plus", "tap_tempo", "shift_resync"):
            src = timing_cfg.get(control_name)
            if not src:
                continue
            key = ("note", src.get("channel", 0), src["note"])
            runtime.action_map[key] = ActionSpec(
                action=action_name,   # nudge_minus / nudge_plus / tap_tempo / tempo_sync
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
    """

    BOOL_PROPS = ("playing", "effects", "transforms", "fft_mask", "color")

    def __init__(self, runtime: MappingRuntime, midi_out: mido.ports.BaseOutput):
        self.runtime = runtime
        self.midi_out = midi_out
        self.groups = [OutputGroupState() for _ in range(8)]

        # Per-note velocity profiles
        self.note_velocity: Dict[int, VelocityProfile] = runtime.note_velocity

        # Global autopilot: both global and local must be True for
        # autopilot to be effectively "on" for a layer.
        self.global_autopilot: bool = True

        self.global_intensity_scale = 1.0

        # Simple global "queue level" placeholder (0/1)
        self.global_queue_level: int = 0

        # Blink engine
        self.fast_blink_on = True
        self.slow_blink_on = True
        self.fast_period = 0.25  # FAST blink
        self.slow_period = 0.75  # SLOW blink
        now = time.monotonic()
        self.last_fast_toggle = now
        self.last_slow_toggle = now

        # Scene snapshots (for scene_launch buttons)
        # scene index is 0-based: 0..4 for scene_launch_1..5
        self.scene_snapshots: Dict[int, list[OutputGroupState]] = {}
        self.active_scene: int | None = None
        self.active_scene_pristine: bool = False  # True until any other change

        # Initialize intensity LEDs to reflect default intensity
        self._init_intensity_preset_leds()
        
    def _mark_state_changed(self):
        """
        Called whenever the user changes state (any group/global toggle,
        slider, etc.). If a scene is currently active, this clears the
        'pristine' flag so the scene LED becomes solid instead of blinking.
        """
        if self.active_scene is not None and self.active_scene_pristine:
            self.active_scene_pristine = False
            self._update_scene_leds()

    # ---- LED init for intensity presets ----

    def _init_intensity_preset_leds(self):
        for gi in range(len(self.groups)):
            self._update_intensity_leds(gi)

    # ---- Gesture handlers ----

    def handle_property_press(self, group_idx: int, prop: str, press_type: str):
        """
        Implement:
          - Short press: if False -> True; if True -> next_<prop>()
          - Long press:  if True  -> False
          - Double press: toggle <prop>_autopilot
        """
        g = self.groups[group_idx]
        value = getattr(g, prop)

        if press_type == "short":
            if not value:
                self.set_property(group_idx, prop, True)
            else:
                self.next_property(group_idx, prop)

        elif press_type == "long":
            if value:
                self.set_property(group_idx, prop, False)

        elif press_type == "double":
            self.toggle_autopilot(group_idx, prop)

    def handle_intensity_preset_press(self, group_idx: int, preset_value: float, press_type: str):
        """
        Intensity preset buttons:

          - Short: set intensity to preset_value
          - Long:  set intensity to 0.0
          - Double: set intensity to a random quantized value
                    from {0.0, 0.25, 0.5, 0.75, 1.0} that differs
                    from the current intensity.
        """
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
        setattr(g, prop, value)
        print(f"[STATE] Group {group_idx + 1} {prop} -> {value}")
        self._update_led_for_property(group_idx, prop)

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
        For now, just log; later this can hook into Resolume/OSC/etc.
        """
        print(f"[STATE] Group {group_idx + 1} NEXT {prop}")

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

    def set_intensity_from_preset(self, group_idx: int, value: float):
        v = max(0.0, min(1.0, value))
        self.groups[group_idx].intensity = v
        print(f"[STATE] Group {group_idx + 1} intensity -> {v:.3f}")
        self._update_intensity_leds(group_idx)

    def _update_intensity_leds(self, group_idx: int):
        """
        Map current intensity (0–1) to a bar across preset buttons:

          0.00 -> 0 lit
          0.25 -> 1 lit
          0.50 -> 2 lit
          0.75 -> 3 lit
          1.00 -> 4 lit
        """
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

    def toggle_effects_all(self):
        """
        Queue button toggles 'effects' across all groups,
        and flips a simple global_queue_level placeholder.
        """
        any_on = any(g.effects for g in self.groups)
        new_val = not any_on
        for idx in range(len(self.groups)):
            self.set_property(idx, "effects", new_val)

        self.global_queue_level = 1 - self.global_queue_level
        print(f"[STATE] All groups effects -> {new_val}, global_queue_level -> {self.global_queue_level}")

    # ---- Global autopilot control ----

    def set_global_autopilot(self, value: bool):
        self.global_autopilot = value
        print(f"[STATE] Global autopilot -> {value}")
        # Recompute LEDs for all boolean props
        for gi in range(len(self.groups)):
            for prop in self.BOOL_PROPS:
                self._update_led_for_property(gi, prop)

    def set_global_autopilot_on(self):
        self.set_global_autopilot(True)

    def set_global_autopilot_off(self):
        self.set_global_autopilot(False)

    # ---- Global clips control ----

    def start_all_clips(self):
        print("[STATE] Start all clips")
        for gi in range(len(self.groups)):
            self.set_property(gi, "playing", True)

    def stop_all_clips(self):
        print("[STATE] Stop all clips")
        for gi in range(len(self.groups)):
            self.set_property(gi, "playing", False)

    # ---- Global tempo / nudge ----

    def global_nudge(self, direction: int):
        """
        direction: -1 for nudge-, +1 for nudge+
        Placeholder: just log for now.
        """
        print(f"[STATE] Global nudge {'+' if direction > 0 else '-'}")

    def global_nudge_plus(self):
        self.global_nudge(+1)

    def global_nudge_minus(self):
        self.global_nudge(-1)

    def global_tap_tempo(self):
        """
        Tap-tempo pulse event.
        No BPM computation; used as a trigger for external sync logic.
        """
        print("[STATE] Tap tempo pulse")

    def global_tempo_sync(self):
        """
        Placeholder for tempo-sync action (e.g., sync to external clock/Resolume).
        """
        print("[STATE] Global tempo sync requested")

    # ---- LED feedback & blinking for boolean layers ----

    def _base_velocity_for_note(self, note: int, prop: str | None = None) -> int:
        """
        Choose base LED velocity (color) for a note when "on".

        If a velocity profile is configured for this note, use its
        resolved_on() value. Otherwise:

          - fft_mask -> ORANGE_VELOCITY
          - others   -> full-bright 127
        """
        profile = self.note_velocity.get(note)
        if profile:
            vel = profile.resolved_on()
            # print(f"[DEBUG] note {note} using mapped velocity {vel} (prop={prop})")
            return vel


        print(f"[DEBUG] note {note} using fallback 127 (prop={prop})")
        return 127




    def _desired_led_on(self, group_idx: int, prop: str) -> bool:
        """
        LED logic for boolean properties:

        - value == False                 -> off
        - global_autopilot == False      -> blink SLOW
        - global_autopilot == True AND local_autopilot == False -> blink FAST
        - both global & local autopilot  -> solid
        """
        g = self.groups[group_idx]
        value = getattr(g, prop)

        if not value:
            return False

        local_auto = getattr(g, f"{prop}_autopilot", True)

        # Global disabled: blink SLOW
        if not self.global_autopilot:
            return self.slow_blink_on

        # Global enabled but local disabled: blink FAST
        if not local_auto:
            return self.fast_blink_on

        # Both global & local autopilot on: solid
        return True

    def _update_led_for_property(self, group_idx: int, prop: str):
        addrs = self.runtime.group_addrs[group_idx]
        if prop not in addrs.prop_notes:
            return
        note = addrs.prop_notes[prop]
        ch = addrs.channel

        on = self._desired_led_on(group_idx, prop)
        if on:
            velocity = self._base_velocity_for_note(note, prop)
        else:
            velocity = 0

        msg = mido.Message("note_on", channel=ch, note=note, velocity=velocity)
        self.midi_out.send(msg)

    def update_blink(self, now: float):
        """
        Blink engine for fast and slow blink states.
        """
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

        # Only recompute LEDs if something changed
        if fast_changed or slow_changed:
            for gi in range(len(self.groups)):
                for prop in self.BOOL_PROPS:
                    self._update_led_for_property(gi, prop)
            self._update_scene_leds()
    # ---- Scene snapshots -------------------------------------------------

    def _snapshot_group_state(self, g: OutputGroupState) -> OutputGroupState:
        """
        Create a copy of a group's state suitable for storing in a scene.

        Requirements: capture all boolean/autopilot toggles,
        but IGNORE opacity and intensity.
        """
        snap = deepcopy(g)
        snap.opacity = 0.0
        snap.intensity = 0.0
        return snap

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

        # Apply all saved group states (booleans + autopilots)
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
        self.active_scene_pristine = True  # no changes yet since launch

        print(f"[SCENE] Applied snapshot for scene {scene_index + 1}")

        # Refresh all LEDs (groups + scenes)
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
        """
        Scene button semantics:

          - Short press:
              if snapshot exists -> apply it
              (button becomes active; blinks if no further changes)

          - Long press:
              save current group toggles to this scene, overwriting any
              existing snapshot. Illuminates the button.

          - Double press:
              clear snapshot for this scene and turn the LED off.
        """
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
        """
        Refresh LEDs for all groups and scenes.
        """
        for gi in range(len(self.groups)):
            for prop in self.BOOL_PROPS:
                self._update_led_for_property(gi, prop)
            self._update_intensity_leds(gi)

        self._update_scene_leds()

    def _update_scene_leds(self):
        """
        Update LEDs for scene buttons.

        Rules:
          - No snapshot stored          -> LED off
          - Snapshot stored, not active -> solid
          - Snapshot stored, active and pristine -> blinking
        """
        for scene_idx, (ch, note) in self.runtime.scene_buttons.items():
            has_snapshot = scene_idx in self.scene_snapshots

            if not has_snapshot:
                vel = 0
            else:
                base = self._base_velocity_for_note(note, prop=None)
                if self.active_scene == scene_idx and self.active_scene_pristine:
                    # Blink using fast blink state
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
    """
    Fire global actions that should not use short/long/double press logic.
    These trigger immediately on note_on.
    """
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
    # Note messages
    if msg.type in ("note_on", "note_off"):
        key = ("note", msg.channel, msg.note)

        # For certain global actions, we do NOT want short/long/double
        # semantics – just fire immediately on note_on.
        if msg.type == "note_on" and msg.velocity > 0:
            spec = runtime.action_map.get(key)
            if spec and spec.scope == "global" and spec.action in IMMEDIATE_GLOBAL_ACTIONS:
                handle_immediate_global_action(spec, sm)
                return  # do not feed this into PressManager

        # Everything else still goes through the gesture engine
        events = press_mgr.handle_note_message(key, msg, now)
        for k, press_type in events:
            dispatch_press(k, press_type, runtime, sm)

    # CC messages -> direct actions (no gesture)
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

        # Boolean property buttons (Arm/Solo/Activator/Clip Stop/Clip Row 1 etc.)
        if spec.property_name in sm.BOOL_PROPS:
            sm.handle_property_press(gi, spec.property_name, press_type)

        # Intensity preset buttons
        elif spec.action == "set_intensity_preset" and spec.intensity_value is not None:
            sm.handle_intensity_preset_press(gi, spec.intensity_value, press_type)

        # Reset button: only react on short press (if configured)
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

    # Load all profiles
    profiles = load_mapping_profiles(mappings_dir)

    # Get MIDI ports so we can auto-match
    midi_input_names = mido.get_input_names()
    midi_output_names = mido.get_output_names()

    # Choose mapping profile
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

    # Build the runtime mapping (includes per-note velocity profiles)
    runtime = build_runtime_mapping(input_cfg, state_cfg)

    in_name = auto_select_port(midi_input_names, controller_name, "input")
    out_name = auto_select_port(midi_output_names, controller_name, "output")

    print(f"\nUsing input:  {in_name}")
    print(f"Using output: {out_name}")
    print("Press Ctrl+C to exit.\n")

    press_mgr = PressManager()

    with mido.open_input(in_name) as in_port, mido.open_output(out_name) as out_port:
        sm = Apc40StateMachine(runtime, out_port)
        try:
            while True:
                now = time.monotonic()

                # Process incoming messages
                for msg in in_port.iter_pending():
                    handle_midi_message(msg, runtime, sm, press_mgr, now)

                # Resolve pending single-clicks (short presses)
                for key, pt in press_mgr.poll(now):
                    dispatch_press(key, pt, runtime, sm)

                # Update blinking LEDs
                sm.update_blink(now)

                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()
