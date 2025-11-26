#!/usr/bin/env python3
"""
APC40 state-machine demo driven by two configs:

- Input mapping:  input_mappings/midi_apc40.yaml
- State mapping:  state_mappings/state_apc40_layers.yaml

Maintains 8 "output groups" with properties:
    playing: bool
    effects: bool
    transforms: bool
    fft_mask: bool
    color: bool
    opacity: float (0-1)
    intensity: float (0-1)

Each boolean property also has an "autopilot" flag:

    <prop>_autopilot: bool

Button gesture semantics for boolean-layer buttons
(Arm, Solo, Activator, Clip Stop, Clip Row 1):

  - Short press:
      if prop is False: set to True
      if prop is True:  call next_<prop>() (hook point)
  - Long press:
      if prop is True:  set to False
  - Double press:
      toggle <prop>_autopilot

Intensity preset semantics for clip_row_2–5:

  - Short press sets intensity to:
      row 2 -> 0.25
      row 3 -> 0.50
      row 4 -> 0.75
      row 5 -> 1.00
  - Long press on any of them -> intensity 0.0
  - Double press -> random quantized intensity != current
  - LEDs show intensity as a red "bar" across the 4 buttons:
      0.00 -> all off
      0.25 -> first 1 red
      0.50 -> first 2 red
      0.75 -> first 3 red
      1.00 -> all 4 red

Global controls:

  - Play          -> set_global_autopilot_on (short)
  - Stop          -> set_global_autopilot_off (short)
  - Stop All Clips:
        short -> start_all_clips()  (set playing=True for all groups)
        long  -> stop_all_clips()   (set playing=False for all groups)
  - Queue Level   -> toggle_effects_all() and flip global_queue_level
  - Nudge -, Nudge +
        short -> global_nudge(-1) / global_nudge(+1)
  - Tap Tempo     -> global_tap_tempo() (compute global_tempo_bpm from taps)
  - Tempo Sync    -> global_tempo_sync() (placeholder hook for external sync)

Autopilot semantics for boolean props:
  - global_autopilot: bool (controlled by Play/Stop)
  - effective_autopilot(prop) = global_autopilot AND <prop>_autopilot

LED behavior for boolean properties:
  - prop == False                      -> off
  - prop == True, global & local on    -> solid
  - prop == True, global_autopilot off -> blink SLOW
  - prop == True, global on, local off -> blink FAST

Colors:
  - fft_mask boolean: orange when "on" (subject to blink)
  - other booleans: full-bright
  - intensity presets: red (static, no blink)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Any

import mido
import yaml
import random

INPUT_MAPPING_FILE = "input_mappings/midi_apc40.yaml"
STATE_MAPPING_FILE = "state_mappings/state_apc40_layers.yaml"

MidiKey = Tuple[str, int, int]  # ("note"|"cc", channel, note_or_cc)

# APC40 color velocities – tweak to taste for your unit
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


@dataclass
class MappingRuntime:
    action_map: Dict[MidiKey, ActionSpec] = field(default_factory=dict)
    group_addrs: Dict[int, GroupControlAddresses] = field(default_factory=dict)


# ---------------------------------------------------------
# Press manager (short/long/double)
# ---------------------------------------------------------

@dataclass
class ButtonPressState:
    is_down: bool = False
    last_down: float = 0.0
    pending_click: bool = False
    pending_click_time: float = 0.0


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

def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_runtime_mapping(input_cfg: Dict[str, Any],
                          state_cfg: Dict[str, Any]) -> MappingRuntime:
    """
    Combine input_mappings + state_mappings into a mapping from
    raw MIDI events -> ActionSpec, plus LED address information.
    """
    runtime = MappingRuntime()

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
        if control_name == "global_slider" and "global_slider" in input_global_cfg:
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

        # Global autopilot: both global and local must be True for
        # autopilot to be effectively "on" for a layer.
        self.global_autopilot: bool = True

        self.global_intensity_scale = 1.0

        # Simple global "queue level" placeholder (0/1)
        self.global_queue_level: int = 0

        # Global tempo from tap-tempo
        self.global_tempo_bpm: float | None = None
        self.tap_times: list[float] = []

        # Blink engine
        self.fast_blink_on = True
        self.slow_blink_on = True
        self.fast_period = 0.25  # FAST blink
        self.slow_period = 0.75  # SLOW blink
        now = time.monotonic()
        self.last_fast_toggle = now
        self.last_slow_toggle = now

        # Initialize intensity LEDs to reflect default intensity
        self._init_intensity_preset_leds()

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
                vel = RED_VELOCITY
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
        Tap-tempo: compute BPM from tap intervals.

        - If gap > 2s since last tap, reset sequence.
        - Use average of last few intervals for BPM.
        """
        now = time.monotonic()
        if self.tap_times and (now - self.tap_times[-1] > 2.0):
            self.tap_times.clear()

        self.tap_times.append(now)

        if len(self.tap_times) >= 2:
            intervals = [
                self.tap_times[i] - self.tap_times[i - 1]
                for i in range(1, len(self.tap_times))
            ]
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval > 0:
                bpm = 60.0 / avg_interval
                self.global_tempo_bpm = bpm
                print(f"[STATE] Global tempo from tap -> {bpm:.2f} BPM "
                      f"(taps={len(self.tap_times)})")
        else:
            print("[STATE] Tap tempo: first tap")


    def global_tempo_sync(self):
        """
        Placeholder for tempo-sync action (e.g., sync to external clock/Resolume).
        """
        print("[STATE] Global tempo sync requested")

    # ---- LED feedback & blinking for boolean layers ----

    def _velocity_for_prop(self, prop: str) -> int:
        """
        Choose base LED velocity (color) for a property when "on".
        """
        if prop == "fft_mask":
            return ORANGE_VELOCITY
        # other layer toggles use full-bright
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
            velocity = self._velocity_for_prop(prop)
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


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

def main():
    input_cfg = load_yaml(INPUT_MAPPING_FILE)
    state_cfg = load_yaml(STATE_MAPPING_FILE)

    controller_name = input_cfg.get("controller_name", "APC40")
    runtime = build_runtime_mapping(input_cfg, state_cfg)

    in_name = auto_select_port(mido.get_input_names(), controller_name, "input")
    out_name = auto_select_port(mido.get_output_names(), controller_name, "output")

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


if __name__ == "__main__":
    main()
