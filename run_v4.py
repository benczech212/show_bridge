#!/usr/bin/env python3
"""
show_bridge v4 – APC40 MK2 runner.

Reuses the core state machine (Apc40StateMachine) from run_v3.py and overrides
only the LED layer to use the MK2's 128-colour RGB grid + hardware pulse.

Grid layout (5 rows × 8 columns):
  Row 0 (notes  0– 7) → colors      (dynamic colour from active clip name)
  Row 1 (notes  8–15) → effects     (orange)
  Row 2 (notes 16–23) → transforms  (cyan)
  Row 3 (notes 24–31) → masks       (purple)
  Row 4 (notes 32–39) → playing     (green)

LED semantics per button:
  enabled=True,  autopilot=False → solid colour  (MIDI ch 0)
  enabled=True,  autopilot=True  → pulse 1/8     (MIDI ch 8)  ← autopilot indicator
  enabled=False                  → off (velocity 0)

Gesture semantics (same for every role button, grid and strip):
  tap    when disabled  → enable
  tap    when enabled   → next clip
  hold   when enabled   → disable
  double                → toggle autopilot
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import mido
import requests
import yaml

# ---------------------------------------------------------------------------
# Reuse stable core from run_v3
# ---------------------------------------------------------------------------
from run_v3 import (
    ClipInfo, LayerInfo, GroupInfo, CompositionInfo, CompositionMapping,
    OutputGroupState, LayerRuntimeState,
    fetch_composition_json, build_composition_model, _guess_composition_name,
    debug_dump_composition_columns,
    load_connections, get_resolume_http_connection, get_resolume_osc_connection,
    get_show_bridge_osc_connection, get_synesthesia_osc_connection, get_osc_client,
    PressManager, MidiKey, auto_select_port,
    Apc40StateMachine,
    MappingRuntime, GroupControlAddresses,
    DEFAULT_CONNECTIONS_PATH, DEFAULT_COMPOSITION_MAPPING_DIR,
    setup_logging, sb_log,
    start_transport_osc_listener,
    make_resolume_base_url, _name_from_field,
    _iter_layergroups, _iter_layers, _iter_clips,
)

from lib.mk2_leds import (
    color_name_to_velocity,
    dim_velocity,
    grid_note,
    ANIM_SOLID, AUTOPILOT_ANIM,
    RAINBOW_VELOCITIES,
    make_sysex_init,
)

# ---------------------------------------------------------------------------
# Property ↔ physical control tables
# ---------------------------------------------------------------------------

# Grid row (0=top) → state property name
ROW_TO_PROP = ["playing", "effects", "masks", "transforms", "color"]

# State property name → grid row
PROP_TO_ROW: dict[str, int] = {p: r for r, p in enumerate(ROW_TO_PROP)}

# State property name → autopilot attribute on OutputGroupState
PROP_AUTOPILOT: dict[str, str] = {
    "color":         "color_autopilot",
    "effects":       "effects_autopilot",
    "transforms":    "transforms_autopilot",
    "masks": "masks_autopilot",
    "playing":       "playing_autopilot",
}

# State property name → fixed LED velocity (colours = None → dynamic)
PROP_ON_VEL: dict[str, Optional[int]] = {
    "color":         None,   # derived from active clip name at runtime
    "transforms":    49,     # violet       #5400FF  hue ~260°
    "masks":         37,     # sky blue     #00A9FF  hue ~200°
    "effects":       29,     # spring green #00FF55  hue ~145°
    "playing":       17,     # chartreuse   #54FF00  hue ~100°
}

# State property name → strip note number
PROP_STRIP_NOTE: dict[str, int] = {
    "color":         0x30,   # 48  ARM
    "effects":       0x31,   # 49  SOLO
    "transforms":    0x32,   # 50  ACTIVATOR
    "masks": 0x33,   # 51  TRACK SELECT
    "playing":       0x34,   # 52  CLIP STOP
}

# Strip note → property name (reverse of above)
STRIP_NOTE_TO_PROP: dict[int, str] = {v: k for k, v in PROP_STRIP_NOTE.items()}

# Output group LED velocities (groups whose name contains "Output -")
OUTPUT_INACTIVE_VEL = 36   # #4CC3FF — light sky cyan  (layer not bypassed but dim)
OUTPUT_ACTIVE_VEL   = 45   # #0000FF — solid blue      (layer active/connected)

# Number of beat4 pulses between autopilot cycles (tune here)
# Autopilot cycle interval in beat4 pulses.
# Scales with intensity: at intensity 1.0 → MIN beats; at 0.0 → MAX beats.
AUTOPILOT_BEAT_MIN =  8   # fastest (high intensity)
AUTOPILOT_BEAT_MAX = 128  # slowest  (low intensity)

# Inactive LED velocities (neither role color nor fully off)
# vel 1  = (30, 30, 30)  — barely visible; role mapped but not enabled
# vel 117 = (64, 64, 64) — dim but distinct; enabled but no clip playing (bypass/none)
LED_IDLE_VEL   = 1    # has role, not enabled
LED_BYPASS_VEL = 117  # enabled, no active clip

# Clip view scroll: fraction of [0,1] range advanced per encoder click
SCROLL_STEP = 0.01

# Composition role name → state property name
ROLE_TO_PROP: dict[str, str] = {
    "fills":         "playing",
    "effects":       "effects",
    "transforms":    "transforms",
    "masks": "masks",
    "colors":        "color",
}

# Global note → action label
GLOBAL_NOTE_ACTION: dict[int, str] = {
    0x5B: "autopilot_play",
    0x62: "tempo_sync",
    0x63: "tap_tempo",
    0x64: "nudge_minus",
    0x65: "nudge_plus",
    0x51: "stop_all",
    0x52: "scene_0",
    0x53: "scene_1",
    0x54: "scene_2",
    0x55: "scene_3",
    0x56: "scene_4",
    0x5E: "nav_up",
    0x5F: "nav_down",
    0x60: "nav_right",
    0x61: "nav_left",
}


# ---------------------------------------------------------------------------
# MIDI clock thread — keeps MK2 LED pulsing synced to BPM
# ---------------------------------------------------------------------------

class MidiClockThread(threading.Thread):
    """Sends MIDI clock (0xF8) at the current BPM so the MK2's pulsing animations
    stay tempo-locked.  Call on_beat() each quarter-note to track live BPM."""

    def __init__(self, midi_out: mido.ports.BaseOutput):
        super().__init__(daemon=True, name="midi-clock")
        self._midi_out  = midi_out
        self._bpm       = 120.0
        self._lock      = threading.Lock()
        self._running   = True
        self._last_beat: Optional[float] = None

    def on_beat(self) -> None:
        """Call on every quarter-note beat to auto-detect BPM."""
        now = time.monotonic()
        with self._lock:
            if self._last_beat is not None:
                interval = now - self._last_beat
                if 0.2 < interval < 3.0:          # sanity: 20–300 BPM
                    self._bpm = 60.0 / interval
            self._last_beat = now

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        clock = mido.Message("clock")
        start = mido.Message("start")
        try:
            self._midi_out.send(start)
        except Exception:
            pass

        next_tick = time.monotonic()
        while self._running:
            now = time.monotonic()
            if now >= next_tick:
                try:
                    self._midi_out.send(clock)
                except Exception:
                    pass
                with self._lock:
                    interval = 60.0 / (self._bpm * 24)   # 24 clocks per quarter note
                next_tick += interval
                if next_tick < now - interval:
                    next_tick = now          # re-anchor if we drifted badly
            else:
                time.sleep(min(next_tick - now, 0.01))


# ---------------------------------------------------------------------------
# Mk2StateMachine
# ---------------------------------------------------------------------------

class Mk2StateMachine(Apc40StateMachine):
    """
    Extends Apc40StateMachine for the APC40 MK2.

    All state-machine logic, OSC control, autopilot, composition model, and
    scene snapshots are inherited.  Only the LED output layer is replaced.
    """

    def __init__(self, midi_out: mido.ports.BaseOutput, **kwargs):
        # Provide a minimal MappingRuntime so the base __init__ is satisfied.
        # All LED methods are overridden, so prop_notes in group_addrs is unused.
        dummy_rt = MappingRuntime()
        for gi in range(8):
            dummy_rt.group_addrs[gi] = GroupControlAddresses(
                channel=gi,
                prop_notes={p: -1 for p in Apc40StateMachine.BOOL_PROPS},
                reset_note=-1,
                slider_cc=0x07,
                intensity_preset_notes=[],
            )
        for si in range(5):
            dummy_rt.scene_buttons[si] = (0, 0x52 + si)

        super().__init__(runtime=dummy_rt, midi_out=midi_out, **kwargs)

        # Guard against base-class attributes that are only set in attach_composition.
        if not hasattr(self, "synesthesia_group_idx"):
            self.synesthesia_group_idx = None

        # Default intensity to 1.0 so fills/color/effects autoplay picks
        # content clips instead of passthrough when first enabled.
        for g in self.groups:
            g.intensity = 1.0

        # Per-group colour velocity for the colours-role button (row 2).
        self.group_color_velocity: list[int] = [5] * 8   # default: red

        # Set by main() after the MIDI clock thread starts.
        self._clock_thread: Optional[MidiClockThread] = None

        # Start at the slowest interval; first beat updates it from intensity.
        self.autopilot_beats_remaining = AUTOPILOT_BEAT_MAX

        # True only during super().on_transport_beat() so autoplay overrides
        # can distinguish autopilot-triggered clip changes from manual presses.
        self._autopilot_running: bool = False

        # Tracks global autopilot state so the PLAY LED can mirror it.
        self._global_autopilot_on: bool = False

        # Beat counter for one-shot autopilot pulse indicator (pulse once per bar).
        self._beat_count: int = 0
        self._autopilot_flash_active: bool = False

        # Groups whose active "colors" clip is named "rainbow" — LED is driven
        # by the cycling thread instead of the normal static velocity.
        self._rainbow_groups: set[int] = set()
        # Per-group step index into RAINBOW_VELOCITIES; staggered so simultaneous
        # rainbow groups show different hues.
        self._rainbow_steps: list[int] = list(range(8))

        # Clip view horizontal scroll position [0.0, 1.0] — driven by CC 0x2F encoder.
        self._clip_scroll_h: float = 0.0

        # Fills-layer direction: 0=reverse, 2=forward (1=stopped, unused)
        self._fills_direction: int = 2
        # Per-group solo/mute state (tracked locally; mirrored to Resolume via OSC)
        self._solo_groups: set[int] = set()
        self._mute_groups: set[int] = set()
        # Groups with no composition mapping — permanently muted indicator, not togglable
        self._unmapped_groups: set[int] = set()

        # Output groups: col_idx (0-based, filled from right) → ordered list of LayerInfo.
        # Keyed by APC column (7=rightmost for first output group, 6 for second, …).
        self._output_group_cols: dict[int, list[LayerInfo]] = {}
        # Per-layer active state for output groups (True = not bypassed).
        self._output_layer_active: dict[int, bool] = {}
        # Display name for each output group column.
        self._output_group_names: dict[int, str] = {}


    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def send_sysex_init(self):
        """Put MK2 in Alternate Ableton Live mode so host controls all LEDs."""
        data = make_sysex_init(mode=0x42)
        self.midi_out.send(mido.Message("sysex", data=data))
        self.logger.info("[MK2] SysEx init sent (mode 0x42 – full host LED control)")

    def clear_all_grid_leds(self) -> None:
        """Blank all 40 clip launch buttons before the first state push."""
        for note in range(40):
            self.midi_out.send(mido.Message("note_on", channel=ANIM_SOLID,
                                            note=note, velocity=0))

    def clear_strip_leds(self) -> None:
        """Blank all strip button LEDs (ARM/SOLO/ACTIVATOR/TRACK_SELECT/CLIP_STOP
        on each of the 8 group channels) plus the direction buttons on ch=0."""
        for ch in range(8):
            for note in (0x30, 0x31, 0x32, 0x33, 0x34):
                self.midi_out.send(mido.Message("note_on", channel=ch,
                                                note=note, velocity=0))
        for note in (0x3C, 0x3D):
            self.midi_out.send(mido.Message("note_on", channel=0, note=note, velocity=0))

    def _recompute_unmapped_groups(self) -> None:
        """Populate _unmapped_groups with indices of groups that have no layer mapping."""
        self._unmapped_groups.clear()
        if not self.group_role_layers:
            return
        for gi in range(len(self.groups)):
            roles = self.group_role_layers.get(gi + 1, {})
            if not any(layers for layers in roles.values()):
                self._unmapped_groups.add(gi)

    def _update_unmapped_group_indicators(self) -> None:
        """For unmapped groups: kill the knob ring and lock the mute LED on.
        Output group columns are skipped — they have their own LED scheme."""
        for gi in range(len(self.groups)):
            if gi in self._unmapped_groups and gi not in self._output_group_cols:
                self.midi_out.send(mido.Message("control_change", channel=0,
                                                control=0x30 + gi, value=0))
                self.midi_out.send(mido.Message("note_on", channel=gi,
                                                note=0x30, velocity=1))

    def _detect_output_groups(self) -> None:
        """Scan composition for groups named 'Output - …' and assign them to APC
        columns from the right (col 7 = group 8 = first output group found)."""
        self._output_group_cols.clear()
        self._output_layer_active.clear()
        self._output_group_names.clear()
        if not self.composition:
            return
        output_groups = [
            g for g in self.composition.groups
            if "output -" in g.name.lower()
        ]
        for i, grp in enumerate(output_groups):
            col = 7 - i
            if col < 0:
                break
            self._output_group_cols[col] = list(grp.layers)
            self._output_group_names[col] = grp.name
            for layer in grp.layers:
                # Init from Resolume's actual connected state, not assumed True
                self._output_layer_active[layer.global_index] = any(
                    c.connected for c in layer.clips
                )

    def _update_output_group_leds(self, col_idx: int) -> None:
        """Render all 5 row LEDs for one output-group column."""
        layers = self._output_group_cols.get(col_idx, [])
        for row in range(5):
            note = grid_note(row, col_idx)
            if row < len(layers):
                active = self._output_layer_active.get(layers[row].global_index, True)
                vel = OUTPUT_ACTIVE_VEL if active else OUTPUT_INACTIVE_VEL
            else:
                vel = 0
            self.midi_out.send(mido.Message("note_on", channel=ANIM_SOLID,
                                            note=note, velocity=vel))

    def _prop_has_active_clip(self, group_idx: int, prop: str) -> bool:
        """True if at least one layer for this prop has a clip currently connected.

        Returns True whenever state is uncertain — bypass gray is only shown when
        we have confirmed layer data that shows no clip is connected.
        """
        if not self.group_role_layers:
            return True   # no composition loaded — assume active
        apc_idx  = group_idx + 1
        role_key = next((rk for rk, p in ROLE_TO_PROP.items() if p == prop), None)
        if not role_key:
            return True
        layers = self.group_role_layers.get(apc_idx, {}).get(role_key, [])
        if not layers:
            return True   # no layers mapped — can't determine, assume active
        for layer in layers:
            ls = self.layer_states.get(layer.global_index)
            if ls is None:
                return True   # no state data yet — assume active (avoids false bypass flash)
            if ls.current_clip_index is not None:
                return True
        return False

    def init_track_knob_rings(self) -> None:
        """Set all 8 track knob LED rings to Volume Style and sync to current intensity."""
        for gi in range(8):
            # CC 0x38–0x3F: ring type — 2 = Volume Style (fills L→R like a level meter)
            self.midi_out.send(mido.Message("control_change", channel=0,
                                            control=0x38 + gi, value=2))
        for gi in range(8):
            # CC 0x30–0x37: ring position — mirror current intensity (1.0 → 127)
            cc_val = int(self.groups[gi].intensity * 127)
            self.midi_out.send(mido.Message("control_change", channel=0,
                                            control=0x30 + gi, value=cc_val))

    def _autopilot_beat_interval(self) -> int:
        """Return the next beat interval scaled by max group intensity."""
        max_int = max((g.intensity for g in self.groups), default=1.0)
        effective = max(0.0, min(1.0, max_int * self.global_intensity_scale))
        span = AUTOPILOT_BEAT_MAX - AUTOPILOT_BEAT_MIN
        return max(1, int(AUTOPILOT_BEAT_MAX - span * effective))

    def _send_play_led(self, on: bool) -> None:
        vel = 1 if on else 0
        self.midi_out.send(mido.Message("note_on", channel=0, note=0x5B, velocity=vel))

    # ------------------------------------------------------------------
    # Direction (Forward / Reverse) LEDs and action
    # ------------------------------------------------------------------

    def _send_direction_leds(self) -> None:
        fwd_vel = 1 if self._fills_direction == 2 else 0
        rev_vel = 1 if self._fills_direction == 0 else 0
        self.midi_out.send(mido.Message("note_on", channel=0, note=0x3D, velocity=fwd_vel))
        self.midi_out.send(mido.Message("note_on", channel=0, note=0x3C, velocity=rev_vel))

    def set_fills_direction(self, direction: int) -> None:
        """Send direction to all fills layers. 0=reverse, 2=forward."""
        self._fills_direction = direction
        if self.osc_client and self.group_role_layers:
            for _, roles in self.group_role_layers.items():
                for layer in roles.get("fills", []):
                    self.osc_client.send_message(
                        f"/composition/layers/{layer.global_index + 1}/direction", direction)
        self._send_direction_leds()

    # ------------------------------------------------------------------
    # Solo / Mute per group
    # ------------------------------------------------------------------

    def _send_solo_led(self, group_idx: int) -> None:
        vel = 1 if group_idx in self._solo_groups else 0
        self.midi_out.send(mido.Message("note_on", channel=group_idx, note=0x31, velocity=vel))

    def _send_mute_led(self, group_idx: int) -> None:
        vel = 1 if group_idx in self._mute_groups else 0
        self.midi_out.send(mido.Message("note_on", channel=group_idx, note=0x30, velocity=vel))

    def _send_activator_led(self, group_idx: int) -> None:
        g = self.groups[group_idx]
        any_autopilot = any(
            getattr(g, ap_attr, False)
            for prop, ap_attr in PROP_AUTOPILOT.items()
            if self._group_has_role(group_idx, prop)
        )
        vel = 1 if any_autopilot else 0
        self.midi_out.send(mido.Message("note_on", channel=group_idx, note=0x32, velocity=vel))

    def _trigger_autopilot_pulse(self) -> None:
        """Flash all autopilot LEDs for one pulse cycle (~0.45 s), then return to solid."""
        if self._autopilot_flash_active:
            return
        self._autopilot_flash_active = True
        self._update_all_leds()

        def _end():
            self._autopilot_flash_active = False
            self._update_all_leds()

        threading.Timer(0.45, _end).start()

    def toggle_group_autopilot(self, group_idx: int) -> None:
        """Toggle all autopilots for a group: if any are on → all off; else all on."""
        if group_idx in self._unmapped_groups:
            return
        g = self.groups[group_idx]
        any_on = any(
            getattr(g, ap_attr, False)
            for prop, ap_attr in PROP_AUTOPILOT.items()
            if self._group_has_role(group_idx, prop)
        )
        new_state = not any_on
        for prop, ap_attr in PROP_AUTOPILOT.items():
            if self._group_has_role(group_idx, prop):
                setattr(g, ap_attr, new_state)
                self._set_resolume_autopilot_for_role(group_idx, prop, new_state)
                if getattr(g, prop, False):
                    self._update_led_for_property(group_idx, prop)
        self._send_activator_led(group_idx)

    def toggle_group_solo(self, group_idx: int) -> None:
        if group_idx in self._unmapped_groups:
            return
        active = group_idx not in self._solo_groups
        if active:
            self._solo_groups.add(group_idx)
        else:
            self._solo_groups.discard(group_idx)
        if self.osc_client and self.composition:
            grp = self.composition.group_for_apc(group_idx + 1)
            if grp is not None:
                self.osc_client.send_message(
                    f"/composition/groups/{grp.index_in_composition + 1}/solo",
                    1 if active else 0)
        self._send_solo_led(group_idx)

    def toggle_group_mute(self, group_idx: int) -> None:
        if group_idx in self._unmapped_groups:
            return
        active = group_idx not in self._mute_groups
        if active:
            self._mute_groups.add(group_idx)
        else:
            self._mute_groups.discard(group_idx)
        if self.osc_client and self.composition:
            grp = self.composition.group_for_apc(group_idx + 1)
            if grp is not None:
                self.osc_client.send_message(
                    f"/composition/groups/{grp.index_in_composition + 1}/bypassed",
                    1 if active else 0)
        self._send_mute_led(group_idx)

    # ------------------------------------------------------------------
    # Resolume autopilot delegation
    # ------------------------------------------------------------------

    def _set_resolume_autopilot_for_role(
            self, group_idx: int, prop: str, enable: bool) -> None:
        """Push autopilot target (3=shuffle / 1=off) to Resolume for all layers of this role."""
        if not self.osc_client or not self.group_role_layers:
            return
        role_key = next((rk for rk, p in ROLE_TO_PROP.items() if p == prop), None)
        if not role_key:
            return
        target = 3 if enable else 0
        for layer in self.group_role_layers.get(group_idx + 1, {}).get(role_key, []):
            self.osc_client.send_message(
                f"/composition/layers/{layer.global_index + 1}/autopilot/target", target)

    def handle_property_press(self, group_idx: int, prop: str, press_type: str) -> None:
        """Intercept double-tap to sync Resolume autopilot after base class toggles the flag."""
        super().handle_property_press(group_idx, prop, press_type)
        if press_type == "double":
            g = self.groups[group_idx]
            ap_attr = PROP_AUTOPILOT.get(prop, "")
            if ap_attr:
                self._set_resolume_autopilot_for_role(
                    group_idx, prop, getattr(g, ap_attr, False))

    def set_global_autopilot_on(self) -> None:
        super().set_global_autopilot_on()
        self._global_autopilot_on = True
        self._send_play_led(True)
        # Enable Resolume autopilot for every role that already has it toggled on
        if self.group_role_layers:
            for gi in range(len(self.groups)):
                g = self.groups[gi]
                for prop, ap_attr in PROP_AUTOPILOT.items():
                    if getattr(g, ap_attr, False) and self._group_has_role(gi, prop):
                        self._set_resolume_autopilot_for_role(gi, prop, True)

    def set_global_autopilot_off(self) -> None:
        super().set_global_autopilot_off()
        self._global_autopilot_on = False
        self._send_play_led(False)
        # Disable Resolume autopilot for all layers across all groups
        if self.osc_client and self.group_role_layers:
            for _, roles in self.group_role_layers.items():
                for role_layers in roles.values():
                    for layer in role_layers:
                        self.osc_client.send_message(
                            f"/composition/layers/{layer.global_index + 1}/autopilot/target", 0)

    def advance_all_autopilots(self) -> None:
        pass  # Resolume autopilot drives clip advancement; show_bridge no longer advances clips

    def enable_all_group_props(self, group_idx: int) -> None:
        """Turn on every mapped role for a group (CLIP STOP tap)."""
        g = self.groups[group_idx]
        for prop in self.BOOL_PROPS:
            if self._group_has_role(group_idx, prop) and not getattr(g, prop, False):
                self.handle_property_press(group_idx, prop, "short")

    def disable_all_group_props(self, group_idx: int) -> None:
        """Turn off every active role for a group (CLIP STOP hold)."""
        g = self.groups[group_idx]
        for prop in self.BOOL_PROPS:
            if self._group_has_role(group_idx, prop) and getattr(g, prop, False):
                self.handle_property_press(group_idx, prop, "long")

    def stop_all_clips(self) -> None:
        """Disconnect every layer across all roles (not just fills)."""
        if self.osc_client and self.group_role_layers:
            for roles in self.group_role_layers.values():
                for role_layers in roles.values():
                    for layer in role_layers:
                        self.osc_client.send_message(
                            f"/composition/layers/{layer.global_index + 1}/clips/1/connect", 1.0)
        for gi in range(len(self.groups)):
            g = self.groups[gi]
            for prop in self.BOOL_PROPS:
                if getattr(g, prop, False):
                    setattr(g, prop, False)
                    self._update_led_for_property(gi, prop)

    def next_all_group_props(self, group_idx: int) -> None:
        """Advance all mapped roles to the next clip (CLIP STOP double-tap)."""
        for prop in self.BOOL_PROPS:
            if self._group_has_role(group_idx, prop):
                self.next_property(group_idx, prop)

    def refresh_group_colors(self):
        """Re-derive every group's colour velocity from the composition model."""
        if not self.composition:
            return
        for gi in range(8):
            self._refresh_one_group_color(gi)

    # ------------------------------------------------------------------
    # BPM tracking — forward beats to the MIDI clock thread
    # ------------------------------------------------------------------

    def on_transport_beat(self, beat_kind: str) -> None:
        if beat_kind == "beat4":
            if self._clock_thread is not None:
                self._clock_thread.on_beat()
            remaining = getattr(self, "autopilot_beats_remaining", "?")
            self.logger.debug(f"[BEAT] beat4 received — autopilot_beats_remaining={remaining}")
        self._beat_count += 1
        if self._beat_count % 4 == 0:
            self._trigger_autopilot_pulse()
        self._autopilot_running = True
        try:
            super().on_transport_beat(beat_kind)
        finally:
            self._autopilot_running = False
        # Base class resets the counter to its hardcoded 64 after each cycle.
        # Override with an intensity-scaled interval for the next cycle.
        if beat_kind == "beat4" and self.autopilot_beats_remaining == 64:
            self.autopilot_beats_remaining = self._autopilot_beat_interval()

    # ------------------------------------------------------------------
    # Color polling — HTTP sync from Resolume every ~2 s
    # ------------------------------------------------------------------

    def start_synestesia_beat_listener(self, port: int = 12000) -> None:
        """
        Listen for Synestesia OSC on *port* and drive autopilot + MIDI clock from it.

        /audio/bpm/bpm    → sets clock thread BPM directly
        /audio/beat/onbeat → rising-edge (≤0 → >0) fires on_transport_beat("beat4")

        A watchdog resets the clock to 240 BPM after 5 s with no BPM message.
        """
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer

        self._syn_last_bpm_ts: float = 0.0
        self._syn_last_onbeat: float = -1.0
        self._syn_last_beat_ts: float = 0.0   # debounce: prevents double-triggers

        dispatcher = Dispatcher()

        def _handle_bpm(address: str, *args):
            if not args:
                return
            try:
                bpm = float(args[0])
            except (TypeError, ValueError):
                return
            if 20.0 < bpm < 400.0 and self._clock_thread is not None:
                with self._clock_thread._lock:
                    self._clock_thread._bpm = bpm
                self._syn_last_bpm_ts = time.monotonic()

        def _handle_onbeat(address: str, *args):
            if not args:
                return
            try:
                val = float(args[0])
            except (TypeError, ValueError):
                return
            now = time.monotonic()
            # Rising edge + 200 ms debounce (guards against noisy zero-crossings;
            # 200 ms = minimum inter-beat interval at 300 BPM).
            if (self._syn_last_onbeat <= 0.0 < val
                    and now - self._syn_last_beat_ts > 0.2):
                self._syn_last_beat_ts = now
                self.on_transport_beat("beat4")
            self._syn_last_onbeat = val

        dispatcher.map("/audio/bpm/bpm", _handle_bpm)
        dispatcher.map("/audio/beat/onbeat", _handle_onbeat)

        def _bpm_watchdog():
            FALLBACK_BPM = 120.0
            TIMEOUT = 5.0
            while True:
                time.sleep(2.0)
                if (time.monotonic() - self._syn_last_bpm_ts > TIMEOUT
                        and self._clock_thread is not None):
                    with self._clock_thread._lock:
                        self._clock_thread._bpm = FALLBACK_BPM

        threading.Thread(target=_bpm_watchdog, daemon=True, name="bpm-watchdog").start()

        try:
            server = ThreadingOSCUDPServer(("0.0.0.0", port), dispatcher)
            threading.Thread(target=server.serve_forever, daemon=True,
                             name="syn-osc").start()
            self.logger.info(f"[MK2] Synestesia beat listener on 0.0.0.0:{port}")
        except OSError as e:
            self.logger.warning(f"[MK2] Could not bind Synestesia listener on port {port}: {e}")

    def start_color_polling(self) -> None:
        """Background thread: query each group's Color layer via HTTP and
        update the corresponding grid LED when the connected clip changes."""
        def _poll():
            while True:
                time.sleep(2.0)
                if not self.resolume_conn or not self.group_role_layers:
                    continue
                try:
                    base_url = make_resolume_base_url(self.resolume_conn)
                    for apc_idx in range(1, 9):
                        gi = apc_idx - 1
                        layers = self.group_role_layers.get(apc_idx, {}).get("colors", [])
                        if not layers:
                            continue
                        layer = layers[0]
                        url = f"{base_url}/composition/layers/{layer.global_index + 1}"
                        try:
                            resp = requests.get(url, timeout=0.5)
                            connected_found = False
                            for clip in resp.json().get("clips", []):
                                conn = clip.get("connected")
                                if not isinstance(conn, dict):
                                    continue
                                if str(conn.get("value", "")).lower() != "connected":
                                    continue
                                connected_found = True
                                name_raw = clip.get("name", "")
                                name = (_name_from_field(name_raw, "")
                                        if isinstance(name_raw, dict) else str(name_raw))
                                if name.strip().lower() == "rainbow":
                                    self._rainbow_groups.add(gi)
                                else:
                                    self._rainbow_groups.discard(gi)
                                    vel = color_name_to_velocity(name)
                                    if vel is not None and vel != self.group_color_velocity[gi]:
                                        self.group_color_velocity[gi] = vel
                                        self._update_led_for_property(gi, "color")
                                break
                            if not connected_found:
                                self._rainbow_groups.discard(gi)
                        except Exception:
                            pass
                except Exception:
                    pass

        t = threading.Thread(target=_poll, daemon=True, name="color-poll")
        t.start()

    def _start_rainbow_cycling(self) -> None:
        """Background thread: advance rainbow hue every 150 ms for active rainbow groups."""
        def _run():
            while True:
                time.sleep(0.15)
                for gi in list(self._rainbow_groups):
                    if gi >= len(self.groups):
                        continue
                    g = self.groups[gi]
                    if not getattr(g, "color", False):
                        continue   # role disabled — _update_led_for_property owns the off state
                    if not self._group_has_role(gi, "color"):
                        continue
                    step = self._rainbow_steps[gi]
                    vel  = RAINBOW_VELOCITIES[step % len(RAINBOW_VELOCITIES)]
                    autopilot = getattr(g, PROP_AUTOPILOT.get("color", ""), False)
                    note = grid_note(PROP_TO_ROW["color"], gi)
                    try:
                        if autopilot:
                            self.midi_out.send(mido.Message("note_on", channel=ANIM_SOLID,
                                                            note=note, velocity=dim_velocity(vel)))
                            self.midi_out.send(mido.Message("note_on", channel=AUTOPILOT_ANIM,
                                                            note=note, velocity=vel))
                        else:
                            self.midi_out.send(mido.Message("note_on", channel=ANIM_SOLID,
                                                            note=note, velocity=vel))
                    except Exception:
                        pass
                    self._rainbow_steps[gi] = (step + 1) % len(RAINBOW_VELOCITIES)

        t = threading.Thread(target=_run, daemon=True, name="rainbow-cycle")
        t.start()

    # ------------------------------------------------------------------
    # LED helpers
    # ------------------------------------------------------------------

    def _on_vel_for(self, prop: str, group_idx: int) -> int:
        if prop == "color":
            return self.group_color_velocity[group_idx]
        return PROP_ON_VEL[prop] or 21

    def _send_grid_led(self, row: int, group_idx: int, enabled: bool, autopilot: bool,
                       on_vel: int):
        note = grid_note(row, group_idx)
        if not enabled:
            ch, vel = ANIM_SOLID, LED_IDLE_VEL    # dim gray — role exists but inactive
        elif autopilot and self._autopilot_flash_active:
            # One-shot flash window: prime dim floor then send pulse animation.
            self.midi_out.send(mido.Message("note_on", channel=ANIM_SOLID, note=note,
                                            velocity=dim_velocity(on_vel)))
            ch, vel = AUTOPILOT_ANIM, on_vel
        else:
            ch, vel = ANIM_SOLID, on_vel
        self.midi_out.send(mido.Message("note_on", channel=ch, note=note, velocity=vel))

    def _send_strip_led(self, group_idx: int, prop: str, enabled: bool):
        note = PROP_STRIP_NOTE[prop]
        vel  = 1 if enabled else 0
        self.midi_out.send(mido.Message("note_on", channel=group_idx, note=note, velocity=vel))

    def _send_scene_led(self, scene_idx: int, has_snapshot: bool, is_active: bool):
        note = 0x52 + scene_idx
        if not has_snapshot:
            vel = 0
        elif is_active:
            vel = 21      # green  – active scene
        else:
            vel = 9       # orange – stored but not active
        self.midi_out.send(mido.Message("note_on", channel=0, note=note, velocity=vel))

    # ------------------------------------------------------------------
    # Override base-class LED methods
    # ------------------------------------------------------------------

    def _group_has_role(self, group_idx: int, prop: str) -> bool:
        """True if the loaded composition provides at least one layer for prop in this group."""
        if not self.group_role_layers:
            return True   # no composition yet — don't suppress anything
        apc_idx = group_idx + 1
        for role_key, p in ROLE_TO_PROP.items():
            if p == prop:
                return bool(self.group_role_layers.get(apc_idx, {}).get(role_key))
        return False

    def _update_led_for_property(self, group_idx: int, prop: str):
        if group_idx in self._output_group_cols:
            return  # output groups have their own LED path
        if prop not in PROP_TO_ROW:
            return
        row = PROP_TO_ROW[prop]
        if not self._group_has_role(group_idx, prop):
            # Blank the grid cell only (strip buttons are unmapped).
            self.midi_out.send(mido.Message("note_on", channel=ANIM_SOLID,
                                            note=grid_note(row, group_idx), velocity=0))
            return
        g         = self.groups[group_idx]
        enabled   = getattr(g, prop, False)
        autopilot = getattr(g, PROP_AUTOPILOT.get(prop, ""), False)

        # Sync color velocity from composition state before reading it so we
        # never flash the stale default (red) when the button is first pressed.
        if prop == "color" and enabled:
            self._sync_color_velocity(group_idx)

        on_vel    = self._on_vel_for(prop, group_idx)

        # Rainbow mode: cycling thread owns the grid LED while the role is enabled.
        # When disabled, show idle gray (role is valid, just inactive).
        if prop == "color" and group_idx in self._rainbow_groups:
            if not enabled:
                self.midi_out.send(mido.Message("note_on", channel=ANIM_SOLID,
                                                note=grid_note(row, group_idx),
                                                velocity=LED_IDLE_VEL))
            return

        self._send_grid_led(row, group_idx, enabled, autopilot, on_vel)

    def _update_all_leds(self):
        for gi in range(len(self.groups)):
            for prop in self.BOOL_PROPS:
                self._update_led_for_property(gi, prop)
            if gi not in self._unmapped_groups and gi not in self._output_group_cols:
                self._send_activator_led(gi)
        self._update_scene_leds()
        self._send_play_led(self._global_autopilot_on)
        for col_idx in self._output_group_cols:
            self._update_output_group_leds(col_idx)

    def _update_intensity_leds(self, group_idx: int):
        pass   # MK2 has no intensity-preset row

    def _init_intensity_preset_leds(self):
        pass   # nothing to initialise

    def _update_scene_leds(self):
        for si in range(5):
            has      = si in self.scene_snapshots
            is_active = (self.active_scene == si and self.active_scene_pristine)
            self._send_scene_led(si, has, is_active)

    def update_blink(self, now: float):
        # MK2 handles pulsing in hardware.  Software blinking not needed.
        pass

    # ------------------------------------------------------------------
    # Refresh colour after next_property changes the active clip
    # ------------------------------------------------------------------

    def next_property(self, group_idx: int, prop: str):
        """Call base, then refresh colour LED if the colors role changed."""
        super().next_property(group_idx, prop)
        if prop == "color":
            self._refresh_one_group_color(group_idx)

    def _sync_color_velocity(self, group_idx: int) -> None:
        """Update group_color_velocity and _rainbow_groups from composition state.
        Pure state sync — no LED messages sent.  Safe to call before rendering.
        """
        if not self.group_role_layers:
            return
        apc_idx = group_idx + 1
        layers  = self.group_role_layers.get(apc_idx, {}).get("colors", [])
        for layer in layers:
            ls = self.layer_states.get(layer.global_index)
            active_col = ls.current_clip_index if ls else None
            if active_col is not None:
                clip = next((c for c in layer.clips if c.column_index == active_col), None)
                if clip and clip.name.strip().lower() == "rainbow":
                    self._rainbow_groups.add(group_idx)
                    return
        self._rainbow_groups.discard(group_idx)
        vel = _color_velocity_from_layers(layers, self.layer_states)
        if vel is not None:
            self.group_color_velocity[group_idx] = vel
        # If vel is None (no active clip with a recognised colour name), keep
        # the previous value so the LED doesn't flash to a random default.

    def _refresh_one_group_color(self, group_idx: int):
        self._sync_color_velocity(group_idx)
        self._update_led_for_property(group_idx, "color")

    # ------------------------------------------------------------------
    # Live clip-change hook — refresh color LED when Resolume fires
    # ------------------------------------------------------------------

    def on_clip_connect(self, address: str, *args):
        super().on_clip_connect(address, *args)
        m = re.match(r"^/composition/layers/(\d+)/clips/(\d+)/connect$", address)
        if not m:
            return
        global_layer_index = int(m.group(1)) - 1   # OSC is 1-based
        for apc_index, role_layers in self.group_role_layers.items():
            for layer in role_layers.get("colors", []):
                if layer.global_index == global_layer_index:
                    self._refresh_one_group_color(apc_index - 1)
                    return

    # ------------------------------------------------------------------
    # Autopilot flash — brief white pulse when a clip is auto-advanced
    # ------------------------------------------------------------------

    def _flash_button(self, group_idx: int, prop: str) -> None:
        """Flash the grid button white for 80 ms then restore its normal colour."""
        if prop not in PROP_TO_ROW:
            return
        note = grid_note(PROP_TO_ROW[prop], group_idx)
        try:
            self.midi_out.send(mido.Message("note_on", channel=ANIM_SOLID,
                                            note=note, velocity=3))  # 3 = white
        except Exception:
            return

        def _restore():
            time.sleep(0.30)
            self._update_led_for_property(group_idx, prop)

        threading.Thread(target=_restore, daemon=True, name="led-flash").start()

    def _restore_scroll_position(self) -> None:
        """Re-assert the user's last scroll position after Resolume auto-scrolls to a new clip."""
        if self.osc_client:
            self.osc_client.send_message(
                "/application/ui/clipsscrollhorizontal", self._clip_scroll_h)

    def _autoplay_role_single_layer(self, group_idx: int, role: str) -> None:
        if not self._autopilot_running:
            # User-triggered (button press / manual next) — connect a clip via OSC.
            super()._autoplay_role_single_layer(group_idx, role)
            self._restore_scroll_position()
        # If _autopilot_running: beat-driven cycle — Resolume autopilot handles it.

    def _autoplay_fill_layers_for_group(self, group_idx: int) -> None:
        if not self._autopilot_running:
            super()._autoplay_fill_layers_for_group(group_idx)
            self._restore_scroll_position()
            if group_idx == self.synesthesia_group_idx and self.syn_osc_client:
                self.syn_osc_client.send_message("/playlist/next", [])
        # If _autopilot_running: beat-driven cycle — Resolume autopilot handles it.

    def disable_all_group_props(self, group_idx: int) -> None:
        """Turn off every active role for a group (CLIP STOP hold)."""
        g = self.groups[group_idx]
        for prop in self.BOOL_PROPS:
            if self._group_has_role(group_idx, prop) and getattr(g, prop, False):
                self.handle_property_press(group_idx, prop, "long")

    # ------------------------------------------------------------------
    # Resync: refresh colours after composition reload
    # ------------------------------------------------------------------

    def resync_with_resolume(self):
        super().resync_with_resolume()
        self._recompute_unmapped_groups()
        self._detect_output_groups()
        self.refresh_group_colors()
        self._update_all_leds()
        self._update_unmapped_group_indicators()


# ---------------------------------------------------------------------------
# Colour velocity derivation
# ---------------------------------------------------------------------------

def _color_velocity_from_layers(
    layers: list[LayerInfo],
    layer_states: dict[int, LayerRuntimeState],
) -> Optional[int]:
    """Return the MK2 palette velocity for the currently active colour clip, or
    None if no active clip has a recognised colour name.

    We deliberately do NOT fall back to scanning all clips — doing so would
    show whichever clip happened to be first in the layer (e.g. "yellow"),
    overwriting the correct displayed colour.
    """
    for layer in layers:
        ls = layer_states.get(layer.global_index)
        if ls is None:
            continue
        active_col = ls.current_clip_index
        if active_col is None:
            continue
        clip = next((c for c in layer.clips if c.column_index == active_col), None)
        if clip:
            vel = color_name_to_velocity(clip.name)
            if vel is not None:
                return vel
    return None


# ---------------------------------------------------------------------------
# MK2 MIDI input handler
# ---------------------------------------------------------------------------

def _toggle_output_layer(sm: Mk2StateMachine, row: int, col_idx: int) -> None:
    """Toggle an output-group layer on/off via clip connect (same as regular buttons)."""
    layers = sm._output_group_cols.get(col_idx, [])
    if row >= len(layers):
        return
    layer = layers[row]
    active = sm._output_layer_active.get(layer.global_index, True)
    new_active = not active
    sm._output_layer_active[layer.global_index] = new_active
    if sm.osc_client:
        osc_layer = layer.global_index + 1
        if new_active:
            # Connect the named content clip (first clip in model, col 1 in Resolume → column_index=2)
            content_col = layer.clips[0].column_index if layer.clips else 2
            sm.osc_client.send_message(
                f"/composition/layers/{osc_layer}/clips/{content_col}/connect", 1.0)
        else:
            # Fire the empty/off slot — always clip 1
            sm.osc_client.send_message(
                f"/composition/layers/{osc_layer}/clips/1/connect", 1.0)
    sm._update_output_group_leds(col_idx)


def _apply_button_press_feedback(msg: mido.Message, sm: Mk2StateMachine) -> None:
    """Light a mapped button white/bright while physically held; restore its LED on release."""
    note = msg.note
    ch   = msg.channel
    pressing  = msg.type == "note_on" and msg.velocity > 0
    releasing = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)

    # Grid clip-launch buttons (notes 0–39, ch=0) — flash white while held
    if 0 <= note <= 39 and ch == 0:
        row, gi = note // 8, note % 8
        if gi in sm._output_group_cols:
            # Output group: only flash rows that have a layer mapped
            if row < len(sm._output_group_cols[gi]):
                if pressing:
                    sm.midi_out.send(mido.Message("note_on", channel=ANIM_SOLID,
                                                  note=note, velocity=3))  # white
                elif releasing:
                    sm._update_output_group_leds(gi)
        else:
            prop = ROW_TO_PROP[row]
            if sm._group_has_role(gi, prop):
                if pressing:
                    sm.midi_out.send(mido.Message("note_on", channel=ANIM_SOLID,
                                                  note=note, velocity=3))  # white
                elif releasing:
                    sm._update_led_for_property(gi, prop)
        return

    # Per-group strip buttons — vel 1 while held, restore specific LED on release
    if note in (0x30, 0x31, 0x32, 0x33, 0x34) and 0 <= ch < 8:
        if ch not in sm._unmapped_groups:
            if pressing:
                sm.midi_out.send(mido.Message("note_on", channel=ch, note=note, velocity=1))
            elif releasing:
                if note == 0x30:
                    sm._send_mute_led(ch)
                elif note == 0x31:
                    sm._send_solo_led(ch)
                elif note == 0x32:
                    sm._send_activator_led(ch)
                else:  # 0x33 track-select, 0x34 clip-stop — transient, restore off
                    sm.midi_out.send(mido.Message("note_on", channel=ch, note=note, velocity=0))
        return

    # Direction buttons
    if ch == 0 and note in (0x3C, 0x3D):
        if pressing:
            sm.midi_out.send(mido.Message("note_on", channel=0, note=note, velocity=1))
        elif releasing:
            sm._send_direction_leds()
        return

    # Global mapped buttons
    if ch == 0 and note in GLOBAL_NOTE_ACTION:
        if pressing:
            sm.midi_out.send(mido.Message("note_on", channel=0, note=note, velocity=1))
        elif releasing:
            action = GLOBAL_NOTE_ACTION[note]
            if action == "autopilot_play":
                sm._send_play_led(sm._global_autopilot_on)
            elif action.startswith("scene_"):
                sm._update_scene_leds()
            else:
                sm.midi_out.send(mido.Message("note_on", channel=0, note=note, velocity=0))


def handle_mk2_midi(
    msg: mido.Message,
    sm: Mk2StateMachine,
    press_mgr: PressManager,
    now: float,
    debug_midi: bool = False,
):
    """Translate incoming MK2 MIDI into state-machine calls."""
    if msg.type not in ("note_on", "note_off", "control_change"):
        return

    # ── Button press feedback ─────────────────────────────────────────
    if msg.type in ("note_on", "note_off"):
        _apply_button_press_feedback(msg, sm)

    # ── CC (sliders / knobs) ─────────────────────────────────────────
    if msg.type == "control_change":
        if msg.control == 0x07 and 0 <= msg.channel < 8:
            # Track fader → group opacity
            sm.set_opacity_from_cc(msg.channel, msg.value)
        elif msg.control == 0x0E:
            # Master fader → global intensity scale
            sm.set_global_intensity_from_cc(msg.value)
        elif 0x30 <= msg.control <= 0x37 and msg.channel == 0:
            # Top-strip knobs (CC 0x30–0x37, all ch=0) → group intensity
            gi = msg.control - 0x30
            if 0 <= gi < len(sm.groups):
                intensity = msg.value / 127.0
                sm.groups[gi].intensity = intensity
                # Echo value back so ring LED tracks position in Mode 2
                sm.midi_out.send(mido.Message("control_change", channel=0,
                                              control=msg.control, value=msg.value))
                if debug_midi:
                    group_name = ""
                    if sm.composition_mapping:
                        group_name = sm.composition_mapping.apc_groups.get(gi + 1, "")
                    print(f"[KNOB] knob {gi+1} (CC 0x{msg.control:02X})  "
                          f"group {gi+1} '{group_name}'  "
                          f"intensity = {intensity:.2f}  (raw {msg.value})")
        elif msg.control == 0x0D and msg.channel == 0:
            # Tempo knob — relative encoder (Relative-2 / two's complement)
            # val 1-63 = CW steps (increment), val 65-127 = CCW steps (decrement)
            if sm.osc_client and msg.value != 64:
                if 1 <= msg.value <= 63:
                    steps = msg.value
                    for _ in range(steps):
                        sm.osc_client.send_message("/composition/tempocontroller/tempo/inc", 1.0)
                elif 65 <= msg.value <= 127:
                    steps = 128 - msg.value
                    for _ in range(steps):
                        sm.osc_client.send_message("/composition/tempocontroller/tempo/dec", 1.0)
        elif msg.control == 0x0F and msg.channel == 0:
            # Layer speed — linear 0.0–1.0 across full knob travel
            speed = msg.value / 127.0
            if sm.osc_client and sm.group_role_layers:
                for _, roles in sm.group_role_layers.items():
                    for layer in roles.get("fills", []):
                        path = f"/composition/layers/{layer.global_index + 1}/speed"
                        sm.osc_client.send_message(path, speed)
                if debug_midi:
                    print(f"[OSC→ARENA] layer speed={speed:.3f}  (cc={msg.value})")
            elif debug_midi:
                print(f"[OSC→ARENA] speed SKIPPED — no osc_client or layers  (cc={msg.value})")
        elif msg.control == 0x2F and msg.channel == 0:
            # Clip view horizontal scroll — relative encoder
            # val 1-63 = scroll right (each unit × SCROLL_STEP), val 65-127 = scroll left
            if msg.value != 64:
                if 1 <= msg.value <= 63:
                    sm._clip_scroll_h = min(1.0, sm._clip_scroll_h + msg.value * SCROLL_STEP)
                else:
                    steps = 128 - msg.value
                    sm._clip_scroll_h = max(0.0, sm._clip_scroll_h - steps * SCROLL_STEP)
                if sm.osc_client:
                    sm.osc_client.send_message("/application/ui/clipsscrollhorizontal",
                                               sm._clip_scroll_h)
                    if debug_midi:
                        print(f"[OSC→ARENA] /application/ui/clipsscrollhorizontal  "
                              f"{sm._clip_scroll_h:.3f}  (cc={msg.value})")
        return

    note = msg.note
    ch   = msg.channel

    # ── Grid buttons: notes 0–39 on channel 0 ────────────────────────
    if 0 <= note <= 39 and ch == 0:
        key: MidiKey = ("note", 0, note)
        for k, press_type in press_mgr.handle_note_message(key, msg, now):
            row       = k[2] // 8
            group_idx = k[2] % 8
            if group_idx in sm._output_group_cols:
                if press_type == "short":
                    _toggle_output_layer(sm, row, group_idx)
            else:
                prop = ROW_TO_PROP[row]
                if sm._group_has_role(group_idx, prop):
                    sm.handle_property_press(group_idx, prop, press_type)
        return

    # ── CLIP STOP buttons: note 0x34, channel = group (0–7) ─────────────
    # tap → enable all roles; hold → disable all roles; double → next clip all roles
    if note == 0x34 and 0 <= ch < 8:
        key: MidiKey = ("note", ch, note)
        for k, press_type in press_mgr.handle_note_message(key, msg, now):
            gi = k[1]
            _handle_clip_stop(sm, gi, press_type)
        return

    # ── Solo: note 0x31, ch = group index ────────────────────────────
    if note == 0x31 and 0 <= ch < 8:
        if msg.type == "note_on" and msg.velocity > 0:
            sm.toggle_group_solo(ch)
        return

    # ── Mute: note 0x30, ch = group index ────────────────────────────
    if note == 0x30 and 0 <= ch < 8:
        if msg.type == "note_on" and msg.velocity > 0:
            sm.toggle_group_mute(ch)
        return

    # ── Track Select: note 0x33, ch = group index → advance all active layers
    if note == 0x33 and 0 <= ch < 8:
        if msg.type == "note_on" and msg.velocity > 0:
            g = sm.groups[ch]
            for prop in sm.BOOL_PROPS:
                if sm._group_has_role(ch, prop) and getattr(g, prop, False):
                    sm.next_property(ch, prop)
        return

    # ── Activator: note 0x32, ch = group index → toggle all autopilots in group
    if note == 0x32 and 0 <= ch < 8:
        if msg.type == "note_on" and msg.velocity > 0:
            sm.toggle_group_autopilot(ch)
        return

    # ── Direction: Reverse (0x3C) / Forward (0x3D), ch=0 ─────────────
    if ch == 0 and note in (0x3C, 0x3D):
        if msg.type == "note_on" and msg.velocity > 0:
            sm.set_fills_direction(0 if note == 0x3C else 2)
        return

    # ── Global buttons on channel 0 ───────────────────────────────────
    if ch == 0 and note in GLOBAL_NOTE_ACTION:
        _handle_global_note(note, msg, sm, press_mgr, now)


def _handle_clip_stop(sm: Mk2StateMachine, group_idx: int, press_type: str) -> None:
    """CLIP STOP button gesture handler (per group).
    short: enable all mapped roles in the group
    long:  disable all mapped roles in the group
    double: advance all mapped roles to the next clip
    """
    if press_type == "short":
        sm.enable_all_group_props(group_idx)
    elif press_type == "long":
        sm.disable_all_group_props(group_idx)
    elif press_type == "double":
        sm.next_all_group_props(group_idx)


def _handle_autopilot_play(sm: Mk2StateMachine, press_type: str) -> None:
    """PLAY button gesture handler.
    short (tap):  OFF → turn ON;  ON → advance all autopilot roles now
    long  (hold): ON  → turn OFF (no-op when already off)
    """
    if press_type == "short":
        if sm._global_autopilot_on:
            sm.advance_all_autopilots()
        else:
            sm.set_global_autopilot_on()
    elif press_type == "long" and sm._global_autopilot_on:
        sm.set_global_autopilot_off()


def _handle_global_note(
    note: int,
    msg: mido.Message,
    sm: Mk2StateMachine,
    press_mgr: PressManager,
    now: float,
):
    action = GLOBAL_NOTE_ACTION[note]

    # Immediate fire on note_on (no gesture needed)
    if msg.type == "note_on" and msg.velocity > 0:
        if action == "tap_tempo":
            sm.global_tap_tempo();         return
        if action == "nudge_minus":
            sm.global_nudge_minus();       return
        if action == "nudge_plus":
            sm.global_nudge_plus();        return
        if action == "tempo_sync":
            sm.global_tempo_sync();        return
        if action == "nav_right":
            sm.scroll_clips_horizontal(1); return
        if action == "nav_left":
            sm.scroll_clips_horizontal(-1); return
        if action == "nav_up":
            sm.scroll_clips_vertical(-1);  return
        if action == "nav_down":
            sm.scroll_clips_vertical(1);   return

    # Gesture-based (autopilot_play, stop_all, scenes)
    key: MidiKey = ("note", 0, note)
    for _, press_type in press_mgr.handle_note_message(key, msg, now):
        if action == "autopilot_play":
            _handle_autopilot_play(sm, press_type)
        elif action == "stop_all":
            if press_type == "short":
                sm.start_all_clips()
            elif press_type == "long":
                sm.stop_all_clips()
        elif action.startswith("scene_"):
            sm.handle_scene_button(int(action.split("_")[1]), press_type)


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

_WEB_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>show_bridge v4</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0e0e0e;color:#ccc;font-family:'Courier New',monospace;padding:16px}
#hdr{display:flex;align-items:center;gap:16px;padding:12px 18px;background:#1a1a1a;
  border-radius:8px;margin-bottom:16px;border:1px solid #2a2a2a;flex-wrap:wrap}
#title{font-size:15px;font-weight:bold;color:#fff;letter-spacing:.05em}
.badge{padding:4px 12px;border-radius:16px;font-size:11px;font-weight:bold}
.bon{background:#1a4a2e;color:#2ecc71;border:1px solid #2ecc71}
.boff{background:#1e1e1e;color:#444;border:1px solid #2a2a2a}
.stat{font-size:12px;color:#555}
.stat b{color:#bbb}
#ts{margin-left:auto;font-size:10px;color:#2a2a2a}
#grid-wrap{overflow-x:auto}
table{border-collapse:separate;border-spacing:4px}
.ch{text-align:center;font-size:10px;color:#444;padding:2px 4px;max-width:80px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:bottom}
.ch-out{color:#4cc3ff}
.rl{text-align:right;font-size:10px;color:#555;padding-right:10px;white-space:nowrap;vertical-align:middle}
/* Role grid cells */
.btn{width:80px;height:48px;border-radius:5px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;font-size:9px;font-weight:bold;
  cursor:default;user-select:none;gap:2px;letter-spacing:.04em}
.bno{background:#0f0f0f;color:#1c1c1c;border:1px dashed #1c1c1c}
.boff2{background:#181818;color:#2e2e2e;border:1px solid #202020}
.bon2{border:none}
@keyframes ap{from{opacity:.5}to{opacity:1}}
.bap{animation:ap .9s ease-in-out infinite alternate}
/* Output group layer cells */
.olayer{width:80px;height:48px;border-radius:5px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;font-size:8px;font-weight:bold;
  cursor:default;user-select:none;gap:2px}
.ol-on{background:#001060;color:#4c88ff;border:1px solid #0055ff}
.ol-off{background:#080810;color:#335566;border:1px solid #223344}
.ol-none{background:#0a0a0a;border:1px dashed #161616}
/* Strip button cells */
.srow{width:80px;height:30px;border-radius:4px;display:flex;align-items:center;
  justify-content:center;font-size:9px;font-weight:bold;cursor:default;user-select:none;gap:4px}
.sm-on{background:#3a0808;color:#e74c3c;border:1px solid #c0392b}
.sm-off{background:#141414;color:#2a2a2a;border:1px solid #1e1e1e}
.ss-on{background:#3a3000;color:#f1c40f;border:1px solid #d4ac0d}
.ss-off{background:#141414;color:#2a2a2a;border:1px solid #1e1e1e}
.sa-on{background:#130a2a;color:#8855ff;border:1px solid #5400ff}
.sa-off{background:#141414;color:#2a2a2a;border:1px solid #1e1e1e}
/* Bars */
.bw{padding:4px 6px}
.bt{font-size:9px;color:#383838}
.bk{width:100%;height:5px;background:#1a1a1a;border-radius:3px;overflow:hidden;margin:3px 0}
.bf{height:100%;border-radius:3px;transition:width .25s}
.bp{font-size:9px;color:#383838;text-align:right}
/* Dividers */
.div-col{width:6px}
.div-row{height:6px}
</style>
</head>
<body>
<div id="hdr">
  <span id="title">show_bridge v4</span>
  <span id="ap" class="badge boff">&#9675; Autopilot OFF</span>
  <span class="stat">BPM&nbsp;<b id="bpm">&#8212;</b></span>
  <span class="stat">Beat&nbsp;<b id="beat">&#8212;</b></span>
  <span id="ts">connecting&hellip;</span>
</div>
<div id="grid-wrap"><table id="tbl"></table></div>
<script>
// Display order = physical top→bottom (row4=top=color, row0=bottom=playing)
const ROLES=['color','transforms','masks','effects','playing'];
const RLABEL={color:'Color',transforms:'Transforms',masks:'Masks',effects:'Effects',playing:'Playing'};
// Actual APC40 MK2 palette colours (vel: color 17=#54ff00 playing, 29=#00ff55 effects,
// 37=#00a9ff masks, 49=#5400ff transforms; color row is dynamic)
const RCOLOR={transforms:'#5400ff',masks:'#00a9ff',effects:'#00ff55',playing:'#54ff00'};

function lum(hex){
  const r=parseInt(hex.slice(1,3),16)/255,g=parseInt(hex.slice(3,5),16)/255,
        b=parseInt(hex.slice(5,7),16)/255;
  return .299*r+.587*g+.114*b;
}

function sep(cols){
  return `<td class="div-col" colspan="${cols}"></td>`;
}

function render(s){
  const G=s.groups, OG=s.output_groups||[], N=G.length, NO=OG.length;
  const hasSep=NO>0;

  // ── Column headers ─────────────────────────────────────────────────
  let h='<thead><tr><td></td>';
  for(const g of G) h+=`<th class="ch" title="${g.name}">${g.name}</th>`;
  if(hasSep) h+=sep(1);
  for(const og of OG) h+=`<th class="ch ch-out" title="${og.name}">${og.name}</th>`;
  h+='</tr></thead><tbody>';

  // ── Role rows (physical top→bottom) ───────────────────────────────
  for(let di=0;di<ROLES.length;di++){
    const role=ROLES[di];
    // Output group layer index: display row 0 (color, top) → physical row 4 → layer index 4
    const layerIdx=ROLES.length-1-di;
    h+=`<tr><td class="rl">${RLABEL[role]}</td>`;
    for(const g of G){
      if(g.is_output){
        // Output col in normal group slots — shouldn't happen, skip
        h+=`<td><div class="btn bno">&#8212;</div></td>`;
        continue;
      }
      const r=g.roles[role];
      let cls,style,lbl;
      if(!r.has_role){cls='btn bno';style='';lbl='&#8212;'}
      else if(!r.enabled){cls='btn boff2';style='';lbl='&middot;'}
      else{
        const c=role==='color'?(r.color||'#ffffff'):(RCOLOR[role]||'#888');
        const tc=lum(c)>.5?'#111':'#eee';
        cls='btn bon2'+(r.autopilot?' bap':'');
        style=`background:${c};color:${tc};`;
        lbl=(r.autopilot?'&#9678; AUTO':'&#9679; ON');
      }
      h+=`<td><div class="${cls}" style="${style}">${lbl}</div></td>`;
    }
    if(hasSep) h+=sep(1);
    // Output group columns
    for(const og of OG){
      const layer=og.layers[layerIdx];
      if(!layer){h+=`<td><div class="olayer ol-none"></div></td>`;continue;}
      const cls2=layer.active?'olayer ol-on':'olayer ol-off';
      const dot=layer.active?'&#9679;':'&#9675;';
      const nm=layer.name.replace(/^(Panel |Beam )/,'');
      h+=`<td><div class="${cls2}">${dot}<span style="font-size:8px;max-width:70px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:block;text-align:center">${nm}</span></div></td>`;
    }
    h+='</tr>';
  }

  // ── Strip button rows ──────────────────────────────────────────────
  h+=`<tr><td colspan="${N+(hasSep?NO+1:0)+1}" class="div-row"></td></tr>`;
  const strips=[
    {key:'muted',        label:'MUTE', on:'sm-on', off:'sm-off'},
    {key:'soloed',       label:'SOLO', on:'ss-on', off:'ss-off'},
    {key:'any_autopilot',label:'AUTO', on:'sa-on', off:'sa-off'},
  ];
  for(const {key,label,on,off} of strips){
    h+=`<tr><td class="rl">${label}</td>`;
    for(const g of G){
      const a=g[key];
      h+=`<td><div class="srow ${a?on:off}">${a?'&#9679;':'&#9675;'}&nbsp;${label}</div></td>`;
    }
    if(hasSep){
      h+=sep(1);
      for(let i=0;i<NO;i++) h+=`<td><div class="srow" style="background:#0a0a0a;border:1px dashed #141414;color:#1a1a1a">&#8212;</div></td>`;
    }
    h+='</tr>';
  }

  // ── Intensity / opacity bars ───────────────────────────────────────
  h+=`<tr><td colspan="${N+(hasSep?NO+1:0)+1}" class="div-row"></td></tr>`;
  for(const [key,label,color] of [['intensity','Intensity','#ccaa00'],['opacity','Opacity','#6655cc']]){
    h+=`<tr><td class="rl">${label}</td>`;
    for(const g of G){
      const p=Math.round(g[key]*100);
      h+=`<td><div class="bw">
        <div class="bt">${label.slice(0,3).toUpperCase()}</div>
        <div class="bk"><div class="bf" style="width:${p}%;background:${color}"></div></div>
        <div class="bp">${p}%</div>
      </div></td>`;
    }
    if(hasSep){h+=sep(1);for(let i=0;i<NO;i++) h+='<td></td>';}
    h+='</tr>';
  }

  document.getElementById('tbl').innerHTML=h+'</tbody>';

  const ap=document.getElementById('ap');
  if(s.autopilot_on){ap.textContent='\\u25cf Autopilot ON';ap.className='badge bon'}
  else{ap.textContent='\\u25cb Autopilot OFF';ap.className='badge boff'}
  document.getElementById('bpm').textContent=s.bpm.toFixed(1);
  document.getElementById('beat').textContent=`${s.beat_remaining}\\u202f/\\u202f${s.beat_interval}`;
}

async function poll(){
  const ts=document.getElementById('ts');
  try{
    const r=await fetch('/api/state');
    if(!r.ok) throw new Error('HTTP '+r.status);
    render(await r.json());
    ts.textContent=new Date().toLocaleTimeString();
    ts.style.color='#2a2a2a';
  }catch(e){ts.textContent='\\u26a0 '+e.message;ts.style.color='#883333'}
}
poll();setInterval(poll,500);
</script>
</body>
</html>"""


def _build_web_state(sm: "Mk2StateMachine") -> dict:
    from lib.mk2_leds import PALETTE
    bpm = 0.0
    if sm._clock_thread is not None:
        with sm._clock_thread._lock:
            bpm = sm._clock_thread._bpm

    groups = []
    for gi in range(8):
        name = ""
        if sm.composition_mapping:
            name = sm.composition_mapping.apc_groups.get(gi + 1, "")
        g = sm.groups[gi]
        roles = {}
        for prop in ROW_TO_PROP:
            enabled   = getattr(g, prop, False)
            autopilot = getattr(g, PROP_AUTOPILOT.get(prop, ""), False)
            has_role  = sm._group_has_role(gi, prop)
            color_hex = None
            if prop == "color" and has_role:
                vel = sm.group_color_velocity[gi]
                rv, gv, bv = PALETTE[max(0, min(127, vel))]
                color_hex = f"#{rv:02X}{gv:02X}{bv:02X}"
            roles[prop] = {
                "enabled":   enabled,
                "autopilot": autopilot,
                "has_role":  has_role,
                "color":     color_hex,
            }
        any_autopilot = any(
            getattr(g, ap_attr, False)
            for prop, ap_attr in PROP_AUTOPILOT.items()
            if sm._group_has_role(gi, prop)
        )
        groups.append({
            "name":          name or f"Grp {gi + 1}",
            "intensity":     round(g.intensity, 3),
            "opacity":       round(getattr(g, "opacity", 1.0), 3),
            "roles":         roles,
            "muted":         gi in sm._mute_groups,
            "soloed":        gi in sm._solo_groups,
            "any_autopilot": any_autopilot,
            "is_output":     gi in sm._output_group_cols,
        })

    output_groups = []
    for col in sorted(sm._output_group_cols.keys()):
        layers = sm._output_group_cols[col]
        output_groups.append({
            "col":    col,
            "name":   sm._output_group_names.get(col, f"Out {col + 1}"),
            "layers": [
                {
                    "name":   layer.name,
                    "active": sm._output_layer_active.get(layer.global_index, True),
                }
                for layer in layers
            ],
        })

    return {
        "autopilot_on":   sm._global_autopilot_on,
        "bpm":            round(bpm, 1),
        "beat_remaining": getattr(sm, "autopilot_beats_remaining", 0),
        "beat_interval":  sm._autopilot_beat_interval(),
        "groups":         groups,
        "output_groups":  output_groups,
    }


def start_web_ui(sm: "Mk2StateMachine", port: int = 8765) -> None:
    try:
        from flask import Flask, jsonify
    except ImportError:
        print("Flask not installed — run:  pip install flask")
        return

    import logging as _log
    _log.getLogger("werkzeug").setLevel(_log.ERROR)

    flask_app = Flask(__name__)

    @flask_app.route("/")
    def index():
        return _WEB_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    @flask_app.route("/api/state")
    def api_state():
        return jsonify(_build_web_state(sm))

    t = threading.Thread(
        target=lambda: flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True, name="web-ui",
    )
    t.start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="show_bridge v4 – APC40 MK2")
    parser.add_argument("--no-resolume", action="store_true",
                        help="Skip Resolume connection (for LED testing only)")
    parser.add_argument("--debug-midi", action="store_true",
                        help="Print every incoming MIDI message (useful for mapping unknown controls)")
    parser.add_argument("--ui", action="store_true",
                        help="Start web UI dashboard (requires flask)")
    parser.add_argument("--web-port", type=int, default=8765,
                        help="Port for the web UI (default: 8765)")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console + file log level (default: WARNING)")
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)
    app_logger = setup_logging(level=log_level)
    sb_log(app_logger, logging.INFO, "BRIDGE", "INIT", "show_bridge v4 (APC40 MK2) starting")

    connections_cfg = load_connections(Path(DEFAULT_CONNECTIONS_PATH))

    # ── OSC / HTTP connections ────────────────────────────────────────
    osc_client = syn_osc_client = sb_osc_client = conn_http = None

    if not args.no_resolume:
        try:
            conn_http  = get_resolume_http_connection(connections_cfg)
            osc_host, osc_port = get_resolume_osc_connection(connections_cfg)
            osc_client = get_osc_client(osc_host, osc_port)
            sb_log(app_logger, logging.INFO, "ARENA", "OSC",
                   f"Resolume OSC → {osc_host}:{osc_port}")
        except Exception as e:
            app_logger.warning(f"[RESOLUME] Could not connect: {e}")

        try:
            syn_host, syn_port = get_synesthesia_osc_connection(connections_cfg)
            syn_osc_client = get_osc_client(syn_host, syn_port)
        except Exception:
            pass

        try:
            sb_host, sb_port = get_show_bridge_osc_connection(connections_cfg)
            sb_osc_client = get_osc_client(sb_host, sb_port)
        except Exception:
            pass

    state_output_cfg: dict = {}
    try:
        cfg_path = Path("settings/state_machine_mappings.yaml")
        if cfg_path.exists():
            state_output_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        app_logger.warning(f"[STATE] state_machine_mappings.yaml: {e}")

    # ── Open MIDI ports (with reconnect) ─────────────────────────────
    in_port = out_port = None
    attempt = 0
    while True:
        try:
            in_names  = mido.get_input_names()
            out_names = mido.get_output_names()
            in_name   = auto_select_port(in_names,  "APC40 MkII", "input")
            out_name  = auto_select_port(out_names, "APC40 MkII", "output")
            in_port   = mido.open_input(in_name)
            out_port  = mido.open_output(out_name)
            sb_log(app_logger, logging.INFO, "MIDI", "PORTS",
                   f"IN='{in_name}'  OUT='{out_name}'")
            break
        except Exception as e:
            attempt += 1
            delay = 5 if attempt == 1 else 15
            app_logger.warning(f"[MIDI] Cannot open MK2 ports: {e} – retry in {delay}s")
            time.sleep(delay)

    # ── Build state machine ───────────────────────────────────────────
    sm = Mk2StateMachine(
        midi_out=out_port,
        osc_client=osc_client,
        resolume_conn=conn_http,
        syn_osc_client=syn_osc_client,
        broadcast_osc_client=sb_osc_client,
        state_output_cfg=state_output_cfg,
    )

    # ── MIDI clock thread (keeps MK2 pulsing synced to BPM) ──────────
    clock = MidiClockThread(out_port)
    clock.start()
    sm._clock_thread = clock

    # ── MK2 initialisation ────────────────────────────────────────────
    sm.send_sysex_init()
    time.sleep(0.1)   # let device apply mode change before LED commands
    sm.clear_all_grid_leds()
    sm.clear_strip_leds()        # blank ARM/SOLO/ACTIVATOR/TRACK_SELECT/CLIP_STOP + direction
    sm.init_track_knob_rings()   # Volume Style rings, position = current intensity
    sm._send_direction_leds()    # Forward lit by default, Reverse off

    # ── Beat source: Synestesia OSC (port 12000), fallback 240 BPM ───
    sm.start_synestesia_beat_listener(port=12000)
    # Keep the Resolume transport listener as a secondary source on port 7001
    start_transport_osc_listener(sm, host="127.0.0.1", port=7001)

    # ── Load composition ──────────────────────────────────────────────
    if conn_http and not args.no_resolume:
        try:
            comp_json    = fetch_composition_json(conn_http)
            comp_name    = _guess_composition_name(comp_json)
            print(f"[RESOLUME] Composition name from Resolume: {comp_name!r}")

            # List every YAML in the mapping dir so mismatches are obvious
            mapping_dir = Path(DEFAULT_COMPOSITION_MAPPING_DIR)
            yaml_names  = [p.name for p in sorted(mapping_dir.glob("*.y*ml"))]
            print(f"[RESOLUME] Mapping files available: {yaml_names}")

            try:
                comp_mapping = CompositionMapping.from_yaml_dir(
                    composition_name=comp_name,
                    mapping_dir=mapping_dir,
                )
                print(f"[RESOLUME] Matched mapping: apc_groups = {comp_mapping.apc_groups}")
            except RuntimeError as mapping_err:
                print(f"[RESOLUME] *** NO MAPPING MATCHED — {mapping_err}")
                print(f"[RESOLUME] *** Check that a YAML in {mapping_dir}/ has "
                      f"composition_name: {comp_name!r}")
                raise

            comp_model = build_composition_model(comp_json, comp_mapping)
            print(f"[RESOLUME] Groups in composition: "
                  f"{[g.name for g in comp_model.groups]}")
            print(f"[RESOLUME] Unmapped groups (not in apc_groups): "
                  f"{[g.name for g in comp_model.groups if g.apc_group_index is None]}")

            sm.attach_composition(comp_model, comp_mapping)
            sm.initialize_state_from_resolume()
            sm._detect_output_groups()
            debug_dump_composition_columns(comp_model, app_logger)
            sb_log(app_logger, logging.INFO, "ARENA", "INIT",
                   f"Composition '{comp_model.name}' – {len(comp_model.groups)} groups")
            sm.refresh_group_colors()
        except Exception as e:
            app_logger.warning(f"[RESOLUME] Composition load failed: {e}")

    # Push initial LED state to the device
    sm._recompute_unmapped_groups()
    sm._update_all_leds()
    sm._update_unmapped_group_indicators()

    # ── Background color polling (keeps color LEDs synced with Resolume)
    if conn_http and not args.no_resolume:
        sm.start_color_polling()

    # ── Rainbow cycling thread (always runs; skips groups not in rainbow mode)
    sm._start_rainbow_cycling()

    # ── Main loop ─────────────────────────────────────────────────────
    press_mgr = PressManager()

    def _one_tick(now: float) -> None:
        for msg in in_port.iter_pending():
            if args.debug_midi and msg.type in ("note_on", "note_off", "control_change"):
                if msg.type == "control_change":
                    print(f"[MIDI] CC  ch={msg.channel}  cc=0x{msg.control:02X} ({msg.control:3d})  val={msg.value}")
                else:
                    print(f"[MIDI] {msg.type:<8s} ch={msg.channel}  note=0x{msg.note:02X} ({msg.note:3d})  vel={msg.velocity}")
            try:
                handle_mk2_midi(msg, sm, press_mgr, now, debug_midi=args.debug_midi)
            except Exception as e:
                app_logger.warning(f"[MIDI] {msg}: {e}")

        for key, press_type in press_mgr.poll(now):
            note, ch = key[2], key[1]
            if 0 <= note <= 39 and ch == 0:
                row, group_idx = note // 8, note % 8
                if group_idx in sm._output_group_cols:
                    if press_type == "short":
                        _toggle_output_layer(sm, row, group_idx)
                else:
                    sm.handle_property_press(group_idx, ROW_TO_PROP[row], press_type)
            elif note == 0x34 and 0 <= ch < 8:
                _handle_clip_stop(sm, ch, press_type)
            elif ch == 0 and note in GLOBAL_NOTE_ACTION:
                action = GLOBAL_NOTE_ACTION[note]
                if action == "autopilot_play":
                    _handle_autopilot_play(sm, press_type)
                elif action == "stop_all":
                    if press_type == "short":
                        sm.start_all_clips()
                    elif press_type == "long":
                        sm.stop_all_clips()
                elif action.startswith("scene_"):
                    sm.handle_scene_button(int(action.split("_")[1]), press_type)

        sm._refresh_routing_layers(now)

    if args.ui:
        start_web_ui(sm, args.web_port)
        print(f"Web UI → http://localhost:{args.web_port}  (Ctrl+C to exit)")
    else:
        print("Running – Ctrl+C to exit")

    try:
        while True:
            now = time.monotonic()
            _one_tick(now)
            time.sleep(0.005)

    except KeyboardInterrupt:
        app_logger.info("Shutting down.")
    finally:
        clock.stop()
        for note in range(40):
            try:
                out_port.send(mido.Message("note_on", channel=0, note=note, velocity=0))
            except Exception:
                pass
        in_port.close()
        out_port.close()


if __name__ == "__main__":
    main()
