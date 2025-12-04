#!/usr/bin/env python3
"""
Visualize MIDI → State Machine → Output mappings as a Graphviz graph.

- Reads a mapping profile from ./mappings/*.yml / *.yaml
- Builds a MappingRuntime (same logic as run.py)
- Emits mapping_graph.dot showing:

    MIDI input  ──▶  Action / Property  ──▶  Effect / OSC hint

Usage:

    python visualize_midi_mappings.py \
        --mappings-dir mappings \
        --device apc40 \
        --out mapping_graph.dot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import yaml

# ------------------------------
# Types & dataclasses (trimmed)
# ------------------------------

MidiKey = Tuple[str, int, int]  # ("note"|"cc", channel, note_or_cc)


@dataclass
class ActionSpec:
    """
    Represents a bound action for a particular MIDI key.
    """
    action: str
    scope: str                   # "group" or "global"
    property_name: str | None = None
    group_index: int | None = None
    intensity_value: float | None = None
    scene_index: int | None = None


@dataclass
class GroupControlAddresses:
    """
    Where to send LED feedback for each boolean property and intensity presets.
    (We don't use this for the graph, but it's needed to build the runtime.)
    """
    channel: int
    prop_notes: Dict[str, int]
    reset_note: int
    slider_cc: int
    intensity_preset_notes: list[int] = field(default_factory=list)


@dataclass
class VelocityProfile:
    """
    Describes how to light a note.
    """
    off: int = 0
    on: int = 127
    colors: list[int] = field(default_factory=list)

    def resolved_on(self) -> int:
        for c in self.colors:
            if c != self.off:
                return c
        return self.on


@dataclass
class MappingRuntime:
    action_map: Dict[MidiKey, ActionSpec] = field(default_factory=dict)
    group_addrs: Dict[int, GroupControlAddresses] = field(default_factory=dict)
    note_velocity: Dict[int, VelocityProfile] = field(default_factory=dict)
    scene_buttons: Dict[int, Tuple[int, int]] = field(default_factory=dict)


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


# ------------------------------
# YAML loading helpers
# ------------------------------

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
    desired_name: Optional[str] = None,
) -> MappingProfile:
    """
    Select a mapping profile.

    Priority:
      1) If desired_name is provided, match by controller_name or file stem.
      2) If there is only one profile, use it.
      3) Otherwise, prompt.
    """
    if desired_name:
        matches = [
            p for p in profiles
            if p.controller_name.lower() == desired_name.lower()
            or p.file_path.stem.lower() == desired_name.lower()
        ]
        if matches:
            p = matches[0]
            print(f"Selected mapping profile: {p.controller_name} ({p.file_path.name})")
            return p
        else:
            print(f"WARNING: No mapping profile named '{desired_name}' found; falling back.")

    if len(profiles) == 1:
        p = profiles[0]
        print(f"Using only available profile: {p.controller_name} ({p.file_path.name})")
        return p

    print("Available mapping profiles:")
    for idx, p in enumerate(profiles):
        print(f"  [{idx}] {p.controller_name}  (file: {p.file_path.name})")

    while True:
        raw = input("Select profile index: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 0 <= idx < len(profiles):
            return profiles[idx]
        print(f"Index out of range. Choose between 0 and {len(profiles) - 1}.")


# ------------------------------
# Velocity map builder (trimmed)
# ------------------------------

def build_note_velocity_map(input_cfg: Dict[str, Any]) -> Dict[int, VelocityProfile]:
    vel_cfg = input_cfg.get("velocity_mappings")
    if not vel_cfg:
        return {}

    maps_cfg = vel_cfg.get("maps", {})
    ranges_cfg = vel_cfg.get("ranges", [])
    notes_cfg = vel_cfg.get("notes", {})

    named_profiles: Dict[str, VelocityProfile] = {}

    for name, m in maps_cfg.items():
        if isinstance(m, int):
            prof = VelocityProfile(off=0, on=int(m), colors=[])
        elif isinstance(m, list):
            colors = [int(v) for v in m]
            prof = VelocityProfile(off=0, on=127, colors=colors)
        elif isinstance(m, dict):
            off_val = int(m.get("off", 0))
            colors_raw = m.get("colors", [])
            colors = [int(v) for v in colors_raw] if isinstance(colors_raw, list) else []
            on_raw = m.get("on")
            if on_raw is None:
                derived = next((c for c in colors if c != off_val), None)
                on_val = derived if derived is not None else 127
            else:
                on_val = int(on_raw)
            prof = VelocityProfile(off=off_val, on=on_val, colors=colors)
        else:
            prof = VelocityProfile()
        named_profiles[name] = prof

    note_map: Dict[int, VelocityProfile] = {}

    for r in ranges_cfg:
        try:
            start = int(r["start"])
            end = int(r["end"])
            map_name = r["map"]
        except KeyError:
            continue
        profile = named_profiles.get(map_name)
        if not profile:
            continue
        a, b = sorted((start, end))
        for note in range(a, b + 1):
            note_map[note] = profile

    for note_str, map_name in notes_cfg.items():
        try:
            note = int(note_str)
        except ValueError:
            continue
        profile = named_profiles.get(map_name)
        if not profile:
            continue
        note_map[note] = profile

    return note_map


# ------------------------------
# Runtime mapping builder
# (essentially copied from run.py, trimmed a bit)
# ------------------------------

def build_runtime_mapping(input_cfg: Dict[str, Any],
                          state_cfg: Dict[str, Any]) -> MappingRuntime:
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

        # 2) reset
        reset_note = -1
        reset_control_name = group_reset_cfg.get("control")
        reset_action_name = group_reset_cfg.get("action")
        if reset_control_name and reset_control_name in control_note_map:
            reset_note = control_note_map[reset_control_name]
            key = ("note", channel, reset_note)
            runtime.action_map[key] = ActionSpec(
                action=reset_action_name,
                scope="group",
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
                continue
            scene_idx, s_entry = target
            key = ("note", s_entry.get("channel", 0), s_entry["note"])
            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="global",
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
            )

        elif control_name in ("play", "stop") and "transport" in input_global_cfg:
            t = input_global_cfg["transport"].get(control_name)
            if not t:
                continue
            key = ("note", t.get("channel", 0), t["note"])
            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="global",
            )

        elif control_name == "stop_all_clips" and "stop_all_clips" in input_global_cfg:
            s = input_global_cfg["stop_all_clips"]
            key = ("note", s.get("channel", 0), s["note"])
            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="global",
            )

        elif control_name in ("nudge_minus", "nudge_plus", "tap_tempo", "shift_resync"):
            src = timing_cfg.get(control_name)
            if not src:
                continue
            key = ("note", src.get("channel", 0), src["note"])
            runtime.action_map[key] = ActionSpec(
                action=action_name,
                scope="global",
            )

    return runtime


# ------------------------------
# Graphviz export
# ------------------------------

# Rough hints for “output” nodes based on action name
ACTION_OUTPUT_HINTS: Dict[str, str] = {
    "set_global_intensity_from_cc": "/composition/master",
    "nudge_minus": "/composition/tempocontroller/tempopull",
    "nudge_plus": "/composition/tempocontroller/tempopush",
    "tap_tempo": "/composition/tempocontroller/tempotap",
    "tempo_sync": "/composition/tempocontroller/resync",
    # Others are more conceptual; we’ll describe them in the node label.
}


def midi_key_to_node_id(key: MidiKey) -> str:
    kind, ch, num = key
    return f"in_{kind}_{ch}_{num}"


def action_to_node_id(spec: ActionSpec) -> str:
    if spec.scope == "group":
        gi = (spec.group_index or 0) + 1
        prop = spec.property_name or spec.action
        return f"act_group{gi}_{prop}"
    else:
        base = spec.action
        if spec.scene_index is not None:
            return f"act_scene_{spec.scene_index+1}"
        return f"act_global_{base}"


def output_to_node_id(spec: ActionSpec) -> Optional[str]:
    """
    For actions where we have a clear OSC endpoint, return an output node id.
    For others, we just annotate the action node.
    """
    if spec.action in ACTION_OUTPUT_HINTS:
        return f"out_{spec.action}"
    return None


def build_action_label(spec: ActionSpec) -> str:
    if spec.scope == "group":
        gi = (spec.group_index or 0) + 1
        if spec.property_name in ("playing", "effects", "transforms", "color", "dynamic_masks"):
            return f"Group {gi}: {spec.property_name}\\n(short: toggle/next, long: off, dbl: auto)"
        if spec.property_name == "opacity":
            return f"Group {gi}: opacity\\nCC → set_opacity_from_cc"
        if spec.action == "set_intensity_preset" and spec.intensity_value is not None:
            return f"Group {gi}: intensity preset {spec.intensity_value:.2f}\\n(short: set, long: 0, dbl: random)"
        if spec.action == "reset_group":
            return f"Group {gi}: reset_group"
        return f"Group {gi}: {spec.action}"
    else:
        if spec.action == "set_global_intensity_from_cc":
            return "Global: master intensity\\nCC → /composition/master"
        if spec.action == "toggle_effects_all":
            return "Global: toggle_effects_all\\n(stop-all semantics)"
        if spec.action == "set_global_autopilot_on":
            return "Global: autopilot ON"
        if spec.action == "set_global_autopilot_off":
            return "Global: autopilot OFF"
        if spec.action == "nudge_minus":
            return "Global: nudge_minus\\n→ tempopull"
        if spec.action == "nudge_plus":
            return "Global: nudge_plus\\n→ tempopush"
        if spec.action == "tap_tempo":
            return "Global: tap_tempo\\n→ tempotap"
        if spec.action == "tempo_sync":
            return "Global: tempo_sync\\n→ resync"
        if spec.action == "control_all_clips":
            return "Global: control_all_clips\\n(start/next/stop fills)"
        if spec.action == "scene_slot" and spec.scene_index is not None:
            return f"Scene {spec.scene_index+1}: load/save/clear"
        return f"Global: {spec.action}"


def build_output_label(spec: ActionSpec) -> str:
    if spec.action in ACTION_OUTPUT_HINTS:
        return ACTION_OUTPUT_HINTS[spec.action]
    # Conceptual outputs for group properties
    if spec.scope == "group" and spec.property_name == "playing":
        gi = (spec.group_index or 0) + 1
        return f"APC group {gi} fill layers\\n/ composition/layers/*/clips/*/connect"
    if spec.scope == "group" and spec.property_name in ("effects", "transforms", "color", "dynamic_masks"):
        gi = (spec.group_index or 0) + 1
        return f"APC group {gi} role='{spec.property_name}'\\n/ composition/layers/*/clips/*/connect"
    if spec.scope == "group" and spec.property_name == "opacity":
        gi = (spec.group_index or 0) + 1
        return f"APC group {gi} Router layer\\n/ composition/layers/router/video/opacity"
    return "internal state / no direct OSC hint"


def export_mapping_graph(
    profile: MappingProfile,
    runtime: MappingRuntime,
    out_path: Path,
) -> None:
    """
    Build a Graphviz DOT file describing:

        MIDI input → action → (optional) output

    Nodes are styled by type:
      - Inputs: ellipse, light blue
      - Actions: box, light orange
      - Outputs: note shape, light gray
    """
    lines: List[str] = []
    lines.append("digraph MidiMapping {")
    lines.append("  rankdir=LR;")
    lines.append("  node [fontname=\"Helvetica\"];")
    lines.append("")
    lines.append(f"  // Controller: {profile.controller_name}")
    lines.append(f"  // Mapping file: {profile.file_path}")
    lines.append("")

    # Collect nodes to avoid duplicates
    input_nodes: Dict[str, str] = {}
    action_nodes: Dict[str, str] = {}
    output_nodes: Dict[str, str] = {}

    # Build nodes & edges
    edges: List[Tuple[str, str]] = []
    output_edges: List[Tuple[str, str]] = []

    for key, spec in runtime.action_map.items():
        kind, ch, num = key
        in_id = midi_key_to_node_id(key)
        in_label = f"{kind.upper()} ch{ch} {num}"

        if in_id not in input_nodes:
            input_nodes[in_id] = in_label

        act_id = action_to_node_id(spec)
        act_label = build_action_label(spec)
        if act_id not in action_nodes:
            action_nodes[act_id] = act_label

        edges.append((in_id, act_id))

        out_id = output_to_node_id(spec)
        if out_id:
            out_label = build_output_label(spec)
            if out_id not in output_nodes:
                output_nodes[out_id] = out_label
            output_edges.append((act_id, out_id))
        else:
            # For some actions, it's still useful to show conceptual output nodes
            conceptual_label = build_output_label(spec)
            if conceptual_label and "no direct OSC" not in conceptual_label:
                out_id = f"out_{act_id}"
                if out_id not in output_nodes:
                    output_nodes[out_id] = conceptual_label
                output_edges.append((act_id, out_id))

    # Emit input nodes
    lines.append("  // MIDI input nodes")
    for nid, label in input_nodes.items():
        lines.append(
            f"  {nid} [shape=ellipse, style=filled, fillcolor=\"#9ecae1\", label=\"{label}\"];"
        )
    lines.append("")

    # Emit action nodes
    lines.append("  // Action / state-machine nodes")
    for nid, label in action_nodes.items():
        lines.append(
            f"  {nid} [shape=box, style=filled, fillcolor=\"#fdd0a2\", label=\"{label}\"];"
        )
    lines.append("")

    # Emit output nodes
    if output_nodes:
        lines.append("  // Output / OSC hint nodes")
        for nid, label in output_nodes.items():
            lines.append(
                f"  {nid} [shape=note, style=filled, fillcolor=\"#f0f0f0\", label=\"{label}\"];"
            )
        lines.append("")

    # Emit edges
    lines.append("  // MIDI -> action edges")
    for src, dst in edges:
        lines.append(f"  {src} -> {dst};")

    if output_edges:
        lines.append("")
        lines.append("  // Action -> output edges")
        for src, dst in output_edges:
            lines.append(f"  {src} -> {dst};")

    # Tiny legend
    lines.append("")
    lines.append("  subgraph cluster_legend {")
    lines.append("    label=\"Legend\";")
    lines.append("    fontsize=10;")
    lines.append("    key_in [shape=ellipse, style=filled, fillcolor=\"#9ecae1\", label=\"MIDI input (note/CC)\"];")
    lines.append("    key_act [shape=box, style=filled, fillcolor=\"#fdd0a2\", label=\"Action / state property\"];")
    lines.append("    key_out [shape=note, style=filled, fillcolor=\"#f0f0f0\", label=\"Effect / OSC hint\"];")
    lines.append("    key_in -> key_act -> key_out [style=invis];")
    lines.append("  }")

    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[GRAPH] Wrote mapping graph to {out_path}")


# ------------------------------
# CLI entrypoint
# ------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualize MIDI → state-machine mappings as Graphviz.")
    parser.add_argument(
        "--mappings-dir",
        type=str,
        default="mappings",
        help="Directory containing *.yaml mapping profiles (default: ./mappings)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Profile/controller name to use (matches controller_name or file stem).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="mapping_graph.dot",
        help="Output DOT file (default: mapping_graph.dot)",
    )
    args = parser.parse_args(argv)

    mappings_dir = Path(args.mappings_dir)
    profiles = load_mapping_profiles(mappings_dir)
    profile = choose_mapping_profile(profiles, desired_name=args.device)

    full_cfg = load_yaml_file(profile.file_path)
    input_cfg = full_cfg.get("input_mappings", {})
    state_cfg = full_cfg.get("state_mappings", {})

    runtime = build_runtime_mapping(input_cfg, state_cfg)

    out_path = Path(args.out)
    export_mapping_graph(profile, runtime, out_path)


if __name__ == "__main__":
    main()
