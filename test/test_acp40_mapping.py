#!/usr/bin/env python3

import time
from pathlib import Path

import mido
import yaml


# ------------------------
# Config + Mapping
# ------------------------

mapping_file = "input_mappings/midi_apc40.yaml"

def load_config(path=mapping_file):
    """Load the raw YAML config so we can see controller_name and the layout."""
    yaml_path = Path(path)
    return yaml.safe_load(yaml_path.read_text())


def build_mapping(config):
    """
    Build a lookup:
      ("note", channel, note) -> "Label"
      ("cc",   channel, cc)   -> "Label"
    """
    mapping = {}

    # --- GROUPS 1-8 ---
    for group_id, g in config["groups"].items():
        channel = g["channel"]

        # Clip buttons
        for clip in g["clip_buttons"]:
            mapping[("note", channel, clip["note"])] = \
                f"Group {group_id} {clip['name']}"

        # Clip stop
        mapping[("note", channel, g["clip_stop"]["note"])] = \
            f"Group {group_id} clip_stop"

        # Track Select (note 51)
        mapping[("note", channel, g["track_select"]["note"])] = \
            f"Group {group_id} track_select"

        # Activator / Solo / Arm
        mapping[("note", channel, g["activator"]["note"])] = \
            f"Group {group_id} activator"
        mapping[("note", channel, g["solo"]["note"])] = \
            f"Group {group_id} solo"
        mapping[("note", channel, g["arm"]["note"])] = \
            f"Group {group_id} arm"

        # Slider (CC7)
        mapping[("cc", channel, g["slider"]["cc"])] = \
            f"Group {group_id} slider"

    # --- GLOBAL CONTROLS ---
    g = config["global"]

    mapping[("note", g["stop_all_clips"]["channel"], g["stop_all_clips"]["note"])] = \
        "GLOBAL stop_all_clips"

    mapping[("note", g["master_track_select"]["channel"], g["master_track_select"]["note"])] = \
        "GLOBAL master_track_select"

    mapping[("note", g["queue_level"]["channel"], g["queue_level"]["note"])] = \
        "GLOBAL queue_level"

    mapping[("cc", g["master_slider"]["channel"], g["master_slider"]["cc"])] = \
        "GLOBAL master_slider"

    # Scene launch 1–5
    for scene in g.get("scene_launch", []):
        mapping[("note", scene["channel"], scene["note"])] = \
            f"GLOBAL {scene['name']}"

    # Navigation: right/left/up/down
    for name, btn in g.get("navigation", {}).items():
        mapping[("note", btn["channel"], btn["note"])] = \
            f"GLOBAL {name}"

    # Timing controls: shift/nudge/tap tempo
    for name, btn in g.get("timing_controls", {}).items():
        mapping[("note", btn["channel"], btn["note"])] = \
            f"GLOBAL {name}"

    # Global buttons rows
    for row_key in ("global_buttons_row1", "global_buttons_row2"):
        for btn in g.get(row_key, []):
            mapping[("note", btn["channel"], btn["note"])] = \
                f"GLOBAL {btn['name']}"

    # Transport: play/stop/rec
    for name, btn in g.get("transport", {}).items():
        mapping[("note", btn["channel"], btn["note"])] = \
            f"GLOBAL {name}"

    # Global slider (CC15)
    if "global_slider" in g:
        gs = g["global_slider"]
        mapping[("cc", gs["channel"], gs["cc"])] = "GLOBAL global_slider"

    return mapping


# ------------------------
# Port selection
# ------------------------

def choose_input_port(preferred_name: str | None = None) -> str:
    """
    Choose a MIDI input port.
    If preferred_name is provided, try to auto-select a port whose name contains it.
    Falls back to interactive selection if no match is found.
    """
    input_names = mido.get_input_names()

    if not input_names:
        raise RuntimeError("No MIDI input ports found.")

    # Try auto-match using controller_name from YAML
    if preferred_name:
        preferred_lower = preferred_name.lower()
        matches = [n for n in input_names if preferred_lower in n.lower()]

        if len(matches) == 1:
            print(f"Automatically selected input '{matches[0]}' "
                  f"for controller_name='{preferred_name}'.")
            return matches[0]
        elif len(matches) > 1:
            # If there are multiple matches, pick the first but warn
            print(f"Multiple ports match controller_name='{preferred_name}':")
            for i, n in enumerate(matches):
                print(f"  [{i}] {n}")
            print(f"Choosing first match: {matches[0]}")
            return matches[0]
        else:
            print(f"No ports matched controller_name='{preferred_name}'. "
                  f"Falling back to manual selection.\n")

    # Manual fallback
    print("Available MIDI input ports:")
    for idx, name in enumerate(input_names):
        print(f"  [{idx}] {name}")

    while True:
        try:
            choice = int(input("Select input port number: "))
            if 0 <= choice < len(input_names):
                return input_names[choice]
            print(f"Enter a number between 0 and {len(input_names) - 1}.")
        except ValueError:
            print("Please enter a valid integer.")


# ------------------------
# Main loop
# ------------------------

def main():
    # Load YAML config and mapping
    config = load_config()
    controller_name = config.get("controller_name")
    mapping = build_mapping(config)

    print(f"Loaded mapping for controller: {controller_name}\n")

    # Auto-select port based on controller_name, with fallback
    port_name = choose_input_port(preferred_name=controller_name)
    print(f"\nListening on MIDI input: {port_name}\n"
          f"Press Ctrl+C to exit.\n")

    with mido.open_input(port_name) as port:
        try:
            while True:
                for msg in port.iter_pending():
                    if msg.type in ("note_on", "note_off"):
                        key = ("note", msg.channel, msg.note)
                        label = mapping.get(key, "Unmapped note")
                        print(f"{label}: {msg}")

                    elif msg.type == "control_change":
                        key = ("cc", msg.channel, msg.control)
                        label = mapping.get(key, "Unmapped CC")
                        print(f"{label}: {msg}")

                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()