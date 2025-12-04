#!/usr/bin/env python3
"""
midi_listener.py

Listen to a MIDI input port (default: "From Resolume") and print all messages.

Usage examples:
  python midi_listener.py
  python midi_listener.py --port "From Resolume"
  python midi_listener.py --list
  python midi_listener.py --port "From Resolume" --raw
"""

import argparse
import sys
import time

import mido


def list_ports() -> None:
    """Print all available MIDI input ports."""
    input_names = mido.get_input_names()
    if not input_names:
        print("No MIDI input ports found.")
        return

    print("Available MIDI input ports:")
    for idx, name in enumerate(input_names):
        print(f"  [{idx}] {name}")


def choose_port_by_name(preferred_name: str | None) -> str:
    """
    Find an input port whose name matches the preferred_name.

    Matching strategy:
      1. Exact match (case-sensitive)
      2. Case-insensitive match
      3. Substring match (case-insensitive)

    Raises SystemExit if no matching port is found.
    """
    ports = mido.get_input_names()

    if not ports:
        print("No MIDI input ports found on this system.", file=sys.stderr)
        sys.exit(1)

    if preferred_name is None:
        # If no name specified, but only one port, use it
        if len(ports) == 1:
            print(f"No port name provided; using only available port: {ports[0]}")
            return ports[0]
        else:
            print(
                "Multiple ports found and no --port specified.\n"
                "Use --list to see ports, then pass --port \"<name>\".",
                file=sys.stderr,
            )
            list_ports()
            sys.exit(1)

    # 1. Exact match (case-sensitive)
    for name in ports:
        if name == preferred_name:
            print(f"Using exact match port: {name}")
            return name

    # 2. Case-insensitive match
    lower_pref = preferred_name.lower()
    for name in ports:
        if name.lower() == lower_pref:
            print(f"Using case-insensitive match port: {name}")
            return name

    # 3. Substring match (case-insensitive)
    substring_matches = [name for name in ports if lower_pref in name.lower()]
    if len(substring_matches) == 1:
        print(f"Using substring match port: {substring_matches[0]}")
        return substring_matches[0]
    elif len(substring_matches) > 1:
        print(
            f"Multiple ports match '{preferred_name}'. Be more specific or use --list.",
            file=sys.stderr,
        )
        for name in substring_matches:
            print(f"  {name}")
        sys.exit(1)

    print(
        f"No MIDI input port found matching '{preferred_name}'.\n"
        "Use --list to see available ports.",
        file=sys.stderr,
    )
    list_ports()
    sys.exit(1)


def print_message(msg: mido.Message, raw: bool = False) -> None:
    """
    Pretty-print a MIDI message.

    If raw=True, also show the underlying bytes in hex.
    """
    ts = time.strftime("%H:%M:%S")
    # mido.Message.__str__ is already nice, e.g. "note_on channel=0 note=60 velocity=64 time=0"
    if raw:
        try:
            b = msg.bytes()  # list[int]
            hex_bytes = " ".join(f"{x:02X}" for x in b)
            print(f"[{ts}] {msg!s} | raw: {hex_bytes}")
        except Exception as e:  # pragma: no cover (very rare)
            print(f"[{ts}] {msg!s} | raw: <error getting bytes: {e}>")
    else:
        print(f"[{ts}] {msg!s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Listen on a MIDI input port and print incoming messages."
    )
    parser.add_argument(
        "--port",
        "-p",
        type=str,
        default="From Resolume",
        help='Name (or partial name) of the MIDI input port to use. '
             'Default: "From Resolume".',
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available MIDI input ports and exit.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print raw MIDI bytes in hex alongside decoded messages.",
    )

    args = parser.parse_args()

    if args.list:
        list_ports()
        sys.exit(0)

    port_name = choose_port_by_name(args.port)

    print(f"Opening MIDI input port: {port_name!r}")
    print("Press Ctrl+C to stop.\n")

    try:
        with mido.open_input(port_name) as inport:
            # mido's input port is iterable; this blocks until a message arrives.
            for msg in inport:
                print_message(msg, raw=args.raw)
    except KeyboardInterrupt:
        print("\nStopping listener. Goodbye.")
    except Exception as e:
        print(f"Error opening or reading from MIDI port '{port_name}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
