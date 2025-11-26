#!/usr/bin/env python3
"""
List MIDI input devices, let the user select one,
then print all incoming messages.

Dependencies:
    pip install mido python-rtmidi
"""

import sys
import time

import mido


def choose_input_port() -> str:
    """List all available MIDI input ports and let user choose one."""
    input_names = mido.get_input_names()

    if not input_names:
        print("No MIDI input ports found.")
        sys.exit(1)

    print("Available MIDI input ports:")
    for idx, name in enumerate(input_names):
        print(f"  [{idx}] {name}")

    while True:
        try:
            choice = input("\nSelect input port number: ").strip()
            idx = int(choice)
            if 0 <= idx < len(input_names):
                port_name = input_names[idx]
                print(f"\nSelected input: {port_name}\n")
                return port_name
            else:
                print(f"Please enter a number between 0 and {len(input_names) - 1}.")
        except ValueError:
            print("Please enter a valid integer.")


def main():
    # Optional: explicitly use the rtmidi backend if needed:
    # mido.set_backend('mido.backends.rtmidi')

    port_name = choose_input_port()

    print("Opening MIDI input port. Press Ctrl+C to quit.\n")

    # Using `open_input` as a context manager ensures clean close.
    with mido.open_input(port_name) as inport:
        try:
            while True:
                # non-blocking, returns an iterator of pending messages
                for msg in inport.iter_pending():
                    # Raw message print
                    print(msg)

                    # If you want more structured info for buttons only:
                    # if msg.type in ("note_on", "note_off"):
                    #     print(
                    #         f"Type={msg.type}, "
                    #         f"channel={msg.channel}, "
                    #         f"note={msg.note}, "
                    #         f"velocity={msg.velocity}"
                    #     )

                # tiny sleep so we don't peg the CPU
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nExiting on user request.")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            print("MIDI input closed.")


if __name__ == "__main__":
    main()
