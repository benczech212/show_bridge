#!/usr/bin/env python3
"""
Tiny CLI helper to:

- Create a Spout receiver
- Enumerate available Spout senders
- Let you choose one by index
- Lock the receiver to that sender

This is intentionally minimal: it just prints sender names and
confirms the chosen one. You can bolt on actual frame receiving
once this part is working.
"""

import sys
import time

try:
    # Depending on the binding you installed, this might be:
    #   from SpoutGL import SpoutReceiver
    #   from spout import SpoutReceiver
    from SpoutGL import SpoutReceiver  # adjust if needed
except ImportError:
    print("Could not import SpoutReceiver. Check that your Spout Python binding is installed.")
    sys.exit(1)


def list_senders(receiver) -> list[str]:
    """
    Try to get a list of sender names from whatever Spout binding is present.
    Supports a couple of common APIs:

      - receiver.getSenderNames() -> list[str]
      - receiver.getSenderCount(), receiver.getSenderName(i) -> str
    """
    # Preferred: single call returning list of names
    if hasattr(receiver, "getSenderNames"):
        names = receiver.getSenderNames()
        # Some bindings return bytes; normalize to str
        names = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                 for n in names]
        return names

    # Fallback: count + indexed access
    if hasattr(receiver, "getSenderCount") and hasattr(receiver, "getSenderName"):
        try:
            count = int(receiver.getSenderCount())
        except Exception:
            count = 0
        names: list[str] = []
        for i in range(count):
            n = receiver.getSenderName(i)
            if isinstance(n, (bytes, bytearray)):
                n = n.decode("utf-8", errors="ignore")
            names.append(str(n))
        return names

    raise RuntimeError(
        "SpoutReceiver does not expose getSenderNames() or "
        "getSenderCount()/getSenderName(). Check the binding docs."
    )


def choose_sender_interactively(names: list[str]) -> str:
    """
    Print the list of senders and let the user pick one by number.
    """
    if not names:
        raise RuntimeError("No Spout senders found. Is Resolume sending Spout?")

    if len(names) == 1:
        print(f"Only one sender found, auto-selecting: {names[0]!r}")
        return names[0]

    print("Available Spout senders:")
    for idx, name in enumerate(names):
        print(f"  [{idx}] {name}")

    while True:
        raw = input("Select sender index: ").strip()
        try:
            i = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue

        if 0 <= i < len(names):
            return names[i]
        print(f"Index out of range. Choose between 0 and {len(names) - 1}.")


def main():
    print("[SPOUT] Creating receiver...")
    receiver = SpoutReceiver()

    print("[SPOUT] Enumerating senders...")
    names = list_senders(receiver)

    if not names:
        print("No Spout senders detected. Make sure Resolume is outputting Spout.")
        return

    chosen = choose_sender_interactively(names)
    print(f"[SPOUT] Chosen sender: {chosen!r}")

    # Different bindings configure the receiver in different ways.
    # Common patterns:
    #   receiver.setReceiverName(chosen)
    #   receiver.setSenderName(chosen)
    #   receiver.createReceiver(chosen, width, height)
    #
    # We'll try a couple of common names and log what we’re doing.

    if hasattr(receiver, "setReceiverName"):
        print("[SPOUT] Using receiver.setReceiverName(...)")
        receiver.setReceiverName(chosen)
    elif hasattr(receiver, "setSenderName"):
        print("[SPOUT] Using receiver.setSenderName(...)")
        receiver.setSenderName(chosen)
    else:
        print(
            "[SPOUT] NOTE: I don't know how this binding selects the sender.\n"
            "Check its docs for the appropriate method (e.g. createReceiver)."
        )

    print("\n[SPOUT] Sender selection done.")
    print("At this point you can add a receive loop, e.g. receiver.receiveImage() or similar.")

    # Optional tiny sleep so the OS has a moment to log things before exit
    time.sleep(0.5)


if __name__ == "__main__":
    main()
