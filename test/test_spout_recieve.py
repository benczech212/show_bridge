# test_spout_receiver.py
import os
import sys

# Make sure SpoutSDK.pyd and Spout.dll are in the same folder as this file
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# If necessary on Python 3.8+, help Windows find the DLLs
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.join(HERE, "spout", "MD", "bin"))

import SpoutSDK  # provided by the Spout-for-Python wrapper

def main():
    # This part depends on the wrapper API;
    # in Spout-for-Python it's typically something like:
    receiver = SpoutSDK.SpoutReceiver()
    receiver.setReceiverName("Resolume Spout Name")

    print("Waiting for Spout frames...")
    while True:
        tex = receiver.receiveFrame()
        if tex is not None:
            # Do something with tex (e.g., inspect size, etc.)
            print("Got frame", tex.shape)
        # add a small sleep, break condition, etc.

if __name__ == "__main__":
    main()
