"""
APC40 MK2 LED palette, color matching, SysEx helpers, and grid layout.

Outbound LED protocol summary:
  Clip launch grid (notes 0x00–0x27):
    - Note = row * 8 + col  (row 0 = top, col 0 = left/group 1)
    - MIDI channel = animation type (0=solid, 6-10=pulsing, 11-15=blinking)
    - Velocity = color from the 128-color palette (0=off)

  Track strip (notes 0x30–0x34):
    - MIDI channel = track 0-7
    - Velocity = 0 (off) or 1+ (on)

  SysEx init must be sent before any LED commands.
"""

from __future__ import annotations

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Animation channel constants (MIDI channel in outbound clip launch Note On)
# ---------------------------------------------------------------------------
ANIM_SOLID    = 0   # primary color, solid
ANIM_PULSE_24 = 6   # pulsing 1/24
ANIM_PULSE_16 = 7   # pulsing 1/16
ANIM_PULSE_8  = 8   # pulsing 1/8  ← default autopilot indicator
ANIM_PULSE_4  = 9   # pulsing 1/4
ANIM_PULSE_2  = 10  # pulsing 1/2
ANIM_BLINK_24 = 11  # blinking 1/24
ANIM_BLINK_16 = 12  # blinking 1/16
ANIM_BLINK_8  = 13  # blinking 1/8
ANIM_BLINK_4  = 14  # blinking 1/4
ANIM_BLINK_2  = 15  # blinking 1/2

# Autopilot animation: 1/4-note pulse = 1 pulse per beat, synced via MIDI clock
AUTOPILOT_ANIM = ANIM_PULSE_4

# ---------------------------------------------------------------------------
# 128-color palette: (velocity, r, g, b)
# Source: APC40 Mk2 Communications Protocol v1.2
# ---------------------------------------------------------------------------
_PALETTE_RAW: list[tuple[int, int, int, int]] = [
    (0,   0x00, 0x00, 0x00),
    (1,   0x1E, 0x1E, 0x1E),
    (2,   0x7F, 0x7F, 0x7F),
    (3,   0xFF, 0xFF, 0xFF),
    (4,   0xFF, 0x4C, 0x4C),
    (5,   0xFF, 0x00, 0x00),
    (6,   0x59, 0x00, 0x00),
    (7,   0x19, 0x00, 0x00),
    (8,   0xFF, 0xBD, 0x6C),
    (9,   0xFF, 0x54, 0x00),
    (10,  0x59, 0x1D, 0x00),
    (11,  0x27, 0x1B, 0x00),
    (12,  0xFF, 0xFF, 0x4C),
    (13,  0xFF, 0xFF, 0x00),
    (14,  0x59, 0x59, 0x00),
    (15,  0x19, 0x19, 0x00),
    (16,  0x88, 0xFF, 0x4C),
    (17,  0x54, 0xFF, 0x00),
    (18,  0x1D, 0x59, 0x00),
    (19,  0x14, 0x2B, 0x00),
    (20,  0x4C, 0xFF, 0x4C),
    (21,  0x00, 0xFF, 0x00),
    (22,  0x00, 0x59, 0x00),
    (23,  0x00, 0x19, 0x00),
    (24,  0x4C, 0xFF, 0x5E),
    (25,  0x00, 0xFF, 0x19),
    (26,  0x00, 0x59, 0x0D),
    (27,  0x00, 0x19, 0x02),
    (28,  0x4C, 0xFF, 0x88),
    (29,  0x00, 0xFF, 0x55),
    (30,  0x00, 0x59, 0x1D),
    (31,  0x00, 0x1F, 0x12),
    (32,  0x4C, 0xFF, 0xB7),
    (33,  0x00, 0xFF, 0x99),
    (34,  0x00, 0x59, 0x35),
    (35,  0x00, 0x19, 0x12),
    (36,  0x4C, 0xC3, 0xFF),
    (37,  0x00, 0xA9, 0xFF),
    (38,  0x00, 0x41, 0x52),
    (39,  0x00, 0x10, 0x19),
    (40,  0x4C, 0x88, 0xFF),
    (41,  0x00, 0x55, 0xFF),
    (42,  0x00, 0x1D, 0x59),
    (43,  0x00, 0x08, 0x19),
    (44,  0x4C, 0x4C, 0xFF),
    (45,  0x00, 0x00, 0xFF),
    (46,  0x00, 0x00, 0x59),
    (47,  0x00, 0x00, 0x19),
    (48,  0x87, 0x4C, 0xFF),
    (49,  0x54, 0x00, 0xFF),
    (50,  0x19, 0x00, 0x64),
    (51,  0x0F, 0x00, 0x30),
    (52,  0xFF, 0x4C, 0xFF),
    (53,  0xFF, 0x00, 0xFF),
    (54,  0x59, 0x00, 0x59),
    (55,  0x19, 0x00, 0x19),
    (56,  0xFF, 0x4C, 0x87),
    (57,  0xFF, 0x00, 0x54),
    (58,  0x59, 0x00, 0x1D),
    (59,  0x22, 0x00, 0x13),
    (60,  0xFF, 0x15, 0x00),
    (61,  0x99, 0x35, 0x00),
    (62,  0x79, 0x51, 0x00),
    (63,  0x43, 0x64, 0x00),
    (64,  0x03, 0x39, 0x00),
    (65,  0x00, 0x57, 0x35),
    (66,  0x00, 0x54, 0x7F),
    (67,  0x00, 0x00, 0xFF),
    (68,  0x00, 0x45, 0x4F),
    (69,  0x25, 0x00, 0xCC),
    (70,  0x7F, 0x7F, 0x7F),
    (71,  0x20, 0x20, 0x20),
    (72,  0xFF, 0x00, 0x00),
    (73,  0xBD, 0xFF, 0x2D),
    (74,  0xAF, 0xED, 0x06),
    (75,  0x64, 0xFF, 0x09),
    (76,  0x10, 0x8B, 0x00),
    (77,  0x00, 0xFF, 0x87),
    (78,  0x00, 0xA9, 0xFF),
    (79,  0x00, 0x2A, 0xFF),
    (80,  0x3F, 0x00, 0xFF),
    (81,  0x7A, 0x00, 0xFF),
    (82,  0xB2, 0x1A, 0x7D),
    (83,  0x40, 0x21, 0x00),
    (84,  0xFF, 0x4A, 0x00),
    (85,  0x88, 0xE1, 0x06),
    (86,  0x72, 0xFF, 0x15),
    (87,  0x00, 0xFF, 0x00),
    (88,  0x3B, 0xFF, 0x26),
    (89,  0x59, 0xFF, 0x71),
    (90,  0x38, 0xFF, 0xCC),
    (91,  0x5B, 0x8A, 0xFF),
    (92,  0x31, 0x51, 0xC6),
    (93,  0x87, 0x7F, 0xE9),
    (94,  0xD3, 0x1D, 0xFF),
    (95,  0xFF, 0x00, 0x5D),
    (96,  0xFF, 0x7F, 0x00),
    (97,  0xB9, 0xB0, 0x00),
    (98,  0x90, 0xFF, 0x00),
    (99,  0x83, 0x5D, 0x07),
    (100, 0x39, 0x2B, 0x00),
    (101, 0x14, 0x4C, 0x10),
    (102, 0x0D, 0x50, 0x38),
    (103, 0x15, 0x15, 0x2A),
    (104, 0x16, 0x20, 0x5A),
    (105, 0x69, 0x3C, 0x1C),
    (106, 0xA8, 0x00, 0x0A),
    (107, 0xDE, 0x51, 0x3D),
    (108, 0xD8, 0x6A, 0x1C),
    (109, 0xFF, 0xE1, 0x26),
    (110, 0x9E, 0xE1, 0x2F),
    (111, 0x67, 0xB5, 0x0F),
    (112, 0x1E, 0x1E, 0x30),
    (113, 0xDC, 0xFF, 0x6B),
    (114, 0x80, 0xFF, 0xBD),
    (115, 0x9A, 0x99, 0xFF),
    (116, 0x8E, 0x66, 0xFF),
    (117, 0x40, 0x40, 0x40),
    (118, 0x75, 0x75, 0x75),
    (119, 0xE0, 0xFF, 0xFF),
    (120, 0xA0, 0x00, 0x00),
    (121, 0x35, 0x00, 0x00),
    (122, 0x1A, 0xD0, 0x00),
    (123, 0x07, 0x42, 0x00),
    (124, 0xB9, 0xB0, 0x00),
    (125, 0x3F, 0x31, 0x00),
    (126, 0xB3, 0x5F, 0x00),
    (127, 0x4B, 0x15, 0x02),
]

# velocity → (r, g, b) lookup table (index = velocity)
PALETTE: list[tuple[int, int, int]] = [(0, 0, 0)] * 128
for _v, _r, _g, _b in _PALETTE_RAW:
    PALETTE[_v] = (_r, _g, _b)


def nearest_velocity(r: int, g: int, b: int, exclude_off: bool = True) -> int:
    """Return the palette velocity whose color is closest (Euclidean RGB distance)."""
    best_vel = 0
    best_dist = float("inf")
    start = 1 if exclude_off else 0
    for vel in range(start, 128):
        pr, pg, pb = PALETTE[vel]
        dist = math.sqrt((r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_vel = vel
    return best_vel


def dim_velocity(vel: int, factor: float = 0.30) -> int:
    """Return the nearest palette velocity for `vel` scaled down to `factor` brightness.

    Used to set the LED primary (dim floor) before enabling the pulsing animation
    so the pulse sweeps dim→bright instead of off→bright.
    """
    if vel == 0:
        return 0
    r, g, b = PALETTE[vel]
    return nearest_velocity(int(r * factor), int(g * factor), int(b * factor))


def hex_to_velocity(hex_color: str) -> int:
    """Convert a '#RRGGBB' hex string to the nearest palette velocity."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return nearest_velocity(r, g, b)


# ---------------------------------------------------------------------------
# CSS / common color name → velocity (for clip names like "Red", "Orange")
# ---------------------------------------------------------------------------
_NAME_TO_HEX: dict[str, str] = {
    "red":     "#FF0000",
    "orange":  "#FF8800",
    "yellow":  "#FFFF00",
    "lime":    "#88FF00",
    "green":   "#00FF00",
    "teal":    "#00FF88",
    "cyan":    "#00FFFF",
    "sky":     "#00AAFF",
    "blue":    "#0000FF",
    "indigo":  "#4400FF",
    "violet":  "#8800FF",
    "purple":  "#AA00FF",
    "magenta": "#FF00FF",
    "pink":    "#FF0088",
    "rose":    "#FF0055",
    "white":   "#FFFFFF",
    "grey":    "#808080",
    "gray":    "#808080",
    "black":   "#000000",
    "amber":   "#FFAA00",
    "gold":    "#FFD700",
    "coral":   "#FF5533",
    "mint":    "#00FF88",
    "aqua":    "#00FFCC",
    "lavender":"#AA88FF",
    "maroon":  "#880000",
    "olive":   "#888800",
    "navy":    "#000088",
}


def color_name_to_velocity(name: str) -> Optional[int]:
    """
    Map a color name (e.g. from a Resolume clip name) to the nearest palette
    velocity.  Returns None if the name is not recognized or is "rainbow"
    (handled separately by the cycling thread).
    """
    key = name.strip().lower()
    if key == "rainbow":
        return None   # special mode — caller must check for "rainbow" explicitly
    # exact match
    if key in _NAME_TO_HEX:
        return hex_to_velocity(_NAME_TO_HEX[key])
    # partial match: "Dark Red" → check if any known color is a substring
    for known, hex_val in _NAME_TO_HEX.items():
        if known in key:
            return hex_to_velocity(hex_val)
    return None


# Rainbow hue cycle — 8 saturated palette entries spanning the visible spectrum.
# Advance one step per tick (~150 ms) so a full cycle takes ~1.2 s.
RAINBOW_VELOCITIES: list[int] = [
    5,    # red     #FF0000
    9,    # orange  #FF5400
    13,   # yellow  #FFFF00
    21,   # green   #00FF00
    37,   # sky     #00A9FF
    45,   # blue    #0000FF
    49,   # purple  #5400FF
    53,   # magenta #FF00FF
]


# ---------------------------------------------------------------------------
# Grid layout helpers
# ---------------------------------------------------------------------------
NUM_ROWS = 5
NUM_GROUPS = 8  # columns


def grid_note(row: int, group_idx: int) -> int:
    """
    Compute the MK2 note number for a clip launch button.

    row       : 0-4 top-to-bottom  (row 0 = colors, row 4 = playing)
    group_idx : 0-7 left-to-right  (0 = group 1)
    """
    return row * NUM_GROUPS + group_idx


# Row index for each role
ROLE_ROW: dict[str, int] = {
    "colors":     0,
    "effects":    1,
    "transforms": 2,
    "masks":      3,
    "playing":    4,
}

# Fixed LED velocities for non-colors roles (colors is dynamic)
ROLE_ON_VELOCITY: dict[str, int] = {
    "effects":    9,   # orange   #FF5400
    "transforms": 37,  # cyan     #00A9FF
    "masks":      49,  # purple   #5400FF
    "playing":    21,  # green    #00FF00
}

# Track strip note numbers (inbound and outbound)
STRIP_NOTE: dict[str, int] = {
    "colors":     0x30,  # 48  ARM
    "effects":    0x31,  # 49  SOLO
    "transforms": 0x32,  # 50  ACTIVATOR
    "masks":      0x33,  # 51  TRACK SELECT
    "playing":    0x34,  # 52  CLIP STOP
}

STRIP_SLIDER_CC = 0x07   # track fader
MASTER_FADER_CC = 0x0E   # 14

# Global button notes (all channel 0)
GLOBAL_NOTE: dict[str, int] = {
    "master":          0x50,  # 80
    "stop_all_clips":  0x51,  # 81
    "scene_1":         0x52,  # 82
    "scene_2":         0x53,  # 83
    "scene_3":         0x54,  # 84
    "scene_4":         0x55,  # 85
    "scene_5":         0x56,  # 86
    "pan":             0x57,  # 87
    "sends":           0x58,  # 88
    "user":            0x59,  # 89
    "metronome":       0x5A,  # 90
    "play":            0x5B,  # 91
    "stop":            0x5C,  # 92
    "record":          0x5D,  # 93
    "up":              0x5E,  # 94
    "down":            0x5F,  # 95
    "right":           0x60,  # 96
    "left":            0x61,  # 97
    "shift":           0x62,  # 98
    "tap_tempo":       0x63,  # 99
    "nudge_minus":     0x64,  # 100
    "nudge_plus":      0x65,  # 101
    "session_record":  0x66,  # 102
}


# ---------------------------------------------------------------------------
# SysEx
# ---------------------------------------------------------------------------
_SYSEX_HEADER = [0x47, 0x7F, 0x29]


def make_sysex_init(mode: int = 0x42) -> list[int]:
    """
    Build the MK2 Introduction SysEx payload (without F0 / F7 framing).

    mode:
        0x40 = Generic (device controls some LEDs)
        0x41 = Ableton Live (host controls most LEDs)
        0x42 = Alternate Ableton Live (host controls ALL LEDs) ← recommended

    Returns the full byte list to pass to mido.Message("sysex", data=...).
    mido adds the F0/F7 framing automatically.
    """
    return _SYSEX_HEADER + [0x60, 0x00, 0x04, mode, 0x01, 0x00, 0x00]
