# show_bridge
A tool for bridging various show software such as Resolume Arena / Touch Designer / Midi Devices, etc.




GROUPS
    The group has properties for Opacity (float) and Intensity (float). 
    Opacity is controlled by Controller 7 / channel 1-8 (depending on group)
    Intensity is controlled by Notes 53-56. 53 = 100%, 54 = 75%, 55 = 50%, 56 = 25%. Short press sets intensity to that value. Long press on any intensity button sets intensity to 0%. double press on any sets random intensity.

    There are a number of groups (8). Each group controls a number of layers roles including:
    - Fill Layers (Controlled by Clip Stop / note 52)
    - Color Layers (Controlled by Arm note 48)
    - Effect Layers (Controlled by Solo / queue note 49)
    - Transform Layers (Controlled by Activator note 50)
    - Dynamic Mask Layers (Controlled by clip_row_1  / note 57)

    For each layer in a group, there is a boolean controling playing on/off and autopilot on/off. There is also a property for current_clip_index and last_clip_index. If the layer is turned ON from OFF, it should play the current_clip_index. If the layer is set to play NEXT, it should use a function to consider intensity and use rng to decide if it should choose passthrough or content. If content, choose a random valid clip index above 2. If passthrough, choose column index 2. If the layer is turned OFF, choose column index 1, but don't save it as the current_clip_index. Only save current clip index if it is passthrough or content.
    
GLOBAL
    Global Opacity (Controller 14 Channel 1) OSC /composition/master
    Global Scroll (Controller 15 / Channel 0) OSC /application/ui/clipsscrollhorizontal
    Global Autopilot ON (Note 91 / Channel 0) 
    Global Autopilot Off (Note 92 / channel 0)
    Global I like this (Note 93 / channel 0) 
    Global Fill Layers (Stop All Clips Note 81 / channel 1) 
    Scene Launch 1-5 (Save / replay state / Note 82-86 channel 1)
    Tempo Tap (Note 99 / channel 0)
    Shift / Resync (Note 98 / channel 0)
    Nudge - (Note 101 / channel 0)
    Nudge + (Note 100 / channel 0)
    Tempo - (Note 95 / channel 0)
    Tempo + (Note 94 / channel 0)
    Tempo double (Note 96 / channel 0)
    Tempo half (Note 97 / channel 0)
