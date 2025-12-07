STILL TO WORKING ON

Pressing global_8 will toggle a new variable "Auto Tempo Tap". If Auto Tempo Tap is OFF, then set the button to OFF. if ON, set the button to ON. Listen for OSC commands Touch Designer 
--------------------
Abstract the state machine logic into a yaml config for easy rewiring / changing of the mapping / actions. 
- file that maps logical actions to OSC commands and/or MIDI commands.



========================================================


Hitting Autopilot On when already on triggers global autopilot advance

Set master slider to Composition opacity
Set horizontal_slider slider to scroll H
Set vertical_slider slider to scroll V
set global_3 to global_previous_clip
set global_4 to global_next_clip


if can't connect to midi controller, show warning, wait 5s try agian, if it fails again try every 10 seconds to reconnect.

Tripple tapping any layer role in any group will that layer roll and set all other layers of the same role in other groups to the same setting. IE tripple tapping an OFF color button sets all groups color buttons to ON. Tripple tapping an ON transform button sets all groups transforms to OFF


Add an OSC out feed that shows the state changes of the state machine. Use something like 

/show_controller/v1/state/group/<group_index>>/<layer_role>/clip_name string
/show_controller/v1/state/group/<group_index>>/<layer_role>/status bool or 0/1
/show_controller/v1/state/group/<group_index>>/<layer_role>/autopilot bool or 0/1
/show_controller/v1/state/group/<group_index>>/opacity float 0-1
/show_controller/v1/state/group/<group_index>>/intensity float 0-1
/show_controller/v1/state/global/opacity float 0-1
/show_controller/v1/state/global/autopilot bool or 0/1
any others I'm not thinking of
