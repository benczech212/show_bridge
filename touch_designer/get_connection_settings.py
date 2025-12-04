# connections_ext
#
# Extension for reading a shared connections.yaml file
# and exposing helpers for TouchDesigner parameters.
#
# Expected YAML structure (like your example):
#
# connections.yaml
# ----------------
# inputs:
#   http:
#     resolume_arena:
#       - name: "arena_http_in_local"
#         host: "127.0.0.1"
#         port: 8080
#         use_https: false
#         api_base: "/api/v1"
#         timeout: 2.0
#
#   osc:
#     touchdesigner:
#       - name: "td_osc_in"
#         host: "127.0.0.1"
#         port: 11001
#         direction: "input"
#
# outputs:
#   osc:
#     touchdesigner:
#       - name: "td_osc_out"
#         host: "127.0.0.1"
#         port: 11000
#         direction: "output"
#
# midi:
#   controllers:
#     - name: "Akai APC40"
#       midi_in_port: "APC40 mk1"
#       midi_out_port: "APC40 mk1"
#

import os

try:
    import yaml
except ImportError:
    yaml = None
    # You can also print to Textport:
    # debug("PyYAML not available; connections_ext will return empty config.")


class ConnectionsExt:
    """
    ConnectionsExt

    Attach this as an extension to your IO root COMP.
    It reads a YAML config file pointed to by the COMP's
    Connectionsfile parameter and provides helpers for
    HTTP/OSC/MIDI lookups.
    """

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self._cache = None  # cached dict from YAML
        self._cache_path = None

    # ----------
    # Core loading
    # ----------

    @property
    def path(self):
        """
        Resolve the path to the YAML file.

        Uses ownerComp.par.Connectionsfile if set, otherwise
        defaults to project.folder + '/connections.yaml'.
        """
        par = getattr(self.ownerComp.par, 'Connectionsfile', None)
        if par is not None and par.eval().strip():
            p = par.eval().strip()
        else:
            # Fallback to project root
            p = os.path.join(project.folder, 'connections.yaml')

        # Normalize path so caching works reliably
        return os.path.normpath(p)

    @property
    def data(self):
        """
        The parsed YAML data as a dict (cached).

        Access via: parent().ext.Connections.data
        """
        if self._cache is None or self._cache_path != self.path:
            self._cache = self._load_yaml()
            self._cache_path = self.path
        return self._cache or {}

    def reload(self):
        """
        Force a reload from disk (e.g. after editing the YAML file).
        Usage in TD: parent().ext.Connections.reload()
        """
        self._cache = None
        self._cache_path = None
        return self.data

    def _load_yaml(self):
        """
        Internal: load YAML from disk, handle errors gracefully.
        """
        if yaml is None:
            debug("ConnectionsExt: PyYAML not available; returning empty config.")
            return {}

        path = self.path
        if not os.path.isfile(path):
            debug("ConnectionsExt: config file not found: {}".format(path))
            return {}

        try:
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            debug("ConnectionsExt: error loading YAML: {}".format(e))
            return {}

        return cfg

    # ----------
    # Generic helpers
    # ----------

    def _find_osc(self, direction, group, name):
        """
        Find an OSC entry.

        direction: 'inputs' or 'outputs'
        group: e.g. 'touchdesigner', 'resolume_arena'
        name: the 'name' string in the YAML list (e.g. 'td_osc_in')

        Returns the dict or {} if not found.
        """
        cfg = self.data
        top = cfg.get(direction, {})
        osc_root = top.get('osc', {})
        group_list = osc_root.get(group, [])

        for item in group_list:
            if item.get('name') == name:
                return item

        # Not found
        return {}

    def _find_http(self, direction, group, name):
        """
        Find an HTTP entry.

        Same pattern as OSC but under 'http'.
        """
        cfg = self.data
        top = cfg.get(direction, {})
        http_root = top.get('http', {})
        group_list = http_root.get(group, [])

        for item in group_list:
            if item.get('name') == name:
                return item

        return {}

    def _find_midi_controller(self, name):
        """
        Find a MIDI controller under inputs.midi.controllers.
        """
        cfg = self.data
        inputs_root = cfg.get('inputs', {})
        midi_root = inputs_root.get('midi', {})
        controllers = midi_root.get('controllers', [])

        for item in controllers:
            if item.get('name') == name:
                return item

        return {}

    def _find_midi_output_device(self, name):
        """
        Find a MIDI output device under outputs.midi.devices.
        """
        cfg = self.data
        outputs_root = cfg.get('outputs', {})
        midi_root = outputs_root.get('midi', {})
        devices = midi_root.get('devices', [])

        for item in devices:
            if item.get('name') == name:
                return item

        return {}

    # ----------
    # OSC helpers for TD parameters
    # ----------

    def OscInPort(self, group='touchdesigner', name='td_osc_in', default=0):
        """
        Get port for an OSC input (use for OSC In CHOP).
        """
        info = self._find_osc('inputs', group, name)
        return info.get('port', default)

    def OscInHost(self, group='touchdesigner', name='td_osc_in', default='127.0.0.1'):
        """
        Get host for an OSC input.
        """
        info = self._find_osc('inputs', group, name)
        return info.get('host', default)

    def OscOutPort(self, group='touchdesigner', name='td_osc_out', default=0):
        """
        Get port for an OSC output (use for OSC Out CHOP).
        """
        info = self._find_osc('outputs', group, name)
        return info.get('port', default)

    def OscOutHost(self, group='touchdesigner', name='td_osc_out', default='127.0.0.1'):
        """
        Get host for an OSC output.
        """
        info = self._find_osc('outputs', group, name)
        return info.get('host', default)

    # ----------
    # HTTP helpers
    # ----------

    def HttpIn(self, group='resolume_arena', name='arena_http_in_local'):
        """
        Return the full HTTP input dict, or {}.
        """
        return self._find_http('inputs', group, name)

    def HttpOut(self, group='resolume_arena', name='arena_http_out_main'):
        """
        Return the full HTTP output dict, or {}.
        """
        return self._find_http('outputs', group, name)

    # ----------
    # MIDI helpers
    # ----------

    def MidiControllerInPort(self, name='Akai APC40', default=''):
        """
        Get the MIDI input port name for a controller
        (inputs.midi.controllers[].midi_in_port).
        """
        info = self._find_midi_controller(name)
        return info.get('midi_in_port', default)

    def MidiControllerOutPort(self, name='Akai APC40', default=''):
        """
        Get the MIDI output port name for a controller
        (inputs.midi.controllers[].midi_out_port).
        """
        info = self._find_midi_controller(name)
        return info.get('midi_out_port', default)

    def MidiDeviceOutPort(self, name='APC40_out', default=''):
        """
        Get the MIDI output port name from outputs.midi.devices[].
        """
        info = self._find_midi_output_device(name)
        return info.get('midi_out_port', default)
