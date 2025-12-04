# connections_ext  (minimal smoke test)

import os

try:
    import yaml
except ImportError:
    yaml = None
    debug("ConnectionsExt: PyYAML not available; will return empty config.")


class ConnectionsExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self._cache = None
        self._cache_path = None

    @property
    def path(self):
        """
        Resolve the YAML path from a parameter called 'Connectionsfile'
        on the owner component. If it's empty, default to project.folder/connections.yaml
        """
        par = getattr(self.ownerComp.par, 'Connectionsfile', None)
        if par is not None:
            raw = par.eval().strip()
            if raw:
                # allow ./relative/path style
                if raw.startswith('.'):
                    return os.path.normpath(os.path.join(project.folder, raw))
                return os.path.normpath(raw)

        # fallback
        return os.path.normpath(os.path.join(project.folder, 'connections.yaml'))

    def _load_yaml(self):
        """Internal: load YAML file from disk."""
        if yaml is None:
            debug("ConnectionsExt: yaml module not available.")
            return {}

        path = self.path
        if not os.path.isfile(path):
            debug("ConnectionsExt: config not found: {}".format(path))
            return {}

        try:
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            debug("ConnectionsExt: error loading YAML: {}".format(e))
            return {}

        return cfg

    @property
    def data(self):
        """Cached YAML as a dict."""
        if self._cache is None or self._cache_path != self.path:
            self._cache = self._load_yaml()
            self._cache_path = self.path
        return self._cache or {}

    def reload(self):
        """Force reload from disk."""
        self._cache = None
        self._cache_path = None
        return self.data

    # Tiny debug helper
    def TopKeys(self):
        return list(self.data.keys())
