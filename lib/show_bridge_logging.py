from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any

import yaml

LOG_DIR = Path("logs")
OSC_LOG_DIR = LOG_DIR / "osc"

# Default color mappings (ANSI color codes)
DEFAULT_LEVEL_COLORS = {
    "DEBUG": "36",
    "INFO": "32",
    "WARNING": "33",
    "ERROR": "31",
    "CRITICAL": "35",
}

DEFAULT_PROGRAM_COLORS = {
    "ARENA": "34",
    "WIRE": "35",
    "SYN": "36",
    "BRIDGE": "37",
    "TD": "32",
}

DEFAULT_CATEGORY_COLORS = {
    "OSC": "36",
    "INIT": "35",
    "MIDI": "33",
    "HTTP": "34",
    "STATE": "37",
    "MONITOR": "33",
}

DEFAULT_CATEGORY_PALETTE = [
    "90",
    "91",
    "92",
    "93",
    "94",
    "95",
    "96",
    "37",
]


def _load_formatting_cfg(path: Path | str = Path("settings/log_formatting.yaml")) -> Dict[str, Any]:
    try:
        p = Path(path)
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {}


class ShowBridgeFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str | None = None, color: bool = False, formatting_cfg: Dict[str, Any] | None = None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.color = color

        cfg = formatting_cfg or _load_formatting_cfg()

        self.LEVEL_COLORS = cfg.get("level_colors", DEFAULT_LEVEL_COLORS)
        self.PROGRAM_COLORS = cfg.get("program_colors", DEFAULT_PROGRAM_COLORS)
        self.CATEGORY_COLORS = cfg.get("category_colors", DEFAULT_CATEGORY_COLORS)
        self.CATEGORY_PALETTE = cfg.get("category_palette", DEFAULT_CATEGORY_PALETTE)

    @staticmethod
    def _colorize(text: str, color_code: str | None) -> str:
        if not color_code:
            return text
        return f"\x1b[{color_code}m{text}\x1b[0m"

    def _color_for_category(self, category: str) -> str:
        if not category:
            return "37"
        idx = abs(hash(category)) % len(self.CATEGORY_PALETTE)
        return self.CATEGORY_PALETTE[idx]

    def format(self, record: logging.LogRecord) -> str:
        # Best-effort: extract bracket token from message into program/category
        try:
            if isinstance(record.msg, str):
                import re

                m = re.match(r"^\s*\[([A-Za-z0-9_-]+)\]\s*(.*)$", record.msg)
                if m:
                    token = m.group(1).upper()
                    remainder = m.group(2)
                    if token in self.PROGRAM_COLORS:
                        if (not hasattr(record, "program") or record.program is None or str(record.program).strip() == "-"):
                            record.program = token
                    else:
                        token_main = token.split("-")[0]
                        if token_main in self.CATEGORY_COLORS or token in self.CATEGORY_COLORS:
                            if (not hasattr(record, "category") or record.category is None or str(record.category).strip() in ("-", "LOG")):
                                record.category = token_main if token_main in self.CATEGORY_COLORS else token
                        else:
                            if (not hasattr(record, "category") or record.category is None or str(record.category).strip() in ("-", "LOG")):
                                record.category = token
                    record.msg = remainder
        except Exception:
            pass

        if not hasattr(record, "program") or record.program is None:
            record.program = "BRIDGE"
        if not hasattr(record, "category") or record.category is None:
            record.category = "LOG"

        if not self.color:
            return super().format(record)

        orig_levelname = record.levelname
        orig_program = record.program
        orig_category = record.category

        try:
            level_color = self.LEVEL_COLORS.get(record.levelname, None)
            record.levelname = self._colorize(record.levelname, level_color)

            program = str(orig_program).upper()
            category = str(orig_category).upper()

            prog_color = self.PROGRAM_COLORS.get(program, None)
            cat_color = self.CATEGORY_COLORS.get(category, None)
            if not cat_color:
                cat_color = self._color_for_category(category)

            record.program = self._colorize(program, prog_color)
            record.category = self._colorize(category, cat_color)

            return super().format(record)
        finally:
            record.levelname = orig_levelname
            record.program = orig_program
            record.category = orig_category


def setup_logging(level: int = logging.INFO, formatting_cfg_path: str | Path | None = None) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    OSC_LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatting_cfg = None
    if formatting_cfg_path is not None:
        try:
            formatting_cfg = _load_formatting_cfg(formatting_cfg_path)
        except Exception:
            formatting_cfg = None

    logger = logging.getLogger("show_bridge")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt_str = "%(asctime)s [%(levelname)s] [%(program)s] [%(category)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    console_fmt = ShowBridgeFormatter(fmt_str, datefmt=datefmt, color=True, formatting_cfg=formatting_cfg)
    file_fmt = ShowBridgeFormatter(fmt_str, datefmt=datefmt, color=False, formatting_cfg=formatting_cfg)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    master_path = LOG_DIR / "show_bridge.log"
    fh = RotatingFileHandler(master_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    osc_loggers = {
        "osc.resolume.out": OSC_LOG_DIR / "resolume_out.log",
        "osc.resolume.in": OSC_LOG_DIR / "resolume_in.log",
        "osc.synesthesia.out": OSC_LOG_DIR / "synesthesia_out.log",
        "osc.show_bridge.out": OSC_LOG_DIR / "show_bridge_out.log",
        "osc.show_bridge.in": OSC_LOG_DIR / "show_bridge_in.log",
    }

    for name, path in osc_loggers.items():
        l = logging.getLogger(name)
        l.setLevel(level)
        fh2 = RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh2.setLevel(level)
        fh2.setFormatter(file_fmt)
        l.addHandler(fh2)
        l.propagate = True

    return logger


def sb_log(logger: logging.Logger, level: int, program: str, category: str, msg: str, *args, **kwargs) -> None:
    extra = kwargs.pop("extra", {})
    extra.setdefault("program", program)
    extra.setdefault("category", category)
    logger.log(level, msg, *args, extra=extra, **kwargs)
