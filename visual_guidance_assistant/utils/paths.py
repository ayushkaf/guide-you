"""Filesystem paths anchored to the package, not the current working directory.

Every data path in this project used to be relative ("memory/...",
"known_faces/..."), so launching the app from anywhere other than
visual_guidance_assistant/ silently created a *second*, empty set of memory and
face-encoding files instead of loading the real ones. Anchoring to this file's
own location makes the app behave identically regardless of where it's started.
"""
from __future__ import annotations

import os

# .../visual_guidance_assistant  (this file lives in .../visual_guidance_assistant/utils/)
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(*parts: str) -> str:
    """Absolute path to a file or directory inside the package."""
    return os.path.join(PACKAGE_ROOT, *parts)


def ensure_parent_dir(path: str) -> str:
    """Create the containing directory for `path` if needed; return `path`."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


# Canonical locations for the data this app persists.
CONFIG_FILE = resource_path("config", "settings.yaml")
MEMORY_DIR = resource_path("memory")
LAYERED_MEMORY_FILE = resource_path("memory", "layered_memory.json")
MEMORY_DATA_FILE = resource_path("memory", "memory_data.json")
KNOWN_FACES_DIR = resource_path("known_faces")
# Legacy pickle store. Only ever read once, during migration — never written.
# Unpickling executes arbitrary code, which is not an acceptable risk for a file
# holding biometric data for a vulnerable person.
LEGACY_ENCODINGS_FILE = resource_path("known_faces", "encodings.pkl")
# Current store: plain JSON, safe to load and readable by a human who wants to
# audit exactly what has been recorded about them.
PROFILES_FILE = resource_path("known_faces", "profiles.json")
DEBUG_DIR = resource_path("debug")
ALERTS_DIR = resource_path("alerts")
ALERT_LOG_FILE = resource_path("alerts", "alert_log.json")

# Downloaded once by fetch_models.py, never at runtime. Absent is a supported
# state: the app warns and falls back to the HOG detector.
# Daily routine prompts. Config is meant to be edited by a carer, so it is
# plain readable JSON and lives beside settings.yaml rather than in code.
ROUTINES_FILE = resource_path("config", "routines.json")
# Which routines have already fired today. Separate from the config so editing
# one never disturbs the other, and so a restart cannot replay the day.
ROUTINE_STATE_FILE = resource_path("config", "routines_state.json")

MODELS_DIR = resource_path("models")
YUNET_MODEL_FILE = resource_path("models", "face_detection_yunet_2023mar.onnx")
