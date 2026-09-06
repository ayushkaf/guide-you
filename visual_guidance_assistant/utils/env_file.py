"""Load configuration secrets from a .env file next to the project.

Why this exists: ANTHROPIC_API_KEY set with `setx` lands in the registry, but a
process only ever sees the environment block its PARENT handed it. A terminal
opened inside an editor inherits the editor's environment, and the editor
inherited Explorer's — so if any ancestor in that chain started before the key
was set, "open a new terminal" does not help. You have to restart the whole
chain, which is a miserable thing to have to know, and worse to have to
rediscover months later when the app silently stops answering.

Reading a file removes the problem entirely: the app works the same way no
matter how it was launched, and it survives reboots without anyone remembering
a setup step.

Precedence is deliberate — a real environment variable always wins, so this can
never silently override a key someone set on purpose.

The .env file is already excluded by .gitignore.
"""
from __future__ import annotations

import os
from typing import Dict, List

from utils.paths import PACKAGE_ROOT

# Checked in order; the first existing file wins. Project root first, since
# that is where people expect a .env to live.
CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(PACKAGE_ROOT), ".env"),
    os.path.join(PACKAGE_ROOT, ".env"),
]

_loaded_from: str | None = None


def _parse(text: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip one matching pair of surrounding quotes, if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env_file(verbose: bool = True) -> List[str]:
    """Read the first .env found and fill in any variables not already set.

    Returns the names of variables it supplied. Safe to call more than once.
    """
    global _loaded_from
    applied: List[str] = []

    for path in CANDIDATE_PATHS:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                values = _parse(handle.read())
        except OSError as exc:
            if verbose:
                print(f"[env] could not read {path}: {exc}")
            return applied

        for key, value in values.items():
            # An existing environment variable always wins.
            if os.environ.get(key):
                continue
            os.environ[key] = value
            applied.append(key)

        _loaded_from = path
        if verbose and applied:
            # Never print the values.
            print(f"[env] loaded {', '.join(applied)} from {path}")
        return applied

    return applied


def loaded_from() -> str | None:
    return _loaded_from
