"""At-rest encryption for the files holding personal data.

Uses Windows DPAPI (CryptProtectData) with the key bound to the current Windows
user account. Nothing to remember, nothing to type, nothing to lose.

WHY NOT A STARTUP PASSPHRASE
A passphrase prompt at launch would give stronger confidentiality, but it makes
the device fail closed: after a power cut, a crash, or a Windows update reboot,
the app would sit at a prompt the person it supports cannot answer — so the help
command, distress detection, and the entire safety net would be dead until a
caregiver arrived. That is exactly the window where they matter. Availability is
part of safety here, so the key is derived from the machine instead of a human.

WHAT THIS PROTECTS AGAINST
  - the laptop being lost, stolen, or its drive removed
  - these files being copied elsewhere, or synced into a cloud backup
  - another Windows account on the same machine reading them
WHAT IT DOES NOT PROTECT AGAINST
  - someone already signed in to this Windows account
That is the honest boundary of a device that has to start unattended.

THE REAL TRADE-OFF: the encryption key belongs to this Windows user account on
this machine. Reinstalling Windows, moving to a new computer, or switching to a
different user account makes these files permanently unreadable and everyone
must be enrolled again. See README.md.

Files stay JSON and self-describing so they can still be identified and
inspected; only the payload is ciphertext.
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Optional

from utils.paths import ensure_parent_dir

SCHEME = "windows-dpapi-user"

try:  # pywin32 is already a dependency (pulled in by pyttsx3)
    import win32crypt

    _DPAPI_AVAILABLE = True
except Exception:  # pragma: no cover - non-Windows or missing pywin32
    win32crypt = None
    _DPAPI_AVAILABLE = False


class DecryptionError(Exception):
    """Raised when an envelope exists but cannot be decrypted on this machine."""


def encryption_available() -> bool:
    return _DPAPI_AVAILABLE


def _encrypt(raw: bytes, description: str) -> bytes:
    return win32crypt.CryptProtectData(raw, description, None, None, None, 0)


def _decrypt(blob: bytes) -> bytes:
    # Returns (description, data)
    return win32crypt.CryptUnprotectData(blob, None, None, None, 0)[1]


def is_encrypted(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            head = json.load(handle)
        return isinstance(head, dict) and head.get("encrypted") is True
    except Exception:
        return False


def read_json(path: str, default: Any = None) -> Any:
    """Load JSON, decrypting transparently if the file is an envelope.

    Plaintext files written before encryption was enabled still load, so
    switching this on does not orphan anyone's existing profiles.
    """
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[secure-store] WARNING: could not read {path}: {exc}")
        return default

    if not (isinstance(data, dict) and data.get("encrypted") is True):
        return data  # legacy plaintext — re-saved encrypted on the next write

    if not _DPAPI_AVAILABLE:
        raise DecryptionError(
            f"{os.path.basename(path)} is encrypted but Windows DPAPI is unavailable "
            "on this system, so it cannot be read here."
        )

    try:
        blob = base64.b64decode(data["payload"])
        return json.loads(_decrypt(blob).decode("utf-8"))
    except Exception as exc:
        # Almost always: the file came from a different Windows account or a
        # different machine. Say so specifically — "decryption failed" sends
        # someone hunting for a corrupt file that is perfectly intact.
        raise DecryptionError(
            f"{os.path.basename(path)} cannot be decrypted by this Windows user "
            f"account on this machine ({type(exc).__name__}). It was almost "
            "certainly written by a different account or on a different computer. "
            "The data is not recoverable here; the people in it must be enrolled "
            "again. See README.md."
        ) from exc


def write_json(path: str, payload: Any, description: str = "Guide YOU personal data") -> None:
    """Write JSON encrypted at rest, atomically."""
    ensure_parent_dir(path)
    raw = json.dumps(payload, indent=2).encode("utf-8")

    if _DPAPI_AVAILABLE:
        envelope = {
            "encrypted": True,
            "scheme": SCHEME,
            "note": (
                "Encrypted at rest, bound to this Windows user account. Not "
                "readable on another account or machine. See README.md."
            ),
            "updated": time.time(),
            "payload": base64.b64encode(_encrypt(raw, description)).decode("ascii"),
        }
        body = json.dumps(envelope, indent=2)
    else:
        # Never lose someone's data just because encryption is unavailable —
        # write it plainly and say so loudly rather than failing the save.
        print(f"[secure-store] WARNING: DPAPI unavailable; writing {os.path.basename(path)} "
              "UNENCRYPTED")
        body = json.dumps(payload, indent=2)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.replace(tmp, path)
