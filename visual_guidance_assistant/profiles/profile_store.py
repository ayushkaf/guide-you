"""JSON-backed store of enrolled people.

Replaces the previous known_faces/encodings.pkl. Two reasons the pickle had to
go: unpickling executes arbitrary code, so a tampered file becomes remote code
execution; and a pickle is opaque, which is the wrong property for a file
recording a vulnerable person's biometrics. JSON is safe to load and anyone can
open it and read exactly what has been stored about them.

One record per person:

    {
      "name":            "Aayush",
      "role":            "patient" | "caregiver",
      "relationship":    "daughter" | null,      # caregivers only, optional
      "contact_note":    "mobile 555-0100" | null,
      "face_encoding":   [128 floats] | null,
      "voice_embedding": [N floats]   | null,
      "has_been_introduced": false,   # patients only — see mark_introduced()
      "archived": false,              # excluded from live matching — see mark_archived()
      "created": <epoch>, "updated": <epoch>
    }
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from utils import secure_store
from utils.paths import LEGACY_ENCODINGS_FILE, PROFILES_FILE, ensure_parent_dir

ROLE_PATIENT = "patient"
ROLE_CAREGIVER = "caregiver"

# Someone the person knows — a daughter, a neighbour, a friend — enrolled so the
# assistant can greet them by name and refer to them warmly. NOT a system user.
#
# This is deliberately a third role rather than a flag on caregiver, because the
# access boundary matters: a known person is usually enrolled from a PHOTO, which
# is the weakest identity claim in the system (a photo can be of anyone, shown by
# anyone, without that person present). So this role grants nothing. It is never
# returned by caregivers(), never satisfies the caregiver checks that gate the
# alert log, and resolves to the gentler patient interaction mode. All it does is
# let the assistant say "hello Sarah" instead of "hello".
ROLE_KNOWN_PERSON = "known_person"

# Whoever set the device up: a family member configuring it, or a developer
# testing it. Everything a caregiver can do, plus the ability to switch
# interaction mode on demand for testing.
ROLE_ADMIN = "admin"

VALID_ROLES = (ROLE_PATIENT, ROLE_CAREGIVER, ROLE_KNOWN_PERSON, ROLE_ADMIN)

# Roles that identify someone as a user OF the system, with an interaction mode.
SYSTEM_ROLES = (ROLE_PATIENT, ROLE_CAREGIVER, ROLE_ADMIN)

# Roles cleared for the caregiver-only features: the alert log read-back, and
# anything else gated on being trusted with the supported person's private
# moments. Checked in one place so a fourth role cannot be added later while
# quietly missing one of the gates.
CARE_ACCESS_ROLES = (ROLE_CAREGIVER, ROLE_ADMIN)

SCHEMA_VERSION = 2


class ProfileStore:
    """Single source of truth for who the assistant knows.

    Construct ONE per process and inject it, for the same reason MemoryStore is
    injected: each instance holds the full list in memory and rewrites the whole
    file on save, so two instances would silently clobber each other.
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or PROFILES_FILE
        self._lock = threading.RLock()
        self._profiles: List[Dict[str, Any]] = []
        # How many are known to be on disk. Guards against blanking the file —
        # see save().
        self._loaded_count = 0
        ensure_parent_dir(self.filepath)
        self.load()

    # ---------- persistence ----------

    def load(self) -> None:
        with self._lock:
            if not os.path.exists(self.filepath):
                self._profiles = []
                self._migrate_legacy_pickle()
                return
            try:
                data = secure_store.read_json(self.filepath, default={})
                self._profiles = data.get("profiles", []) if isinstance(data, dict) else []
                self._loaded_count = len(self._profiles)
            except secure_store.DecryptionError as exc:
                # The file is intact but belongs to another account or machine.
                # Keep running with nobody enrolled rather than taking the whole
                # device down — but be unmistakable about why.
                print(f"[profiles] CANNOT DECRYPT PROFILES: {exc}")
                print("[profiles] continuing with no enrolled people this session")
                self._profiles = []
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[profiles] WARNING: could not read {self.filepath}: {exc}")
                print("[profiles] continuing with no enrolled people this session")
                self._profiles = []

    def save(self, allow_empty: bool = False) -> None:
        """Persist. Refuses to blank a populated store unless told to.

        Enrolling a face takes a person sitting down and being photographed
        several times; erasing it takes one stray assignment. That asymmetry
        bit for real — a batch of test scripts each ended with
        `store._profiles = []; store.save()` against the DEFAULT path, so every
        run silently destroyed whoever had been enrolled, and it looked from
        the outside like enrolment was not persisting at all.

        Clearing on purpose is still one call: clear(). This only blocks the
        accidental version.
        """
        with self._lock:
            if not self._profiles and self._loaded_count and not allow_empty:
                print(f"[profiles] REFUSING to overwrite {self._loaded_count} "
                      f"enrolled profile(s) with an empty list. If this is "
                      f"deliberate, call clear() or save(allow_empty=True).")
                return
            payload = {
                "version": SCHEMA_VERSION,
                "updated": time.time(),
                "profiles": self._profiles,
            }
            # Encrypted at rest and written atomically, so an interrupted write
            # cannot leave a truncated profile file behind.
            secure_store.write_json(
                self.filepath, payload, description="Guide YOU enrolled profiles"
            )
            # What is now on disk, so a later empty save is judged against the
            # real file rather than against whatever was there at startup.
            self._loaded_count = len(self._profiles)

    def clear(self) -> None:
        """Deliberately remove everyone. The explicit form of an empty save."""
        with self._lock:
            removed = len(self._profiles)
            self._profiles = []
            self.save(allow_empty=True)
            print(f"[profiles] cleared {removed} profile(s) deliberately")

    def _migrate_legacy_pickle(self) -> None:
        """One-time, read-only look at the old pickle.

        The single entry in the existing file was enrolled under an empty name
        (the old console-input() bug), which cannot be matched to anyone and
        cannot be repaired — there is no way to recover whose face it is. It is
        discarded, as instructed. This runs once; the pickle is never written.
        """
        if not os.path.exists(LEGACY_ENCODINGS_FILE):
            return
        print(f"[profiles] found legacy pickle: {LEGACY_ENCODINGS_FILE}")
        try:
            import pickle  # imported here so the module isn't loaded in normal operation

            with open(LEGACY_ENCODINGS_FILE, "rb") as handle:
                legacy = pickle.load(handle)
            names = legacy.get("names", []) or []
            encodings = legacy.get("encodings", []) or []
        except Exception as exc:
            print(f"[profiles] could not read legacy pickle ({exc}); skipping migration")
            return

        migrated = discarded = 0
        for name, encoding in zip(names, encodings):
            clean = (name or "").strip()
            if not clean:
                discarded += 1
                continue
            self._profiles.append(
                _new_record(
                    name=clean,
                    # Role was not captured by the old flow. Patient is the safe
                    # assumption — see resolve_role() for why.
                    role=ROLE_PATIENT,
                    face_encoding=[float(v) for v in encoding],
                )
            )
            migrated += 1

        print(f"[profiles] migration: {migrated} profile(s) carried over, "
              f"{discarded} unusable (empty-name) entry discarded")
        self.save()
        print(f"[profiles] wrote {self.filepath}; the pickle is no longer read or written")

    # ---------- queries ----------

    def all_profiles(self) -> List[Dict[str, Any]]:
        """Every profile, archived or not. This is the raw truth of what's on
        disk — used by the admin profile audit (which deliberately shows
        archived entries, marked as such) and by mark_archived()/mark_
        introduced() style lookups by exact name, which need to find a
        profile regardless of its archived state to be able to act on it.

        Nothing that can end up in SPOKEN output should call this directly —
        use active_profiles() (or one of names()/caregivers()/known_people()/
        admins()/system_users(), which already do) instead. An archived
        profile keeping its full data on disk is the point of archiving
        rather than deleting; it surfacing again in an ordinary conversation
        is the exact bug that not filtering here caused once already.
        """
        with self._lock:
            return [dict(p) for p in self._profiles]

    def active_profiles(self) -> List[Dict[str, Any]]:
        """all_profiles() minus anything archived — the view every live,
        spoken-facing feature should be built on. See mark_archived()."""
        return [p for p in self.all_profiles() if not p.get("archived")]

    def names(self) -> List[str]:
        with self._lock:
            return [p["name"] for p in self._profiles if not p.get("archived")]

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        if not name:
            return None
        with self._lock:
            for profile in self._profiles:
                if profile["name"].lower() == name.lower():
                    return dict(profile)
        return None

    def role_of(self, name: str) -> Optional[str]:
        profile = self.get(name)
        return profile.get("role") if profile else None

    def mark_introduced(self, name: str) -> bool:
        """Record that this person has now heard the first-meeting
        introduction, so it is never spoken to them again. One-way: there is
        no un-introduce, by design — a repeated lengthy self-introduction
        would read as the assistant not recognising someone it already knows,
        which is the opposite of reassuring."""
        with self._lock:
            for profile in self._profiles:
                if profile["name"].lower() == (name or "").strip().lower():
                    profile["has_been_introduced"] = True
                    profile["updated"] = time.time()
                    self.save()
                    return True
        return False

    def mark_archived(self, name: str, archived: bool = True) -> bool:
        """Archive (or restore) a profile WITHOUT deleting anything.

        An archived profile keeps its full record on disk — face samples,
        voice, role, relationship, everything — but is excluded from
        face_index()/voice_index(), so it can never again be compared against
        during live recognition. That is the one thing deleting a profile and
        archiving one have in common; everything else about the data stays.

        Exists for the case where two profiles turn out to be the same real
        face (or, in principle, the same voice) enrolled twice: live matching
        then always finds a rival within FACE_MATCH_MARGIN of the correct
        match, and refuses to answer confidently — correctly, since from the
        matcher's point of view two profiles that close together genuinely
        are ambiguous. Removing the account of one of the two duplicate
        enrolments from the comparison, without touching what's recorded
        about the person, is what fixes that without deleting anyone's data.

        Two-way on purpose: mark_archived(name, False) restores a profile to
        full live matching, unlike mark_introduced()'s one-way flag.
        """
        with self._lock:
            for profile in self._profiles:
                if profile["name"].lower() == (name or "").strip().lower():
                    profile["archived"] = bool(archived)
                    profile["updated"] = time.time()
                    self.save()
                    return True
        return False

    def caregivers(self) -> List[Dict[str, Any]]:
        """Exact match on ROLE_CAREGIVER only. A known_person is never a carer,
        however they were enrolled — see ROLE_KNOWN_PERSON. Archived excluded:
        this feeds who gets offered as an emergency contact by name."""
        return [p for p in self.active_profiles() if p.get("role") == ROLE_CAREGIVER]

    def known_people(self) -> List[Dict[str, Any]]:
        """People to greet by name, who are not users of the system.
        Archived excluded — this is read straight into spoken output."""
        return [p for p in self.active_profiles() if p.get("role") == ROLE_KNOWN_PERSON]

    def admins(self) -> List[Dict[str, Any]]:
        """Archived excluded — see caregivers()."""
        return [p for p in self.active_profiles() if p.get("role") == ROLE_ADMIN]

    @staticmethod
    def has_care_access(role: Optional[str]) -> bool:
        """Cleared for caregiver-only features. Admin counts; known_person and
        patient never do."""
        return role in CARE_ACCESS_ROLES

    def system_users(self) -> List[Dict[str, Any]]:
        """Archived excluded — see caregivers()."""
        return [p for p in self.active_profiles() if p.get("role") in SYSTEM_ROLES]

    @staticmethod
    def interaction_role(role: Optional[str]) -> str:
        """Which conversational mode a stored role maps to.

        admin maps to caregiver: direct, full-information answers, not the
        gentle patient style.

        known_person maps to patient — the gentler mode — for the same reason an
        unrecognised speaker does: it is the conservative choice when the claim
        is weak, and photo enrolment is the weakest claim here. Only an explicit
        caregiver or admin enrolment gets caregiver mode.
        """
        return ROLE_CAREGIVER if role in CARE_ACCESS_ROLES else ROLE_PATIENT

    @staticmethod
    def _encodings_of(profile: Dict[str, Any]) -> List[List[float]]:
        """Every face encoding held for one person.

        Enrolment stores several samples under "face_encodings". Records written
        before that (and the singular "face_encoding" field) still load, so
        nobody has to re-enrol because the shape changed.
        """
        many = profile.get("face_encodings") or []
        one = profile.get("face_encoding")
        if one and not many:
            return [one]
        return list(many)

    def face_index(self) -> tuple[List[str], List[List[float]]]:
        """Parallel name/encoding lists, in the shape the face matcher wants.

        A person enrolled from several angles contributes several entries under
        the same name, so the matcher takes the best of them. That is how the
        same-person variation (measured up to 0.75 on this camera) is absorbed
        without loosening the match tolerance and risking wrong identifications.

        Archived profiles are skipped entirely — see mark_archived(). Their
        encodings still exist in the record, they are just never handed to the
        matcher, so an archived duplicate can no longer be the rival that makes
        a genuine match come back ambiguous.
        """
        with self._lock:
            names, encodings = [], []
            for profile in self._profiles:
                if profile.get("archived"):
                    continue
                for encoding in self._encodings_of(profile):
                    names.append(profile["name"])
                    encodings.append(encoding)
            return names, encodings

    def voice_index(self) -> tuple[List[str], List[List[float]]]:
        """Same archived exclusion as face_index(), and for the identical
        reason — voice matching has its own ambiguity-margin tie-break
        (voice_id/voice_embedding.py:best_match), which an archived duplicate
        would defeat in exactly the same way if left in."""
        with self._lock:
            names, embeddings = [], []
            for profile in self._profiles:
                if profile.get("archived"):
                    continue
                embedding = profile.get("voice_embedding")
                if embedding:
                    names.append(profile["name"])
                    embeddings.append(embedding)
            return names, embeddings

    # ---------- mutation ----------

    def upsert(
        self,
        name: str,
        role: str,
        relationship: Optional[str] = None,
        contact_note: Optional[str] = None,
        face_encoding: Optional[List[float]] = None,
        voice_embedding: Optional[List[float]] = None,
        face_encodings: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """Create or update a person. Re-enrolling the same name updates that
        record rather than adding a duplicate, so 'remember my face' stays a
        single, repeatable command."""
        clean = (name or "").strip()
        if not clean:
            raise ValueError("a profile must have a name")
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")

        with self._lock:
            for profile in self._profiles:
                if profile["name"].lower() == clean.lower():
                    profile["name"] = clean
                    profile["role"] = role
                    if relationship is not None:
                        profile["relationship"] = relationship or None
                    if contact_note is not None:
                        profile["contact_note"] = contact_note or None
                    if face_encodings is not None:
                        profile["face_encodings"] = _clean_encodings(face_encodings)
                        profile.pop("face_encoding", None)
                    elif face_encoding is not None:
                        profile["face_encodings"] = [[float(v) for v in face_encoding]]
                        profile.pop("face_encoding", None)
                    if voice_embedding is not None:
                        profile["voice_embedding"] = [float(v) for v in voice_embedding]
                    profile["updated"] = time.time()
                    self.save()
                    return dict(profile)

            if face_encodings is not None:
                stored = _clean_encodings(face_encodings)
            elif face_encoding is not None:
                stored = [[float(v) for v in face_encoding]]
            else:
                stored = None
            record = _new_record(
                name=clean,
                role=role,
                relationship=relationship,
                contact_note=contact_note,
                face_encodings=stored,
                voice_embedding=[float(v) for v in voice_embedding] if voice_embedding else None,
            )
            self._profiles.append(record)
            self.save()
            return dict(record)

    def set_voice_embedding(self, name: str, embedding: List[float]) -> bool:
        with self._lock:
            for profile in self._profiles:
                if profile["name"].lower() == (name or "").strip().lower():
                    profile["voice_embedding"] = [float(v) for v in embedding]
                    profile["updated"] = time.time()
                    self.save()
                    return True
        return False

    def set_azure_person_id(self, name: str, person_id: str) -> bool:
        """Records which Azure Face PersonGroup Person this profile mirrors
        to, once recognition/azure_face_client.py has synced it. Purely
        informational metadata — role/access decisions never read this
        field; it only lets the migration sweep and identify() avoid
        re-creating an Azure Person that already exists for someone."""
        with self._lock:
            for profile in self._profiles:
                if profile["name"].lower() == (name or "").strip().lower():
                    profile["azure_person_id"] = person_id
                    profile["updated"] = time.time()
                    self.save()
                    return True
        return False


def _clean_encodings(encodings: List[List[float]]) -> List[List[float]]:
    return [[float(v) for v in enc] for enc in (encodings or []) if enc is not None]


def _new_record(
    name: str,
    role: str,
    relationship: Optional[str] = None,
    contact_note: Optional[str] = None,
    face_encoding: Optional[List[float]] = None,
    voice_embedding: Optional[List[float]] = None,
    face_encodings: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    now = time.time()
    if face_encodings is None and face_encoding is not None:
        face_encodings = [[float(v) for v in face_encoding]]
    return {
        "name": name,
        "role": role,
        "relationship": relationship or None,
        "contact_note": contact_note or None,
        "face_encodings": face_encodings,
        "voice_embedding": voice_embedding,
        "has_been_introduced": False,
        "archived": False,
        # Set by recognition/azure_face_client.py once this person is
        # mirrored into Azure's PersonGroup — see set_azure_person_id().
        "azure_person_id": None,
        "created": now,
        "updated": now,
    }
