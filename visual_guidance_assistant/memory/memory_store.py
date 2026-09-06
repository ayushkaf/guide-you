"""Memory store for alarms, reminders, notes, and conversations.

IMPORTANT: construct exactly ONE of these per process and inject it into
everything that needs it. Each instance keeps its own in-memory list and writes
the whole list back to disk on every save, so two instances against the same
file silently overwrite each other's work — that is what used to make voice-set
alarms never fire and reminders disappear.
"""
import json
import os
import time
import threading

from utils.paths import MEMORY_DATA_FILE, ensure_parent_dir


class MemoryStore:
    def __init__(self, filepath=None):
        self.filepath = filepath or MEMORY_DATA_FILE
        self.lock = threading.Lock()
        ensure_parent_dir(self.filepath)
        self.memories = []
        self.load()
    
    def add_alarm(self, message, trigger_time):
        with self.lock:
            self.memories.append({
                "type": "alarm",
                "message": message,
                "trigger_time": trigger_time,
                "created": time.time()
            })
            self.save()
    
    def add_reminder(self, message):
        with self.lock:
            self.memories.append({
                "type": "reminder",
                "message": message,
                "trigger_time": None,
                "created": time.time()
            })
            self.save()
    
    def get_due_alarms(self):
        now = time.time()
        due = []
        with self.lock:
            for mem in self.memories:
                if mem["type"] == "alarm" and mem["trigger_time"] and mem["trigger_time"] <= now:
                    due.append(mem)
            for mem in due:
                self.memories.remove(mem)
            if due:
                self.save()
        return due
    
    def get_all(self):
        with self.lock:
            return list(self.memories)
    
    def add_conversation(self, user_text, ai_response, profile=None):
        """`profile` is whoever was recognised as speaking (current_speaker),
        or None if nobody was. Every conversation used to land in this one
        list with no attribution at all — every enrolled person's exchanges
        mixed together, unreadable as "whose memory is this". Tagged the same
        way LayeredMemory already tags its layer entries, for the same reason:
        one shared file, isolated by a field per entry, not a file per person.
        """
        with self.lock:
            self.memories.append({
                "type": "conversation",
                "user": user_text,
                "ai": ai_response,
                "profile": profile,
                "created": time.time()
            })
            convs = [m for m in self.memories if m["type"] == "conversation"]
            if len(convs) > 50:
                self.memories.remove(convs[0])
            self.save()

    def get_recent_conversations(self, limit=10, profile=None):
        """With `profile` given: that person's own conversations, plus any
        recorded before a speaker was recognised (profile=None or missing —
        legacy entries from before this field existed load the same way).
        Mirrors LayeredMemory._items_for's visibility rule exactly, so a
        caregiver's exchanges never show up inside the patient's memory card."""
        with self.lock:
            convs = [m for m in self.memories if m["type"] == "conversation"]
            if profile:
                wanted = profile.lower()
                convs = [m for m in convs
                         if not m.get("profile")
                         or str(m.get("profile", "")).lower() == wanted]
            return convs[-limit:]
    
    def add_greeting(self, greeting_type):
        with self.lock:
            self.memories.append({
                "type": "greeting",
                "message": greeting_type,
                "created": time.time()
            })
            self.save()
    
    def get_last_greeting_time(self):
        with self.lock:
            greetings = [m for m in self.memories if m["type"] == "greeting"]
            if greetings:
                return greetings[-1]["created"]
            return 0
    
    def save(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, indent=2)

    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)