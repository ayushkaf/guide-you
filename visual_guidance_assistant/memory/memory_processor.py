"""Processes memory-related voice commands."""
import time
import re
from memory.memory_store import MemoryStore


class MemoryProcessor:
    def __init__(self, memory_store: MemoryStore):
        # Injected, never constructed here — a second MemoryStore against the
        # same file would overwrite alarms and reminders written by the first.
        self.memory = memory_store

    def process_command(self, text, scene_context=None):
        if not isinstance(text, str):
            return None
            
        text_lower = text.lower()

        alarm_match = re.search(r'(?:wake me up|set alarm|alarm)\s+(?:at\s+)?(\d{1,2})\s*(am|pm|o\'?clock)?', text_lower)
        if alarm_match:
            hour = int(alarm_match.group(1))
            ampm = alarm_match.group(2)
            if ampm and 'pm' in ampm and hour != 12:
                hour += 12
            elif ampm and 'am' in ampm and hour == 12:
                hour = 0

            now = time.localtime()
            trigger = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hour, 0, 0, now.tm_wday, now.tm_yday, now.tm_isdst))
            if trigger < time.time():
                trigger += 86400

            self.memory.add_alarm(f"Wake up! It's {hour}:00", trigger)
            return f"Alarm set for {hour}:00."

        remind_match = re.search(r'remind me to (.+)', text_lower)
        if remind_match:
            message = remind_match.group(1)
            self.memory.add_reminder(message)
            return f"I'll remind you to {message}."

        if "any alarm" in text_lower or "my alarms" in text_lower:
            all_items = self.memory.get_all()
            alarms = [m for m in all_items if m['type'] == 'alarm']
            if alarms:
                times = [time.strftime('%H:%M', time.localtime(a['trigger_time'])) for a in alarms]
                return f"You have alarms at: {', '.join(times)}."
            return "You have no alarms set."

        return None