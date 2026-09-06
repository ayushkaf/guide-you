"""Morning routine handler."""
import time
import threading


class MorningRoutine(threading.Thread):
    def __init__(self, memory_store, speech_queue):
        super().__init__(daemon=True)
        self.memory = memory_store
        self.speech_queue = speech_queue
        self.running = False

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            now = time.localtime()
            if now.tm_hour == 7 and now.tm_min == 0:
                last_greeting = self.memory.get_last_greeting_time()
                hours_since = (time.time() - last_greeting) / 3600 if last_greeting > 0 else 999
                if hours_since > 8:
                    msg = "Good morning! How was your sleep?"
                    try:
                        self.speech_queue.put_nowait(msg)
                    except:
                        pass
                    self.memory.add_greeting("morning_alarm")
            time.sleep(60)