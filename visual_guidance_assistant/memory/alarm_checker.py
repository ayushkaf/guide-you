"""Thread that checks for due alarms."""
import threading
import time

class AlarmChecker(threading.Thread):
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
            due = self.memory.get_due_alarms()
            for alarm in due:
                msg = f"Alarm! {alarm['message']}"
                try:
                    self.speech_queue.put_nowait(msg)
                except:
                    pass
            time.sleep(10)
