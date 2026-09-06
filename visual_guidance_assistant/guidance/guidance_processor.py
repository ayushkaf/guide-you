"""GuidanceProcessing thread: converts tracks into speech messages."""
from __future__ import annotations

import threading
import queue
import logging
import time
from typing import List

from guidance.scene_engine import SceneEngine


class GuidanceProcessor(threading.Thread):
    def __init__(self, tracking_queue: "queue.Queue", speech_queue: "queue.Queue", config: dict, display_messages: list = None, display_lock: threading.Lock = None) -> None:
        super().__init__(name="GuidanceProcessor", daemon=True)
        self.log = logging.getLogger("guidance.processor")
        self.tracking_queue = tracking_queue
        self.speech_queue = speech_queue
        self.display_messages = display_messages
        self.display_lock = display_lock
        self.engine = SceneEngine(config)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        print("DEBUG: GuidanceProcessor started")
        while not self._stop_event.is_set():
            try:
                tracked = self.tracking_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            print(f"DEBUG: Got {len(tracked)} tracked objects")

            try:
                self.engine.process(tracked)

                any_msg = False
                while True:
                    try:
                        msg = self.engine.message_queue.pop()
                        if msg is None:
                            break
                    except:
                        break

                    any_msg = True
                    print(f"DEBUG: Sending to speech: {msg}")

                    try:
                        self.speech_queue.put_nowait(msg)
                    except queue.Full:
                        print("DEBUG: Speech queue FULL, dropping oldest")
                        try:
                            _ = self.speech_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.speech_queue.put_nowait(msg)
                        except queue.Full:
                            print("DEBUG: Still full, message dropped!")

                    if self.display_messages is not None and self.display_lock is not None:
                        with self.display_lock:
                            self.display_messages.append(f"{time.strftime('%H:%M:%S')} {msg}")
                            if len(self.display_messages) > 10:
                                self.display_messages.pop(0)

                if not any_msg:
                    print("DEBUG: No messages produced by engine")

            except Exception:
                self.log.exception("Error while processing tracked objects")