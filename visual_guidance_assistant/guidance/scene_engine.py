import time
from typing import List, Dict
from guidance.cooldown_manager import CooldownManager
from guidance.priority_queue import PriorityMessageQueue


class SceneEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.cooldown_manager = CooldownManager(config)
        self.message_queue = PriorityMessageQueue()

    def process(self, tracked_objects: List[Dict]) -> None:
        # Silent mode - LLM handles all conversation
        pass