"""Dialogue manager for generating responses."""
from guidance.message_constructor import build_scene_summary


class DialogueManager:
    def __init__(self, config):
        self.config = config

    def generate_response(self, text, scene_context):
        if not text:
            return None
        
        text_lower = text.lower().strip()

        # What do you see?
        if any(w in text_lower for w in ["what do you see", "what's around", "describe", "what is around"]):
            if scene_context:
                return build_scene_summary(scene_context)
            return "I don't see anything right now."

        # Where is X?
        if "where is" in text_lower or "where's" in text_lower:
            target = text_lower.replace("where is", "").replace("where's", "").replace("my", "").strip()
            for zone, items in scene_context.items():
                for item in items:
                    name = item if isinstance(item, str) else item.get("class_name", "")
                    if target in name.lower():
                        zone_text = "in front of you" if zone == "center" else f"on your {zone}"
                        return f"Your {target} is {zone_text}."
            return f"I don't see a {target}."

        # Is there X?
        if "is there" in text_lower:
            target = text_lower.replace("is there", "").replace("a", "").replace("an", "").strip()
            for zone, items in scene_context.items():
                for item in items:
                    name = item if isinstance(item, str) else item.get("class_name", "")
                    if target in name.lower():
                        zone_text = "in front of you" if zone == "center" else f"on your {zone}"
                        return f"Yes, there is a {target} {zone_text}."
            return f"No, I don't see a {target}."

        # What's in front?
        if any(w in text_lower for w in ["what's in front", "what is in front"]):
            items = scene_context.get("center", [])
            if items:
                names = [i if isinstance(i, str) else i.get("class_name", "") for i in items]
                return f"In front of you: {', '.join(names)}."
            return "Nothing in front of you."

        # Greeting
        if any(w in text_lower for w in ["hello", "hi", "hey"]):
            return "Hello! I'm your visual assistant. Ask me what's around you."

        # Goodbye
        if any(w in text_lower for w in ["bye", "goodbye"]):
            return "Goodbye! Take care."

        return "I didn't understand. Try: what do you see?"