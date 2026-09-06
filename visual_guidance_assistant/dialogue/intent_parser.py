import re

def parse_intent(text: str) -> dict:
    text = text.lower().strip()
    # what do you see / describe
    if re.search(r"what.*see|what.*around|describe|what.*room", text):
        return {"intent": "what_do_you_see", "target": None, "zone": None}
    # where is ...
    m = re.search(r"where is (my |the )?(?P<target>\w+)", text)
    if m:
        return {"intent": "where_is", "target": m.group("target"), "zone": None}
    # is there ...
    m = re.search(r"is there (a |an |the )?(?P<target>\w+)", text)
    if m:
        return {"intent": "is_there", "target": m.group("target"), "zone": None}
    # what's in front/left/right
    if re.search(r"what.*front", text):
        return {"intent": "whats_in_front", "target": None, "zone": "center"}
    if re.search(r"what.*left", text):
        return {"intent": "whats_in_front", "target": None, "zone": "left"}
    if re.search(r"what.*right", text):
        return {"intent": "whats_in_front", "target": None, "zone": "right"}
    # greeting
    if re.search(r"hello|hi|hey", text):
        return {"intent": "greeting", "target": None, "zone": None}
    # goodbye
    if re.search(r"bye|goodbye|see you", text):
        return {"intent": "goodbye", "target": None, "zone": None}
    # fallback
    return {"intent": "unknown", "target": None, "zone": None}
