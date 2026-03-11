import json
import random
from pathlib import Path

_DATA_PATH = Path(__file__).parent.parent / "data" / "responses.json"

with open(_DATA_PATH, "r", encoding="utf-8") as f:
    RESPONSES: dict = json.load(f)


def generate_response(mood: str) -> str:
    if mood not in RESPONSES:
        mood = "NEUTRAL"
    return random.choice(RESPONSES[mood])
