import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
