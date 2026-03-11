from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None
    history: Optional[List[Dict[str, Any]]] = None


class SentimentResult(BaseModel):
    score: float
    mood: str


class ChatResponse(BaseModel):
    sentiment: SentimentResult
    bot_reply: str
    recommendations: list = []
