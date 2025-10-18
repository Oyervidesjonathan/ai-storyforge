# backend/app.py
import os
from typing import List, Optional
from fastapi import FastAPI, Body
from pydantic import BaseModel

APP_NAME = os.getenv("APP_NAME", "AI StoryForge")
app = FastAPI(title=APP_NAME)

class StoryRequest(BaseModel):
    prompt: str
    age_range: Optional[str] = "4-7"
    style: Optional[str] = "bedtime, friendly"
    chapters: int = 1

class StoryResponse(BaseModel):
    title: str
    chapters: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"service": APP_NAME, "docs": "/docs", "health": "/health"}

@app.post("/stories", response_model=StoryResponse)
def generate_story(req: StoryRequest = Body(...)):
    """
    Placeholder story generator.
    Replace with real LLM calls later. Must return fast for Railway health checks.
    """
    title = f"{req.prompt.strip().title()} (A {req.age_range} Story)"
    chapters = [f"Chapter {i+1}: This is a placeholder for '{req.style}' style."
                for i in range(max(1, int(req.chapters)))]
    return StoryResponse(title=title, chapters=chapters)
