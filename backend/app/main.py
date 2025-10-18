# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import uuid

app = FastAPI(title="AI StoryForge", version="1.0.0")

# --- logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("storyforge")

# --- CORS (adjust origins for your frontend URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas
class StoryRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="What the story should be about")
    length: Optional[int] = Field(300, ge=50, le=5000, description="Target length in words")
    style: Optional[str] = Field("default", description="e.g., 'noir', 'children', 'sci-fi'")

class StoryResponse(BaseModel):
    id: str
    prompt: str
    text: str
    length: int
    style: str

# --- in-memory store (swap for DB as needed)
DB: dict[str, StoryResponse] = {}

@app.get("/", tags=["meta"])
def root():
    return {"service": "AI StoryForge", "docs": "/docs", "health": "/health"}

@app.get("/health", tags=["meta"])
def health():
    return {"ok": True}

@app.get("/stories", response_model=List[StoryResponse], tags=["stories"])
def list_stories():
    return list(DB.values())

@app.get("/stories/{story_id}", response_model=StoryResponse, tags=["stories"])
def get_story(story_id: str):
    if story_id not in DB:
        raise HTTPException(status_code=404, detail="Not found")
    return DB[story_id]

@app.post("/stories", response_model=StoryResponse, status_code=201, tags=["stories"])
def create_story(req: StoryRequest):
    try:
        # TODO: replace with your real generator call (LLM, etc.)
        # Keep it quick to avoid 502 timeouts.
        body = f"Once upon a time... {req.prompt}\n\n(This is a {req.style} story ~{req.length} words.)"
        story = StoryResponse(
            id=str(uuid.uuid4()),
            prompt=req.prompt,
            text=body,
            length=req.length or len(body.split()),
            style=req.style or "default",
        )
        DB[story.id] = story
        log.info("story created id=%s", story.id)
        return story
    except Exception as e:
        log.exception("story creation failed")
        raise HTTPException(status_code=500, detail=f"Story generation failed: {e}")
