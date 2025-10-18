# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging, uuid, time

app = FastAPI(title="AI StoryForge", version="1.0.0")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("storyforge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend origin if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoryRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    length: Optional[int] = Field(300, ge=50, le=5000)
    style: Optional[str] = Field("default")

class StoryResponse(BaseModel):
    id: str
    prompt: str
    text: str
    length: int
    style: str
    status: str = "ready"

DB: Dict[str, StoryResponse] = {}

@app.get("/")
def root():
    return {"service": "AI StoryForge", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"ok": True}

# ✅ Fixes the 405
@app.get("/stories", response_model=List[StoryResponse])
def list_stories():
    return list(DB.values())

@app.get("/stories/{story_id}", response_model=StoryResponse)
def get_story(story_id: str):
    if story_id not in DB:
        raise HTTPException(status_code=404, detail="Not found")
    return DB[story_id]

# ✅ Wraps errors so they don’t bubble up as 502
@app.post("/stories", response_model=StoryResponse, status_code=201)
async def create_story(req: StoryRequest, request: Request):
    ct = request.headers.get("content-type", "")
    if "application/json" not in ct:
        # Prevents 422 and gives a clear message when UI sends form-data
        raise HTTPException(status_code=415, detail="Use Content-Type: application/json")

    t0 = time.time()
    try:
        # TODO: replace with your real generator call
        body = f"Once upon a time… {req.prompt}\n\n({req.style} ~{req.length} words)"
        story = StoryResponse(
            id=str(uuid.uuid4()),
            prompt=req.prompt,
            text=body,
            length=req.length or len(body.split()),
            style=req.style or "default",
        )
        DB[story.id] = story
        log.info("POST /stories ok id=%s in %.2fs", story.id, time.time() - t0)
        return story
    except Exception as e:
        log.exception("POST /stories failed in %.2fs", time.time() - t0)
        # Return 500 (so proxy won’t show 502)
        raise HTTPException(status_code=500, detail=f"Story generation failed: {e}")
