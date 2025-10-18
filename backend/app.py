# backend/app.py
import os, time, uuid
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os


SERVICE_NAME = "AI StoryForge"
APP_VERSION = "0.2.0"

# ---------- OpenAI client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")

client: Optional[OpenAI]
if not OPENAI_API_KEY:
    client = None  # /health still works; story endpoints will 500 with clear message
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- FastAPI ----------
app = FastAPI(
    title=SERVICE_NAME,
    version=APP_VERSION,
    description="Generate children's stories and illustrations.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

# ---------- Serve static frontend ----------
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/ui")
def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


# ---------- Schemas ----------
class StoryRequest(BaseModel):
    prompt: constr(strip_whitespace=True, min_length=3) = Field(..., example="A shy firefly who learns to shine with new friends")
    age_range: constr(strip_whitespace=True) = Field("4-7", example="4-7")
    style: constr(strip_whitespace=True) = Field("bedtime, friendly", example="bedtime, friendly")
    chapters: int = Field(1, ge=1, le=10, description="Number of chapters")

class StoryResponse(BaseModel):
    title: str
    chapters: List[str]

# NEW: brief item for listing
class StoryItem(BaseModel):
    id: str
    title: str
    chapters_count: int
    created_at: int  # unix seconds

class ImageRequest(BaseModel):
    prompt: constr(strip_whitespace=True, min_length=3)
    aspect: constr(strip_whitespace=True) = Field("square", description="square|portrait|landscape")

class ImageResponse(BaseModel):
    image_url: str

class BookRequest(BaseModel):
    story: StoryRequest
    images: bool = True

class BookResponse(BaseModel):
    id: str
    title: str
    chapters: List[str]
    image_urls: Optional[List[str]] = None

# ---------- In-memory store (simple) ----------
# id -> {"title": str, "chapters": List[str], "created_at": int}
STORIES: Dict[str, Dict] = {}

# ---------- Helpers ----------
def _title_from_prompt(prompt: str, age_range: str) -> str:
    return f"{prompt.strip().capitalize()} (A {age_range} Story)"

def _chapter_prompts(base_prompt: str, style: str, chapters: int) -> List[str]:
    return [
        f"Write Chapter {i+1} of a children's story.\n"
        f"Base premise: {base_prompt}\n"
        f"Style: {style}. Tone: wholesome, simple words.\n"
        f"Length: 120-200 words. End with a gentle hook unless it’s final chapter."
        for i in range(chapters)
    ]

def _image_prompt_from_chapter(title: str, chapter_text: str, style_hint: str) -> str:
    return (
        f"Children's book illustration for: '{title}'. "
        f"Scene summary: {chapter_text[:240]}. "
        f"Style: {style_hint}. Ultra kid-friendly, bright, cozy, no text overlay."
    )

def _image_size(aspect: str) -> str:
    a = aspect.lower()
    if a.startswith("port"): return "1024x1344"
    if a.startswith("land"): return "1344x1024"
    return "1024x1024"

def _require_client():
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured in Railway Variables.")

# ---------- Routes ----------
@app.get("/", tags=["Root"])
def root():
    return {"service": SERVICE_NAME, "docs": "/docs", "health": "/health"}

@app.get("/health", tags=["Root"])
def health():
    return {"ok": True, "ts": int(time.time())}

# NEW: list stories (fixes 405)
@app.get("/stories", response_model=List[StoryItem], tags=["Stories"])
def list_stories():
    items: List[StoryItem] = []
    for i, data in STORIES.items():
        items.append(StoryItem(
            id=i,
            title=data["title"],
            chapters_count=len(data["chapters"]),
            created_at=data["created_at"],
        ))
    # newest first
    items.sort(key=lambda x: x.created_at, reverse=True)
    return items

# NEW: get one story by id
@app.get("/stories/{story_id}", response_model=StoryResponse, tags=["Stories"])
def get_story(story_id: str):
    data = STORIES.get(story_id)
    if not data:
        raise HTTPException(status_code=404, detail="Not found")
    return StoryResponse(title=data["title"], chapters=data["chapters"])

@app.post("/stories", response_model=StoryResponse, tags=["Stories"])
def generate_story(req: StoryRequest):
    _require_client()

    title = _title_from_prompt(req.prompt, req.age_range)
    chapter_prompts = _chapter_prompts(req.prompt, req.style, req.chapters)

    chapters: List[str] = []
    try:
        for cp in chapter_prompts:
            resp = client.chat.completions.create(
                model=OPENAI_TEXT_MODEL,
                temperature=0.9,
                messages=[
                    {"role": "system", "content":
                        "You are a children’s author. Write warm, age-appropriate prose. "
                        "Avoid scary content, complex vocabulary, and brand names."
                    },
                    {"role": "user", "content": cp},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            chapters.append(text if text else "…")
    except Exception as e:
        # Return 500 (so the proxy doesn't map it as a 502 Bad Gateway)
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # NEW: persist so GET endpoints work
    story_id = str(uuid.uuid4())
    STORIES[story_id] = {"title": title, "chapters": chapters, "created_at": int(time.time())}

    return StoryResponse(title=title, chapters=chapters)

@app.post("/images", response_model=ImageResponse, tags=["Images"])
def generate_image(req: ImageRequest):
    _require_client()
    prompt = req.prompt.strip()
    try:
        img = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size=_image_size(req.aspect),
            quality="high",
        )
        url = img.data[0].url
        return ImageResponse(image_url=url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image error: {e}")

@app.post("/books", response_model=BookResponse, tags=["Books"])
def generate_book(req: BookRequest):
    # 1) story
    story = generate_story(req.story)
    # 2) images (optional)
    urls: Optional[List[str]] = None
    if req.images:
        urls = []
        for ch in story.chapters:
            iprompt = _image_prompt_from_chapter(story.title, ch, req.story.style)
            url = generate_image(ImageRequest(prompt=iprompt, aspect="square")).image_url
            urls.append(url)

    book_id = str(uuid.uuid4())
    return BookResponse(id=book_id, title=story.title, chapters=story.chapters, image_urls=urls)
