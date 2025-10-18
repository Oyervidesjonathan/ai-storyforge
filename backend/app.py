# backend/app.py
import os, time, uuid, json, base64
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

SERVICE_NAME = "AI StoryForge"
APP_VERSION = "0.3.0"

# ---------- OpenAI client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")

client: Optional[OpenAI]
if not OPENAI_API_KEY:
    client = None
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

# ---------- Static mounts ----------
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# serve all persisted files under backend/data -> /media/...
DATA_ROOT = Path(__file__).parent / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=DATA_ROOT), name="media")

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

class StoryItem(BaseModel):
    id: str
    title: str
    chapters_count: int
    created_at: int  # unix seconds

class ImageRequest(BaseModel):
    prompt: constr(strip_whitespace=True, min_length=3)
    aspect: constr(strip_whitespace=True) = Field("square", description="square|portrait|landscape")

class ImageResponse(BaseModel):
    image_url: str  # self-hosted /media/... URL (PNG or SVG)

class BookRequest(BaseModel):
    story: StoryRequest
    images: bool = True

class BookResponse(BaseModel):
    id: str
    title: str
    chapters: List[str]
    image_urls: Optional[List[str]] = None  # list of /media/... URLs

# ---------- Story Storage (JSON) ----------
STORIES: Dict[str, Dict] = {}
STORIES_FILE = Path(__file__).parent / "stories.json"

def save_stories():
    try:
        with open(STORIES_FILE, "w", encoding="utf-8") as f:
            json.dump(STORIES, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(STORIES)} stories")
    except Exception as e:
        print(f"⚠️ Failed to save stories: {e}")

def load_stories():
    global STORIES
    try:
        if STORIES_FILE.exists():
            with open(STORIES_FILE, "r", encoding="utf-8") as f:
                STORIES = json.load(f)
            print(f"✅ Loaded {len(STORIES)} stories")
        else:
            STORIES = {}
            print("🆕 No stories.json found. Starting fresh.")
    except Exception as e:
        print(f"⚠️ Failed to load stories: {e}")
        STORIES = {}

# Load on startup
load_stories()

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
    a = (aspect or "square").lower()
    if a.startswith("port"): return "1024x1344"
    if a.startswith("land"): return "1344x1024"
    return "1024x1024"

def _image_size_px(aspect: str) -> Tuple[int, int]:
    a = (aspect or "square").lower()
    if a.startswith("port"): return (1024, 1344)
    if a.startswith("land"): return (1344, 1024)
    return (1024, 1024)

def _placeholder_svg(title: str, w: int, h: int) -> str:
    t = (title or "StoryForge").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#fde68a"/>
      <stop offset="100%" stop-color="#fca5a5"/>
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#g)" rx="24" ry="24"/>
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
        font-family="Arial, sans-serif" font-size="{int(min(w,h)*0.08)}" fill="#1f2937">
    {t[:50]}
  </text>
  <text x="50%" y="62%" dominant-baseline="middle" text-anchor="middle"
        font-family="Arial, sans-serif" font-size="{int(min(w,h)*0.04)}" fill="#374151">
    (placeholder illustration)
  </text>
</svg>"""

def _gen_image_to_file(prompt: str, aspect: str, dest_stem: Path) -> Path:
    """
    Try to generate PNG via OpenAI image API; on failure, save a local SVG placeholder.
    Returns the absolute path to the saved file (PNG or SVG).
    """
    w, h = _image_size_px(aspect)
    dest_stem.parent.mkdir(parents=True, exist_ok=True)

    # Try OpenAI first (if configured)
    if client is not None and OPENAI_IMAGE_MODEL:
        try:
            resp = client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size=_image_size(aspect),
                quality="high",
                response_format="b64_json",
            )
            b64 = resp.data[0].b64_json
            img_bytes = base64.b64decode(b64)
            out = dest_stem.with_suffix(".png")
            with open(out, "wb") as f:
                f.write(img_bytes)
            return out
        except Exception as e:
            print(f"⚠️ Image API failed, using placeholder: {e}")

    # Fallback: SVG placeholder
    out = dest_stem.with_suffix(".svg")
    out.write_text(_placeholder_svg(prompt, w, h), encoding="utf-8")
    return out

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

# ---- Stories (persistent) ----
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
    items.sort(key=lambda x: x.created_at, reverse=True)
    return items

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
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    story_id = str(uuid.uuid4())
    STORIES[story_id] = {"title": title, "chapters": chapters, "created_at": int(time.time())}
    save_stories()
    return StoryResponse(title=title, chapters=chapters)

@app.delete("/stories/{story_id}", tags=["Stories"])
def delete_story(story_id: str):
    if story_id not in STORIES:
        raise HTTPException(status_code=404, detail="Story not found")
    del STORIES[story_id]
    save_stories()
    return {"ok": True, "deleted_id": story_id}

# ---- Images (now saves to disk and returns self-hosted URL) ----
@app.post("/images", response_model=ImageResponse, tags=["Images"])
def generate_image(req: ImageRequest):
    """
    Generates an image to backend/data/tmp/<uuid>.png (or .svg placeholder),
    and returns a public /media/... URL you can embed.
    """
    tmp_dir = DATA_ROOT / "tmp"
    dest_stem = tmp_dir / str(uuid.uuid4())
    abs_path = _gen_image_to_file(req.prompt.strip(), req.aspect, dest_stem)

    # convert absolute path under data/ to /media/ URL
    rel = Path(abs_path).relative_to(DATA_ROOT)
    return ImageResponse(image_url=f"/media/{rel.as_posix()}")

# ---- Books (one image per chapter; returns URLs) ----
@app.post("/books", response_model=BookResponse, tags=["Books"])
def generate_book(req: BookRequest):
    # 1) make the story via text model
    story = generate_story(req.story)

    urls: Optional[List[str]] = None
    if req.images:
        # each book gets its own folder under data/books/<id>/
        book_id = str(uuid.uuid4())  # temp id for paths
        book_dir = DATA_ROOT / "books" / book_id
        urls = []
        for idx, ch in enumerate(story.chapters, start=1):
            iprompt = _image_prompt_from_chapter(story.title, ch, req.story.style)
            dest_stem = book_dir / f"page_{idx:02d}"
            abs_path = _gen_image_to_file(iprompt, "square", dest_stem)
            rel = Path(abs_path).relative_to(DATA_ROOT)
            urls.append(f"/media/{rel.as_posix()}")
        # we’ll reuse the same id for response so the folder matches
        final_id = book_id
    else:
        final_id = str(uuid.uuid4())

    return BookResponse(id=final_id, title=story.title, chapters=story.chapters, image_urls=urls)
