# backend/app.py
import os, time, uuid, json, base64
from typing import List, Optional, Dict, Tuple
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field, constr
from openai import OpenAI

SERVICE_NAME = "AI StoryForge"
APP_VERSION  = "0.4.3"  # tolerant /books/list variants

# ---------- OpenAI client ----------
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORG_ID      = os.getenv("OPENAI_ORG_ID", "")
OPENAI_TEXT_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")  # b64_json output

client: Optional[OpenAI]
if not OPENAI_API_KEY:
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY, organization=(OPENAI_ORG_ID or None))

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

# Use mounted /data if present, else local ./data
DEFAULT_MEDIA = Path(__file__).parent / "data"
DATA_ROOT = Path(os.getenv("MEDIA_ROOT", "/data"))
if not DATA_ROOT.exists():
    DATA_ROOT = DEFAULT_MEDIA
DATA_ROOT.mkdir(parents=True, exist_ok=True)

print("📂 MEDIA ROOT =", str(DATA_ROOT.resolve()))
print("📦 Exists?", DATA_ROOT.exists(), "Writable?", os.access(DATA_ROOT, os.W_OK))

# Serve persisted files under /media
app.mount("/media", StaticFiles(directory=DATA_ROOT), name="media")

@app.get("/ui")
def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

# ---------- Schemas ----------
class StoryRequest(BaseModel):
    prompt: constr(strip_whitespace=True, min_length=3)
    age_range: constr(strip_whitespace=True) = Field("4-7")
    style: constr(strip_whitespace=True) = Field("bedtime, friendly")
    chapters: int = Field(1, ge=1, le=10)

class StoryResponse(BaseModel):
    title: str
    chapters: List[str]

class StoryItem(BaseModel):
    id: str
    title: str
    chapters_count: int
    created_at: int

class ImageRequest(BaseModel):
    prompt: constr(strip_whitespace=True, min_length=3)
    aspect: constr(strip_whitespace=True) = Field("square", description="square|portrait|landscape")

class ImageResponse(BaseModel):
    image_url: str  # always a /media/... URL

class BookRequest(BaseModel):
    story: StoryRequest
    images: bool = True

# compose models
class Page(BaseModel):
    chapter_index: int
    text: str
    image_urls: List[str]  # 1–3 per chapter

class BookComposeRequest(BaseModel):
    story: StoryRequest
    images_per_chapter: int = Field(2, ge=1, le=3)

class BookComposeResponse(BaseModel):
    id: str
    title: str
    pages: List[Page]
    cover_url: Optional[str] = None

class BookResponse(BaseModel):
    id: str
    title: str
    chapters: List[str]
    image_urls: Optional[List[str]] = None

# list item for shelf
class BookListItem(BaseModel):
    id: str
    title: str
    cover_url: Optional[str] = None
    created_at: Optional[int] = None
    pages: int

# ---------- Story Storage ----------
STORIES: Dict[str, Dict] = {}
STORIES_FILE = DATA_ROOT / "stories.json"

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

def _image_prompts_for_chapter(title: str, chapter_text: str, style_hint: str, n: int) -> List[str]:
    base = _image_prompt_from_chapter(title, chapter_text, style_hint)
    return [f"{base} Depict key moment {i}." for i in range(1, n + 1)]

def _image_size(aspect: str) -> str:
    a = (aspect or "square").lower()
    if a.startswith("port"): return "1024x1536"
    if a.startswith("land"): return "1536x1024"
    return "1024x1024"

def _image_size_px(aspect: str) -> Tuple[int, int]:
    a = (aspect or "square").lower()
    if a.startswith("port"): return (1024, 1536)
    if a.startswith("land"): return (1536, 1024)
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
    w, h = _image_size_px(aspect)
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        if client is None or not OPENAI_IMAGE_MODEL:
            raise RuntimeError("Images API not configured")
        resp = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size=_image_size(aspect),
            quality="high",
        )
        datum = resp.data[0] if getattr(resp, "data", None) else None
        b64 = getattr(datum, "b64_json", None)
        if not isinstance(b64, str) or not b64.strip():
            raise ValueError("No b64_json in image response")
        out = dest_stem.with_suffix(".png")
        with open(out, "wb") as f:
            f.write(base64.b64decode(b64))
        return out
    except Exception as e:
        print(f"⚠️ Image API failed, using placeholder: {e}")
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

# Add near the other book routes in backend/app.py

@app.get("/books/list", tags=["Books"])
@app.get("/books/list/", tags=["Books"])
def list_books_tolerant():
    return _scan_books()


# ---- Stories ----
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

# ---- Images ----
@app.post("/images", response_model=ImageResponse, tags=["Images"])
def generate_image(req: ImageRequest):
    tmp_dir = DATA_ROOT / "tmp"
    dest_stem = tmp_dir / str(uuid.uuid4())
    abs_path = _gen_image_to_file(req.prompt.strip(), req.aspect, dest_stem)
    rel = Path(abs_path).relative_to(DATA_ROOT)
    return ImageResponse(image_url=f"/media/{rel.as_posix()}")

# ---- Books (legacy: one image/chapter) ----
@app.post("/books", response_model=BookResponse, tags=["Books"])
def generate_book(req: BookRequest):
    story = generate_story(req.story)
    urls: Optional[List[str]] = None
    if req.images:
        book_id = str(uuid.uuid4())
        book_dir = DATA_ROOT / "books" / book_id
        urls = []
        for idx, ch in enumerate(story.chapters, start=1):
            iprompt = _image_prompt_from_chapter(story.title, ch, req.story.style)
            dest_stem = book_dir / f"page_{idx:02d}"
            abs_path = _gen_image_to_file(iprompt, "square", dest_stem)
            rel = Path(abs_path).relative_to(DATA_ROOT)
            urls.append(f"/media/{rel.as_posix()}")
        final_id = book_id
    else:
        final_id = str(uuid.uuid4())
    return BookResponse(id=final_id, title=story.title, chapters=story.chapters, image_urls=urls)

# ---- Compose multi-image book & save book.json ----
@app.post("/books/compose", response_model=BookComposeResponse, tags=["Books"])
def compose_book(req: BookComposeRequest):
    s = generate_story(req.story)
    book_id = str(uuid.uuid4())
    book_dir = DATA_ROOT / "books" / book_id
    book_dir.mkdir(parents=True, exist_ok=True)

    pages: List[Page] = []
    cover_url: Optional[str] = None

    for idx, ch_text in enumerate(s.chapters, start=1):
        prompts = _image_prompts_for_chapter(s.title, ch_text, req.story.style, req.images_per_chapter)
        urls: List[str] = []
        for j, pmt in enumerate(prompts, start=1):
            dest = book_dir / f"ch{idx:02d}_img{j}"
            abs_path = _gen_image_to_file(pmt, "square", dest)
            rel = abs_path.relative_to(DATA_ROOT).as_posix()
            url = f"/media/{rel}"
            urls.append(url)
            if cover_url is None:
                cover_url = url
        pages.append(Page(chapter_index=idx-1, text=ch_text, image_urls=urls))

    meta = {
        "id": book_id,
        "title": s.title,
        "pages": [page.dict() for page in pages],
        "cover_url": cover_url,
        "created_at": int(time.time()),
    }
    with open(book_dir / "book.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return BookComposeResponse(id=book_id, title=s.title, pages=pages, cover_url=cover_url)

@app.get("/books/{book_id}", response_model=BookComposeResponse, tags=["Books"])
def read_book(book_id: str):
    book_file = DATA_ROOT / "books" / book_id / "book.json"
    if not book_file.exists():
        raise HTTPException(status_code=404, detail="Book not found")
    meta = json.loads(book_file.read_text(encoding="utf-8"))
    return BookComposeResponse(
        id=meta["id"],
        title=meta["title"],
        pages=[Page(**p) for p in meta["pages"]],
        cover_url=meta.get("cover_url"),
    )

# ---- Books shelf listing (tolerant endpoints) ----
def _scan_books() -> List[BookListItem]:
    books_dir = DATA_ROOT / "books"
    if not books_dir.exists():
        return []
    items: List[BookListItem] = []
    for sub in books_dir.iterdir():
        if not sub.is_dir():
            continue
        jf = sub / "book.json"
        if not jf.exists():
            continue
        try:
            meta = json.loads(jf.read_text(encoding="utf-8"))
            items.append(BookListItem(
                id=meta.get("id", sub.name),
                title=meta.get("title", "Untitled"),
                cover_url=meta.get("cover_url"),
                created_at=meta.get("created_at"),
                pages=len(meta.get("pages", [])),
            ))
        except Exception:
            continue
    items.sort(key=lambda x: x.created_at or 0, reverse=True)
    return items

# Accept GET/POST/OPTIONS and with/without trailing slash
@app.api_route("/books/list", methods=["GET", "POST", "OPTIONS"], tags=["Books"])
@app.api_route("/books/list/", methods=["GET", "POST", "OPTIONS"], tags=["Books"])
def list_books_tolerant():
    return _scan_books()

# Keep simple aliases too (GET)
@app.get("/books", response_model=List[BookListItem], tags=["Books"])
def list_books_alias_2():
    return _scan_books()

@app.get("/bookshelf", response_model=List[BookListItem], tags=["Books"])
def list_books_alias_3():
    return _scan_books()

# ---- Debug helpers ----
@app.get("/debug/media")
def debug_media():
    items = []
    for p in DATA_ROOT.rglob("*"):
        if p.is_file():
            rel = p.relative_to(DATA_ROOT).as_posix()
            items.append({"url": f"/media/{rel}", "bytes": p.stat().st_size})
    def mtime(item):
        try:
            return (DATA_ROOT / item["url"].replace("/media/", "")).stat().st_mtime
        except Exception:
            return 0
    items.sort(key=mtime, reverse=True)
    return {"root": str(DATA_ROOT), "count": len(items), "items": items[:200]}

@app.get("/debug/read")
def debug_read_media(path: str = Query(..., description="Path relative to /media, e.g. tmp/abc.png or books/<id>/ch01_img1.png")):
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise HTTPException(status_code=400, detail="bad path")
    target = DATA_ROOT / p
    if not target.exists():
        raise HTTPException(status_code=404, detail="not found")
    return {"root": str(DATA_ROOT), "path": path, "exists": True, "bytes": target.stat().st_size}

# ---- Route list on startup ----
@app.on_event("startup")
def _log_routes():
    paths = sorted({getattr(r, 'path', '') for r in app.routes})
    print("📚 Registered routes:")
    for p in paths:
        if p:
            print("  •", p)
