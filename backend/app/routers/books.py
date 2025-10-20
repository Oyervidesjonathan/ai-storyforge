import os, time, uuid, json, base64, mimetypes
from typing import List, Optional, Dict, Tuple
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, constr
from openai import OpenAI

# ----- formatter deps -----
from jinja2 import Template
from weasyprint import HTML

# -------- OpenAI config --------
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORG_ID      = os.getenv("OPENAI_ORG_ID", "")
OPENAI_TEXT_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")

client: Optional[OpenAI]
if not OPENAI_API_KEY:
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY, organization=(OPENAI_ORG_ID or None))

# -------- Storage / media root --------
DEFAULT_MEDIA = Path(__file__).resolve().parents[1] / "data"
DATA_ROOT = Path(os.getenv("MEDIA_ROOT", "/data"))
if not DATA_ROOT.exists():
    DATA_ROOT = DEFAULT_MEDIA
DATA_ROOT.mkdir(parents=True, exist_ok=True)

BOOKS_DIR = DATA_ROOT / "books"
BOOKS_DIR.mkdir(parents=True, exist_ok=True)

# ✅ persist stories in /data
STORIES_FILE = DATA_ROOT / "stories.json"
STORIES: Dict[str, Dict] = {}

def _load_stories():
    global STORIES
    try:
        if STORIES_FILE.exists():
            STORIES = json.loads(STORIES_FILE.read_text(encoding="utf-8"))
            print(f"✅ Loaded {len(STORIES)} stories from {STORIES_FILE}")
        else:
            STORIES = {}
            print("🆕 No stories.json found. Starting fresh.")
    except Exception as e:
        STORIES = {}
        print(f"⚠️ Failed to load stories: {e}")

def _save_stories():
    try:
        STORIES_FILE.write_text(json.dumps(STORIES, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 Saved {len(STORIES)} stories -> {STORIES_FILE}")
    except Exception as e:
        print(f"⚠️ Failed to save stories: {e}")

_load_stories()

# --------- models ---------
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
    image_url: str

class Page(BaseModel):
    chapter_index: int
    text: str
    image_urls: List[str]

class BookComposeRequest(BaseModel):
    story: StoryRequest
    images_per_chapter: int = Field(2, ge=1, le=3)

class BookComposeResponse(BaseModel):
    id: str
    title: str
    pages: List[Page]
    cover_url: Optional[str] = None

class BookListItem(BaseModel):
    id: str
    title: str
    cover_url: Optional[str] = None
    created_at: Optional[int] = None
    pages: int

# ----- formatter options -----
class FormatOpts(BaseModel):
    # Trim
    trim: str = "8.5x8.5"            # "8x10", "7x10"
    bleed: bool = False

    # Typography
    font: str = "Literata"
    line_height: float = 1.6

    # Background behind text
    bg_style: str = "blur"           # "blur" | "tint" | "none"

    # Layout
    layout: str = "alt"              # "right" | "alt" | "full_bleed"
    image_scale: float = 0.92        # 0.70–0.98, relative to column height
    v_align: str = "center"          # "top" | "center" | "bottom"

    # NEW: art shape
    image_style: str = "rounded"     # "rounded" | "blob" | "square"

TRIMS: Dict[str, Tuple[float, float]] = {
    "8.5x8.5": (8.5, 8.5),
    "8x10":    (8.0, 10.0),
    "7x10":    (7.0, 10.0),
}

# ---------- HTML template with BLOBBY clip-path ----------
TEMPLATE_HTML = Template(r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    @page { size: {{ page_w_in }}in {{ page_h_in }}in; margin: 0; }
    :root{
      --safe: 0.375in;
      --lh: {{ line_height }};
      --font: '{{ font }}', Georgia, serif;
      --imgScale: {{ image_scale }};
    }
    * { box-sizing: border-box; }
    body { margin:0; font-family: var(--font); color:#1b1510; }

    .page{
      position:relative;
      width:{{ page_w_in }}in; height:{{ page_h_in }}in;
      display:grid; grid-template-columns: 1fr 1fr;
      align-items: stretch; /* let art span full column height */
    }
    .page.flip .textbox { grid-column: 2; }
    .page.flip .art     { grid-column: 1; }

    .bg{
      position:absolute; inset:0; background-size:cover; background-position:center;
      {% if bg_style == 'blur' %}filter: blur(18px) brightness(1.05) saturate(1.05); transform: scale(1.08);{% endif %}
    }
    .tint{
      position:absolute; inset:0;
      background: {% if bg_style == 'tint' %}#fbf6ef{% else %}transparent{% endif %};
      opacity: {% if bg_style == 'tint' %}.92{% else %}1{% endif %};
    }

    .textbox { position:relative; padding: var(--safe); }
    .box{
      background: rgba(255,255,255,.92);
      padding:.50in;
      border-radius:.10in;
      line-height:var(--lh);
      font-size:12.5pt;
      box-shadow: 0 0.03in 0.08in rgba(0,0,0,.08);
    }
    h1{ font-size:18pt; margin:0 0 .15in 0; }

    .art{
      position:relative;
      padding: var(--safe);
      display:flex;
      justify-content:center;
      {% if v_align == 'top' %}align-items:flex-start;
      {% elif v_align == 'bottom' %}align-items:flex-end;
      {% else %}align-items:center;{% endif %}
    }

    /* Default <img> presentation */
    .art img{
      display:block;
      width: auto;
      height: calc((100% - var(--safe)*2) * var(--imgScale));
      max-height: calc((100% - var(--safe)*2) * var(--imgScale));
      object-fit: contain;
      border-radius:
        {% if image_style == 'square' %}0{% else %}.08in{% endif %};
      background:#fff;
      box-shadow: 0 0.03in 0.08in rgba(0,0,0,.08);
    }

    /* When using the BLOBBY style we render an SVG wrapper */
    .blob-wrap{
      width: 100%;
      height: calc((100% - var(--safe)*2) * var(--imgScale));
      max-height: calc((100% - var(--safe)*2) * var(--imgScale));
    }
    .blob-svg{
      width:100%; height:100%;
      display:block;
      box-shadow: 0 0.03in 0.08in rgba(0,0,0,.08);
      background:#fff;
      border-radius:.02in; /* subtle feather on edges */
    }

    /* Full-bleed variant */
    .page.full{ grid-template-columns: 1fr; }
    .page.full .art{ padding:0; align-items:stretch; justify-content:stretch; }
    .page.full .art img{ width:100%; height:100%; max-height:none; object-fit:cover; border-radius:0; box-shadow:none; }
    .page.full .textbox{
      position:absolute; left: var(--safe); top: var(--safe);
      width: calc(50% - var(--safe)); max-width: 60%;
      padding:0;
    }
    .page.full .box{ background: rgba(255,255,255,.90); }

    .spacer{ page-break-after: always; }
  </style>
</head>
<body>

  <!-- SVG defs for blob clip path (re-used across pages) -->
  <svg width="0" height="0" style="position:absolute">
    <defs>
      <!-- organic blob shape (normalized 1000x1000 viewBox) -->
      <clipPath id="sf-blob-clip" clipPathUnits="objectBoundingBox">
        <!-- objectBoundingBox path version -->
        <path d="M0.145,0.281 C0.221,0.093,0.407,0.013,0.598,0.039 C0.744,0.059,0.889,0.161,0.949,0.306
                 C1.007,0.447,0.988,0.624,0.894,0.752 C0.793,0.889,0.611,0.969,0.43,0.953
                 C0.248,0.935,0.095,0.822,0.038,0.673 C-0.016,0.535,0.060,0.450,0.145,0.281 Z"/>
      </clipPath>
    </defs>
  </svg>

  {% for ch in chapters %}
    <section class="page{% if ch.flip %} flip{% endif %}{% if ch.full_bleed %} full{% endif %}">
      {% if ch.bg_src %}<div class="bg" style="background-image:url('{{ ch.bg_src }}');"></div>{% endif %}
      <div class="tint"></div>

      {% if not ch.full_bleed %}
        <div class="textbox">
          <div class="box">
            <h1>Chapter {{ loop.index }}</h1>
            {{ ch.text | replace('\n','<br>') | safe }}
          </div>
        </div>

        <div class="art">
          {% if ch.hero_src %}
            {% if image_style == 'blob' %}
              <!-- Blob-masked image via SVG -->
              <div class="blob-wrap">
                <svg class="blob-svg" viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid slice">
                  <defs>
                    <clipPath id="blob-{{ loop.index }}">
                      <!-- more detailed blob path at 1000x1000 -->
                      <path d="M145,281 C221,93 407,13 598,39 C744,59 889,161 949,306
                               C1007,447 988,624 894,752 C793,889 611,969 430,953
                               C248,935 95,822 38,673 C-16,535 60,450 145,281 Z"/>
                    </clipPath>
                  </defs>
                  <image x="0" y="0" width="1000" height="1000"
                         xlink:href="{{ ch.hero_src }}"
                         clip-path="url(#blob-{{ loop.index }})"
                         preserveAspectRatio="xMidYMid slice" />
                </svg>
              </div>
            {% else %}
              <img src="{{ ch.hero_src }}">
            {% endif %}
          {% endif %}
        </div>
      {% else %}
        <div class="art">{% if ch.hero_src %}<img src="{{ ch.hero_src }}">{% endif %}</div>
        <div class="textbox">
          <div class="box">
            <h1>Chapter {{ loop.index }}</h1>
            {{ ch.text | replace('\n','<br>') | safe }}
          </div>
        </div>
      {% endif %}
    </section>
    <div class="spacer"></div>
  {% endfor %}
</body>
</html>
""")

# --------- helpers ---------
def _title_from_prompt(prompt: str, age: str) -> str:
    return f"{prompt.strip().capitalize()} (A {age} Story)"

def _chapter_prompts(base_prompt: str, style: str, chapters: int) -> List[str]:
    return [
        f"Write Chapter {i+1} of a children's story.\n"
        f"Premise: {base_prompt}\n"
        f"Style: {style}. Tone: gentle, cozy, simple words.\n"
        f"Length: 120-200 words. Avoid brand names and scary stuff."
        for i in range(chapters)
    ]

def _image_prompt(title: str, chapter_text: str, style: str, note: str = "") -> str:
    base = (
        f"Children's picture book illustration for '{title}'. "
        f"Scene: {chapter_text[:240]}. "
        f"Style: {style}. Ultra kid-friendly, warm, watercolor, no text overlay."
    )
    if note:
        base += f" {note}"
    return base

def _img_size(aspect: str) -> str:
    a = (aspect or "square").lower()
    if a.startswith("port"): return "1024x1536"
    if a.startswith("land"): return "1536x1024"
    return "1024x1024"

def _img_size_px(aspect: str) -> Tuple[int, int]:
    a = (aspect or "square").lower()
    if a.startswith("port"): return (1024, 1536)
    if a.startswith("land"): return (1536, 1024)
    return (1024, 1024)

def _placeholder_svg(title: str, w: int, h: int) -> str:
    t = (title or "StoryForge").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0%" stop-color="#fde68a"/><stop offset="100%" stop-color="#fca5a5"/></linearGradient></defs>
  <rect width="100%" height="100%" fill="url(#g)" rx="24" ry="24"/>
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
        font-family="Arial" font-size="{int(min(w,h)*0.08)}" fill="#1f2937">{t[:50]}</text>
  <text x="50%" y="62%" dominant-baseline="middle" text-anchor="middle"
        font-family="Arial" font-size="{int(min(w,h)*0.04)}" fill="#374151">(placeholder illustration)</text>
</svg>"""

def _gen_image_to_file(prompt: str, aspect: str, dest_stem: Path) -> Path:
    w, h = _img_size_px(aspect)
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        if client is None or not OPENAI_IMAGE_MODEL:
            raise RuntimeError("Images API not configured")
        resp = client.images.generate(model=OPENAI_IMAGE_MODEL, prompt=prompt, size=_img_size(aspect), quality="high")
        b64 = getattr(resp.data[0], "b64_json", None)
        if not isinstance(b64, str) or not b64.strip():
            raise ValueError("No b64_json in image response")
        out = dest_stem.with_suffix(".png")
        out.write_bytes(base64.b64decode(b64))
        return out
    except Exception as e:
        print(f"⚠️ Image generation failed -> placeholder: {e}")
        out = dest_stem.with_suffix(".svg")
        out.write_text(_placeholder_svg(prompt, w, h), encoding="utf-8")
        return out

def _require_client():
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in Railway Variables.")

def _media_url_to_abs(media_url: str) -> Path:
    if not media_url.startswith("/media/"):
        raise HTTPException(status_code=400, detail="expected a /media/... url")
    rel = media_url[len("/media/"):]  # e.g. 'books/<id>/ch01_img1.png'
    return DATA_ROOT / rel

def _media_url_to_rel(media_url: str) -> str:
    if not media_url:
        return ""
    if not media_url.startswith("/media/"):
        raise HTTPException(status_code=400, detail="expected a /media/... url")
    return media_url[len("/media/"):]

def _file_to_data_uri(p: Path) -> Optional[str]:
    try:
        if not p.exists():
            return None
        mt, _ = mimetypes.guess_type(str(p))
        if not mt:
            mt = "image/png"
        b = p.read_bytes()
        return f"data:{mt};base64,{base64.b64encode(b).decode('ascii')}"
    except Exception as e:
        print(f"⚠️ embed failed for {p}: {e}")
        return None

# -------- Router --------
router = APIRouter()

# ===== Stories =====
@router.get("/stories", response_model=List[StoryItem], tags=["Stories"])
def list_stories():
    items: List[StoryItem] = [
        StoryItem(id=i, title=d["title"], chapters_count=len(d["chapters"]), created_at=d["created_at"])
        for i, d in STORIES.items()
    ]
    items.sort(key=lambda x: x.created_at, reverse=True)
    return items

@router.get("/stories/{story_id}", response_model=StoryResponse, tags=["Stories"])
def get_story(story_id: str):
    data = STORIES.get(story_id)
    if not data:
        raise HTTPException(status_code=404, detail="Not found")
    return StoryResponse(title=data["title"], chapters=data["chapters"])

@router.post("/stories", response_model=StoryResponse, tags=["Stories"])
def generate_story(req: StoryRequest):
    _require_client()
    title = _title_from_prompt(req.prompt, req.age_range)
    prompts = _chapter_prompts(req.prompt, req.style, req.chapters)
    chapters: List[str] = []
    try:
        for cp in prompts:
            resp = client.chat.completions.create(
                model=OPENAI_TEXT_MODEL, temperature=0.9,
                messages=[
                    {"role": "system", "content":
                        "You are a children’s author. Warm, age-appropriate prose. Avoid brand names and anything scary."
                    },
                    {"role": "user", "content": cp},
                ],
            )
            chapters.append((resp.choices[0].message.content or "").strip() or "…")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    story_id = str(uuid.uuid4())
    STORIES[story_id] = {"title": title, "chapters": chapters, "created_at": int(time.time())}
    _save_stories()
    print(f"📝 Story created id={story_id} title={title!r} chapters={len(chapters)}")
    return StoryResponse(title=title, chapters=chapters)

@router.delete("/stories/{story_id}", tags=["Stories"])
def delete_story(story_id: str):
    if story_id not in STORIES:
        raise HTTPException(status_code=404, detail="Story not found")
    del STORIES[story_id]
    _save_stories()
    print(f"🗑️ Story deleted id={story_id}")
    return {"ok": True, "deleted_id": story_id}

# ===== Images (utility) =====
@router.post("/images", response_model=ImageResponse, tags=["Images"])
def generate_image(req: ImageRequest):
    dest = DATA_ROOT / "tmp" / str(uuid.uuid4())
    out = _gen_image_to_file(req.prompt.strip(), req.aspect, dest)
    rel = out.relative_to(DATA_ROOT).as_posix()
    return ImageResponse(image_url=f"/media/{rel}")

# ===== Books: compose -> persist book.json =====
@router.post("/books/compose", response_model=BookComposeResponse, tags=["Books"])
def compose_book(req: BookComposeRequest):
    s = generate_story(req.story)  # create story + persist

    book_id = str(uuid.uuid4())
    book_dir = BOOKS_DIR / book_id
    book_dir.mkdir(parents=True, exist_ok=True)

    pages: List[Page] = []
    cover_url: Optional[str] = None

    for idx, ch_text in enumerate(s.chapters, start=1):
        urls: List[str] = []
        for j in range(1, req.images_per_chapter + 1):
            pmt = _image_prompt(s.title, ch_text, req.story.style, f"Panel {j} of {req.images_per_chapter}.")
            out = _gen_image_to_file(pmt, "square", book_dir / f"ch{idx:02d}_img{j}")
            rel = out.relative_to(DATA_ROOT).as_posix()
            url = f"/media/{rel}"
            urls.append(url)
            if cover_url is None:
                cover_url = url
        pages.append(Page(chapter_index=idx-1, text=ch_text, image_urls=urls))

    meta = {
        "id": book_id,
        "title": s.title,
        "pages": [p.dict() for p in pages],
        "cover_url": cover_url,
        "created_at": int(time.time()),
    }
    (book_dir / "book.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"📘 Composed book id={book_id} title={s.title!r} pages={len(pages)} dir={book_dir}")
    return BookComposeResponse(id=book_id, title=s.title, pages=pages, cover_url=cover_url)

@router.get("/books/{book_id}", response_model=BookComposeResponse, tags=["Books"])
def read_book(book_id: str):
    jf = BOOKS_DIR / book_id / "book.json"
    if not jf.exists():
        raise HTTPException(status_code=404, detail="Book not found")
    meta = json.loads(jf.read_text(encoding="utf-8"))
    return BookComposeResponse(
        id=meta["id"], title=meta["title"],
        pages=[Page(**p) for p in meta.get("pages", [])],
        cover_url=meta.get("cover_url")
    )

# ===== Books: tolerant listing endpoints =====
def _scan_books() -> List[BookListItem]:
    items: List[BookListItem] = []
    if not BOOKS_DIR.exists():
        return items
    for sub in BOOKS_DIR.iterdir():
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

@router.get("/books", tags=["Books"])
def list_books_alias():
    return _scan_books()

@router.get("/bookshelf", tags=["Books"])
def list_bookshelf():
    return _scan_books()

@router.get("/books/list", tags=["Books"])
@router.get("/books/list/", tags=["Books"])
def list_books_tolerant():
    return _scan_books()

# ---- Debug helpers ----
@router.get("/debug/media")
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

@router.get("/debug/read")
def debug_read_media(path: str = Query(..., description="Path relative to /media, e.g. books/<id>/ch01_img1.png")):
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise HTTPException(status_code=400, detail="bad path")
    target = DATA_ROOT / p
    if not target.exists():
        raise HTTPException(status_code=404, detail="not found")
    return {"root": str(DATA_ROOT), "path": path, "exists": True, "bytes": target.stat().st_size}

# ========= INLINE EDIT + CHAPTER APIS =========
def _book_dir(book_id: str) -> Path:
    return BOOKS_DIR / book_id

def _book_json_path(book_id: str) -> Path:
    return _book_dir(book_id) / "book.json"

def _load_book_json(book_id: str) -> dict:
    jf = _book_json_path(book_id)
    if not jf.exists():
        raise HTTPException(status_code=404, detail="book.json not found")
    try:
        return json.loads(jf.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read book.json: {e}")

def _save_book_json(book_id: str, data: dict) -> None:
    jf = _book_json_path(book_id)
    jf.parent
