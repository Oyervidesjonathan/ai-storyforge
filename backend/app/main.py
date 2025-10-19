# backend/app/main.py
import os, time, json, re
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

SERVICE_NAME = "AI StoryForge"
APP_VERSION  = "0.6.0"

# ------- MEDIA ROOT -------
DEFAULT_MEDIA = Path(__file__).resolve().parent.parent / "data"
DATA_ROOT = Path(os.getenv("MEDIA_ROOT", "/data"))
if not DATA_ROOT.exists():
    DATA_ROOT = DEFAULT_MEDIA
DATA_ROOT.mkdir(parents=True, exist_ok=True)

BOOKS_DIR = DATA_ROOT / "books"
BOOKS_DIR.mkdir(parents=True, exist_ok=True)

print("📂 MEDIA ROOT =", str(DATA_ROOT.resolve()))
print("📦 Exists? True Writable?", os.access(DATA_ROOT, os.W_OK))

app = FastAPI(
    title=SERVICE_NAME,
    version=APP_VERSION,
    description="Generate children's stories and illustrations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------- Static (frontend) + media ----------
frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if not frontend_dir.exists():
    print(f"⚠️ frontend dir not found at {frontend_dir}")

# Serve your static frontend assets (index.html, editor_pro.html, etc.)
# html=True lets /static/ render index.html automatically if requested directly.
app.mount("/static", StaticFiles(directory=str(frontend_dir), html=True), name="static")

# Serve generated images and exported PDFs
app.mount("/media", StaticFiles(directory=str(DATA_ROOT)), name="media")

# ---------- Simple health/root ----------
@app.get("/")
def root():
    return {"service": SERVICE_NAME, "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

# ---------- UI ----------
# Old editor/landing
@app.get("/ui")
def ui():
    return FileResponse(str(frontend_dir / "index.html"))

# New Pro Editor UI (served directly). Falls back to /ui if missing.
@app.get("/editor")
def editor_pro():
    p = frontend_dir / "editor_pro.html"
    if p.exists():
        return FileResponse(str(p))
    # graceful fallback if the pro editor file isn't present
    return FileResponse(str(frontend_dir / "index.html"))

# ---------- Include feature router that creates/edits/books + formatter ----------
from backend.app.routers.books import router as books_router
app.include_router(books_router)

# ---------- Models used by the shelf/list ----------
class BookListItem(BaseModel):
    id: str
    title: str
    pages: int
    cover_url: Optional[str] = None
    created_at: Optional[int] = None
    source: str  # "json" or "legacy"

LEGACY_CH_PAT  = re.compile(r"^ch(\d{2})_img(\d+)\.(?:png|jpg|jpeg|webp|svg)$", re.I)
LEGACY_PG_PAT  = re.compile(r"^page_(\d{2})\.(?:png|jpg|jpeg|webp|svg)$", re.I)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".svg")

def _dir_mtime(p: Path) -> float:
    try:
        return max((f.stat().st_mtime for f in p.glob("**/*") if f.is_file()), default=p.stat().st_mtime)
    except Exception:
        return 0.0

def _list_images(d: Path) -> List[Path]:
    imgs = []
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            imgs.append(f)
    return imgs

def _as_media_url(p: Path) -> str:
    rel = p.relative_to(DATA_ROOT).as_posix()
    return f"/media/{rel}"

def _load_json_book(folder: Path) -> Optional[BookListItem]:
    jf = folder / "book.json"
    if not jf.exists():
        return None
    try:
        meta = json.loads(jf.read_text(encoding="utf-8"))
        return BookListItem(
            id=meta.get("id", folder.name),
            title=meta.get("title", "Untitled"),
            pages=len(meta.get("pages", [])),
            cover_url=meta.get("cover_url"),
            created_at=meta.get("created_at"),
            source="json",
        )
    except Exception as e:
        print(f"⚠️ bad book.json in {folder}: {e}")
        return None

def _probe_legacy_book(folder: Path) -> Optional[BookListItem]:
    """
    Detect legacy book folders that only contain images (no book.json).
    We consider two patterns:
      - chXX_imgY.png
      - page_XX.png
    """
    if (folder / "book.json").exists():
        return None

    imgs = _list_images(folder)
    if not imgs:
        return None

    has_legacy = False
    chapters_seen: Dict[str, int] = {}
    pages_seen: Dict[str, int] = {}

    for f in imgs:
        m1 = LEGACY_CH_PAT.match(f.name)
        m2 = LEGACY_PG_PAT.match(f.name)
        if m1:
            has_legacy = True
            chapters_seen[m1.group(1)] = chapters_seen.get(m1.group(1), 0) + 1
        elif m2:
            has_legacy = True
            pages_seen[m2.group(1)] = 1

    if not has_legacy:
        return None

    pages_count = max(len(chapters_seen), len(pages_seen), 1)
    cover_url = _as_media_url(imgs[0])
    return BookListItem(
        id=folder.name,
        title=f"Book {folder.name}",
        pages=pages_count,
        cover_url=cover_url,
        created_at=int(_dir_mtime(folder)),
        source="legacy",
    )

def _scan_books() -> List[BookListItem]:
    items: List[BookListItem] = []
    if not BOOKS_DIR.exists():
        return items

    for sub in BOOKS_DIR.iterdir():
        if not sub.is_dir():
            continue
        item = _load_json_book(sub)
        if item:
            items.append(item)
            continue
        legacy = _probe_legacy_book(sub)
        if legacy:
            items.append(legacy)

    items.sort(key=lambda x: x.created_at or 0, reverse=True)
    print(f"📚 Books listed: {len(items)}  (json={sum(1 for i in items if i.source=='json')}, legacy={sum(1 for i in items if i.source=='legacy')})")
    return items

@app.get("/books/list")
@app.get("/books/list/")
@app.get("/bookshelf")
@app.get("/books")   # convenient alias for shelf in UI
def list_books_top_level():
    return _scan_books()

# --------- Reindex: create book.json for legacy folders ----------
class ReindexResult(BaseModel):
    created: int
    updated_ids: List[str]

@app.post("/books/reindex", response_model=ReindexResult)
def reindex_legacy_to_json():
    created = 0
    updated_ids: List[str] = []
    for sub in BOOKS_DIR.iterdir():
        if not sub.is_dir():
            continue
        if (sub / "book.json").exists():
            continue

        legacy = _probe_legacy_book(sub)
        if not legacy:
            continue

        # Build a lightweight book.json from legacy images so it appears in shelf and viewer
        imgs = _list_images(sub)
        pages = []
        # Prefer grouping by chapter (chXX_imgY). Fallback to page_XX; otherwise 1 image per page.
        ch_groups: Dict[str, List[str]] = {}
        for f in imgs:
            m = LEGACY_CH_PAT.match(f.name)
            if m:
                ch = m.group(1)
                ch_groups.setdefault(ch, []).append(_as_media_url(f))
        if ch_groups:
            for idx, ch in enumerate(sorted(ch_groups.keys())):
                pages.append({"chapter_index": idx, "text": "", "image_urls": ch_groups[ch]})
        else:
            pg_sorted = []
            for f in imgs:
                m = LEGACY_PG_PAT.match(f.name)
                if m:
                    pg_sorted.append((int(m.group(1)), f))
            if pg_sorted:
                for idx, (_, f) in enumerate(sorted(pg_sorted)):
                    pages.append({"chapter_index": idx, "text": "", "image_urls": [_as_media_url(f)]})
            else:
                for idx, f in enumerate(imgs):
                    pages.append({"chapter_index": idx, "text": "", "image_urls": [_as_media_url(f)]})

        meta = {
            "id": sub.name,
            "title": legacy.title,
            "pages": pages,
            "cover_url": legacy.cover_url,
            "created_at": legacy.created_at or int(time.time()),
        }
        try:
            (sub / "book.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            created += 1
            updated_ids.append(sub.name)
            print(f"🧭 reindexed legacy folder -> {sub / 'book.json'}")
        except Exception as e:
            print(f"⚠️ failed to write book.json for {sub}: {e}")

    return ReindexResult(created=created, updated_ids=updated_ids)

# ---------- Route log on startup ----------
@app.on_event("startup")
def _log_routes():
    paths = sorted({getattr(r, "path", "") for r in app.routes if getattr(r, "path", "")})
    print("📚 Registered routes:")
    for p in paths:
        print("  •", p)
