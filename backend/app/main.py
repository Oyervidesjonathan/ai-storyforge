# backend/app/main.py
import os, time, json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

SERVICE_NAME = "AI StoryForge"
APP_VERSION  = "0.5.0"

# ------- MEDIA ROOT -------
DEFAULT_MEDIA = Path(__file__).resolve().parent.parent / "data"
DATA_ROOT = Path(os.getenv("MEDIA_ROOT", "/data"))
if not DATA_ROOT.exists():
    DATA_ROOT = DEFAULT_MEDIA
DATA_ROOT.mkdir(parents=True, exist_ok=True)

print("📂 MEDIA ROOT =", str(DATA_ROOT.resolve()))
print("📦 Exists?", DATA_ROOT.exists(), "Writable?", os.access(DATA_ROOT, os.W_OK))

app = FastAPI(
    title=SERVICE_NAME,
    version=APP_VERSION,
    description="Generate children's stories and illustrations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

# Mount static (your index.html lives under frontend/)
frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Mount /media so saved files & books are directly served
app.mount("/media", StaticFiles(directory=str(DATA_ROOT)), name="media")

# -------- Simple health/root --------
@app.get("/")
def root():
    return {"service": SERVICE_NAME, "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

# -------- UI passthrough (serves frontend/index.html) --------
from fastapi.responses import FileResponse
@app.get("/ui")
def ui():
    return FileResponse(str(frontend_dir / "index.html"))

# -------- Include Routers --------
from .routers.books import router as books_router
app.include_router(books_router)

# --------- Log routes on startup ---------
@app.on_event("startup")
def _log_routes():
    paths = sorted({getattr(r, "path", "") for r in app.routes})
    print("📚 Registered routes:")
    for p in paths:
        if p:
            print("  •", p)
