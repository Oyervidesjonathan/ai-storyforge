# backend/app.py
import os, time, uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from openai import OpenAI

SERVICE_NAME = "AI StoryForge"
APP_VERSION = "0.2.0"

# ---------- OpenAI client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")         # good + cheap
OPENAI_IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")         # DALLE-style

if not OPENAI_API_KEY:
    # We still boot so /health works, but story endpoints will 500 with a clear message.
    client: Optional[OpenAI] = None
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

# ---------- Schemas ----------
class StoryRequest(BaseModel):
    prompt: constr(strip_whitespace=True, min_length=3) = Field(..., example="A shy firefly who learns to shine with new friends")
    age_range: constr(strip_whitespace=True) = Field("4-7", example="4-7")
    style: constr(strip_whitespace=True) = Field("bedtime, friendly", example="bedtime, friendly")
    chapters: int = Field(1, ge=1, le=10, description="Number of chapters")

class StoryResponse(BaseModel):
    title: str
    chapters: List[str]

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

# ---------- Helpers ----------
def _title_from_prompt(prompt: str, age_range: str) -> str:
    return f"{prompt.strip().capitalize()} (A {age_range} Story)"

def _chapter_prompts(base_prompt: str, style: str, chapters: int) -> List[str]:
    """Make compact prompts so we stay fast + cheap."""
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
    # fast response is important for Railway health checks
    return {"ok": True, "ts": int(time.time())}

@app.post("/stories", response_model=StoryResponse, tags=["Stories"])
def generate_story(req: StoryRequest):
    _require_client()

    title = _title_from_prompt(req.prompt, req.age_range)
    chapter_prompts = _chapter_prompts(req.prompt, req.style, req.chapters)

    chapters: List[str] = []
    try:
        # One request per chapter keeps latency bounded and makes retries simple
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
            if not text:
                text = "…"  # never return empty
            chapters.append(text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

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
        raise HTTPException(status_code=502, detail=f"Image error: {e}")

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
