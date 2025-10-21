# routers/format.py
import os, json
from pathlib import Path
from typing import Dict, Tuple, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from jinja2 import Template
from weasyprint import HTML

router = APIRouter()

# -------- Storage root (mount this as a Railway volume) --------
DATA = Path(os.getenv("MEDIA_ROOT", "/data"))
DATA.mkdir(parents=True, exist_ok=True)

# ----- formatter options -----
class FormatOpts(BaseModel):
    trim: str = "8.5x8.5"        # supported: "8.5x8.5", "8x10", "7x10"
    bleed: bool = False          # adds 0.125" each side
    font: str = "Literata"
    line_height: float = 1.6
    bg_style: str = "tint"       # "blur" | "tint" | "none"
    layout: str = "right"        # "right" | "alt" | "full_bleed"
    image_scale: float = 0.98    # 0.70–0.98
    v_align: str = "center"      # "top" | "center" | "bottom"

TRIMS: Dict[str, Tuple[float, float]] = {
    "8.5x8.5": (8.5, 8.5),
    "8x10":    (8.0, 10.0),
    "7x10":    (7.0, 10.0),
}

# --- helper: turn /media/... or /static/... URLs into paths relative to DATA ---
def _to_rel_src(url: str, book_id: str) -> str:
    """
    Return a path relative to DATA so WeasyPrint (base_url=DATA) resolves it.
    Accepts '/media/books/<id>/...','/static/books/<id>/...','books/<id>/...' or a bare filename.
    """
    if not url:
        return ""
    if url.startswith("/media/"):
        return url[len("/media/"):]
    if url.startswith("/static/"):
        return url[len("/static/"):]
    if url.startswith("books/"):
        return url
    return f"books/{book_id}/{url}"

# ---------- PDF TEMPLATE (flex columns + safe stacking) ----------
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

  /* Spread container — FLEX is more reliable than 2-col grid in WeasyPrint */
  .page{
    position:relative;
    width:{{ page_w_in }}in; height:{{ page_h_in }}in;
    display:flex; flex-direction:row; align-items:stretch;
  }
  .textbox, .art { width:50%; }

  /* flip (text right, image left) */
  .page.flip .textbox { order:2; }
  .page.flip .art     { order:1; }

  /* Background wash */
  .bg{
    position:absolute; inset:0; background-size:cover; background-position:center;
    {% if bg_style == 'blur' %}filter: blur(18px) brightness(1.05) saturate(1.05); transform: scale(1.08);{% endif %}
  }
  .tint{
    position:absolute; inset:0;
    background: {% if bg_style == 'tint' %}#fbf6ef{% elif bg_style == 'none' %}transparent{% else %}transparent{% endif %};
    opacity: {% if bg_style == 'tint' %}.92{% else %}1{% endif %};
  }

  /* Stack order so text always paints above */
  .bg, .tint { position:absolute; inset:0; z-index:1; }
  .art       { position:relative;               z-index:2; }
  .textbox   { position:relative;               z-index:3; }

  /* Text column */
  .textbox { padding: var(--safe); }
  .box{
    background: rgba(255,255,255,.92);
    padding:.50in;
    border-radius:.10in;
    line-height:var(--lh);
    font-size:12.5pt;
    box-shadow: 0 0.03in 0.08in rgba(0,0,0,.08);
    height: 100%;
  }
  h1{ font-size:18pt; margin:0 0 .15in 0; }

  /* Art column (robust vertical centering) */
  .art{ padding: var(--safe); height: 100%; }
  .art-frame{ display: table; width: 100%; height: calc(100% - var(--safe)*0); table-layout: fixed; }
  .art-cell{
    display: table-cell; text-align: center; padding: 0;
    vertical-align: {% if v_align == 'top' %}top{% elif v_align == 'bottom' %}bottom{% else %}middle{% endif %};
  }
  .art-img{
    display:inline-block; max-width: 100%; max-height: calc(100% * var(--imgScale));
    width: auto; height: auto; object-fit: contain;
    border-radius:.08in; background:#fff; box-shadow: 0 0.03in 0.08in rgba(0,0,0,.10);
  }

  /* Full-bleed variant */
  .page.full{ display:block; }
  .page.full .art{ padding:0; width:100%; }
  .page.full .art-frame, .page.full .art-cell{ height:100%; }
  .page.full .art-img{
    width:100%; height:100%; max-height:none; object-fit: cover; border-radius:0; box-shadow:none;
  }
  .page.full .textbox{
    position:absolute; left: var(--safe); top: var(--safe); z-index:4;
    width: calc(50% - var(--safe)); max-width: 60%; padding:0;
  }
  .page.full .box{ background: rgba(255,255,255,.90); }

  .spacer{ page-break-after: always; }
</style>
</head>
<body>
  {% for ch in chapters %}
    <section class="page{% if ch.flip %} flip{% endif %}{% if ch.full_bleed %} full{% endif %}">
      {% if ch.bg_src %}<div class="bg" style="background-image:url('{{ ch.bg_src }}');"></div>{% endif %}
      <div class="tint"></div>

      {% if not ch.full_bleed %}
        <div class="textbox"><div class="box">
          <h1>Chapter {{ loop.index }}</h1>
          {{ ch.text | replace('\n','<br>') | safe }}
        </div></div>

        <div class="art">
          <div class="art-frame"><div class="art-cell">
            {% if ch.hero_src %}<img class="art-img" src="{{ ch.hero_src }}">{% endif %}
          </div></div>
        </div>
      {% else %}
        <div class="art">
          <div class="art-frame"><div class="art-cell">
            {% if ch.hero_src %}<img class="art-img" src="{{ ch.hero_src }}">{% endif %}
          </div></div>
        </div>
        <div class="textbox"><div class="box">
          <h1>Chapter {{ loop.index }}</h1>
          {{ ch.text | replace('\n','<br>') | safe }}
        </div></div>
      {% endif %}
    </section>
    <div class="spacer"></div>
  {% endfor %}
</body>
</html>
""")

@router.post("/{book_id}/format")
def format_book(book_id: str, opts: FormatOpts):
    meta = DATA / "books" / book_id / "book.json"
    if not meta.exists():
        raise HTTPException(status_code=404, detail="book not found")

    book = json.loads(meta.read_text(encoding="utf-8"))

    # Dimensions
    trim_w_in, trim_h_in = TRIMS.get(opts.trim, TRIMS["8.5x8.5"])
    bleed_in = 0.125 if opts.bleed else 0.0
    page_w_in = trim_w_in + (bleed_in * 2)
    page_h_in = trim_h_in + (bleed_in * 2)

    # Build chapters list from PAGES (your composer shape)
    chapters: List[dict] = []
    pages = book.get("pages") or []
    for i, p in enumerate(pages):
        hero_url = (p.get("image_urls") or [None])[0]
        hero_src = _to_rel_src(hero_url, book_id) if hero_url else ""
        bg_src = hero_src or ""
        # layout decisions (flip every other when 'alt')
        if opts.layout == "alt":
            flip = (i % 2 == 1)
            full_bleed = False
        elif opts.layout == "full_bleed":
            flip = False
            full_bleed = True
        else:
            flip = False
            full_bleed = False
        chapters.append({
            "text": p.get("text", "") or "",
            "hero_src": hero_src,
            "bg_src": bg_src,
            "flip": flip,
            "full_bleed": full_bleed,
        })

    if not chapters:
        raise HTTPException(status_code=400, detail="No pages to format. Compose or reindex the book first.")

    # Render + write
    html = TEMPLATE_HTML.render(
        chapters=chapters,
        page_w_in=page_w_in, page_h_in=page_h_in,
        line_height=opts.line_height,
        font=opts.font,
        bg_style=opts.bg_style,
        image_scale=max(0.70, min(0.98, float(getattr(opts, "image_scale", 0.98)))),
        v_align=getattr(opts, "v_align", "center"),
    )

    out_dir = DATA / "books" / book_id / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_title = (book.get("title") or "book").replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_pdf = out_dir / f"{safe_title}_kdp.pdf"

    # IMPORTANT: base_url should be DATA root (since src are relative like 'books/<id>/...')
    HTML(string=html, base_url=str(DATA.resolve())).write_pdf(str(out_pdf))

    # If your app serves DATA at /static, keep this. If at /media, switch accordingly.
    return {"pdf_url": f"/static/books/{book_id}/exports/{out_pdf.name}"}
