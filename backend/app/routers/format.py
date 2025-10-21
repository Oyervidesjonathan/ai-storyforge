# ====== FORMATTER (three style packs) ======
from typing import Dict, Tuple, List
from pydantic import BaseModel, Field
from jinja2 import Template
from weasyprint import HTML
from fastapi import HTTPException

# ----- formatter options (adds style_pack) -----
class FormatOpts(BaseModel):
    trim: str = "8.5x8.5"        # "8x10", "7x10"
    bleed: bool = False
    font: str = "Literata"
    line_height: float = 1.6
    bg_style: str = "blur"       # "blur" | "tint" | "none"
    layout: str = "right"        # "right" | "alt" | "full_bleed" (kept for side/flip behavior)
    image_scale: float = 0.98     # 0.70–0.98
    v_align: str = "center"       # "top" | "center" | "bottom"
    # ✅ NEW: which styling approach to use
    style_pack: str = "overlay_full"  # "overlay_full" | "side_extend" | "float_wrap"

    image_scale: float = 0.98        # 0.70–0.98 (non-full-bleed)
    v_align: str = "center"          # "top" | "center" | "bottom"

TRIMS: Dict[str, Tuple[float, float]] = {
    "8.5x8.5": (8.5, 8.5),
    "8x10":    (8.0, 10.0),
    "7x10":    (7.0, 10.0),
}

def _to_rel_src(url: str, book_id: str) -> str:
    """Return a path relative to DATA so WeasyPrint (base_url=DATA) resolves it."""
    if not url:
        return ""
    if url.startswith("/media/"):
        return url[len("/media/"):]
    if url.startswith("/static/"):
        return url[len("/static/"):]
    if url.startswith("books/"):
        return url
    return f"books/{book_id}/{url}"

# ---------- PDF TEMPLATE (flex columns + safe stacking; 3 style packs) ----------
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

  /* page container uses FLEX (WeasyPrint reliable) */
  .page{
    position:relative;
    width:{{ page_w_in }}in; height:{{ page_h_in }}in;
    display:flex; flex-direction:row; align-items:stretch;
  }
  .textbox, .art { width:50%; }

  /* flip (text right, image left) */
  .page.flip .textbox { order:2; }
  .page.flip .art     { order:1; }

  /* background layer (used by overlay & side-extend) */
  .bg{ position:absolute; inset:0; background-size:cover; background-position:center; }
  .tint{ position:absolute; inset:0; background:transparent; }

  /* stacking: text above art/background */
  .bg, .tint { z-index:1; }
  .art       { position:relative; z-index:2; }
  .textbox   { position:relative; z-index:3; }

  /* text column — NO CARDS */
  .textbox { padding: var(--safe); }
  .copy{ line-height:var(--lh); font-size:12.5pt; background: transparent; }
  .copy h1{ font-size:18pt; margin:0 0 .15in 0; }

  /* subtle readability helpers */
  .copy.soft-shadow{ text-shadow: 0 1px 2px rgba(255,255,255,.85), 0 0 18px rgba(255,255,255,.55); }
  .copy.light-ink  { color:#f7faf8; text-shadow: 0 1px 2px rgba(0,0,0,.55), 0 0 14px rgba(0,0,0,.45); }

  /* art column */
  .art{ padding: var(--safe); height: 100%; }
  .art-frame{ display: table; width: 100%; height: calc(100% - var(--safe)*0); table-layout: fixed; }
  .art-cell{
    display: table-cell; text-align: center; padding: 0;
    vertical-align: {% if v_align == 'top' %}top{% elif v_align == 'bottom' %}bottom{% else %}middle{% endif %};
  }
  .art-img{
    display:inline-block;
    max-width: 100%; max-height: calc(100% * var(--imgScale));
    width: auto; height: auto; object-fit: contain;
    border-radius:.08in; background: transparent;
    box-shadow: 0 0.03in 0.08in rgba(0,0,0,.08);
  }

  /* ===== STYLE PACKS ===== */

  /* 1) overlay_full: art covers the page; text sits directly on it (no box) */
  .page.overlay_full{ display:block; }
  .page.overlay_full .art{ padding:0; width:100%; }
  .page.overlay_full .art-frame, .page.overlay_full .art-cell{ height:100%; }
  .page.overlay_full .art-img{
    width:100%; height:100%; max-height:none; object-fit: cover; border-radius:0; box-shadow:none;
  }
  .page.overlay_full .textbox{
    position:absolute; left: var(--safe); right: var(--safe); top: var(--safe);
    width:auto; max-width: 62%;
  }

  /* 2) side_extend: image on one side; blurred clone under text (no box) */
  .page.side_extend .textbox{ position:relative; }
  .page.side_extend .copy{ position:relative; z-index:1; }

  /* 3) float_wrap: anchored image; clean text field (no box) */
  .page.float_wrap .textbox{ width:60%; }
  .page.float_wrap .art     { width:40%; }

  .spacer{ page-break-after: always; }
</style>
</head>
<body>
  {% for ch in chapters %}
    <section class="page
      {% if style_pack == 'overlay_full' %} overlay_full{% endif %}
      {% if style_pack == 'side_extend' %} side_extend{% endif %}
      {% if style_pack == 'float_wrap' %} float_wrap{% endif %}
      {% if ch.flip %} flip{% endif %}">

      {% if style_pack != 'float_wrap' and ch.bg_src %}
        <div class="bg" style="background-image:url('{{ ch.bg_src }}');"></div>
      {% endif %}
      <div class="tint"></div>

      {% if style_pack == 'overlay_full' %}
        <div class="art">
          <div class="art-frame"><div class="art-cell">
            {% if ch.hero_src %}<img class="art-img" src="{{ ch.hero_src }}">{% endif %}
          </div></div>
        </div>
        <div class="textbox">
          <div class="copy soft-shadow">
            <h1>Chapter {{ loop.index }}</h1>
            {{ ch.text | replace('\n','<br>') | safe }}
          </div>
        </div>

      {% elif style_pack == 'side_extend' %}
        <div class="textbox">
          <!-- per-page blurred clone under the text -->
          {% if ch.bg_src %}
            <div style="
              position:absolute; inset:0; z-index:0;
              background:url('{{ ch.bg_src }}') center/cover no-repeat;
              filter: blur(18px) brightness(1.06) saturate(1.04);
              opacity:.78;"></div>
          {% endif %}
          <div class="copy">
            <h1>Chapter {{ loop.index }}</h1>
            {{ ch.text | replace('\n','<br>') | safe }}
          </div>
        </div>
        <div class="art">
          <div class="art-frame"><div class="art-cell">
            {% if ch.hero_src %}<img class="art-img" src="{{ ch.hero_src }}">{% endif %}
          </div></div>
        </div>

      {% else %} {# float_wrap #}
        <div class="textbox">
          <div class="copy">
            <h1>Chapter {{ loop.index }}</h1>
            {{ ch.text | replace('\n','<br>') | safe }}
          </div>
        </div>
        <div class="art">
          <div class="art-frame"><div class="art-cell">
            {% if ch.hero_src %}<img class="art-img" src="{{ ch.hero_src }}">{% endif %}
          </div></div>
        </div>
      {% endif %}
    </section>
    <div class="spacer"></div>
  {% endfor %}
</body>
</html>
""")

@router.post("/books/{book_id}/format", tags=["Books"])
def format_book(book_id: str, opts: FormatOpts):
    jf = BOOKS_DIR / book_id / "book.json"
    if not jf.exists():
        raise HTTPException(status_code=404, detail="Book not found")

    book = json.loads(jf.read_text(encoding="utf-8"))

    # Page metrics
    trim_w_in, trim_h_in = TRIMS.get(opts.trim, TRIMS["8.5x8.5"])
    bleed_in = 0.125 if opts.bleed else 0.0
    page_w_in = trim_w_in + (bleed_in * 2)
    page_h_in = trim_h_in + (bleed_in * 2)

    # Build render data
    chapters: List[dict] = []
    pages = book.get("pages") or []
    for i, p in enumerate(pages):
        hero_url = (p.get("image_urls") or [None])[0]
        hero_src = _to_rel_src(hero_url, book_id) if hero_url else ""
        bg_src = hero_src or ""
        # rhythm (flip for ALT, handle full_bleed legacy)
        if opts.layout == "alt":
            flip = (i % 2 == 1)
        else:
            flip = False
        chapters.append({
            "text": p.get("text", "") or "",
            "hero_src": hero_src,
            "bg_src": bg_src,
            "flip": flip,
        })

    if not chapters:
        raise HTTPException(status_code=400, detail="No pages to format. Compose or reindex the book first.")

    html = TEMPLATE_HTML.render(
        chapters=chapters,
        style_pack=opts.style_pack,
        page_w_in=page_w_in, page_h_in=page_h_in,
        line_height=opts.line_height,
        font=opts.font,
        image_scale=max(0.70, min(0.98, float(opts.image_scale or 0.98))),
        v_align=opts.v_align if opts.v_align in ("top","center","bottom") else "center",
        style_pack=opts.style_pack if opts.style_pack in ("overlay_full","side_extend","float_wrap") else "overlay_full",
    )

    out_dir = BOOKS_DIR / book_id / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_title = (book.get("title") or "book").replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_pdf = out_dir / f"{safe_title}_kdp.pdf"

    # IMPORTANT: base_url should be DATA_ROOT (BOOKS_DIR.parent) so 'books/<id>/...' resolves
    HTML(string=html, base_url=str(DATA_ROOT.resolve())).write_pdf(str(out_pdf))
    rel = out_pdf.relative_to(DATA_ROOT).as_posix()
    return {"pdf_url": f"/media/{rel}"}
