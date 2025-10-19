# routers/format.py
--font: '{{ font }}', Georgia, serif;
--lh: {{ line_height }};
}
body { margin: 0; font-family: var(--font); }
.page {
position: relative; width: {{ page_w_in }}in; height: {{ page_h_in }}in; overflow: hidden;
display: grid; grid-template-columns: 1fr 1fr;
}
.bg { position:absolute; inset:0; background-size:cover; background-position:center; filter: {% if bg_style=='blur' %} blur(18px) brightness(1.05) saturate(1.05) {% else %} none {% endif %}; transform: scale(1.08); }
.tint { position:absolute; inset:0; background: {% if bg_style=='tint' %} #fbf6ef {% else %} transparent {% endif %}; opacity: .9; }
.textbox { position:relative; padding: calc(var(--safe)); }
.textbox-inner { background: rgba(255,255,255,0.88); padding: .5in; border-radius: 0.1in; line-height: var(--lh); font-size: 12.5pt; }
.art { display:flex; align-items:center; justify-content:center; }
.art img { max-width: 100%; max-height: 100%; }
.spacer { page-break-after: always; }
</style>
</head>
<body>
{% for ch in chapters %}
<section class="page">
<div class="bg" style="background-image:url('{{ ch.hero|e }}');"></div>
<div class="tint"></div>
<div class="textbox">
<div class="textbox-inner">{{ ch.text | replace('\n','<br>') | safe }}</div>
</div>
<div class="art"><img src="{{ ch.hero|e }}"></div>
</section>
<div class="spacer"></div>
{% endfor %}
</body></html>
""")


@router.post("/{book_id}/format")
def format_book(book_id: str, opts: FormatOpts):
meta = DATA / book_id / "book.json"
if not meta.exists():
raise HTTPException(404, "book not found")
book = json.loads(meta.read_text())


trim_w_in, trim_h_in = TRIMS.get(opts.trim, TRIMS["8.5x8.5"])
bleed_in = 0.125 if opts.bleed else 0.0
page_w_in = trim_w_in + (bleed_in * 2)
page_h_in = trim_h_in + (bleed_in * 2)


# Build chapters list with main hero per chapter (use first image as hero by default)
chapters = []
for ch in book.get("chapters", []):
hero = (ch.get("images") or [None])[0]
chapters.append({"text": ch.get("text",""), "hero": hero})


html = TEMPLATE_HTML.render(
chapters=chapters,
page_w_in=page_w_in, page_h_in=page_h_in,
trim_w_in=trim_w_in, trim_h_in=trim_h_in,
bleed_in=bleed_in,
font=opts.font,
line_height=opts.line_height,
bg_style=opts.bg_style,
)


out_dir = DATA / book_id / "exports"; out_dir.mkdir(parents=True, exist_ok=True)
out_pdf = out_dir / f"{book.get('title','book').replace(' ','_')}_kdp.pdf"


HTML(string=html, base_url=str((DATA / book_id).resolve())).write_pdf(str(out_pdf))
return {"pdf_url": f"/static/books/{book_id}/exports/{out_pdf.name}"}
