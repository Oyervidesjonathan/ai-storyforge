import os, json
from celery import Celery
from .services import llm, images, storage, moderation

REDIS_URL = os.getenv("REDIS_URL","redis://localhost:6379/0")
celery_app = Celery("kids_pub", broker=REDIS_URL, backend=REDIS_URL)

@celery_app.task
def generate_book(payload: dict) -> dict:
    theme = payload["theme"]; style = payload.get("style","watercolor")
    # 1) outline
    outline_json = llm.chat([
        {"role":"system","content":"You write safe, kind children's stories."},
        {"role":"user","content": f"...(outline prompt here with theme={theme})"}
    ])
    outline = json.loads(outline_json)
    # 2) prose
    story_md = llm.chat([
        {"role":"system","content":"Children's author; keep sentences short and friendly."},
        {"role":"user","content": f"Expand this outline into Markdown:\n{outline_json}"}
    ])
    # 3) moderation
    report = moderation.basic_checks(story_md)
    # 4) per-page images
    images_out = []
    for p in outline["pages"]:
        prompt = f"Style: {style}. Scene: {p['scene']}. Warm, friendly, safe."
        img_bytes = images.generate_image(prompt)
        url = storage.put_image(img_bytes, ext="png")
        images_out.append({"page_title": p["title"], "prompt": prompt, "image_url": url})
    return {"title": outline["title"], "outline": outline, "story_markdown": story_md, "images": images_out, "safety_report": report}
