def basic_checks(story_md: str) -> dict:
    # cheap heuristics; you can add provider moderation endpoint too.
    banned = ["violence","blood","weapon","drugs","sex","gambling"]
    hits = [w for w in banned if w in story_md.lower()]
    return {"banned_hits": hits, "ok": len(hits)==0}
