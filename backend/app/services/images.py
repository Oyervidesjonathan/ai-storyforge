import os, base64, requests
IMG_BASE  = os.getenv("IMG_BASE", "https://api.openai.com/v1")
IMG_KEY   = os.getenv("IMG_KEY", "")
IMG_MODEL = os.getenv("IMG_MODEL", "gpt-image-1")  # or "stable-diffusion-xl"

def generate_image(prompt: str, size: str="1024x1024") -> bytes:
    url = f"{IMG_BASE}/images/generations"
    headers = {"Authorization": f"Bearer {IMG_KEY}"}
    r = requests.post(url, json={"model": IMG_MODEL, "prompt": prompt, "size": size}, headers=headers, timeout=120)
    r.raise_for_status()
    data = r.json()["data"][0]
    if "b64_json" in data:
        return base64.b64decode(data["b64_json"])
    # if your provider returns URL, fetch it
    if "url" in data:
        return requests.get(data["url"], timeout=60).content
    raise RuntimeError("no image payload")
