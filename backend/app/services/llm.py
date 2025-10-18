import os
from typing import Dict, Any
import requests

LLM_BASE = os.getenv("LLM_BASE", "https://api.openai.com/v1")
LLM_KEY  = os.getenv("LLM_KEY", "")
LLM_MODEL= os.getenv("LLM_MODEL", "gpt-4o-mini")  # name you have access to

def chat(messages: list[Dict[str,str]], **kwargs) -> str:
    url = f"{LLM_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_KEY}"}
    payload = {"model": LLM_MODEL, "messages": messages} | kwargs
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
