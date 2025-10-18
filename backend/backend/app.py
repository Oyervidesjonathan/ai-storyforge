from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"service": "ai-storyforge backend", "message": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}
