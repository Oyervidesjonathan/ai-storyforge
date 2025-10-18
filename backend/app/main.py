from fastapi import FastAPI
from .routers import books

app = FastAPI(title="Kids Publishing Bot")
app.include_router(books.router)

@app.get("/")
def root():
    return {"ok": True}
