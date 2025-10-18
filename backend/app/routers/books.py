from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..worker import generate_book

router = APIRouter(prefix="/books", tags=["books"])

class CreateBookReq(BaseModel):
    theme: str
    audience_age: int = 5
    style: str = "watercolor"

@router.post("/generate")
def create_and_generate(req: CreateBookReq):
    job = generate_book.delay(req.model_dump())
    return {"job_id": job.id}

@router.get("/job/{job_id}")
def job_status(job_id: str):
    from ..worker import celery_app
    res = celery_app.AsyncResult(job_id)
    if res.failed():
        raise HTTPException(500, str(res.result))
    return {"ready": res.ready(), "result": res.result if res.ready() else None}
