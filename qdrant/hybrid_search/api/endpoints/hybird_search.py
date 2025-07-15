from fastapi import FastAPI

# The file where HybridSearcher is stored
from model.hybrid_search import HybridSearcher
from fastapi import APIRouter, Depends,File, Form, HTTPException, UploadFile
from utils.indexing_pipeline import IndexingPipeline
from core.config import settings
from pathlib import Path

search_router = APIRouter(prefix="/search", tags=["Search"])


# Create a neural searcher instance
indexer = IndexingPipeline(qdrant_url=settings.qdrant_url,)


UPLOAD_DIR = Path("file_upload")
UPLOAD_DIR.mkdir(exist_ok=True)

@search_router.get("/api/search")
def search_startup(query: str):
    return {"result": indexer.search(text=query)}


@search_router.post("/api/index")
async def index_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/markdown"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    # Index file vào Qdrant
    try:
        count = indexer.index_file(filepath=str(file_path), tenant_id=tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return {
        "message": "File uploaded and indexed",
        "filename": file.filename,
        # "tenant_id": tenant_id,
        "indexed_chunks": count
    }


@search_router.get("/query")
def search(query: str = Form(...), tenant_id: str = Form("common"), limit: int = 5):
    hits = indexer.search(query=query, tenant_id=tenant_id, limit=limit)
    # format payload-only response
    result = [
        {
            "payload": hit.payload,
            "score": getattr(hit, "score", None)
        } for hit in hits
    ]
    return {"results": result}