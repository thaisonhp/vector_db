from fastapi import APIRouter

from .endpoints.hybird_search import search_router


# --------------------------------------
api_v2 = APIRouter()

api_v2.include_router(search_router)