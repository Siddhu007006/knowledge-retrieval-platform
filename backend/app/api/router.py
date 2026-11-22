"""API router aggregation."""

from fastapi import APIRouter

from app.api.routes import documents, query

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(query.router, tags=["query"])
