"""FastAPI application factory and entry point."""
import truststore

truststore.inject_into_ssl()

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import AppError
from app.core.logging import logger

import pydantic
import traceback


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting", extra={"stage": "startup"})
    yield
    logger.info("Application shutting down", extra={"stage": "shutdown"})


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI-Based Knowledge Retrieval Platform",
        description="Multi-agent RAG knowledge retrieval platform with document ingestion, semantic search, citations, clarification, and conversation memory.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=_status_for_code(exc.code), content=exc.to_dict())

    @app.exception_handler(pydantic.ValidationError)
    async def validation_error_handler(request: Request, exc: pydantic.ValidationError) -> JSONResponse:
        for err in exc.errors():
            ctx = err.get("ctx", {})
            exc_obj = ctx.get("error")
            if isinstance(exc_obj, AppError):
                return JSONResponse(
                    status_code=_status_for_code(exc_obj.code),
                    content=exc_obj.to_dict(),
                )
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "Request validation failed.", "details": exc.errors()}},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all: ensures unhandled Python exceptions still go through CORS middleware.

        Without this handler FastAPI's default Starlette error handler returns a
        plain 500 *before* the CORS middleware can inject the Allow-Origin header,
        which means the browser sees a CORS error instead of the real 500.
        """
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected internal error occurred.",
                    "details": [],
                }
            },
        )

    return app


def _status_for_code(code: str) -> int:
    mapping = {
        "invalid_file": 400,
        "unsupported_file_type": 415,
        "empty_file": 400,
        "file_too_large": 413,
        "malformed_document": 422,
        "empty_query": 400,
        "query_too_long": 400,
        "invalid_document_scope": 400,
        "document_not_found": 404,
        "query_validation_error": 400,
        "insufficient_context": 422,
        "embedding_failure": 502,
        "vector_search_failure": 500,
        "reranker_failure": 500,
        "generation_failure": 502,
        "grounding_failure": 500,
        "provider_timeout": 504,
        "provider_rate_limit": 429,
        "provider_error": 502,
        "malformed_model_response": 502,
    }
    return mapping.get(code, 500)


app = create_app()
