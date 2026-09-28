"""
Combines every route module into a single APIRouter, so main.py only has
to do one `app.include_router(api_router)` call. Add new route modules
here as the API grows (e.g. app/api/conversations.py for Phase 17's chat
history).
"""

from fastapi import APIRouter

from app.api import chat, ingestion, health

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(ingestion.router)
api_router.include_router(chat.router)
