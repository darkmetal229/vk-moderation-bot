from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from database import get_db, Comment, AdminDecision, hash_text
from vk_client import get_vk_client
from config import settings

router = APIRouter()

@router.post("/api/comments/{comment_id}/verdict")
async def set_verdict(comment_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json()
    verdict = body.get("verdict")
    admin_name = body.get("admin", "admin")

    result = await db.execute(select(Comment).where(Comment.id == comment_id))
    comment = result.scalars().first()
    if not comment:
        return JSONResponse({"error": "Комментарий не найден"}, status_code=404)

    comment.manual_verdict = verdict
    comment.status = "reviewed"
    comment.reviewed_at = datetime.utcnow()
    comment.reviewed_by = admin_name

    deleted_vk = False

    if verdict in ("spam", "negative", "ok") and comment.vk_comment_id:
        vk = get_vk_client()
        owner_id = -abs(settings.vk_group_id)
        deleted_vk = await vk.delete_comment(owner_id, comment.vk_comment_id)
        if deleted_vk:
            comment.status = "deleted"

    if verdict in ("spam", "negative", "ok"):
        text_hash = hash_text(comment.text)
        existing = await db.execute(
            select(AdminDecision).where(AdminDecision.text_hash == text_hash)
        )
        if not existing.scalars().first():
            db.add(AdminDecision(
                text_hash=text_hash,
                text_preview=comment.text[:300],
                verdict=verdict,
            ))

    await db.commit()

    return {
        "success": True,
        "deleted_vk": deleted_vk,
        "verdict": verdict,
        "message": "Комментарий удалён из ВК" if deleted_vk else "Комментарий помечен, но не удалён из ВК"
    }
