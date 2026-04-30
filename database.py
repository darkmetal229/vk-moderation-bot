from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import enum
import hashlib
from config import settings

# Для PostgreSQL меняем схему подключения на asyncpg
db_url = settings.database_url
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)

engine = create_async_engine(db_url, echo=settings.debug)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class VerdictEnum(str, enum.Enum):
    ok = "ok"
    spam = "spam"
    negative = "negative"
    pending = "pending"

class StatusEnum(str, enum.Enum):
    new = "new"
    reviewed = "reviewed"
    deleted = "deleted"
    ignored = "ignored"

class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    vk_comment_id = Column(Integer, unique=True, index=True)
    vk_post_id = Column(Integer, index=True)
    vk_from_id = Column(Integer, index=True)
    author_name = Column(String(255), nullable=True)
    text = Column(Text, nullable=False)
    spam_score = Column(Float, default=0.0)
    negative_score = Column(Float, default=0.0)
    auto_verdict = Column(String(20), default=VerdictEnum.ok)
    manual_verdict = Column(String(20), nullable=True)
    status = Column(String(20), default=StatusEnum.new)
    notified_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)

class FAQ(Base):
    __tablename__ = "faq"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    keywords = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class BotSettings(Base):
    __tablename__ = "bot_settings"

    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True)
    comment_id = Column(Integer, nullable=True)
    message = Column(Text, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    delivered = Column(Boolean, default=False)

class AdminDecision(Base):
    """Запомненные решения админа для авто-удаления"""
    __tablename__ = "admin_decisions"

    id = Column(Integer, primary_key=True)
    text_hash = Column(String(64), unique=True, index=True)
    text_preview = Column(String(500))
    verdict = Column(String(20))
    decided_at = Column(DateTime, default=datetime.utcnow)

def hash_text(text: str) -> str:
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(select(BotSettings).limit(1))
        if not result.scalars().first():
            defaults = [
                BotSettings(key="spam_threshold", value="0.7", description="Порог для спама"),
                BotSettings(key="negative_threshold", value="0.65", description="Порог для негатива"),
                BotSettings(key="auto_delete_spam", value="false", description="Автоудаление спама"),
                BotSettings(key="faq_enabled", value="true", description="Автоответы FAQ"),
                BotSettings(key="notify_on_negative", value="true", description="Уведомлять о негативе"),
            ]
            session.add_all(defaults)
            await session.commit()
    print("✅ База данных инициализирована")
