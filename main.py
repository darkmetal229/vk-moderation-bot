import logging
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select

from database import create_tables, get_db, Comment, BotSettings
from analyzer import get_analyzer
from vk_client import get_vk_client
from config import settings
from routes_extra import router as extra_router
from database import hash_text, AdminDecision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск ВГТУ VK Moderation Bot...")
    await create_tables()
    logger.info("✅ База данных готова")
    yield
    logger.info("🛑 Завершение работы")


app = FastAPI(title="ВГТУ VK Moderation Bot", version="2.0.0", lifespan=lifespan)

app.include_router(extra_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_ngrok_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

# ─── Анализ текста ───────────────────────────────────────────
@app.post("/analyze")
async def analyze_comment(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    text = data.get("text", "")

    rows = await db.execute(BotSettings.__table__.select())
    settings_dict = {row.key: row.value for row in rows}

    analyzer = get_analyzer()
    analyzer.update_thresholds(
        float(settings_dict.get("spam_threshold", 0.7)),
        float(settings_dict.get("negative_threshold", 0.65))
    )
    result = analyzer.analyze(text)

    return {
        "text": text[:200],
        "verdict": result.verdict,
        "spam_score": result.spam_score,
        "negative_score": result.negative_score,
        "method": result.method,
        "details": result.details,
    }


# ─── ML метрики ──────────────────────────────────────────────
@app.get("/api/ml-metrics")
async def get_ml_metrics():
    try:
        with open("ml_metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Модель ещё не обучена. Запустите train_model.py"}


# ─── VK Callback ─────────────────────────────────────────────
@app.post("/vk/callback", include_in_schema=False)
async def vk_callback(request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.json()
    logger.info(f"VK event: {body.get('type')}")

    if body.get("type") == "confirmation":
        return PlainTextResponse(
            content=settings.vk_confirmation_token, media_type="text/plain"
        )

    if body.get("type") == "wall_reply_new":
        obj = body.get("object", {})
        text = obj.get("text", "")
        from_id = obj.get("from_id", 0)
        post_id = obj.get("post_id", 0)
        comment_id = obj.get("id", 0)

        if not text:
            return PlainTextResponse("ok")

        text_hash = hash_text(text)
        decision_result = await db.execute(
            select(AdminDecision).where(AdminDecision.text_hash == text_hash)
        )
        past_decision = decision_result.scalars().first()

        if past_decision:
            vk = get_vk_client()
            await vk.delete_comment(-abs(settings.vk_group_id), comment_id)
            logger.info(f"Авто-удалён повторный {past_decision.verdict}: {text[:80]}")
            return PlainTextResponse("ok")

        vk = get_vk_client()
        user_info = await vk.get_user_info(from_id)
        author_name = user_info.get("name", str(from_id)) if user_info else str(from_id)

        analyzer = get_analyzer()
        result = analyzer.analyze(text)

        new_comment = Comment(
            vk_comment_id=comment_id,
            vk_post_id=post_id,
            vk_from_id=from_id,
            author_name=author_name,
            text=text,
            spam_score=result.spam_score,
            negative_score=result.negative_score,
            auto_verdict=result.verdict,
            status="new",
        )
        db.add(new_comment)
        await db.commit()

        if result.verdict in ("spam", "negative"):
            emoji = "📢" if result.verdict == "spam" else "⚠️"
            await vk.notify_admin(
                f"{emoji} Обнаружен {result.verdict}!\n"
                f"Автор: {author_name} (id{from_id})\n"
                f"Текст: {text[:200]}\n"
                f"Уверенность: {max(result.spam_score, result.negative_score):.0%}"
            )

    return PlainTextResponse("ok")


# ─── API для дашборда ─────────────────────────────────────────────
@app.get("/api/comments")
async def get_comments(
    limit: int = 100,
    verdict: str = "",
    search: str = "",
    db: AsyncSession = Depends(get_db)
):
    q = Comment.__table__.select().order_by(Comment.created_at.desc())
    if verdict:
        q = q.where(Comment.auto_verdict == verdict)
    if search:
        q = q.where(Comment.text.contains(search))
    q = q.limit(limit)
    rows = await db.execute(q)
    return [
        {
            "id": r.id,
            "text": r.text[:300],
            "auto_verdict": r.auto_verdict,
            "manual_verdict": r.manual_verdict,
            "spam_score": r.spam_score,
            "negative_score": r.negative_score,
            "status": r.status,
            "author_name": r.author_name,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@app.get("/api/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    total    = (await db.execute(select(func.count(Comment.id)))).scalar() or 0
    spam     = (await db.execute(select(func.count(Comment.id)).where(Comment.auto_verdict == "spam"))).scalar() or 0
    negative = (await db.execute(select(func.count(Comment.id)).where(Comment.auto_verdict == "negative"))).scalar() or 0
    pending  = (await db.execute(select(func.count(Comment.id)).where(Comment.auto_verdict == "pending"))).scalar() or 0
    ok       = (await db.execute(select(func.count(Comment.id)).where(Comment.auto_verdict == "ok"))).scalar() or 0
    return {"total": total, "spam": spam, "negative": negative, "pending": pending, "ok": ok}


@app.get("/api/settings")
async def get_settings(db: AsyncSession = Depends(get_db)):
    rows = await db.execute(BotSettings.__table__.select())
    return {row.key: row.value for row in rows}


@app.put("/api/settings/{key}")
async def update_setting(key: str, request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    value = str(data.get("value", ""))
    await db.execute(
        BotSettings.__table__.update().where(BotSettings.key == key).values(value=value)
    )
    await db.commit()
    return {"updated": key, "value": value}


# ─── Auth ─────────────────────────────────────────────────────
ADMIN_LOGIN = "admin"
ADMIN_PASSWORD = "1234"  # Измените на свой пароль

@app.post("/api/auth")
async def auth(request: Request):
    data = await request.json()
    if data.get("login") == ADMIN_LOGIN and data.get("password") == ADMIN_PASSWORD:
        return {"success": True, "token": "vgtu_admin_token_2024"}
    return JSONResponse({"success": False, "error": "Неверный логин или пароль"}, status_code=401)

# ─── FAQ API ─────────────────────────────────────────────────
@app.get("/api/faq")
async def get_faq(db: AsyncSession = Depends(get_db)):
    from database import FAQ
    rows = await db.execute(FAQ.__table__.select().where(FAQ.is_active == True).order_by(FAQ.id.desc()))
    return [{"id": r.id, "question": r.question, "answer": r.answer, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]

@app.post("/api/faq")
async def create_faq(request: Request, db: AsyncSession = Depends(get_db)):
    from database import FAQ
    data = await request.json()
    q = data.get("question", "").strip()
    a = data.get("answer", "").strip()
    if not q or not a:
        return JSONResponse({"error": "Вопрос и ответ обязательны"}, status_code=400)
    item = FAQ(question=q, answer=a)
    db.add(item)
    await db.commit()
    return {"success": True, "id": item.id}

@app.delete("/api/faq/{faq_id}")
async def delete_faq(faq_id: int, db: AsyncSession = Depends(get_db)):
    from database import FAQ
    result = await db.execute(select(FAQ).where(FAQ.id == faq_id))
    item = result.scalars().first()
    if not item:
        return JSONResponse({"error": "Не найдено"}, status_code=404)
    item.is_active = False
    await db.commit()
    return {"success": True}

@app.get("/")
async def root():
    return {"status": "ok", "service": "ВГТУ VK Moderation Bot v2.0", "ml": "Random Forest"}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)

# ✅ FAQ страница — без авторизации, без кнопки "назад в панель"
@app.get("/faq", response_class=HTMLResponse)
async def faq_page():
    return HTMLResponse(content=FAQ_PAGE_HTML)


# ─── FAQ PAGE HTML (публичная, без авторизации) ───────────────
FAQ_PAGE_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ВГТУ — Часто задаваемые вопросы</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f0f4f8;
            padding: 40px 20px;
            color: #1a1f2e;
        }
        .container {
            max-width: 860px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #0099aa 0%, #5248e0 100%);
            color: white;
            padding: 36px 32px;
            text-align: center;
        }
        .header h1 { font-size: 26px; font-weight: 800; margin-bottom: 8px; }
        .header p { opacity: 0.88; font-size: 14px; }
        .faq-list { padding: 8px 0 24px; }
        .faq-item {
            border-bottom: 1px solid #e8edf3;
            margin: 0 24px;
        }
        .faq-question {
            font-weight: 700;
            font-size: 15px;
            color: #1a1f2e;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 0;
            gap: 16px;
        }
        .faq-question:hover { color: #0099aa; }
        .faq-answer {
            padding: 0 0 16px;
            color: #4a5568;
            line-height: 1.7;
            font-size: 14px;
            display: none;
        }
        .faq-answer.show { display: block; }
        .faq-question .icon {
            font-size: 18px;
            color: #0099aa;
            transition: transform 0.25s;
            flex-shrink: 0;
        }
        .faq-question.open .icon { transform: rotate(180deg); }
        .footer {
            background: #f7f9fc;
            padding: 20px 24px;
            text-align: center;
            font-size: 13px;
            color: #6b7280;
            border-top: 1px solid #e8edf3;
        }
        .loader { text-align: center; padding: 48px; color: #9ca3af; font-size: 15px; }
        .empty { text-align: center; padding: 48px 24px; color: #9ca3af; }
        .empty-icon { font-size: 40px; margin-bottom: 12px; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>❓ Часто задаваемые вопросы</h1>
        <p>ВГТУ — ответы на популярные вопросы студентов и абитуриентов</p>
    </div>
    <div class="faq-list" id="faqList">
        <div class="loader">⏳ Загрузка вопросов...</div>
    </div>
    <div class="footer">
        Вопрос не нашли? Напишите в сообщения сообщества ВКонтакте
    </div>
</div>

<script>
    async function loadFaq() {
        try {
            const response = await fetch('/api/faq');
            const items = await response.json();
            const container = document.getElementById('faqList');

            if (!items.length) {
                container.innerHTML = '<div class="empty"><div class="empty-icon">📭</div>Пока нет вопросов-ответов. Администратор добавит их позже.</div>';
                return;
            }

            container.innerHTML = items.map(item => `
                <div class="faq-item">
                    <div class="faq-question" onclick="toggleAnswer(this)">
                        <span>${escapeHtml(item.question)}</span>
                        <span class="icon">▼</span>
                    </div>
                    <div class="faq-answer">${escapeHtml(item.answer)}</div>
                </div>
            `).join('');
        } catch(e) {
            document.getElementById('faqList').innerHTML = '<div class="empty" style="color:#e02d3c">⚠️ Ошибка загрузки. Попробуйте обновить страницу.</div>';
        }
    }

    function toggleAnswer(el) {
        const answer = el.nextElementSibling;
        answer.classList.toggle('show');
        el.classList.toggle('open');
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    loadFaq();
</script>
</body>
</html>"""

print("✅ FAQ_PAGE_HTML добавлен (без авторизации, без кнопки 'назад')")


# ─── DASHBOARD HTML ────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ВГТУ — Moderation Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Manrope:wght@400;500;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #eef1f6;
    --surface: #ffffff;
    --surface2: #f3f6fa;
    --surface3: #e8ecf4;
    --border: #cdd4e0;
    --border-strong: #b0bdd0;
    --accent: #0099aa;
    --accent2: #5248e0;
    --danger: #d92d3c;
    --warn: #c97000;
    --ok: #1a9e4a;
    --text: #1a1f2e;
    --muted: #6b7280;
    --font: 'Manrope', sans-serif;
    --mono: 'JetBrains Mono', monospace;
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.08);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.12);
    --shadow-lg: 0 8px 28px rgba(0,0,0,0.15);
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: var(--font); background: var(--bg); color: var(--text); min-height: 100vh; }

  /* Auth Overlay */
  .auth-overlay {
    position: fixed; inset: 0; background: var(--bg);
    display: flex; align-items: center; justify-content: center;
    z-index: 9999;
  }
  .auth-overlay.hidden { display: none; }
  .auth-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 44px;
    width: 380px;
    box-shadow: var(--shadow-lg);
  }
  .auth-logo { text-align: center; font-size: 44px; margin-bottom: 16px; }
  .auth-title { text-align: center; font-size: 22px; font-weight: 800; margin-bottom: 4px; }
  .auth-sub { text-align: center; font-size: 12px; color: var(--muted); margin-bottom: 28px; font-family: var(--mono); }
  .auth-field { margin-bottom: 14px; }
  .auth-field label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: .6px; font-weight: 600; }
  .auth-input { width: 100%; background: var(--surface2); border: 1.5px solid var(--border); border-radius: 10px; padding: 12px 14px; color: var(--text); font-size: 14px; font-family: var(--font); outline: none; transition: border-color .2s, box-shadow .2s; }
  .auth-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(0,153,170,.12); }
  .auth-btn { width: 100%; background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; border: none; border-radius: 10px; padding: 13px; font-size: 14px; font-weight: 700; cursor: pointer; margin-top: 8px; transition: opacity .15s, transform .1s; font-family: var(--font); letter-spacing: .3px; }
  .auth-btn:hover { opacity: 0.92; transform: translateY(-1px); }
  .auth-error { color: var(--danger); font-size: 12px; text-align: center; margin-top: 10px; min-height: 18px; }

  /* Header */
  .header {
    background: var(--surface);
    border-bottom: 2px solid var(--border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
    box-shadow: var(--shadow-sm);
  }
  .header-left { display: flex; align-items: center; gap: 14px; }
  .logo { width: 40px; height: 40px; background: linear-gradient(135deg, var(--accent), var(--accent2)); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 4px 12px rgba(0,153,170,.3); }
  .title { font-size: 18px; font-weight: 800; }
  .title span { color: var(--accent); }
  .subtitle { font-size: 12px; color: var(--muted); font-family: var(--mono); }
  .ml-badge { background: linear-gradient(135deg, var(--accent2), var(--accent)); color: white; padding: 5px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 1px; box-shadow: 0 2px 8px rgba(82,72,224,.3); }

  /* Layout */
  .main { display: flex; height: calc(100vh - 70px); }
  .sidebar { width: 210px; background: var(--surface); border-right: 2px solid var(--border); padding: 16px 0; flex-shrink: 0; box-shadow: 2px 0 8px rgba(0,0,0,0.04); }
  .nav-item { display: flex; align-items: center; gap: 10px; padding: 13px 20px; cursor: pointer; color: var(--muted); font-size: 13px; font-weight: 600; transition: all .2s; border-left: 3px solid transparent; }
  .nav-item:hover { color: var(--text); background: var(--surface2); }
  .nav-item.active { color: var(--accent); background: rgba(0,153,170,.08); border-left-color: var(--accent); }
  .nav-icon { font-size: 17px; }
  .content { flex: 1; overflow-y: auto; padding: 28px 32px; background: var(--bg); }

  /* Stats Grid */
  .stats-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 28px; }
  .stat-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 20px 20px 18px;
    position: relative;
    overflow: hidden;
    transition: box-shadow .2s, border-color .2s, transform .15s;
    box-shadow: var(--shadow-sm);
  }
  .stat-card:hover { border-color: var(--accent); box-shadow: var(--shadow-md); transform: translateY(-2px); }
  .stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
  }
  .stat-card.total::before  { background: #64748b; }
  .stat-card.ok::before     { background: var(--ok); }
  .stat-card.spam::before   { background: var(--warn); }
  .stat-card.negative::before { background: var(--danger); }
  .stat-card.pending::before  { background: var(--accent2); }
  .stat-label { font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; margin-bottom: 10px; margin-top: 2px; }
  .stat-value { font-size: 38px; font-weight: 800; font-family: var(--mono); line-height: 1; }
  .stat-card.total .stat-value   { color: #64748b; }
  .stat-card.spam .stat-value    { color: var(--warn); }
  .stat-card.negative .stat-value{ color: var(--danger); }
  .stat-card.pending .stat-value { color: var(--accent2); }
  .stat-card.ok .stat-value      { color: var(--ok); }

  /* Section Card — улучшенные тени и выделение */
  .section-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 24px;
    box-shadow: var(--shadow-md);
  }
  .section-header {
    padding: 16px 22px;
    border-bottom: 2px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(to right, var(--surface2), var(--surface));
  }
  .section-title { font-size: 14px; font-weight: 800; display: flex; align-items: center; gap: 8px; color: var(--text); letter-spacing: .2px; }

  /* Toolbar */
  .toolbar { padding: 14px 20px; border-bottom: 1.5px solid var(--border); display: flex; gap: 10px; align-items: center; flex-wrap: wrap; background: var(--surface2); }
  .search-input { background: var(--surface); border: 1.5px solid var(--border); border-radius: 9px; padding: 9px 14px; color: var(--text); font-size: 13px; font-family: var(--font); outline: none; flex: 1; min-width: 180px; transition: border-color .2s; }
  .search-input:focus { border-color: var(--accent); }
  .select-filter { background: var(--surface); border: 1.5px solid var(--border); border-radius: 9px; padding: 9px 14px; color: var(--text); font-size: 13px; outline: none; cursor: pointer; }
  .btn { padding: 8px 16px; border-radius: 9px; border: none; font-size: 12px; font-weight: 700; cursor: pointer; transition: all .15s; font-family: var(--font); letter-spacing: .2px; }
  .btn-primary { background: linear-gradient(135deg, var(--accent), #007a8a); color: #fff; box-shadow: 0 2px 6px rgba(0,153,170,.25); }
  .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn-ghost { background: var(--surface); color: var(--text); border: 1.5px solid var(--border); }
  .btn-ghost:hover { border-color: var(--accent); color: var(--accent); background: rgba(0,153,170,.05); }
  .btn-danger { background: rgba(217,45,60,.1); color: var(--danger); border: 1.5px solid rgba(217,45,60,.25); }
  .btn-danger:hover { background: rgba(217,45,60,.2); }

  /* Table */
  .data-table { width: 100%; border-collapse: collapse; }
  .data-table th { background: var(--surface3); padding: 11px 16px; text-align: left; font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .7px; border-bottom: 2px solid var(--border); }
  .data-table td { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 13px; vertical-align: middle; }
  .data-table tr:hover td { background: rgba(0,153,170,.03); }
  .text-cell { max-width: 360px; }
  .text-preview { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 360px; display: block; }

  /* Badges */
  .badge { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
  .badge-ok       { background: rgba(26,158,74,.12);  color: var(--ok);     border: 1.5px solid rgba(26,158,74,.3); }
  .badge-spam     { background: rgba(201,112,0,.12);  color: var(--warn);   border: 1.5px solid rgba(201,112,0,.3); }
  .badge-negative { background: rgba(217,45,60,.12);  color: var(--danger); border: 1.5px solid rgba(217,45,60,.3); }
  .badge-pending  { background: rgba(82,72,224,.12);  color: var(--accent2);border: 1.5px solid rgba(82,72,224,.3); }
  .badge-rf       { background: rgba(0,153,170,.1);   color: var(--accent); border: 1.5px solid rgba(0,153,170,.25); }

  /* Score Bar */
  .score-bar { display: flex; align-items: center; gap: 8px; }
  .score-track { flex: 1; height: 5px; background: var(--surface3); border-radius: 3px; overflow: hidden; }
  .score-fill { height: 100%; border-radius: 3px; transition: width .3s; }
  .score-fill.spam { background: var(--warn); }
  .score-fill.negative { background: var(--danger); }
  .score-text { font-size: 11px; font-family: var(--mono); color: var(--muted); min-width: 34px; text-align: right; }

  /* Analyse Panel */
  .analyze-area { padding: 22px; }
  .analyze-input { width: 100%; background: var(--surface2); border: 1.5px solid var(--border); border-radius: 12px; padding: 14px; color: var(--text); font-size: 14px; font-family: var(--font); resize: vertical; min-height: 100px; outline: none; transition: border-color .2s; }
  .analyze-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(0,153,170,.08); }
  .analyze-result { margin-top: 16px; background: var(--surface2); border-radius: 12px; padding: 18px; border: 1.5px solid var(--border); font-family: var(--mono); font-size: 12px; line-height: 1.8; display: none; box-shadow: var(--shadow-sm); }
  .analyze-result.show { display: block; }
  .result-verdict { font-size: 20px; font-weight: 800; margin-bottom: 8px; font-family: var(--font); }

  /* ML Grid */
  .ml-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 22px; }
  .ml-metric { background: var(--surface2); border-radius: 12px; padding: 18px; border: 1.5px solid var(--border); box-shadow: var(--shadow-sm); }
  .ml-metric-title { font-size: 11px; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: .7px; margin-bottom: 14px; }
  .progress-item { margin-bottom: 10px; }
  .progress-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
  .progress-bar { height: 7px; background: var(--border); border-radius: 4px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 4px; }
  .fill-ok       { background: var(--ok); }
  .fill-spam     { background: var(--warn); }
  .fill-negative { background: var(--danger); }
  .fill-accent   { background: var(--accent); }

  /* Settings Grid */
  .settings-grid { padding: 22px; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .setting-item { background: var(--surface2); border: 1.5px solid var(--border); border-radius: 12px; padding: 18px; box-shadow: var(--shadow-sm); border-left: 4px solid var(--accent); }
  .setting-key { font-size: 11px; color: var(--muted); font-family: var(--mono); margin-bottom: 6px; }
  .setting-val { font-size: 17px; font-weight: 800; color: var(--accent); }
  .setting-desc { font-size: 11px; color: var(--muted); margin-top: 6px; }

  /* FAQ Tab */
  .faq-add-form { padding: 22px; display: flex; flex-direction: column; gap: 12px; border-bottom: 2px solid var(--border); background: var(--surface2); }
  .faq-form-title { font-size: 11px; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: .6px; }
  .faq-input { background: var(--surface); border: 1.5px solid var(--border); border-radius: 10px; padding: 11px 14px; color: var(--text); font-size: 13px; font-family: var(--font); outline: none; width: 100%; transition: border-color .2s; }
  .faq-input:focus { border-color: var(--accent); }
  .faq-textarea { resize: vertical; min-height: 72px; }
  .faq-list { padding: 18px 22px; display: flex; flex-direction: column; gap: 14px; }
  .faq-item { background: var(--surface2); border: 1.5px solid var(--border); border-radius: 12px; padding: 16px 18px; box-shadow: var(--shadow-sm); border-left: 4px solid var(--accent2); }
  .faq-q { font-weight: 700; font-size: 13px; margin-bottom: 8px; color: var(--accent2); }
  .faq-a { font-size: 13px; color: var(--text); line-height: 1.6; }
  .faq-item-footer { display: flex; justify-content: flex-end; margin-top: 12px; }
  .btn-del-faq { background: rgba(217,45,60,.08); color: var(--danger); border: 1.5px solid rgba(217,45,60,.2); padding: 5px 12px; border-radius: 7px; font-size: 11px; cursor: pointer; font-family: var(--font); font-weight: 600; }
  .btn-del-faq:hover { background: rgba(217,45,60,.2); }

  .verdict-actions { display: flex; gap: 6px; margin-top: 4px; }
  .btn-confirm { background: rgba(26,158,74,.12); color: var(--ok); border: 1.5px solid rgba(26,158,74,.3); padding: 5px 12px; border-radius: 7px; font-size: 11px; font-weight: 700; cursor: pointer; transition: background .15s; }
  .btn-confirm:hover { background: rgba(26,158,74,.25); }
  .btn-reject  { background: rgba(107,114,128,.1); color: var(--muted); border: 1.5px solid rgba(107,114,128,.25); padding: 5px 12px; border-radius: 7px; font-size: 11px; font-weight: 700; cursor: pointer; transition: background .15s; }
  .btn-reject:hover  { background: rgba(107,114,128,.2); }
  .btn-spam     { background: rgba(201,112,0,.12); color: var(--warn); border: 1.5px solid rgba(201,112,0,.3); padding: 5px 12px; border-radius: 7px; font-size: 11px; font-weight: 700; cursor: pointer; transition: background .15s; }
  .btn-spam:hover     { background: rgba(201,112,0,.25); }
  .btn-negative { background: rgba(217,45,60,.12); color: var(--danger); border: 1.5px solid rgba(217,45,60,.3); padding: 5px 12px; border-radius: 7px; font-size: 11px; font-weight: 700; cursor: pointer; transition: background .15s; }
  .btn-negative:hover { background: rgba(217,45,60,.25); }
  .verdict-auto-ok { font-size: 11px; font-weight: 700; color: var(--ok); padding: 5px 8px; }
  .verdict-done { font-size: 11px; color: var(--muted); font-style: italic; }

  .tab-page { display: none; }
  .tab-page.active { display: block; }
  .loader { text-align: center; padding: 44px; color: var(--muted); font-size: 13px; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
</style>
</head>
<body>

<div class="auth-overlay" id="authOverlay">
  <div class="auth-box">
    <div class="auth-logo">🔐</div>
    <div class="auth-title">ВГТУ Moderation</div>
    <div class="auth-sub">Панель администратора</div>
    <div class="auth-field">
      <label>Логин</label>
      <input class="auth-input" type="text" id="authLogin" placeholder="admin">
    </div>
    <div class="auth-field">
      <label>Пароль</label>
      <input class="auth-input" type="password" id="authPassword" placeholder="••••••••" onkeydown="if(event.key==='Enter')doAuth()">
    </div>
    <button class="auth-btn" onclick="doAuth()">Войти →</button>
    <div class="auth-error" id="authError"></div>
  </div>
</div>

<div class="header">
  <div class="header-left">
    <div class="logo">🔍</div>
    <div>
      <div class="title">ВГТУ <span>VK</span> Moderation</div>
      <div class="subtitle">Система модерации комментариев</div>
    </div>
  </div>
  <div class="ml-badge">🤖 ML ACTIVE</div>
</div>

<div class="main">
  <nav class="sidebar">
    <div class="nav-item active" onclick="showPage('comments')" id="nav-comments">
      <span class="nav-icon">💬</span> Комментарии
    </div>
    <div class="nav-item" onclick="showPage('analyze')" id="nav-analyze">
      <span class="nav-icon">🧪</span> Анализ
    </div>
    <div class="nav-item" onclick="showPage('ml')" id="nav-ml">
      <span class="nav-icon">🤖</span> ML Метрики
    </div>
    <div class="nav-item" onclick="showPage('settings')" id="nav-settings">
      <span class="nav-icon">⚙️</span> Состояние
    </div>
    <div class="nav-item" onclick="showPage('faq')" id="nav-faq">
      <span class="nav-icon">❓</span> FAQ
    </div>
  </nav>

  <div class="content">
    <!-- Comments Tab -->
    <div class="tab-page active" id="page-comments">
      <div class="stats-row" id="statsRow">
        <div class="stat-card total"><div class="stat-label">Всего</div><div class="stat-value" id="st-total">—</div></div>
        <div class="stat-card ok"><div class="stat-label">Нормальные</div><div class="stat-value" id="st-ok">—</div></div>
        <div class="stat-card spam"><div class="stat-label">Спам</div><div class="stat-value" id="st-spam">—</div></div>
        <div class="stat-card negative"><div class="stat-label">Негатив</div><div class="stat-value" id="st-negative">—</div></div>
        <div class="stat-card pending"><div class="stat-label">Проверка</div><div class="stat-value" id="st-pending">—</div></div>
      </div>
      <div class="section-card">
        <div class="section-header">
          <div class="section-title">💬 Комментарии</div>
          <button class="btn btn-ghost" onclick="loadComments()">🔄 Обновить</button>
        </div>
        <div class="toolbar">
          <select class="select-filter" id="filterVerdict" onchange="loadComments()">
            <option value="">Все вердикты</option>
            <option value="ok">✅ Нормальные</option>
            <option value="spam">📢 Спам</option>
            <option value="negative">⚠️ Негативные</option>
            <option value="pending">🔍 На проверке</option>
          </select>
          <input class="search-input" type="text" id="filterText" placeholder="🔍 Поиск по тексту..." onkeyup="debounceSearch()">
          <button class="btn btn-primary" onclick="loadComments()">Поиск</button>
        </div>
        <div id="commentsList"><div class="loader">Загрузка...</div></div>
      </div>
    </div>

    <!-- Analyze Tab -->
    <div class="tab-page" id="page-analyze">
      <div class="section-card">
        <div class="section-header">
          <div class="section-title">🧪 Тестирование модели</div>
          <span class="badge badge-rf">RF Model</span>
        </div>
        <div class="analyze-area">
          <textarea class="analyze-input" id="analyzeText" placeholder="Введите комментарий для анализа..."></textarea>
          <div style="margin-top:12px; display:flex; gap:10px;">
            <button class="btn btn-primary" onclick="runAnalysis()">🤖 Анализировать</button>
            <button class="btn btn-ghost" onclick="fillExample('spam')">Пример спама</button>
            <button class="btn btn-ghost" onclick="fillExample('negative')">Пример негатива</button>
            <button class="btn btn-ghost" onclick="fillExample('ok')">Нормальный</button>
          </div>
          <div class="analyze-result" id="analyzeResult"></div>
        </div>
      </div>
    </div>

    <!-- ML Metrics Tab -->
    <div class="tab-page" id="page-ml">
      <div class="section-card">
        <div class="section-header">
          <div class="section-title">🤖 ML — Метрики модели</div>
        </div>
        <div id="mlContent"><div class="loader">Загрузка метрик...</div></div>
      </div>
    </div>

    <!-- Settings Tab -->
    <div class="tab-page" id="page-settings">
      <div class="section-card">
        <div class="section-header">
          <div class="section-title">⚙️ Состояние бота</div>
        </div>
        <div id="settingsContent"><div class="loader">Загрузка...</div></div>
      </div>
    </div>

    <!-- FAQ Tab -->
    <div class="tab-page" id="page-faq">
      <div class="section-card">
        <div class="section-header">
          <div class="section-title">❓ Управление FAQ</div>
          <a href="/faq" target="_blank" class="btn btn-ghost" style="font-size:12px;text-decoration:none">🔗 Открыть FAQ для подписчиков</a>
        </div>
        <div class="faq-add-form">
          <div class="faq-form-title">Добавить вопрос-ответ</div>
          <input class="faq-input" type="text" id="faqQuestion" placeholder="Вопрос...">
          <textarea class="faq-input faq-textarea" id="faqAnswer" placeholder="Ответ..."></textarea>
          <button class="btn btn-primary" onclick="addFaqItem()" style="align-self:flex-start">+ Добавить</button>
        </div>
        <div id="faqAdminList"><div class="loader">Загрузка...</div></div>
      </div>
    </div>
  </div>
</div>

<script>
const API = window.location.origin;
let _searchTimer = null;

(function() {
  const token = sessionStorage.getItem('vgtu_token');
  if (token) document.getElementById('authOverlay').classList.add('hidden');
})();

async function doAuth() {
  const login = document.getElementById('authLogin').value.trim();
  const password = document.getElementById('authPassword').value;
  const errEl = document.getElementById('authError');
  errEl.textContent = '';
  try {
    const res = await fetch(`${API}/api/auth`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({login, password})
    }).then(r=>r.json());
    if (res.success) {
      sessionStorage.setItem('vgtu_token', res.token);
      document.getElementById('authOverlay').classList.add('hidden');
      loadStats();
      loadComments();
    } else {
      errEl.textContent = res.error || 'Ошибка авторизации';
    }
  } catch(e) {
    errEl.textContent = 'Ошибка соединения с сервером';
  }
}

function showPage(page) {
  document.querySelectorAll('.tab-page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.getElementById('nav-' + page).classList.add('active');
  if (page === 'settings') loadSettings();
  if (page === 'ml') loadMLMetrics();
  if (page === 'faq') loadFaqAdmin();
}

async function loadStats() {
  try {
    const d = await fetch(`${API}/api/stats`).then(r=>r.json());
    document.getElementById('st-total').textContent = d.total ?? '—';
    document.getElementById('st-ok').textContent = d.ok ?? '—';
    document.getElementById('st-spam').textContent = d.spam ?? '—';
    document.getElementById('st-negative').textContent = d.negative ?? '—';
    document.getElementById('st-pending').textContent = d.pending ?? '—';
  } catch(e) {}
}

function debounceSearch() {
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(loadComments, 400);
}

async function loadComments() {
  const verdict = document.getElementById('filterVerdict').value;
  const search = document.getElementById('filterText').value;
  let url = `${API}/api/comments?limit=100`;
  if (verdict) url += `&verdict=${verdict}`;
  if (search) url += `&search=${encodeURIComponent(search)}`;

  document.getElementById('commentsList').innerHTML = '<div class="loader">Загрузка...</div>';
  try {
    const comments = await fetch(url).then(r=>r.json());
    if (!comments.length) {
      document.getElementById('commentsList').innerHTML = '<div class="loader">Нет данных</div>';
      return;
    }
    document.getElementById('commentsList').innerHTML = `
      <table class="data-table">
        <thead><tr>
          <th>Дата</th><th>Автор</th><th>Текст</th>
          <th>Вердикт ML</th><th>Спам</th><th>Негатив</th><th>Действие</th>
        </tr></thead>
        <tbody>${comments.map(c => `
          <tr id="row-${c.id}">
            <td style="white-space:nowrap;font-size:12px;font-family:var(--mono);color:var(--muted)">
              ${c.created_at ? new Date(c.created_at).toLocaleString('ru') : '—'}
             </td>
            <td style="font-size:13px;font-weight:600">${escHtml(c.author_name || '—')}</td>
            <td class="text-cell"><span class="text-preview" title="${escHtml(c.text)}">${escHtml(c.text)}</span></td>
            <td>${verdictBadge(c.auto_verdict)}</td>
            <td>${scoreBar(c.spam_score, 'spam')}</td>
            <td>${scoreBar(c.negative_score, 'negative')}</td>
            <td>${verdictActions(c)}</td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  } catch(e) {
    document.getElementById('commentsList').innerHTML = '<div class="loader" style="color:var(--danger)">Ошибка загрузки</div>';
  }
}

function verdictActions(c) {
  if (c.manual_verdict) {
    const icons = {ok:'✅ Одобрено', spam:'🗑️ Спам', negative:'🗑️ Негатив'};
    return `<span class="verdict-done">${icons[c.manual_verdict] || c.manual_verdict}</span>`;
  }
  if (c.auto_verdict === 'ok') {
    return `<div class="verdict-actions">
      <span class="verdict-auto-ok">✅ Норма</span>
      <button class="btn-spam"     onclick="setVerdict(${c.id}, 'spam',     this)">📢 Спам</button>
      <button class="btn-negative" onclick="setVerdict(${c.id}, 'negative', this)">⚠️ Негатив</button>
    </div>`;
  }
  return `<div class="verdict-actions">
    <button class="btn-spam"     onclick="setVerdict(${c.id}, 'spam',     this)">📢 Спам</button>
    <button class="btn-negative" onclick="setVerdict(${c.id}, 'negative', this)">⚠️ Негатив</button>
    <button class="btn-reject"   onclick="setVerdict(${c.id}, 'ok',       this)">✗ Оставить</button>
  </div>`;
}

async function setVerdict(commentId, verdict, btn) {
  const actionCell = btn.closest('td');
  actionCell.innerHTML = '<span style="color:var(--muted);font-size:12px">⏳ Обработка...</span>';
  try {
    const res = await fetch(`${API}/api/comments/${commentId}/verdict`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({verdict, admin: 'admin'})
    }).then(r=>r.json());
    if (res.success) {
      if (verdict === 'spam' || verdict === 'negative') {
        const deleted = res.deleted_vk ? ' · удалён из ВК ✓' : ' · не удалён из ВК ⚠️';
        actionCell.innerHTML = `<span class="verdict-done" style="color:var(--danger)">🗑️ Удалён${deleted}</span>`;
      } else {
        actionCell.innerHTML = `<span class="verdict-done" style="color:var(--muted)">✗ Оставлен</span>`;
      }
      loadStats();
    } else {
      actionCell.innerHTML = `<span style="color:var(--danger);font-size:12px">Ошибка: ${res.error||'?'}</span>`;
    }
  } catch(e) {
    actionCell.innerHTML = `<span style="color:var(--danger);font-size:12px">Ошибка сети</span>`;
  }
}

function verdictBadge(v) {
  const map = {ok:'✅ Норма', spam:'📢 Спам', negative:'⚠️ Негатив', pending:'🔍 Проверка'};
  return `<span class="badge badge-${v}">${map[v]||v}</span>`;
}

function scoreBar(score, type) {
  const pct = Math.round((score||0)*100);
  return `<div class="score-bar">
    <div class="score-track"><div class="score-fill ${type}" style="width:${pct}%"></div></div>
    <span class="score-text">${pct}%</span>
  </div>`;
}

function escHtml(str) {
  return (str||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

const EXAMPLES = {
  spam: 'Купи iPhone 15 дёшево! Переходи на t.me/best_deals и получи скидку 70% прямо сейчас!',
  negative: 'Ужасный университет, ненавижу это место. Расписание кошмарное, всё отвратительно.',
  ok: 'Добрый день, подскажите пожалуйста когда будет следующая встреча студенческого совета?'
};

function fillExample(type) {
  document.getElementById('analyzeText').value = EXAMPLES[type];
}

async function runAnalysis() {
  const text = document.getElementById('analyzeText').value.trim();
  if (!text) return;
  const el = document.getElementById('analyzeResult');
  el.classList.add('show');
  el.innerHTML = '⏳ Анализируем с помощью Random Forest...';
  try {
    const res = await fetch(`${API}/analyze`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text})
    }).then(r=>r.json());
    const vColors = {ok:'var(--ok)', spam:'var(--warn)', negative:'var(--danger)', pending:'var(--accent2)'};
    const vEmoji = {ok:'✅', spam:'📢', negative:'⚠️', pending:'🔍'};
    const color = vColors[res.verdict] || 'var(--text)';
    el.innerHTML = `
      <div class="result-verdict" style="color:${color}">${vEmoji[res.verdict]||''} ${res.verdict?.toUpperCase()}</div>
      <div style="color:var(--muted);margin-bottom:12px">Метод: <span style="color:var(--accent);font-weight:700">${res.method}</span></div>
      <div>Спам: <b style="color:var(--warn)">${Math.round((res.spam_score||0)*100)}%</b></div>
      <div>Негатив: <b style="color:var(--danger)">${Math.round((res.negative_score||0)*100)}%</b></div>
      <div style="margin-top:8px;color:var(--muted);font-size:11px">${res.details||''}</div>
    `;
  } catch(e) {
    el.innerHTML = `<span style="color:var(--danger)">Ошибка: ${e.message}</span>`;
  }
}

async function loadFaqAdmin() {
  const el = document.getElementById('faqAdminList');
  el.innerHTML = '<div class="loader">Загрузка...</div>';
  try {
    const items = await fetch(`${API}/api/faq`).then(r=>r.json());
    if (!items.length) {
      el.innerHTML = '<div class="loader">Нет вопросов-ответов. Добавьте первый!</div>';
      return;
    }
    el.innerHTML = `<div class="faq-list">${items.map(f=>`
      <div class="faq-item" id="faq-${f.id}">
        <div class="faq-q">❓ ${escHtml(f.question)}</div>
        <div class="faq-a">${escHtml(f.answer)}</div>
        <div class="faq-item-footer">
          <button class="btn-del-faq" onclick="deleteFaq(${f.id})">🗑 Удалить</button>
        </div>
      </div>`).join('')}</div>`;
  } catch(e) {
    el.innerHTML = '<div class="loader" style="color:var(--danger)">Ошибка загрузки FAQ</div>';
  }
}

async function addFaqItem() {
  const q = document.getElementById('faqQuestion').value.trim();
  const a = document.getElementById('faqAnswer').value.trim();
  if (!q || !a) { alert('Заполните вопрос и ответ'); return; }
  try {
    const res = await fetch(`${API}/api/faq`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:q, answer:a})
    }).then(r=>r.json());
    if (res.success) {
      document.getElementById('faqQuestion').value = '';
      document.getElementById('faqAnswer').value = '';
      loadFaqAdmin();
    }
  } catch(e) { alert('Ошибка при добавлении'); }
}

async function deleteFaq(id) {
  if (!confirm('Удалить этот вопрос-ответ?')) return;
  await fetch(`${API}/api/faq/${id}`, {method:'DELETE'});
  document.getElementById('faq-'+id)?.remove();
}

async function loadMLMetrics() {
  document.getElementById('mlContent').innerHTML = '<div class="loader">Загрузка метрик...</div>';
  try {
    const d = await fetch(`${API}/api/ml-metrics`).then(r=>r.json());
    if (d.error) {
      document.getElementById('mlContent').innerHTML = `
        <div style="padding:30px;text-align:center;">
          <div style="font-size:40px;margin-bottom:12px">🤖</div>
          <div style="color:var(--muted);margin-bottom:16px">${d.error}</div>
          <div style="font-size:12px;color:var(--muted);font-family:var(--mono)">Запустите: python train_model.py</div>
        </div>`;
      return;
    }
    const rep = d.report || {};
    const dist = d.distribution || {};
    const total = d.total_samples || 0;
    const classes = ['ok','spam','negative'];
    document.getElementById('mlContent').innerHTML = `
      <div class="ml-grid">
        <div class="ml-metric">
          <div class="ml-metric-title">📊 Датасет</div>
          <div style="font-size:30px;font-weight:800;font-family:var(--mono);color:var(--accent);margin-bottom:12px">${total.toLocaleString()}</div>
          <div style="font-size:11px;color:var(--muted);margin-bottom:14px">обучающих примеров</div>
          ${classes.map(c => {
            const cnt = dist[c] || 0;
            const pct = total ? Math.round(cnt/total*100) : 0;
            return `<div class="progress-item">
              <div class="progress-label"><span>${c}</span><span style="font-family:var(--mono)">${cnt} (${pct}%)</span></div>
              <div class="progress-bar"><div class="progress-fill fill-${c==='ok'?'ok':c==='spam'?'spam':'negative'}" style="width:${pct}%"></div></div>
            </div>`;
          }).join('')}
        </div>
        <div class="ml-metric">
          <div class="ml-metric-title">🎯 Точность по классам</div>
          ${classes.map(c => {
            const m = rep[c] || {};
            const prec = Math.round((m.precision||0)*100);
            const rec = Math.round((m.recall||0)*100);
            const f1 = Math.round((m['f1-score']||0)*100);
            return `<div style="margin-bottom:14px">
              <div style="font-size:12px;font-weight:700;margin-bottom:6px;color:var(--text)">${c.toUpperCase()}</div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
                ${[['Precision',prec],['Recall',rec],['F1',f1]].map(([l,v])=>`
                  <div style="background:var(--surface);border-radius:8px;padding:10px;text-align:center;border:1px solid var(--border)">
                    <div style="font-size:20px;font-weight:800;font-family:var(--mono);color:var(--accent)">${v}%</div>
                    <div style="font-size:10px;color:var(--muted);margin-top:2px">${l}</div>
                  </div>`).join('')}
              </div>
            </div>`;
          }).join('')}
        </div>
      </div>
      <div style="padding:0 22px 20px;font-size:11px;color:var(--muted);font-family:var(--mono)">
        Алгоритм: Random Forest (300 деревьев) · TF-IDF char n-grams (2-4) · Ручные признаки (12) · HuggingFace datasets
      </div>`;
  } catch(e) {
    document.getElementById('mlContent').innerHTML = `<div class="loader" style="color:var(--danger)">Ошибка: ${e.message}</div>`;
  }
}

async function loadSettings() {
  try {
    const s = await fetch(`${API}/api/settings`).then(r=>r.json());
    const descMap = {
      spam_threshold: 'Порог классификации спама (0.0–1.0)',
      negative_threshold: 'Порог классификации негатива (0.0–1.0)',
      auto_delete_spam: 'Автоматически удалять спам',
      faq_enabled: 'Включить автоответы FAQ',
      notify_on_negative: 'Уведомлять админа о негативе',
    };
    document.getElementById('settingsContent').innerHTML = `
      <div class="settings-grid">
        ${Object.entries(s).map(([k,v])=>`
          <div class="setting-item">
            <div class="setting-key">${k}</div>
            <div class="setting-val">${escHtml(v)}</div>
            <div class="setting-desc">${descMap[k]||''}</div>
          </div>`).join('')}
      </div>`;
  } catch(e) {
    document.getElementById('settingsContent').innerHTML = '<div class="loader" style="color:var(--danger)">Ошибка загрузки</div>';
  }
}

loadStats();
loadComments();
setInterval(()=>{ loadStats(); }, 15000);
</script>
</body>
</html>"""

print("✅ DASHBOARD_HTML и FAQ_PAGE_HTML добавлены в main.py")
