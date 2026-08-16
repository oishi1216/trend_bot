from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .config import Settings
from .adaptive_dashboard import load_adaptive_dashboard
from .research_dashboard import load_research_dashboard
from .engine import TradingEngine
from .feedback import FeedbackService
from .storage import Storage

settings = Settings.from_env()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
storage = Storage(settings.db_path, settings.paper_initial_balance)
engine = TradingEngine(settings, storage)
feedback_service = FeedbackService(settings, storage)
templates = Jinja2Templates(directory="app/templates")
run_lock = asyncio.Lock()
adaptive_db_path = "data/adaptive_paper.sqlite3"
adaptive_task_log_dir = "data/task_logs"
research_data_dir = "data/fx_research"
research_cache_path = "data/research/fx_adaptive_backtest.json"


async def execute_run(force: bool = False):
    if run_lock.locked():
        raise HTTPException(status_code=409, detail="Engine is already running")
    async with run_lock:
        summary = await asyncio.to_thread(engine.run_once, force)
    # Trading has already completed; feedback failure must never affect execution.
    feedback = await asyncio.to_thread(
        feedback_service.analyze_run, summary, force=False
    )
    summary["openai_feedback"] = feedback
    return summary


async def scheduler_loop() -> None:
    while True:
        try:
            now = datetime.now(timezone.utc)
            due = (now.hour, now.minute) >= (
                settings.schedule_hour_utc,
                settings.schedule_minute_utc,
            )
            key = f"scheduled_run:{now.date().isoformat()}"
            weekday = now.weekday() < 5
            if weekday and due and storage.get_kv(key) != "done":
                try:
                    await execute_run(force=False)
                    storage.set_kv(key, "done")
                except HTTPException:
                    pass
                except Exception as exc:
                    logger.exception("Scheduled run failed")
                    storage.add_event("ERROR", "scheduled_run_failed", {"error": str(exc)})
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Scheduler loop error")
            await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(scheduler_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"app_name": settings.app_name},
    )


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/status")
def status():
    return engine.status()


@app.get("/api/events")
def events(limit: int = Query(default=50, ge=1, le=500)):
    return storage.recent_events(limit)


@app.get("/api/adaptive/status")
def adaptive_status():
    dashboard = load_adaptive_dashboard(adaptive_db_path, adaptive_task_log_dir)
    return {
        "status": dashboard.status,
        "runs": dashboard.runs,
        "latest_run": dashboard.latest_run,
        "task_status": dashboard.task_status,
    }



@app.get("/api/research/status")
def research_status():
    dashboard = load_research_dashboard(research_data_dir, research_cache_path, settings)
    return dashboard.to_dict()

@app.post("/api/run")
async def run(force: bool = Query(default=False)):
    return await execute_run(force=force)


@app.get("/api/feedback/latest")
def latest_feedback():
    event = storage.latest_event("openai_feedback")
    if event is None:
        raise HTTPException(status_code=404, detail="No OpenAI feedback has been generated")
    return event


@app.post("/api/feedback/run")
async def run_feedback(force: bool = Query(default=True)):
    event = storage.latest_event("engine_run")
    if event is None:
        raise HTTPException(status_code=404, detail="No engine run report is available")
    return await asyncio.to_thread(
        feedback_service.analyze_run, event["payload"], force=force
    )
