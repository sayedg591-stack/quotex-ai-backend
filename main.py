import asyncio
from datetime import datetime, timezone
import logging
import os
import time
from typing import Any, Dict, Literal, Optional
import uuid

from dotenv import load_dotenv
   
# 1. Environment Variables Load - MUST BE FIRST
load_dotenv()

# 2. Logging Config BEFORE custom module imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("ai_signal_system")

from contextlib import asynccontextmanager
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Security,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn

# 3. Custom Module Imports using Central Config Architecture
from config import settings
from pocket_feed import PocketOptionFeed
from live_fetcher import LiveFetcher
from strategy_engine import StrategyEngine
from trust_engine import TrustEngine

# Global Settings & Dual Source Fallbacks
IS_DEBUG = settings.ENV.lower() in ("development", "debug")
API_KEY = getattr(settings, "API_KEY", "") or os.getenv("API_KEY", "")

# Fix #2: Correct Data Pipeline Assembly (PocketOptionFeed -> LiveFetcher -> StrategyEngine)
try:
    pocket_feed = PocketOptionFeed(allow_otc=settings.ALLOW_OTC)
    live_fetcher = LiveFetcher(feed=pocket_feed)
except Exception as init_err:
    logger.critical(f"PocketOptionFeed/LiveFetcher initialization failed: {init_err}", exc_info=True)
    pocket_feed = None
    live_fetcher = None

strategy_engine = StrategyEngine(fetcher=live_fetcher)
trust_engine = TrustEngine(
    github_token=settings.GITHUB_TOKEN,
    repo_owner=settings.REPO_OWNER,
    repo_name=settings.REPO_NAME,
    branch=settings.GITHUB_BRANCH,
)

# In-Memory Rate Limiter (Prevents Feed Abuse)
RATE_LIMIT_STORE: Dict[str, list] = {}
MAX_REQUESTS_PER_MINUTE = 15


async def rate_limiter(request: Request):
    """Fix #1: Real Client IP Resolver behind Reverse Proxies (Cloudflare/Render/Nginx)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.headers.get("CF-Connecting-IP") or (
            request.client.host if request.client else "unknown"
        )

    now = time.time()
    timestamps = RATE_LIMIT_STORE.get(client_ip, [])
    timestamps = [ts for ts in timestamps if now - ts < 60]

    if len(timestamps) >= MAX_REQUESTS_PER_MINUTE:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 15 requests per minute allowed.",
        )

    timestamps.append(now)
    RATE_LIMIT_STORE[client_ip] = timestamps


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Signal System Backend (Pocket Option Engine)...")

    missing_configs = []
    if not settings.GITHUB_TOKEN:
        missing_configs.append("GITHUB_TOKEN")
    if not settings.REPO_OWNER:
        missing_configs.append("REPO_OWNER")
    if not settings.REPO_NAME:
        missing_configs.append("REPO_NAME")
    if not API_KEY:
        missing_configs.append("API_KEY")

    if missing_configs:
        err_msg = f"CRITICAL STARTUP WARNING: Missing env variables: {', '.join(missing_configs)}"
        logger.warning(err_msg)
        if not IS_DEBUG:
            logger.error("Running in production with missing critical security tokens!")

    # Explicit WebSocket connect on startup
    if pocket_feed:
        connected = await pocket_feed.connect()
        if connected:
            logger.info("[STARTUP] Pocket Option WebSocket Feed connected successfully.")
        else:
            logger.warning("[STARTUP WARNING] Pocket Option Feed connection deferred/failed.")

    yield

    logger.info("Shutting down AI Signal System Backend...")
    if pocket_feed and hasattr(pocket_feed, "close"):
        try:
            await pocket_feed.close()
        except Exception as close_err:
            logger.warning(f"Error during PocketOptionFeed closure: {close_err}")


app = FastAPI(
    title="Enterprise AI Signal System API",
    description="Production Signal API with Key Protection, Rate Limiting & GitHub Audit Trail",
    version="2.0.0",
    lifespan=lifespan,
)

# API Key Security Setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    if not API_KEY:
        logger.error("API_KEY is not configured on the server.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server security configuration error.",
        )

    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing X-API-Key header.",
        )
    return api_key


# Fix #4: Production-Safe CORS Config
raw_origins = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


# Pydantic Response Schemas
class RootResponse(BaseModel):
    status: str = Field(..., examples=["online"])
    message: str = Field(..., examples=["AI Signal Server is Running"])


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    service: str = Field(..., examples=["AI Signal Backend"])
    fetcher_status: str = Field(..., examples=["ready"])
    timestamp: str = Field(..., examples=["2026-10-02 12:00:00 UTC"])


class SignalResponse(BaseModel):
    status: str = Field(..., examples=["SUCCESS"])
    id: Optional[str] = Field(None, examples=["SIG-1727616600-A1B2C3D4"])
    asset: Optional[str] = Field(None, examples=["EURUSD_otc"])
    direction: Optional[str] = Field(None, examples=["CALL"])
    score: Optional[str] = Field(None, examples=["8.5/10"])
    percentage: Optional[str] = Field(None, examples=["85.0%"])
    timeframe: Optional[str] = Field(None, examples=["1m"])
    timestamp: Optional[str] = Field(None, examples=["2026-10-02 12:00:00 UTC"])
    hash: Optional[str] = Field(None, examples=["a1b2c3d4..."])
    github_status: Optional[str] = Field(None, examples=["QUEUED"])
    message: Optional[str] = Field(None, examples=["65%+ confirmation pawa jayni"])


def parse_score_to_percentage(score_val: Any) -> float:
    """Safely converts scores ('8.5/10', '80%', 0.8, 80) into a clean percentage float."""
    try:
        if isinstance(score_val, (int, float)):
            val = float(score_val)
            return val * 100.0 if val <= 1.0 else val

        if isinstance(score_val, str):
            clean_str = score_val.strip().replace("%", "")
            if "/" in clean_str:
                parts = clean_str.split("/")
                if len(parts) == 2:
                    num, den = float(parts[0]), float(parts[1])
                    if den > 0:
                        return (num / den) * 100.0
            else:
                val = float(clean_str)
                return val * 100.0 if val <= 1.0 else val
    except Exception as err:
        logger.warning(f"Score parsing failed for '{score_val}': {err}")
    return 0.0


async def async_github_commit(signal_data: Dict[str, Any], signal_hash: str):
    """Executes async GitHub commit via TrustEngine directly."""
    try:
        url = await trust_engine.commit_to_github(signal_data, signal_hash)
        logger.info(f"Signal {signal_data.get('id')} committed to GitHub: {url}")
    except Exception as err:
        logger.error(f"GitHub commit failed for {signal_data.get('id')}: {err}")


@app.get("/", response_model=RootResponse)
def root():
    return RootResponse(status="online", message="AI Signal Server is Running")


@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
async def health_check():
    fetcher_ok = pocket_feed is not None and getattr(pocket_feed, "_is_connected", False)
    current_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return HealthResponse(
        status="healthy",
        service="AI Signal Backend",
        fetcher_status="ready" if fetcher_ok else "degraded",
        timestamp=current_utc,
    )


@app.get(
    "/api/v1/get-signal",
    response_model=SignalResponse,
    dependencies=[Depends(verify_api_key), Depends(rate_limiter)],
)
async def get_on_demand_signal(
    background_tasks: BackgroundTasks,
    timeframe: Literal["1m", "5m", "10m", "15m", "30m", "1hr"] = "1m",
):
    try:
        # Fix #3: Auto-reconnect check before throwing 503
        if not pocket_feed:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API DISCONNECTED: Market feed is uninitialized.",
            )

        if not getattr(pocket_feed, "_is_connected", False):
            logger.info("Market feed disconnected. Attempting auto-reconnect...")
            connected = await pocket_feed.connect()
            if not connected:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="API DISCONNECTED: Market feed is offline and reconnect failed.",
                )

        live_scan = await strategy_engine.scan_best_stable_market(timeframe=timeframe)

        if not live_scan or live_scan.get("action") == "HOLD" or live_scan.get("status") != "SIGNAL":
            reason_msg = (
                live_scan.get("reason", "65%+ confirmation pawa jayni")
                if live_scan
                else "No market data found"
            )
            return SignalResponse(status="NO_SIGNAL", timeframe=timeframe, message=reason_msg)

        symbol = live_scan.get("symbol") or live_scan.get("pair")
        if not symbol:
            return SignalResponse(
                status="NO_SIGNAL",
                timeframe=timeframe,
                message="Asset symbol missing in market scan data.",
            )

        raw_score = live_scan.get("score", "0/10")
        direction = live_scan.get("direction") or live_scan.get("action") or "HOLD"

        score_percentage = parse_score_to_percentage(raw_score)

        if score_percentage < 65.0 or direction in ["NO_SIGNAL", "HOLD"]:
            return SignalResponse(
                status="NO_SIGNAL",
                timeframe=timeframe,
                message=f"65%+ confirmation pawa jayni (Current Score: {raw_score} / {score_percentage:.1f}%)",
            )

        current_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        signal_id = f"SIG-{int(time.time())}-{uuid.uuid4().hex[:8].upper()}"

        signal_data = {
            "id": signal_id,
            "timestamp": current_utc_str,
            "asset": symbol,
            "direction": direction,
            "score": str(raw_score),
            "timeframe": timeframe,
        }

        signal_hash = trust_engine.generate_signal_hash(signal_data)
        background_tasks.add_task(async_github_commit, signal_data, signal_hash)

        return SignalResponse(
            status="SUCCESS",
            id=signal_data["id"],
            asset=signal_data["asset"],
            direction=signal_data["direction"],
            score=signal_data["score"],
            percentage=f"{score_percentage:.1f}%",
            timeframe=timeframe,
            timestamp=signal_data["timestamp"],
            hash=signal_hash,
            github_status="QUEUED",
        )

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Unhandled Engine Error: {err}", exc_info=True)
        error_message = (
            f"Engine Error: {str(err)}"
            if IS_DEBUG
            else "An internal server error occurred while processing signal."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message,
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=IS_DEBUG)
