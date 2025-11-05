"""
FastAPI application for llama-cpp chat interface.

This module provides a REST API for interacting with the llama-cpp language model.
It automatically downloads the model if not present and exposes endpoints for
health checks and chat completion.
"""

import json
import logging
import os
import re
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------
# Configuration
# -------------------------
class Config:
    """Application configuration."""

    MODEL_FILE: str = "phi-2.Q4_K_M.gguf"
    MODEL_URL: str = (
        "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
    )
    MODEL_DIR: Path = Path("models")
    N_CTX: int = 2048
    N_THREADS: int = 2
    CHAT_FORMAT: str = "llama-2"
    MAX_DOWNLOAD_RETRIES: int = 3
    DOWNLOAD_TIMEOUT: int = 3600  # 1 hour
    MAX_RESPONSE_LENGTH: int = 2000

    @property
    def model_path(self) -> Path:
        """Get full model path."""
        return self.MODEL_DIR / self.MODEL_FILE


config = Config()


# -------------------------
# Pydantic Models
# -------------------------
class HealthInsight(BaseModel):
    """Schema for a single health insight."""

    type: Literal["info", "warning", "alert", "success"] = Field(
        description="Type of health insight"
    )
    message: str = Field(
        min_length=1, max_length=200, description="Short headline for the insight"
    )
    priority: Literal["critical", "high", "medium", "low"] = Field(
        description="Priority level of the insight"
    )
    details: str = Field(
        min_length=1,
        max_length=500,
        description="Detailed explanation with actionable advice",
    )
    metric: str = Field(
        min_length=1,
        max_length=100,
        description="Health metric this insight relates to",
    )


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    prompt: str = Field(
        min_length=1,
        max_length=5000,
        description="User prompt for generating health insights",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and sanitize prompt."""
        v = v.strip()
        if not v:
            raise ValueError("Prompt cannot be empty or only whitespace")
        return v


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    insights: List[HealthInsight] = Field(
        description="List of generated health insights"
    )


class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str
    model: str
    model_loaded: bool
    model_path: str


# -------------------------
# Model Manager
# -------------------------
class ModelManager:
    """Manages model download and initialization."""

    def __init__(self, cfg: Config):
        self.config = cfg
        self._model: Optional[Llama] = None

    @property
    def model(self) -> Llama:
        """Get the loaded model instance."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    async def download_model(self) -> None:
        """Download model file if not present."""
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = self.config.model_path

        if model_path.exists():
            logger.info(f"Model already exists at {model_path}")
            return

        logger.info(f"Downloading model from {self.config.MODEL_URL}")

        # Try wget first
        if self._try_wget_download(model_path):
            return

        # Fallback to httpx for streaming download
        await self._httpx_download(model_path)

    def _try_wget_download(self, model_path: Path) -> bool:
        """Attempt download using wget."""
        try:
            result = subprocess.run(
                ["wget", "--version"], capture_output=True, timeout=5, check=False
            )
            if result.returncode != 0:
                return False

            logger.info("Using wget for download")
            result = subprocess.run(
                [
                    "wget",
                    "-O",
                    str(model_path),
                    "--progress=bar:force",
                    self.config.MODEL_URL,
                ],
                check=True,
                timeout=self.config.DOWNLOAD_TIMEOUT,
            )
            logger.info("Model downloaded successfully via wget")
            return True

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            logger.warning(f"wget download failed: {e}")
            return False

    async def _httpx_download(self, model_path: Path) -> None:
        """Download model using httpx with progress tracking."""
        try:
            async with httpx.AsyncClient(
                timeout=self.config.DOWNLOAD_TIMEOUT
            ) as client:
                async with client.stream("GET", self.config.MODEL_URL) as response:
                    response.raise_for_status()

                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    with open(model_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                if (
                                    downloaded % (1024 * 1024 * 10) == 0
                                ):  # Log every 10MB
                                    logger.info(f"Download progress: {progress:.1f}%")

                    logger.info("Model downloaded successfully via httpx")

        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download model: {e}") from e

    def load_model(self) -> None:
        """Load the model into memory."""
        model_path = self.config.model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        logger.info(f"Loading model from {model_path}")

        try:
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.N_CTX,
                n_threads=self.config.N_THREADS,
                chat_format=self.config.CHAT_FORMAT,
                verbose=False,
            )
            logger.info("Model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    async def initialize(self) -> None:
        """Initialize model (download if needed and load)."""
        try:
            await self.download_model()
            self.load_model()
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise


# -------------------------
# Chat Service
# -------------------------
class ChatService:
    """Service for generating chat completions."""

    SYSTEM_PROMPT = """You produce 3-7 structured health insights based on user data.
Return ONLY a valid JSON array. No explanation, no markdown, no code blocks.
Each item must look exactly like:

{
  "type": "info",
  "message": "Short headline",
  "priority": "medium",
  "details": "One sentence explanation with actionable suggestion.",
  "metric": "The health metric name"
}

Valid types: info, warning, alert, success
Valid priorities: critical, high, medium, low"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def _clean_json_response(self, raw: str) -> str:
        """Clean and extract JSON from model response."""
        # Remove markdown code blocks
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        # Try to find JSON array in response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return match.group(0)

        return raw

    def _create_fallback_insight(self, raw_response: str) -> List[Dict[str, Any]]:
        """Create fallback insight when JSON parsing fails."""
        return [
            {
                "type": "info",
                "message": "General Health Insight",
                "priority": "medium",
                "details": raw_response[: config.MAX_RESPONSE_LENGTH],
                "metric": "general",
            }
        ]

    def generate_insights(self, prompt: str) -> List[HealthInsight]:
        """Generate health insights from prompt."""
        if not self.model_manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        try:
            response = self.model_manager.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": prompt.strip()},
                ],
                temperature=0.7,
                max_tokens=2048,
                top_p=0.95,
            )

            raw_content = response["choices"][0]["message"]["content"].strip()
            cleaned_content = self._clean_json_response(raw_content)

            try:
                parsed = json.loads(cleaned_content)

                # Ensure it's a list
                if not isinstance(parsed, list):
                    parsed = [parsed]

                # Validate each insight
                insights = [HealthInsight(**item) for item in parsed]

                logger.info(f"Generated {len(insights)} insights")
                return insights

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"JSON parsing/validation failed: {e}. Raw: {raw_content[:200]}"
                )
                fallback_data = self._create_fallback_insight(raw_content)
                return [HealthInsight(**item) for item in fallback_data]

        except Exception as e:
            logger.error(f"Error generating insights: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate insights: {str(e)}",
            ) from e


# -------------------------
# Application Lifecycle
# -------------------------
model_manager = ModelManager(config)
chat_service = ChatService(model_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting application")
    try:
        await model_manager.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        # Don't crash the app, allow health checks to work

    yield

    # Shutdown
    logger.info("Shutting down application")


# -------------------------
# FastAPI Application
# -------------------------
app = FastAPI(
    title="Health Insights API",
    description="Generate health insights using Phi-2 language model",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# API Endpoints
# -------------------------
@app.get(
    "/",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check if the API and model are operational",
)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint."""
    return HealthCheckResponse(
        status="ok",
        model=config.MODEL_FILE,
        model_loaded=model_manager.is_loaded,
        model_path=str(config.model_path),
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Generate Health Insights",
    description="Generate structured health insights based on provided health data",
    status_code=status.HTTP_200_OK,
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Generate health insights from user prompt."""
    insights = chat_service.generate_insights(request.prompt)
    return ChatResponse(insights=insights)


# -------------------------
# Application Entry Point
# -------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
