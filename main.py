"""
FastAPI application for llama-cpp chat interface.

This module provides a REST API for interacting with the llama-cpp language model.
It automatically downloads the model if not present and exposes endpoints for
health checks and chat completion.
"""

import os
import json
import subprocess
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# -------------------------
# MODEL CONFIG
# -------------------------
MODEL_FILE = "mistral-7b-instruct-q4_0.gguf"
MODEL_URL = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

# Global model variable
model: Optional[Llama] = None


# -------------------------
# DOWNLOAD MODEL IF MISSING
# -------------------------
def download_model_if_missing():
    """Download the model file if it doesn't exist locally."""
    if not os.path.exists(MODEL_FILE):
        print(f"⚠ Model not found. Downloading {MODEL_FILE} ...")
        try:
            # Use subprocess instead of os.system for better error handling
            subprocess.run(
                ["wget", "-O", MODEL_FILE, MODEL_URL],
                check=True,
                capture_output=True,
                text=True,
            )
            print("✅ Model downloaded successfully.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"❌ Model download failed: {e.stderr}")
        except FileNotFoundError:
            # wget not available, try with curl or urllib
            try:
                import urllib.request

                print("wget not found, using urllib instead...")
                urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
                print("✅ Model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"❌ Model download failed: {str(e)}")


# -------------------------
# LOAD MODEL
# -------------------------
def load_model() -> Llama:
    """Load the Llama model into memory."""
    download_model_if_missing()
    print("⏳ Loading model... This may take 20–60 seconds.")
    try:
        llm = Llama(
            model_path=MODEL_FILE,
            n_ctx=4096,
            n_threads=4,
            chat_format="mistral-instruct",  # Fixed: correct chat format
        )
        print("✅ Model loaded & ready.")
        return llm
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {str(e)}")


# -------------------------
# STARTUP EVENT
# -------------------------
@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global model
    try:
        model = load_model()
    except Exception as e:
        print(f"❌ Error during startup: {str(e)}")
        # Allow app to start even if model fails to load
        # This way you can still access health check endpoint


# -------------------------
# ENABLE CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# REQUEST BODY
# -------------------------
class Prompt(BaseModel):
    prompt: str


# -------------------------
# HEALTH CHECK ROUTE
# -------------------------
@app.get("/")
def home():
    """Health check endpoint."""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "ok",
        "message": "Mistral API is running ✅",
        "model_status": model_status,
    }


# -------------------------
# MAIN CHAT / INSIGHT ROUTE
# -------------------------
@app.post("/chat")
def chat(request: Prompt):
    """
    Generate health insights based on provided data.

    Args:
        request: Prompt object containing the user's input

    Returns:
        JSON response with health insights
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait or restart the service.",
        )

    # System prompt for structured JSON output
    system_prompt = """You generate 3-7 health insights based on provided data. Output ONLY a valid JSON array. Each item must follow exactly:

{
  "type": "info" | "warning" | "alert" | "success",
  "message": "Short headline",
  "priority": "critical" | "high" | "medium" | "low",
  "details": "1-2 sentence explanation with advice",
  "metric": "The health metric this relates to"
}

Do not include any text before or after the JSON array."""

    try:
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        raw = response["choices"][0]["message"]["content"].strip()
        # Remove markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()
        # Try to parse JSON
        try:
            parsed = json.loads(raw)
            # Ensure it's a list
            if not isinstance(parsed, list):
                parsed = [parsed]
            return {"insights": parsed}
        except json.JSONDecodeError as e:
            # If JSON parsing fails, wrap the response
            print(f"⚠ JSON parsing failed: {e}")
            return {
                "insights": [
                    {
                        "type": "info",
                        "message": "General Health Insight",
                        "priority": "medium",
                        "details": raw[:500],  # Limit length
                        "metric": "general",
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
