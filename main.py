"""
FastAPI application for llama-cpp chat interface.

This module provides a REST API for interacting with the llama-cpp language model.
It automatically downloads the model if not present and exposes endpoints for
health checks and chat completion.
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

MODEL_FILE = "mistral-7b-instruct-q4_0.gguf"
MODEL_URL = (
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/"
    "resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)


def download_model_if_missing():
    """Download the model if it does not exist yet."""
    if not os.path.exists(MODEL_FILE):
        print(f"⚠ Model not found. Downloading {MODEL_FILE} ...")
        exit_code = os.system(f"wget -O {MODEL_FILE} {MODEL_URL}")
        if exit_code != 0:
            raise RuntimeError("❌ Model download failed. URL may have changed.")
        print("✅ Download complete.")


def load_model():
    """Ensure model exists and then load it."""
    download_model_if_missing()
    print("⏳ Loading model (this can take 15–40s)...")
    loaded_model = Llama(
        model_path=MODEL_FILE,
        n_ctx=4096,
        n_threads=4,
        chat_format="mistral"
    )
    print("✅ Model loaded and ready!")
    return loaded_model


model = load_model()


class Prompt(BaseModel):
    """Request model for chat endpoint containing the user's prompt."""
    prompt: str


@app.get("/")
def home():
    """Root endpoint to check if the API is running."""
    return {"message": "Mistral API online ✅"}


@app.post("/chat")
def chat(request: Prompt):
    """
    Chat endpoint that accepts a prompt and returns a Mistral generated response.
    
    Args:
        request: Prompt object containing the user's prompt text
        
    Returns:
        dict: Dictionary containing the model's response
    """
    output = model.create_chat_completion(
        messages=[{"role": "user", "content": request.prompt}]
    )
    reply = output["choices"][0]["message"]["content"]
    return {"response": reply}
