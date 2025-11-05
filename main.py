"""
FastAPI application for GPT4All chat interface.

This module provides a REST API for interacting with the GPT4All language model.
It automatically downloads the model if not present and exposes endpoints for
health checks and chat completion.
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from gpt4all import GPT4All

app = FastAPI()

# The filename we will store the model as locally:
MODEL_FILE = "gpt4all-groovy.gguf"

# Direct download URL (works without login/token)
MODEL_URL = "https://huggingface.co/nomic-ai/gpt4all-j/resolve/main/ggml-gpt4all-j-v1.3-groovy.gguf"


def download_model_if_missing():
    """Download the model if it does not exist yet."""
    if not os.path.exists(MODEL_FILE):
        print(f"⚠ Model not found. Downloading {MODEL_FILE} ...")
        exit_code = os.system(f"wget -O {MODEL_FILE} {MODEL_URL}")
        if exit_code != 0:
            raise RuntimeError("❌ Model download failed. Check your internet connection or URL.")
        print("✅ Model downloaded successfully.")


def load_model():
    """Ensure model exists and then load it."""
    download_model_if_missing()
    print("⏳ Loading model… this may take a moment.")
    # Fixed: GPT4All expects 'model_name' parameter, not 'model_path'
    # And it should just be the filename if it's in the current directory
    loaded_model = GPT4All(model_name=MODEL_FILE)
    print("✅ Model loaded.")
    return loaded_model


model = load_model()


class Prompt(BaseModel):
    """Request model for chat endpoint containing the user's prompt."""
    prompt: str


@app.get("/")
def root():
    """Root endpoint to check if the API is running."""
    return {"message": "GPT4All API is running!"}


@app.post("/chat")
def chat(req: Prompt):
    """
    Chat endpoint that accepts a prompt and returns a GPT4All generated response.
    
    Args:
        req: Prompt object containing the user's prompt text
        
    Returns:
        dict: Dictionary containing the model's response
    """
    with model.chat_session():
        response = model.generate(req.prompt, max_tokens=200)
    return {"response": response}
