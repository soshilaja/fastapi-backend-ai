"""
FastAPI application for llama-cpp chat interface.

This module provides a REST API for interacting with the GPT4All language model.
It automatically downloads the model if not present and exposes endpoints for
health checks and chat completion.
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

MODEL_FILE = "mistral-7b-instruct-q4_0.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"


def download_model_if_missing():
    if not os.path.exists(MODEL_FILE):
        print(f"⚠ Model not found. Downloading {MODEL_FILE} ...")
        exit_code = os.system(f"wget -O {MODEL_FILE} {MODEL_URL}")
        if exit_code != 0:
            raise RuntimeError("❌ Model download failed. URL may have changed.")
        print("✅ Download complete.")


def load_model():
    download_model_if_missing()
    print("⏳ Loading model (this can take 15–40s)...")
    model = Llama(
        model_path=MODEL_FILE,
        n_ctx=4096,
        n_threads=4,
        chat_format="mistral"
    )
    print("✅ Model loaded and ready!")
    return model


model = load_model()


class Prompt(BaseModel):
    prompt: str


@app.get("/")
def home():
    return {"message": "Mistral API online ✅"}


@app.post("/chat")
def chat(request: Prompt):
    output = model.create_chat_completion(
        messages=[{"role": "user", "content": request.prompt}]
    )
    reply = output["choices"][0]["message"]["content"]
    return {"response": reply}
