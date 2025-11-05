"""
FastAPI application for llama-cpp chat interface.

This module provides a REST API for interacting with the llama-cpp language model.
It automatically downloads the model if not present and exposes endpoints for
health checks and chat completion.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

LLAMA_SERVER_URL = "http://localhost:8080/completion"


class PromptBody(BaseModel):
    prompt: str


@app.post("/ai-insights")
def ai_insights(body: PromptBody):
    response = requests.post(
        LLAMA_SERVER_URL,
        json={"prompt": body.prompt, "temperature": 0.7, "n_predict": 256},
    ).json()
    return {"insights": response["content"].strip()}
