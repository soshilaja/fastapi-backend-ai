#!/bin/bash

# Download model if not exists
if [ ! -f smollm.gguf ]; then
  echo "Downloading model..."
  curl -L -o smollm.gguf https://huggingface.co/nbroad/SmolLM-1.7B-Instruct-GGUF/resolve/main/smollm-1.7b-instruct-q4_k_m.gguf
fi

# Start model server in background
./llama-server -m smollm.gguf -p 8080 --ctx-size 2048 &

# Wait 2 seconds for server to start
sleep 2

# Start FastAPI
uvicorn main:app --host 0.0.0.0 --port $PORT

