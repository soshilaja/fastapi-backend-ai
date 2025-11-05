#!/bin/bash

./llama-server \
  -m smollm.gguf \
  -c 2048 \
  --port 8080 \
  &

uvicorn main:app --host 0.0.0.0 --port $PORT
