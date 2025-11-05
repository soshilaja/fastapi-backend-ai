FROM ubuntu:22.04

# System dependencies
RUN apt update && apt install -y \
    git wget curl build-essential cmake python3 python3-pip

WORKDIR /app

# Download model (no login required)
RUN wget https://huggingface.co/TheBloke/SmolLM-1.7B-Instruct-GGUF/resolve/main/smollm-1.7b-instruct.q4_k_s.gguf -O smollm.gguf

# Build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    mkdir build && cd build && \
    cmake .. && make -j && \
    cp server/llama-server /app/llama-server

# Copy source files
COPY . /app

# Install dependencies
RUN pip3 install -r requirements.txt

EXPOSE 8080
EXPOSE 8000

CMD ["bash", "start.sh"]
