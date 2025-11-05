from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from gpt4all import GPT4All

app = FastAPI()

# Allow your React frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change "*" to your frontend URL when ready
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GPT4All model (you can use any .gguf model)
model_name = "ggml-gpt4all-j-v1.3-groovy.gguf"

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = GPT4All(model_name)
        print(f"✅ Loaded model: {model_name}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")

@app.get("/")
async def root():
    return {"message": "✅ FastAPI running on Render!"}

@app.post("/insight")
async def generate_insight(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "Give me a healthy lifestyle tip.")
    output = model.generate(prompt, max_tokens=200)
    return {"insight": output}
