import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing.")

app = FastAPI(title="Steelnova AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://steelnova.ma",
        "https://www.steelnova.ma",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=api_key)

knowledge_file = "steelnova_catalog_knowledge.md"
system_instruction = "Vous êtes l'assistant IA officiel de STEELNOVA SOLUTIONS, entreprise spécialisée dans la distribution d'acier au Maroc."

if os.path.exists(knowledge_file):
    with open(knowledge_file, "r", encoding="utf-8") as f:
        catalog_content = f.read()
        system_instruction += f"\n\nVoici le catalogue et la base de connaissances de STEELNOVA SOLUTIONS:\n{catalog_content}"

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"status": "ok", "service": "Steelnova AI Backend is live"}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    # Free tier automatic retry loop for rate limits (429 / 503)
    max_retries = 2
    delay = 2
    
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=req.message,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                )
            )
            return {"reply": response.text}
        except Exception as e:
            error_msg = str(e)
            # If it's a rate limit or high demand error and we have retries left, wait and loop
            if ("503" in error_msg or "429" in error_msg or "UNAVAILABLE" in error_msg) and attempt < max_retries:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            
            # If out of retries, return a friendly message instead of a crash
            if "503" in error_msg or "429" in error_msg or "UNAVAILABLE" in error_msg:
                return {"reply": "L'assistant est très demandé pour le moment. Veuillez patienter une petite seconde et réessayer."}
            
            raise HTTPException(status_code=500, detail=f"AI Service Error: {error_msg}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
