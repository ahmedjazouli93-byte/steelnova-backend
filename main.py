import os
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
base_prompt = (
    "Vous êtes l'assistant commercial virtuel officiel de STEELNOVA SOLUTIONS à Casablanca. "
    "Répondez de manière professionnelle, très concise et claire en français. "
    "N'utilisez JAMAIS d'en-têtes Markdown complexes (comme ### ou ---). "
    "Présentez vos réponses sous forme de courts paragraphes ou de simples puces sans surcharger le texte."
)

if os.path.exists(knowledge_file):
    with open(knowledge_file, "r", encoding="utf-8") as f:
        catalog_content = f.read()
        system_instruction = f"{base_prompt}\n\nBase de connaissances STEELNOVA:\n{catalog_content[:3000]}"
else:
    system_instruction = base_prompt

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"status": "ok", "service": "Steelnova AI Backend is live"}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=req.message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            )
        )
        return {"reply": response.text}
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "429" in error_msg or "UNAVAILABLE" in error_msg:
            return {"reply": "Veuillez patienter quelques secondes entre vos messages."}
        raise HTTPException(status_code=500, detail=f"AI Service Error: {error_msg}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
