import os
import smtplib
from typing import List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing.")

# Email Configuration
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "contact@steelnova.ma")
SMTP_USER = os.environ.get("SMTP_USER", "contact@steelnova.ma")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
SMTP_HOST = os.environ.get("SMTP_HOST", "webhosting3007.is.cc")

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

# Load knowledge base ONCE at startup for maximum speed
knowledge_file = "steelnova_catalog_knowledge.md"
base_prompt = (
    "Vous êtes l'assistant commercial virtuel officiel de STEELNOVA SOLUTIONS, entreprise spécialisée dans la distribution d'acier et de produits sidérurgiques au Maroc (Casablanca).\n"
    "INFORMATIONS DE CONTACT OFFICIELLES (À utiliser impérativement quand on vous demande les coordonnées) :\n"
    "- Téléphone : +212 700 345 839\n"
    "- Email : contact@steelnova.ma\n"
    "- Ne donnez JAMAIS d'autres adresses email (interdit d'ajouter 'solutions' dans le domaine).\n\n"
    "DIRECTIVES DE STYLE IMPÉRATIVES:\n"
    "1. Ne vous représentez JAMAIS au milieu d'une conversation.\n"
    "2. Allez DIRECTEMENT au but de la question posée.\n"
    "3. Interdit d'utiliser des séparateurs (---), des en-têtes (###) ou des émojis excessifs.\n"
    "4. Rédigez des réponses courtes, élégantes et bien aérées pour un widget de chat."
)

if os.path.exists(knowledge_file):
    with open(knowledge_file, "r", encoding="utf-8") as f:
        catalog_content = f.read()
        system_instruction = f"{base_prompt}\n\nBase de connaissances :\n{catalog_content[:3000]}"
else:
    system_instruction = base_prompt

class ChatRequest(BaseModel):
    message: str

class MessageItem(BaseModel):
    sender: str  # 'user' or 'bot'
    text: str

class TranscriptRequest(BaseModel):
    messages: List[MessageItem]
    user_contact: Optional[str] = None

def send_full_transcript_email(messages: List[MessageItem], user_contact: Optional[str] = None):
    """Sends a single summary email using SSL (Port 465) for webhosting3007.is.cc."""
    if not all([NOTIFICATION_EMAIL, SMTP_USER, SMTP_PASSWORD]):
        print("Email environment variables missing. Transcript email skipped.")
        return

    formatted_transcript = ""
    for msg in messages:
        label = "👤 CLIENT" if msg.sender.lower() == "user" else "🤖 ASSISTANT"
        formatted_transcript += f"{label}:\n{msg.text}\n\n"

    subject = "📋 HISTORIQUE COMPLET DE CONVERSATION - STEELNOVA CHATBOT"
    body = f"""
Bonjour,

Une session de chat vient de se terminer sur steelnova.ma. Voici l'historique complet des échanges :

==================================================
TRANSCRIPTION DE LA CONVERSATION
==================================================

{formatted_transcript}
==================================================
Coordonnées identifiées : {user_contact if user_contact else "Non spécifiées"}
==================================================

-- 
Steelnova AI Assistant
    """

    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = NOTIFICATION_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        server = smtplib.SMTP_SSL(SMTP_HOST, 465)
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, NOTIFICATION_EMAIL, msg.as_string())
        server.quit()
        print("Full conversation transcript email successfully sent to contact@steelnova.ma.")
    except Exception as e:
        print(f"Failed to send transcript email: {str(e)}")

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
                temperature=0.2,
            )
        )
        return {"reply": response.text}
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "429" in error_msg or "UNAVAILABLE" in error_msg:
            return {"reply": "Veuillez patienter quelques secondes entre vos messages."}
        raise HTTPException(status_code=500, detail=f"AI Service Error: {error_msg}")

@app.post("/api/send-transcript")
async def send_transcript(req: TranscriptRequest, background_tasks: BackgroundTasks):
    if not req.messages:
        return {"status": "ignored", "reason": "No messages to log"}
    
    background_tasks.add_task(send_full_transcript_email, req.messages, req.user_contact)
    return {"status": "ok", "message": "Transcript queued for email dispatch"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
