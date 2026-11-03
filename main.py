
import os
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("steelnova-ai")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing from environment variables."
    )

client = genai.Client(api_key=API_KEY)

app = FastAPI(
    title="Steelnova AI Backend",
    version="2.0.0"
)

# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://steelnova.ma",
        "https://www.steelnova.ma",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# --------------------------------------------------
# STEELNOVA KNOWLEDGE BASE
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "steelnova_catalog_knowledge.md"

SYSTEM_INSTRUCTION = """
Vous êtes l'assistant IA officiel de STEELNOVA SOLUTIONS,
une entreprise spécialisée dans la distribution d'acier au Maroc.

VOTRE MISSION :

1. Accueillir chaleureusement les visiteurs du site.
2. Présenter STEELNOVA SOLUTIONS de manière professionnelle.
3. Répondre aux questions sur les produits et services.
4. Aider les clients à comprendre les caractéristiques des produits.
5. Orienter les visiteurs souhaitant obtenir un devis vers l'équipe commerciale.
6. Répondre en français par défaut.
7. Vous pouvez répondre en arabe ou en anglais si le visiteur le demande.

RÈGLES IMPORTANTES :

- Soyez professionnel, clair et accueillant.
- Donnez des réponses utiles et faciles à comprendre.
- Ne créez jamais de faux prix, stocks ou spécifications.
- N'inventez jamais des informations sur l'entreprise.
- Si une information est inconnue, dites-le honnêtement.
- Encouragez les visiteurs intéressés à contacter STEELNOVA
  pour obtenir un devis ou davantage d'informations.
"""

if KNOWLEDGE_FILE.exists():
    try:
        catalog_content = KNOWLEDGE_FILE.read_text(
            encoding="utf-8"
        )

        SYSTEM_INSTRUCTION += (
            "\n\nBASE DE CONNAISSANCES STEELNOVA :\n"
            + catalog_content
        )

        logger.info("Steelnova knowledge base loaded.")

    except Exception:
        logger.exception("Could not load knowledge file.")

else:
    logger.warning("Knowledge file not found: %s", KNOWLEDGE_FILE)

# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000
    )

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Steelnova AI Backend",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# --------------------------------------------------
# AI CHAT ENDPOINT
# --------------------------------------------------

@app.post("/api/chat")
def chat(req: ChatRequest):

    message = req.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:
        logger.info("Received chat request.")

        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=message,
            system_instruction=SYSTEM_INSTRUCTION
        )

        reply = interaction.output_text

        if not reply:
            logger.error("Gemini returned an empty response.")

            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response."
            )

        return {
            "reply": reply,
            "status": "success"
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Gemini API request failed.")

        raise HTTPException(
            status_code=502,
            detail="AI service temporarily unavailable. Please try again."
        )

# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
