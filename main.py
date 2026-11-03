import os  
from fastapi import FastAPI, HTTPException  
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel  
from google import genai  
from dotenv import load_dotenv  
load_dotenv()  
app = FastAPI()  
client = genai.Client()  
@app.post("/api/chat")  
async def chat(req: BaseModel):  
    res = client.models.generate_content(model='gemini-2.5-flash', contents=str(req))  
    return {"reply": res.text} 
