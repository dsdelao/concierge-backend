import os
import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# 1. Configuración
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GENAI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Base de Datos (Omitida por brevedad, asumo que ya tienes el places.json bien cargado)
# ... (Tu código de carga de JSON va aquí, no cambia) ...
# Para que no te de error si copias todo, pongo esto dummy:
PLACES_DB = [] 
json_path = os.path.join(current_dir, 'data', 'places.json')
try:
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            PLACES_DB = json.load(f)
except: pass

class UserMessage(BaseModel):
    message: str
    history: List[str] = []

# --- HERRAMIENTA DE DIAGNÓSTICO ---
@app.get("/api/models")
def list_available_models():
    """Pregunta a Google qué modelos están disponibles para tu API Key"""
    if not GENAI_API_KEY:
        return {"error": "No hay API KEY"}
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GENAI_API_KEY}"
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# 3. EL MOTOR (Ajustado para usar el modelo que descubramos)
def llamar_gemini_directo(prompt, api_key):
    # INTENTO 1: Probamos con 'gemini-pro' (El más estándar y viejo confiable)
    # Si este falla, tú cambiarás este nombre por el que salga en /api/models
    //MODEL_NAME = "gemini-2.5-flash" 
    MODEL_NAME = "gemini-flash-latest"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        # Si falla, imprimimos el error para verlo en el chat
        if response.status_code != 200:
            return f"Error Google ({response.status_code}): {response.text}"
            
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error conectando con Gemini: {str(e)}"

# 4. Endpoints del Chat
@app.post("/api/chat")
async def chat_endpoint(user_input: UserMessage):
    # ... (Tu lógica de búsqueda de lugares va aquí) ...
    # Simplificado para probar conexión:
    
    system_instruction = f"Actúa como guía turístico. Usuario dice: {user_input.message}"
    
    ai_text = llamar_gemini_directo(system_instruction, GENAI_API_KEY)

    return {
        "response": ai_text,
        "recommended_places": [] # Aquí irían tus places reales
    }

@app.get("/")
def read_root():
    return {"status": "Mayan Concierge Diagnóstico Activo 🗿"}