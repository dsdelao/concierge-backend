import os
import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="Mayan Concierge API", version="3.0-Fallback")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserMessage(BaseModel):
    message: str
    history: List[str] = []

class ChatResponse(BaseModel):
    response: str
    recommended_places: List[dict]

@lru_cache(maxsize=1)
def cargar_places_db():
    json_path = os.path.join(BASE_DIR, 'data', 'places.json')
    try:
        if not os.path.exists(json_path):
            return []
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def buscar_lugares_relevantes(query: str, places: List[dict]):
    if not places:
        return [], "AVISO: Base de datos vacía."
    
    query = query.lower()
    keywords = query.split()
    scored_places = []
    
    for place in places:
        score = 0
        full_text = str(place).lower()
        for word in keywords:
            if len(word) > 3 and word in full_text:
                score += 1
        if score > 0:
            scored_places.append((score, place))
            
    scored_places.sort(key=lambda x: x[0], reverse=True)
    results = [p[1] for p in scored_places[:4]]
    context = json.dumps(results, ensure_ascii=False) if results else "No encontré coincidencias."
    return results, context

# --- LA MAGIA: INTENTA VARIOS MODELOS HASTA QUE UNO FUNCIONE ---
def llamar_gemini_robusto(prompt: str, api_key: str):
    # Lista de modelos ordenada por prioridad (del más probable al menos probable)
    # Basado en la lista que tu cuenta mostró disponible
    MODELS_TO_TRY = [
        "gemini-flash-latest",    # Intento 1: El alias oficial
        "gemini-1.5-flash",       # Intento 2: El estándar
        "gemini-pro",             # Intento 3: El clásico (suele tener buena cuota)
        "gemini-2.0-flash-exp",   # Intento 4: Experimental
        "gemini-1.5-pro-latest"   # Intento 5: Último recurso
    ]
    
    last_error = ""

    for model in MODELS_TO_TRY:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 400}
        }
        
        try:
            logger.info(f"🔄 Intentando con modelo: {model}...")
            response = requests.post(url, headers=headers, json=payload, params={"key": api_key})
            
            # Si funciona (200), rompemos el ciclo y regresamos el texto
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ ÉXITO con modelo: {model}")
                return data['candidates'][0]['content']['parts'][0]['text']
            
            # Si falla, guardamos el error y seguimos al siguiente modelo
            error_msg = response.text
            last_error = f"{model} falló ({response.status_code})"
            logger.warning(f"⚠️ Falló {model}: {error_msg}")
            
        except Exception as e:
            last_error = str(e)
            continue

    # Si llegamos aquí, fallaron TODOS los modelos
    return f"Lo siento, la selva está muy densa hoy. (Error técnico: {last_error})"

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(user_input: UserMessage):
    places_db = cargar_places_db()
    relevant, context = buscar_lugares_relevantes(user_input.message, places_db)
    
    system_instruction = f"""
    Eres 'Mayan Concierge', guía de Tenosique.
    DATOS: {context}
    Usuario: "{user_input.message}"
    Responde breve y amable con emojis 🌴. Si no hay datos, sugiere el Cañón.
    """

    ai_text = llamar_gemini_robusto(system_instruction, GENAI_API_KEY)

    return ChatResponse(
        response=ai_text,
        recommended_places=relevant
    )

@app.get("/")
def health_check():
    return {"status": "Mayan Concierge Online 3.0"}
