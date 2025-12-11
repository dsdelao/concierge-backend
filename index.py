import os
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CORRECCIÓN 1: RUTA RELATIVA SIMPLE ---
# Como index.py está en la raíz, la carpeta actual es la base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Mayan Concierge API", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render inyecta la variable automáticamente, no necesitamos cargar .env aquí si falla
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")

class UserMessage(BaseModel):
    message: str
    history: List[str] = []

class ChatResponse(BaseModel):
    response: str
    recommended_places: List[dict]

# --- CORRECCIÓN 2: CARGA DE DATOS ROBUSTA ---
@lru_cache(maxsize=1)
def cargar_places_db():
    # Buscamos 'data/places.json' desde la raíz
    json_path = os.path.join(BASE_DIR, 'data', 'places.json')
    try:
        if not os.path.exists(json_path):
            logger.error(f"❌ NO ENCUENTRO EL ARCHIVO: {json_path}")
            # Intento de respaldo por si la estructura es distinta
            return []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"✅ DB Cargada: {len(data)} lugares.")
            return data
    except Exception as e:
        logger.error(f"❌ Error DB: {e}")
        return []

def buscar_lugares_relevantes(query: str, places: List[dict]):
    if not places:
        return [], "AVISO: La base de datos está vacía. No puedo recomendar nada específico."
    
    query = query.lower()
    keywords = query.split()
    scored_places = []
    
    for place in places:
        score = 0
        # Convertimos todo el lugar a texto para buscar
        full_text = str(place).lower()
        
        for word in keywords:
            if len(word) > 3 and word in full_text:
                score += 1
                
        if score > 0:
            scored_places.append((score, place))
            
    # Ordenamos por score
    scored_places.sort(key=lambda x: x[0], reverse=True)
    results = [p[1] for p in scored_places[:4]] # Top 4
    
    context = json.dumps(results, ensure_ascii=False) if results else "No encontré coincidencias exactas."
    return results, context

def llamar_gemini(prompt: str, api_key: str):
    # Usamos el alias estable
    model = "gemini-flash-latest"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            # --- CORRECCIÓN 3: MÁS ESPACIO PARA HABLAR ---
            "maxOutputTokens": 800, 
            "topP": 0.9
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, params={"key": api_key})
        
        if response.status_code != 200:
            return f"Error IA ({response.status_code}): {response.text}"
            
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error de conexión: {str(e)}"

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(user_input: UserMessage):
    places_db = cargar_places_db()
    
    # Búsqueda
    relevant, context = buscar_lugares_relevantes(user_input.message, places_db)
    
    # Prompt
    system_instruction = f"""
    Eres 'Mayan Concierge', guía de Tenosique, Tabasco.
    
    INFORMACIÓN ENCONTRADA:
    {context}
    
    INSTRUCCIONES:
    1. Si hay lugares en la lista de arriba, recomiéndalos con entusiasmo.
    2. Si la lista está vacía o dice "No encontré", sugiere visitar el Cañón del Usumacinta o el Centro.
    3. Sé amable y usa emojis 🌴.
    4. NO digas "según mis datos", actúa natural.
    
    Usuario: "{user_input.message}"
    Respuesta:
    """

    ai_text = llamar_gemini(system_instruction, GENAI_API_KEY)

    return ChatResponse(
        response=ai_text,
        recommended_places=relevant
    )

@app.get("/")
def health_check():
    db = cargar_places_db()
    return {"status": "Online", "lugares_en_db": len(db)}
