import os
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carga de variables de entorno
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)

app = FastAPI(title="Mayan Concierge API", version="2.0")

# CORS configurado de forma más segura
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: especifica dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY no encontrada en .env")

# Modelos Pydantic
class UserMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    history: List[str] = Field(default_factory=list, max_items=20)

class ChatResponse(BaseModel):
    response: str
    recommended_places: List[dict]
    sources_used: int

# Carga de base de datos con caché
@lru_cache(maxsize=1)
def cargar_places_db():
    """Carga places.json una sola vez y lo cachea"""
    json_path = os.path.join(current_dir, 'data', 'places.json')
    try:
        if not os.path.exists(json_path):
            logger.error(f"❌ Archivo no encontrado: {json_path}")
            return []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"✅ Cargados {len(data)} lugares desde places.json")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parseando JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Error inesperado cargando DB: {e}")
        return []

# Motor de búsqueda semántica simple
def buscar_lugares_relevantes(query: str, places: List[dict], max_results: int = 5) -> tuple:
    """
    Busca lugares relevantes en la base de datos.
    Retorna: (lugares_encontrados, texto_contexto)
    """
    if not places:
        return [], "No hay lugares disponibles en la base de datos."
    
    query_lower = query.lower()
    keywords = query_lower.split()
    
    # Puntuación de relevancia
    scored_places = []
    for place in places:
        score = 0
        place_text = json.dumps(place, ensure_ascii=False).lower()
        
        # Busca coincidencias de palabras clave
        for keyword in keywords:
            if len(keyword) > 2:  # Ignora palabras muy cortas
                score += place_text.count(keyword) * 2
        
        # Boost para coincidencias en campos importantes
        if 'nombre' in place and any(k in place['nombre'].lower() for k in keywords):
            score += 10
        if 'categoria' in place and any(k in place['categoria'].lower() for k in keywords):
            score += 5
        
        if score > 0:
            scored_places.append((score, place))
    
    # Ordena por relevancia
    scored_places.sort(reverse=True, key=lambda x: x[0])
    top_places = [p[1] for p in scored_places[:max_results]]
    
    # Genera texto de contexto
    if top_places:
        context = json.dumps(top_places, ensure_ascii=False, indent=2)
    else:
        context = "No se encontraron lugares específicos para esta consulta."
    
    return top_places, context

# Cliente Gemini mejorado
def llamar_gemini(prompt: str, api_key: str, model: str = "gemini-flash-latest") -> str:
    """
    Llama a Gemini API con manejo robusto de errores.
    Modelos recomendados: gemini-1.5-flash, gemini-1.5-pro, gemini-pro
    """
    if not api_key:
        raise ValueError("API Key de Gemini no configurada")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 200,  # Respuestas concisas
            "topP": 0.9
        }
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            params={"key": api_key},  # API key como parámetro, más seguro
            timeout=10
        )
        
        if response.status_code == 400:
            logger.error(f"Error 400: {response.text}")
            return "⚠️ Modelo no disponible. Verifica /api/models para ver opciones válidas."
        
        if response.status_code == 429:
            return "⏳ Demasiadas peticiones. Intenta en un momento."
        
        if response.status_code != 200:
            logger.error(f"Error {response.status_code}: {response.text}")
            return f"Error del servidor de IA ({response.status_code})"
        
        data = response.json()
        
        # Extrae texto de la respuesta
        if 'candidates' in data and len(data['candidates']) > 0:
            candidate = data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                return candidate['content']['parts'][0]['text']
        
        return "No pude generar una respuesta. Intenta reformular tu pregunta."
        
    except requests.Timeout:
        return "⏱️ La solicitud tardó demasiado. Intenta de nuevo."
    except requests.RequestException as e:
        logger.error(f"Error de red: {e}")
        return "Error de conexión con el servicio de IA."
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        return "Ocurrió un error inesperado."

# Endpoint principal de chat
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(user_input: UserMessage):
    """Endpoint principal del chatbot"""
    
    if not GENAI_API_KEY:
        raise HTTPException(status_code=500, detail="API Key no configurada")
    
    query = user_input.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Mensaje vacío")
    
    # Carga DB y busca lugares relevantes
    places_db = cargar_places_db()
    relevant_places, context_text = buscar_lugares_relevantes(query, places_db)
    
    # Construye el prompt optimizado
    system_instruction = f"""Eres 'Mayan Concierge', guía digital de Tenosique, Tabasco.

**TU BASE DE DATOS:**
{context_text}

**REGLAS ESTRICTAS:**
1. SOLO usa información del JSON arriba
2. Si NO hay datos relevantes, responde: "Ese dato no lo tengo, pero te recomiendo el Cañón del Usumacinta 🛶🐆"
3. Respuestas de máximo 40 palabras
4. Sé directo, sin saludos largos
5. Usa 1-2 emojis relevantes: 🌴🐆🍫🛶☀️

**PREGUNTA DEL TURISTA:**
"{query}"

**TU RESPUESTA:**"""

    # Llama a Gemini
    ai_response = llamar_gemini(system_instruction, GENAI_API_KEY)
    
    return ChatResponse(
        response=ai_response,
        recommended_places=relevant_places[:3],  # Top 3 más relevantes
        sources_used=len(relevant_places)
    )

# Endpoint de diagnóstico
@app.get("/api/models")
def list_available_models():
    """Lista modelos disponibles en tu API Key"""
    if not GENAI_API_KEY:
        return {"error": "GEMINI_API_KEY no configurada"}
    
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        response = requests.get(url, params={"key": GENAI_API_KEY}, timeout=10)
        
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}", "detail": response.text}
        
        data = response.json()
        
        # Filtra solo modelos de generación de contenido
        models = [
            {
                "name": m.get("name", ""),
                "displayName": m.get("displayName", ""),
                "description": m.get("description", "")[:100]
            }
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        
        return {
            "available_models": models,
            "recommended": ["gemini-1.5-flash", "gemini-1.5-pro"],
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Error listando modelos: {e}")
        return {"error": str(e)}

# Health check
@app.get("/")
def health_check():
    places_count = len(cargar_places_db())
    return {
        "status": "✅ Mayan Concierge API Online",
        "version": "2.0",
        "places_loaded": places_count,
        "gemini_configured": bool(GENAI_API_KEY)
    }

# Endpoint de prueba sin IA
@app.get("/api/test-search")
def test_search(q: str = "restaurante"):
    """Prueba la búsqueda sin llamar a Gemini"""
    places_db = cargar_places_db()
    relevant_places, context = buscar_lugares_relevantes(q, places_db)
    
    return {
        "query": q,
        "found": len(relevant_places),
        "places": relevant_places,
        "context_preview": context[:500]
    }
