import os
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import logging

# ============================================
# CONFIGURACIÓN
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)

app = FastAPI(
    title="Mayan Concierge API",
    version="2.1",
    description="API inteligente para turismo en Tenosique, Tabasco"
)

# CORS optimizado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: lista específica
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY no encontrada en .env")

# ============================================
# MODELOS PYDANTIC
# ============================================
class UserMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    history: List[str] = Field(default_factory=list, max_items=20)

class Place(BaseModel):
    id: str
    nombre: str
    categoria: str
    tags: List[str]
    precio_promedio: str
    horario: str
    vibe: str
    ubicacion_texto: str
    google_maps: str
    telefono: Optional[str] = None
    tip_secreto: str
    promo_activa: Optional[str] = None
    imagen_url: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    recommended_places: List[Dict]
    sources_used: int
    search_metadata: Optional[Dict] = None

# ============================================
# BASE DE DATOS CON CACHÉ
# ============================================
@lru_cache(maxsize=1)
def cargar_places_db() -> List[Place]:
    """Carga places.json una sola vez y lo cachea como objetos Place"""
    json_path = os.path.join(current_dir, 'data', 'places.json')
    
    try:
        if not os.path.exists(json_path):
            logger.error(f"❌ Archivo no encontrado: {json_path}")
            return []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            places = [Place(**place) for place in data]
            logger.info(f"✅ Cargados {len(places)} lugares desde places.json")
            return places
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parseando JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return []

# ============================================
# MOTOR DE BÚSQUEDA SEMÁNTICA MEJORADO
# ============================================
def buscar_lugares_relevantes(
    query: str, 
    places: List[Place], 
    max_results: int = 5
) -> tuple:
    """
    Búsqueda semántica avanzada con scoring ponderado.
    Retorna: (lugares_encontrados, texto_contexto, metadata)
    """
    if not places:
        return [], "No hay lugares disponibles.", {}
    
    query_lower = query.lower()
    keywords = [k for k in query_lower.split() if len(k) > 2]
    
    # Mapeo de intenciones comunes
    intent_map = {
        'comer': ['gastronomía', 'comida', 'restaurante', 'desayuno', 'almuerzo'],
        'dormir': ['hotel', 'hospedaje', 'alojamiento'],
        'aventura': ['kayak', 'tirolesa', 'naturaleza', 'rio', 'senderismo'],
        'barato': ['economico', '$'],
        'fotos': ['mirador', 'vista', 'romantico'],
        'cultura': ['museo', 'arte', 'galeria', 'cultural'],
        'comprar': ['mercado', 'artesanias', 'souvenirs'],
    }
    
    # Detecta intenciones
    detected_intents = []
    for key, values in intent_map.items():
        if key in query_lower or any(v in query_lower for v in values):
            detected_intents.extend(values)
    
    # Scoring de lugares
    scored_places = []
    for place in places:
        score = 0
        
        # Texto completo del lugar para buscar
        place_text = f"{place.nombre} {place.categoria} {' '.join(place.tags)} {place.vibe} {place.precio_promedio}".lower()
        
        # 1. Coincidencias en keywords (peso: 2)
        for keyword in keywords:
            score += place_text.count(keyword) * 2
        
        # 2. Coincidencias en intenciones detectadas (peso: 3)
        for intent in detected_intents:
            if intent in place_text:
                score += 3
        
        # 3. Coincidencias en nombre (peso: 10)
        if any(k in place.nombre.lower() for k in keywords):
            score += 10
        
        # 4. Coincidencias en categoría (peso: 8)
        if any(k in place.categoria.lower() for k in keywords):
            score += 8
        
        # 5. Coincidencias exactas en tags (peso: 5)
        for tag in place.tags:
            if tag in keywords or tag in detected_intents:
                score += 5
        
        # 6. Boost por promoción activa (peso: 2)
        if place.promo_activa:
            score += 2
        
        # 7. Boost por precio económico si busca barato (peso: 4)
        if any(k in query_lower for k in ['barato', 'economico', 'cheap']):
            if '$' in place.precio_promedio and '$$' not in place.precio_promedio:
                score += 4
        
        if score > 0:
            scored_places.append((score, place))
    
    # Ordena por relevancia
    scored_places.sort(reverse=True, key=lambda x: x[0])
    top_places = [p[1] for p in scored_places[:max_results]]
    
    # Genera contexto para Gemini
    if top_places:
        context_data = []
        for p in top_places:
            context_data.append({
                "nombre": p.nombre,
                "categoria": p.categoria,
                "tags": p.tags,
                "precio": p.precio_promedio,
                "horario": p.horario,
                "vibe": p.vibe,
                "ubicacion": p.ubicacion_texto,
                "tip": p.tip_secreto,
                "promo": p.promo_activa
            })
        context = json.dumps(context_data, ensure_ascii=False, indent=2)
    else:
        context = "No se encontraron lugares específicos para esta consulta."
    
    # Metadata de búsqueda
    metadata = {
        "total_places": len(places),
        "matches_found": len(top_places),
        "keywords_used": keywords,
        "intents_detected": list(set(detected_intents))
    }
    
    return top_places, context, metadata

# ============================================
# CLIENTE GEMINI OPTIMIZADO
# ============================================
def llamar_gemini(
    prompt: str, 
    api_key: str, 
    model: str = "gemini-1.5-flash"
) -> tuple:
    """
    Llama a Gemini con manejo robusto de errores.
    Retorna: (respuesta_texto, status_ok)
    """
    if not api_key:
        return "⚠️ API Key no configurada", False
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.8,  # Más creativo para respuestas naturales
            "maxOutputTokens": 250,
            "topP": 0.9,
            "topK": 40
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            params={"key": api_key},
            timeout=12
        )
        
        # Manejo de errores específicos
        if response.status_code == 400:
            logger.error(f"Error 400: {response.text}")
            return "⚠️ Modelo no disponible. Usa /api/models para ver opciones.", False
        
        if response.status_code == 429:
            return "⏳ Muchas peticiones. Espera unos segundos.", False
        
        if response.status_code == 403:
            return "🔑 API Key inválida o sin permisos.", False
        
        if response.status_code != 200:
            logger.error(f"Error {response.status_code}: {response.text}")
            return f"Error del servidor ({response.status_code})", False
        
        data = response.json()
        
        # Extrae texto
        if 'candidates' in data and len(data['candidates']) > 0:
            candidate = data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                text = candidate['content']['parts'][0].get('text', '')
                return text, True
        
        return "No pude generar respuesta. Intenta reformular.", False
        
    except requests.Timeout:
        return "⏱️ Tiempo de espera agotado. Intenta de nuevo.", False
    except requests.RequestException as e:
        logger.error(f"Error de red: {e}")
        return "🌐 Error de conexión con IA.", False
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        return "❌ Error inesperado.", False

# ============================================
# SISTEMA DE PROMPTS OPTIMIZADO
# ============================================
def construir_prompt(query: str, context: str, metadata: Dict) -> str:
    """Construye prompt optimizado según los datos encontrados"""
    
    # Prompt base
    base_prompt = f"""Eres 'Mayan Concierge', el guía digital de Tenosique, Tabasco.

**DATOS DISPONIBLES:**
{context}

**REGLAS ESTRICTAS:**
1. USA SOLO la información del JSON arriba
2. Si NO hay datos relevantes, responde: "Ese dato no lo tengo, pero te recomiendo el Cañón del Usumacinta 🛶🐆"
3. Respuestas MÁXIMO 50 palabras
4. Sé directo, sin saludos largos
5. Usa 1-2 emojis relevantes: 🌴🐆🍫🛶☀️🌮🏨

**PREGUNTA DEL TURISTA:**
"{query}"

**TU RESPUESTA:**"""

    # Si hay lugares con promociones, mencionarlo
    if "promo" in context and context.count('"promo":') > 0:
        base_prompt = base_prompt.replace(
            "**TU RESPUESTA:**",
            "**IMPORTANTE:** Menciona las promociones activas si las hay.\n\n**TU RESPUESTA:**"
        )
    
    return base_prompt

# ============================================
# ENDPOINTS PRINCIPALES
# ============================================
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(user_input: UserMessage):
    """Endpoint principal del chatbot con búsqueda semántica"""
    
    if not GENAI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="API Key de Gemini no configurada"
        )
    
    query = user_input.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Mensaje vacío")
    
    logger.info(f"📩 Query recibida: {query}")
    
    # Carga DB y busca lugares
    places_db = cargar_places_db()
    relevant_places, context_text, metadata = buscar_lugares_relevantes(
        query, 
        places_db,
        max_results=5
    )
    
    logger.info(f"🔍 Encontrados {len(relevant_places)} lugares relevantes")
    logger.info(f"🎯 Intenciones detectadas: {metadata.get('intents_detected', [])}")
    
    # Construye prompt
    prompt = construir_prompt(query, context_text, metadata)
    
    # Llama a Gemini
    ai_response, success = llamar_gemini(prompt, GENAI_API_KEY)
    
    if not success:
        logger.warning(f"⚠️ Gemini falló: {ai_response}")
    
    # Convierte lugares a dict para respuesta
    places_dict = [place.dict() for place in relevant_places[:3]]
    
    return ChatResponse(
        response=ai_response,
        recommended_places=places_dict,
        sources_used=len(relevant_places),
        search_metadata=metadata
    )

# ============================================
# ENDPOINTS DE UTILIDAD
# ============================================
@app.get("/api/models")
def list_available_models():
    """Lista modelos disponibles en Gemini"""
    if not GENAI_API_KEY:
        return {"error": "GEMINI_API_KEY no configurada"}
    
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        response = requests.get(
            url, 
            params={"key": GENAI_API_KEY}, 
            timeout=10
        )
        
        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}", 
                "detail": response.text
            }
        
        data = response.json()
        
        models = [
            {
                "name": m.get("name", "").split('/')[-1],
                "displayName": m.get("displayName", ""),
                "description": m.get("description", "")[:150]
            }
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        
        return {
            "available_models": models,
            "recommended": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Error listando modelos: {e}")
        return {"error": str(e)}

@app.get("/api/places")
def list_all_places(
    categoria: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 10
):
    """Lista todos los lugares con filtros opcionales"""
    places = cargar_places_db()
    
    # Aplica filtros
    filtered = places
    if categoria:
        filtered = [p for p in filtered if p.categoria.lower() == categoria.lower()]
    if tag:
        filtered = [p for p in filtered if tag.lower() in [t.lower() for t in p.tags]]
    
    return {
        "total": len(places),
        "filtered": len(filtered),
        "places": [p.dict() for p in filtered[:limit]]
    }

@app.get("/api/places/{place_id}")
def get_place_by_id(place_id: str):
    """Obtiene un lugar específico por ID"""
    places = cargar_places_db()
    place = next((p for p in places if p.id == place_id), None)
    
    if not place:
        raise HTTPException(status_code=404, detail="Lugar no encontrado")
    
    return place.dict()

@app.get("/api/search")
def search_places(q: str, max_results: int = 5):
    """Búsqueda directa sin IA"""
    places_db = cargar_places_db()
    relevant_places, context, metadata = buscar_lugares_relevantes(
        q, 
        places_db, 
        max_results
    )
    
    return {
        "query": q,
        "found": len(relevant_places),
        "metadata": metadata,
        "places": [p.dict() for p in relevant_places]
    }

@app.get("/api/stats")
def get_stats():
    """Estadísticas de la base de datos"""
    places = cargar_places_db()
    
    categorias = {}
    tags_count = {}
    con_promo = 0
    
    for place in places:
        # Cuenta categorías
        categorias[place.categoria] = categorias.get(place.categoria, 0) + 1
        
        # Cuenta tags
        for tag in place.tags:
            tags_count[tag] = tags_count.get(tag, 0) + 1
        
        # Cuenta promos
        if place.promo_activa:
            con_promo += 1
    
    return {
        "total_places": len(places),
        "categorias": categorias,
        "top_tags": dict(sorted(tags_count.items(), key=lambda x: x[1], reverse=True)[:10]),
        "places_con_promo": con_promo,
        "gemini_configured": bool(GENAI_API_KEY)
    }

@app.get("/")
def health_check():
    """Health check con información del sistema"""
    places_count = len(cargar_places_db())
    return {
        "status": "✅ Mayan Concierge API Online",
        "version": "2.1",
        "places_loaded": places_count,
        "gemini_configured": bool(GENAI_API_KEY),
        "endpoints": {
            "chat": "/api/chat",
            "places": "/api/places",
            "search": "/api/search",
            "stats": "/api/stats",
            "models": "/api/models"
        }
    }

# ============================================
# MANEJO DE ERRORES GLOBAL
# ============================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no capturado: {exc}")
    return {
        "error": "Error interno del servidor",
        "detail": str(exc)
    }
