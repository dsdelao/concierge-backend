"""
index.py — Mayan Concierge API v2.0
Sin SDK de Supabase. Usa la API REST directamente con 'requests'.
Cero dependencias nuevas respecto a la v1.
"""

import os
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mayan Concierge API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Variables de entorno (configurar en Render Dashboard)
# ============================================================
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL  = os.getenv("https://kixzeoduohupvyapbssd.supabase.co")      # ej: https://xxxx.supabase.co
SUPABASE_KEY  = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtpeHplb2R1b2h1cHZ5YXBic3NkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzIzNDM1MjgsImV4cCI6MjA4NzkxOTUyOH0.FRKTo3Jx-B9aqE5AjI13SR67uEuR2Ih9DGNbVvBgyKA") # la anon/public key


# ============================================================
# SUPABASE REST — Cliente minimalista con requests
# (Evita instalar el SDK que tiene dependencias con errores de compilación)
# ============================================================

def supabase_headers() -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json"
    }

def supabase_get(table: str, params: dict = None) -> list:
    """SELECT a Supabase via REST."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    try:
        r = requests.get(url, headers=supabase_headers(), params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Supabase GET {table}: {e}")
        return []

def supabase_post(table: str, data: dict) -> dict:
    """INSERT a Supabase via REST."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {**supabase_headers(), "Prefer": "return=representation"}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=10)
        r.raise_for_status()
        result = r.json()
        return result[0] if isinstance(result, list) else result
    except Exception as e:
        logger.error(f"Supabase POST {table}: {e}")
        raise


# ============================================================
# MODELOS
# ============================================================

class UserMessage(BaseModel):
    message: str
    history: List[str] = []

class ChatResponse(BaseModel):
    response: str
    recommended_places: List[dict]

class NuevaResena(BaseModel):
    negocio_id: str
    texto: str = Field(..., max_length=300, min_length=5)
    rating: int = Field(..., ge=1, le=5)
    autor_alias: str = Field(default="Viajero Anonimo", max_length=50)


# ============================================================
# LOGICA DE BUSQUEDA Y RAG
# ============================================================

def buscar_lugares_relevantes(query: str):
    places = supabase_get("negocios", params={"aprobado": "eq.true", "select": "*"})

    if not places:
        return [], "AVISO: La base de datos esta vacia."

    query_lower = query.lower()
    keywords = [w for w in query_lower.split() if len(w) > 3]
    scored = []

    for place in places:
        score = 0
        full_text = str(place).lower()
        for kw in keywords:
            if kw in full_text:
                score += 1
        if score > 0:
            scored.append((score, place))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [p[1] for p in scored[:4]]
    context = json.dumps(results, ensure_ascii=False) if results else "No encontre coincidencias exactas."
    return results, context


def obtener_resenas_recientes(negocio_id: str, limite: int = 5) -> list:
    return supabase_get("resenas", params={
        "negocio_id": f"eq.{negocio_id}",
        "select":     "texto,rating,autor_alias,created_at",
        "order":      "created_at.desc",
        "limit":      limite
    })


# ============================================================
# GEMINI
# ============================================================

def llamar_gemini(prompt: str, api_key: str) -> str:
    model = "gemini-flash-latest"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 900, "topP": 0.9}
    }
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"},
                          json=payload, params={"key": api_key}, timeout=20)
        if r.status_code != 200:
            return f"Error IA ({r.status_code}): {r.text}"
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error de conexion con Gemini: {str(e)}"


# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(user_input: UserMessage):

    relevant, _ = buscar_lugares_relevantes(user_input.message)

    # RAG Dinamico: enriquecer con resenas reales
    contexto_enriquecido = []
    for place in relevant[:3]:
        resenas = obtener_resenas_recientes(place["id"], limite=5)
        entry = dict(place)

        if resenas:
            promedio = sum(r["rating"] for r in resenas) / len(resenas)
            textos = "\n".join(
                f'  * "{r["texto"]}" - {r["autor_alias"]} ({r["rating"]} estrellas)'
                for r in resenas
            )
            entry["_veredicto_comunidad"] = f"Promedio: {promedio:.1f} estrellas de {len(resenas)} resena(s).\n{textos}"
        else:
            entry["_veredicto_comunidad"] = "Sin resenas todavia."

        contexto_enriquecido.append(entry)

    system_instruction = f"""
Eres 'Mayan Concierge', el guia turistico mas querido de Tenosique, Tabasco.
Conoces cada rincon, eres calido, entusiasta y hablas con personalidad.

LUGARES ENCONTRADOS (con datos reales y opiniones de turistas):
{json.dumps(contexto_enriquecido, ensure_ascii=False, indent=2)}

INSTRUCCIONES:
1. Recomienda los lugares con entusiasmo y personalidad propia.
2. Si el lugar tiene resenas en '_veredicto_comunidad', menciona UNA o DOS opiniones 
   de forma natural. Ejemplo: "Los turistas destacan su salsa verde, aunque varios 
   mencionaron que cierran temprano."
3. Si no hay resenas, recomienda igual usando el 'tip_secreto' y el 'vibe'.
4. Si no hay lugares relevantes, sugiere el Canon del Usumacinta o el Centro.
5. Usa emojis, se conciso (max 3 parrafos). NUNCA digas "segun mis datos".

Pregunta del turista: "{user_input.message}"
Respuesta:
"""

    ai_text = llamar_gemini(system_instruction, GENAI_API_KEY)
    return ChatResponse(response=ai_text, recommended_places=relevant)


@app.post("/api/resenas")
async def guardar_resena(resena: NuevaResena):
    check = supabase_get("negocios", params={
        "id":       f"eq.{resena.negocio_id}",
        "aprobado": "eq.true",
        "select":   "id"
    })
    if not check:
        raise HTTPException(status_code=404, detail="Negocio no encontrado")

    try:
        result = supabase_post("resenas", {
            "negocio_id":  resena.negocio_id,
            "texto":       resena.texto,
            "rating":      resena.rating,
            "autor_alias": resena.autor_alias
        })
        return {"success": True, "id": result.get("id")}
    except Exception:
        raise HTTPException(status_code=500, detail="Error al guardar la resena")


@app.get("/")
def health_check():
    places = supabase_get("negocios", params={"aprobado": "eq.true", "select": "id"})
    return {"status": "Online", "version": "2.0", "negocios_activos": len(places)}

@app.get("/debug")
def debug_env():
    """Endpoint temporal para verificar variables de entorno en Render."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    return {
        "SUPABASE_URL":      url if url else "NO ENCONTRADA",
        "SUPABASE_ANON_KEY": f"{key[:10]}..." if key else "NO ENCONTRADA",
        "GEMINI_API_KEY":    "OK" if os.getenv("GEMINI_API_KEY") else "NO ENCONTRADA",
    }