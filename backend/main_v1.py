"""
FastAPI — основной сервер GeoAI.
"""
from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from ee_service import ee_service
from llm_service import generate_response
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = FastAPI(title="GeoAI Service")

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))


# ── Модели запросов ──

class ChatRequest(BaseModel):
    message: str
    territory_data: Optional[Dict] = None
    chat_history: Optional[List[Dict]] = None


# ── Startup ──

@app.on_event("startup")
async def startup():
    try:
        ee_service.initialize()
    except Exception as e:
        print("[WARN] GEE не инициализирован: {}".format(e))
        print("[WARN] UI загрузится, но слои GEE не будут работать")


# ── Страницы ──

@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"), media_type="text/html")


# ── API: Слои ──

@app.get("/api/layers")
async def get_layers():
    try:
        layers = ee_service.get_available_layers()
        return {"layers": layers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tiles/{layer_id}")
async def get_tile_url(layer_id: str):
    try:
        url = ee_service.get_tile_url(layer_id)
        return {"layer_id": layer_id, "tile_url": url}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── API: Данные по полигону (для отправки в LLM) ──

@app.post("/api/territory-data")
async def get_territory_data(req: Dict):
    """Получить данные всех слоёв для полигона (без буфера)."""
    try:
        geojson = req.get("geojson")
        if not geojson:
            raise HTTPException(status_code=400, detail="geojson required")
        # buffer_km=None чтобы НЕ вызывать buffer(0)
        results = ee_service.analyze_all_layers_polygon(geojson, buffer_km=None)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── API: Чат с ИИ ──

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Отправка сообщения в YandexGPT с контекстом территории."""
    try:
        result = generate_response(
            user_message=req.message,
            territory_data=req.territory_data,
            chat_history=req.chat_history
        )

        if result["success"]:
            return {
                "success": True,
                "response": result["text"]
            }
        else:
            return {
                "success": False,
                "error": result["error"]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)