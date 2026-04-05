"""
FastAPI — основной сервер GeoAI.
"""
from __future__ import annotations

import os
import httpx
from fastapi import FastAPI, HTTPException, Query
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


# ── API: Конфигурация (передаём YANDEX_API_KEY на фронтенд) ──

@app.get("/api/config")
async def get_config():
    """Отдаёт публичные настройки для фронтенда, включая YANDEX_API_KEY."""
    yandex_key = os.environ.get("YANDEX_API_KEY", "")
    return {
        "yandex_api_key": yandex_key
    }


# ── API: Геокодирование через Яндекс (fallback для фронтенда) ──

@app.get("/api/geocode")
async def geocode(q: str = Query(..., min_length=1, description="Поисковый запрос")):
    """
    Проксирует запрос к Яндекс Геокодер API.
    Используется как fallback, если фронтенд не может обратиться к Яндексу напрямую.
    """
    yandex_key = os.environ.get("YANDEX_API_KEY", "")
    if not yandex_key:
        raise HTTPException(status_code=500, detail="YANDEX_API_KEY не настроен")

    url = "https://geocode-maps.yandex.ru/1.x/"
    params = {
        "apikey": yandex_key,
        "geocode": q,
        "format": "json",
        "results": 7,
        "lang": "ru_RU",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        members = (
            data.get("response", {})
            .get("GeoObjectCollection", {})
            .get("featureMember", [])
        )

        results = []
        for m in members:
            obj = m.get("GeoObject", {})
            pos = obj.get("Point", {}).get("pos", "0 0").split(" ")
            meta = (
                obj.get("metaDataProperty", {})
                .get("GeocoderMetaData", {})
            )
            results.append({
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "lon": float(pos[0]),
                "lat": float(pos[1]),
                "kind": meta.get("kind", ""),
            })

        return {"results": results}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail="Яндекс Геокодер: {}".format(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ошибка геокодирования: {}".format(e))


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