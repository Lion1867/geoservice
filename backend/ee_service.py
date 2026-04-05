"""
Google Earth Engine — сервис получения тайлов и аналитики.
Совместимо с Python 3.8+
"""

from __future__ import annotations

import ee
import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class EarthEngineService:

    def __init__(self):
        self._initialized = False
        self._tile_url_cache = {}

    def initialize(self):
        if self._initialized:
            return
        project = os.getenv("GEE_PROJECT", "my-gee-terrain")
        key_path = os.getenv("GEE_SERVICE_ACCOUNT_KEY", "")
        try:
            if key_path and os.path.exists(key_path):
                with open(key_path) as f:
                    key_data = json.load(f)
                credentials = ee.ServiceAccountCredentials(key_data["client_email"], key_path)
                ee.Initialize(credentials, project=project)
            else:
                try:
                    ee.Initialize(project=project)
                except Exception:
                    ee.Authenticate()
                    ee.Initialize(project=project)
            self._initialized = True
            print("[GEE] Инициализация успешна")
        except Exception as e:
            print("[GEE] Ошибка инициализации: {}".format(e))
            raise

    def _get_layer_config(self):
        return {
            "elevation": {
                "name": "Высота над уровнем моря",
                "category": "Рельеф",
                "builder": self._layer_elevation,
            },
            "slope": {
                "name": "Уклон (градусы)",
                "category": "Рельеф",
                "builder": self._layer_slope,
            },
            "aspect": {
                "name": "Экспозиция склона",
                "category": "Рельеф",
                "builder": self._layer_aspect,
            },
            "hillshade": {
                "name": "Теневой рельеф",
                "category": "Рельеф",
                "builder": self._layer_hillshade,
            },
            "curvature": {
                "name": "Кривизна поверхности",
                "category": "Рельеф",
                "builder": self._layer_curvature,
            },
            "landcover": {
                "name": "Тип земного покрова",
                "category": "Покров",
                "builder": self._layer_landcover,
            },
            "forest": {
                "name": "Лесной покров (%)",
                "category": "Покров",
                "builder": self._layer_forest,
            },
            "forest_loss": {
                "name": "Потеря леса",
                "category": "Покров",
                "builder": self._layer_forest_loss,
            },
            "water": {
                "name": "Водные объекты",
                "category": "Вода",
                "builder": self._layer_water,
            },
            "water_occurrence": {
                "name": "Частота затопления",
                "category": "Вода",
                "builder": self._layer_water_occurrence,
            },
            "population": {
                "name": "Плотность населения",
                "category": "Население",
                "builder": self._layer_population,
            },
            "built_area": {
                "name": "Застроенные территории",
                "category": "Население",
                "builder": self._layer_built_area,
            },
            "nightlights": {
                "name": "Ночные огни (VIIRS)",
                "category": "Население",
                "builder": self._layer_nightlights,
            },
            "ndvi": {
                "name": "Индекс вегетации (NDVI)",
                "category": "Растительность",
                "builder": self._layer_ndvi,
            },
            "temperature": {
                "name": "Температура поверхности",
                "category": "Климат",
                "builder": self._layer_temperature,
            },
            "precipitation": {
                "name": "Осадки (среднегодовые)",
                "category": "Климат",
                "builder": self._layer_precipitation,
            },
            "soil_organic": {
                "name": "Органический углерод почвы",
                "category": "Почвы",
                "builder": self._layer_soil_organic,
            },
        }

    def _dem(self):
        return ee.Image("NASA/NASADEM_HGT/001").select("elevation")

    def _layer_elevation(self):
        img = self._dem()
        vis = {"min": 0, "max": 3000, "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"]}
        return img, vis

    def _layer_slope(self):
        img = ee.Terrain.slope(self._dem())
        vis = {"min": 0, "max": 60, "palette": ["00ff00", "ffff00", "ff8800", "ff0000", "8B0000"]}
        return img, vis

    def _layer_aspect(self):
        img = ee.Terrain.aspect(self._dem())
        vis = {"min": 0, "max": 360, "palette": ["blue", "green", "yellow", "red", "blue"]}
        return img, vis

    def _layer_hillshade(self):
        img = ee.Terrain.hillshade(self._dem())
        vis = {"min": 0, "max": 255}
        return img, vis

    def _layer_curvature(self):
        dem = self._dem()
        kernel = ee.Kernel.laplacian8()
        img = dem.convolve(kernel)
        vis = {"min": -10, "max": 10, "palette": ["0000ff", "ffffff", "ff0000"]}
        return img, vis

    def _layer_landcover(self):
        img = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
        vis = {"min": 10, "max": 100, "palette": ["006400","ffbb22","ffff4c","f096ff","fa0000","b4b4b4","f0f0f0","0064c8","0096a0","00cf75","fae6a0"]}
        return img, vis

    def _layer_forest(self):
        img = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").select("treecover2000")
        vis = {"min": 0, "max": 100, "palette": ["FFFFCC", "006600"]}
        return img, vis

    def _layer_forest_loss(self):
        img = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").select("loss")
        vis = {"min": 0, "max": 1, "palette": ["000000", "FF0000"]}
        return img, vis

    def _layer_water(self):
        img = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent")
        vis = {"min": 0, "max": 1, "palette": ["ffffff", "0000ff"]}
        return img, vis

    def _layer_water_occurrence(self):
        img = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        vis = {"min": 0, "max": 100, "palette": ["ffffff", "ffbbbb", "0000ff"]}
        return img, vis

    def _layer_population(self):
        img = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate("2020-01-01", "2021-01-01").mosaic()
        vis = {"min": 0, "max": 500, "palette": ["FFFFE0", "FFA500", "FF0000", "8B0000"]}
        return img, vis

    def _layer_built_area(self):
        img = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate("2023-01-01", "2024-01-01").select("built").median()
        vis = {"min": 0, "max": 1, "palette": ["000000", "FF0000"]}
        return img, vis

    def _layer_nightlights(self):
        img = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG").filterDate("2023-01-01", "2024-01-01").select("avg_rad").median()
        vis = {"min": 0, "max": 60, "palette": ["000000", "FFFF00", "FFFFFF"]}
        return img, vis

    def _layer_ndvi(self):
        img = ee.ImageCollection("MODIS/061/MOD13A2").filterDate("2023-01-01", "2024-01-01").select("NDVI").median().multiply(0.0001)
        vis = {"min": -0.1, "max": 0.9, "palette": ["CE7E45","DF923D","F1B555","FCD163","99B718","74A901","66A000","529400","3E8601","207401","056201","004C00","023B01","012E01","011D01","011301"]}
        return img, vis

    def _layer_temperature(self):
        img = ee.ImageCollection("MODIS/061/MOD11A1").filterDate("2023-06-01", "2023-09-01").select("LST_Day_1km").median().multiply(0.02).subtract(273.15)
        vis = {"min": -10, "max": 45, "palette": ["040274","0502a3","0502ce","0602ff","307ef3","30c8e2","32d3ef","3ae237","86e26f","b5e22e","d6e21f","fff705","ffd611","ffb613","ff8b13","ff6e08","ff500d","ff0000","de0101","c21301"]}
        return img, vis

    def _layer_precipitation(self):
        img = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate("2023-01-01", "2024-01-01").sum()
        vis = {"min": 0, "max": 2000, "palette": ["FFFFFF","00FFFF","0080FF","DA00FF","FFA400","FF0000"]}
        return img, vis

    def _layer_soil_organic(self):
        img = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select("b0")
        vis = {"min": 0, "max": 120, "palette": ["ffffa0","f7fcb9","addd8e","41ab5d","006837","004529"]}
        return img, vis

    # ═══════════════════════════════════════
    # Публичное API
    # ═══════════════════════════════════════

    def get_available_layers(self):
        config = self._get_layer_config()
        result = []
        for layer_id, info in config.items():
            result.append({
                "id": layer_id,
                "name": info["name"],
                "category": info["category"],
            })
        return result

    def get_tile_url(self, layer_id):
        if layer_id in self._tile_url_cache:
            return self._tile_url_cache[layer_id]
        config = self._get_layer_config()
        if layer_id not in config:
            raise ValueError("Неизвестный слой: {}".format(layer_id))
        builder = config[layer_id]["builder"]
        image, vis_params = builder()
        map_id = image.getMapId(vis_params)
        url = map_id["tile_fetcher"].url_format
        self._tile_url_cache[layer_id] = url
        return url

    def analyze_polygon(self, layer_id, geojson, buffer_km=None):
        """
        Статистика слоя внутри полигона.
        buffer_km=None или 0 — без буфера.
        buffer_km>0 — с буферной зоной.
        """
        config = self._get_layer_config()
        if layer_id not in config:
            raise ValueError("Неизвестный слой: {}".format(layer_id))

        geometry = ee.Geometry(geojson)

        builder = config[layer_id]["builder"]
        image, _ = builder()

        # Статистика по полигону
        print("[GEE] Анализ слоя '{}' для полигона...".format(config[layer_id]["name"]))

        stats_inner = image.reduceRegion(
            reducer=ee.Reducer.mean()
                .combine(ee.Reducer.minMax(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=geometry,
            scale=100,
            maxPixels=1e8,
            bestEffort=True,
        ).getInfo()

        area_inner = geometry.area().getInfo() / 1e6

        print("[GEE]   -> stats: {}".format(str(stats_inner)[:200]))

        result = {
            "layer_id": layer_id,
            "layer_name": config[layer_id]["name"],
            "polygon": {
                "area_km2": round(area_inner, 4),
                "stats": stats_inner,
            },
        }

        # Буфер — ТОЛЬКО если задан и > 0
        if buffer_km and buffer_km > 0:
            print("[GEE]   -> буфер {} км...".format(buffer_km))
            buffered = geometry.buffer(buffer_km * 1000)

            stats_buffer = image.reduceRegion(
                reducer=ee.Reducer.mean()
                    .combine(ee.Reducer.minMax(), sharedInputs=True)
                    .combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=buffered,
                scale=100,
                maxPixels=1e8,
                bestEffort=True,
            ).getInfo()

            area_buffer = buffered.area().getInfo() / 1e6

            result["buffer"] = {
                "buffer_km": buffer_km,
                "area_km2": round(area_buffer, 4),
                "stats": stats_buffer,
            }

        return result

    def analyze_all_layers_polygon(self, geojson, buffer_km=None):
        """Статистика по ВСЕМ слоям для полигона."""
        print("[GEE] ═══ Начало анализа всех слоёв ═══")
        print("[GEE] buffer_km = {}".format(buffer_km))

        results = []
        config = self._get_layer_config()
        for layer_id in config:
            try:
                r = self.analyze_polygon(layer_id, geojson, buffer_km)
                results.append(r)
            except Exception as e:
                print("[GEE] ОШИБКА слоя '{}': {}".format(layer_id, e))
                results.append({
                    "layer_id": layer_id,
                    "layer_name": config[layer_id]["name"],
                    "error": str(e),
                })

        ok_count = len([r for r in results if "error" not in r])
        err_count = len([r for r in results if "error" in r])
        print("[GEE] ═══ Анализ завершён: {} OK, {} ошибок ═══".format(ok_count, err_count))

        return results


ee_service = EarthEngineService()