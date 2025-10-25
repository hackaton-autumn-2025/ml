from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routers.route_optimization import router as route_router

app = FastAPI(
    title="Route Optimization API",
    description="API для оптимизации маршрутов с использованием Graph Neural Networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(route_router)

@app.get("/")
async def root():
    """Корневой эндпоинт приложения"""
    return {
        "message": "Route Optimization API",
        "version": "1.0.0",
        "description": "Система оптимизации маршрутов на основе Graph Neural Networks",
        "endpoints": {
            "api_docs": "/docs",
            "api_redoc": "/redoc",
            "health": "/api/v1/health",
            "optimize_route": "/api/v1/optimize-route",
            "dataset_info": "/api/v1/dataset-info"
        },
        "architecture": {
            "schemas": "app/schemas/",
            "models": "app/models/",
            "routers": "app/routers/",
            "views": "app/views/",
            "tests": "tests/",
            "data": "data/",
            "education": "educate/"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )