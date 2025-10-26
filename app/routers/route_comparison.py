from fastapi import APIRouter, HTTPException
from typing import List, Dict, Tuple, Optional
from datetime import time, datetime
import json

from app.schemas.models import (
    RoutePoint, RouteRequest, RouteResponse, RouteComparison, 
    ClientLevel, TransportMode, OptimizationMethod
)
from app.models.data_processor import DataProcessor
from app.models.improved_gnn_optimizer import ImprovedRouteOptimizer
from app.models.tsp_optimizer import TSPOptimizer

router = APIRouter(prefix="/api/v1", tags=["route-comparison"])

data_processor = DataProcessor()
gnn_optimizer = ImprovedRouteOptimizer(model_path="models/improved_route_optimization_gnn.pth")
tsp_optimizer = TSPOptimizer()

@router.get("/")
async def root():
    """Корневой эндпоинт API сравнения маршрутов"""
    return {
        "message": "Route Comparison Demo API",
        "version": "1.0.0",
        "description": "Демонстрация сравнения GNN алгоритма с TSP на заданном маршруте",
        "endpoints": {
            "get_route_schemas": "/api/v1/get",
            "compare_demo_routes": "/api/v1/compare-demo",
            "health": "/api/v1/health",
            "docs": "/docs"
        }
    }

@router.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "message": "Сервис сравнения маршрутов работает корректно",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/get")
async def get_route_schemas():
    """
    Получение схем маршрутов для демонстрации
    
    Возвращает:
    - GNN маршрут с учетом внешних факторов
    - TSP маршрут без учета внешних факторов
    - Сравнительные метрики
    """
    try:
        # Фиксированные параметры для демо
        max_points = 12
        traffic_level = 3
        transport_mode = "car"
        
        # Загружаем датасет
        df = data_processor.load_dataset("data/dataset.csv")
        
        # Ограничиваем количество точек для демо
        if len(df) > max_points:
            df = df.head(max_points)
        
        # Подготавливаем точки
        points = data_processor.prepare_route_points(df)
        
        # Создаем матрицу времени
        time_matrix = data_processor.create_time_matrix(
            points, 
            transport_mode, 
            traffic_level
        )
        
        # GNN маршрут с учетом внешних факторов
        gnn_route = gnn_optimizer.optimize_route(
            points, 
            time_matrix, 
            start_point_idx=0, 
            method='genetic'
        )
        gnn_metrics = calculate_route_metrics(gnn_route, points, time_matrix, time(9, 0))
        
        # TSP маршрут без учета внешних факторов
        tsp_route = tsp_optimizer.optimize_route(points, method='nearest_neighbor')
        tsp_metrics = calculate_route_metrics(tsp_route, points, time_matrix, time(9, 0))
        
        # Вычисляем метрики сравнения
        comparison_metrics = calculate_comparison_metrics(gnn_metrics, tsp_metrics)
        
        # Создаем детальную информацию о точках
        def create_route_details(route, metrics, algorithm_name):
            details = []
            for i, point_idx in enumerate(route):
                point = points[point_idx]
                details.append({
                    "order": i + 1,
                    "id": point.id,
                    "address": point.address,
                    "coordinates": (point.latitude, point.longitude),
                    "arrival_time": metrics["arrival_times"][i],
                    "client_level": point.client_level,
                    "work_hours": f"{point.work_start}-{point.work_end}",
                    "stop_duration": point.stop_duration,
                    "algorithm": algorithm_name
                })
            return details
        
        gnn_details = create_route_details(gnn_route, gnn_metrics, "GNN")
        tsp_details = create_route_details(tsp_route, tsp_metrics, "TSP")
        
        return {
            "success": True,
            "message": "Схемы маршрутов успешно получены",
            "demo_info": {
                "total_points": len(points),
                "traffic_level": traffic_level,
                "transport_mode": transport_mode,
                "vip_clients": sum(1 for p in points if p.client_level == ClientLevel.VIP),
                "standard_clients": sum(1 for p in points if p.client_level == ClientLevel.STANDARD)
            },
            "gnn_route": {
                "algorithm": "GNN (Graph Neural Networks)",
                "description": "Учитывает внешние факторы: трафик, приоритет VIP, временные окна",
                "optimized_order": gnn_route,
                "total_distance": gnn_metrics["total_distance"],
                "total_time": gnn_metrics["total_time"],
                "route_coordinates": gnn_metrics["route_coordinates"],
                "route_details": gnn_details
            },
            "tsp_route": {
                "algorithm": "TSP (Traveling Salesman Problem)",
                "description": "Базовый алгоритм без учета внешних факторов",
                "optimized_order": tsp_route,
                "total_distance": tsp_metrics["total_distance"],
                "total_time": tsp_metrics["total_time"],
                "route_coordinates": tsp_metrics["route_coordinates"],
                "route_details": tsp_details
            },
            "comparison": {
                "distance_improvement": comparison_metrics["distance_improvement"],
                "time_improvement": comparison_metrics["time_improvement"],
                "efficiency_score": comparison_metrics["efficiency_score"],
                "distance_improvement_percent": round((comparison_metrics["distance_improvement"] / tsp_metrics["total_distance"]) * 100, 1) if tsp_metrics["total_distance"] > 0 else 0,
                "time_improvement_percent": round((comparison_metrics["time_improvement"] / tsp_metrics["total_time"]) * 100, 1) if tsp_metrics["total_time"] > 0 else 0,
                "recommendation": generate_recommendation(comparison_metrics, {})
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении схем маршрутов: {str(e)}"
        )

@router.get("/compare-demo")
async def compare_demo_routes():
    """
    Демонстрационное сравнение маршрутов GNN vs TSP
    """
    try:
        # Фиксированные параметры для демо
        max_points = 12
        traffic_level = 3
        transport_mode = "car"
        
        # Загружаем датасет
        df = data_processor.load_dataset("data/dataset.csv")
        
        # Ограничиваем количество точек для демо
        if len(df) > max_points:
            df = df.head(max_points)
        
        # Подготавливаем точки
        points = data_processor.prepare_route_points(df)
        
        # Создаем матрицу времени
        time_matrix = data_processor.create_time_matrix(
            points, 
            transport_mode, 
            traffic_level
        )
        
        # GNN маршрут с учетом внешних факторов
        gnn_route = gnn_optimizer.optimize_route(
            points, 
            time_matrix, 
            start_point_idx=0, 
            method='genetic'
        )
        gnn_metrics = calculate_route_metrics(gnn_route, points, time_matrix, time(9, 0))
        
        # TSP маршрут без учета внешних факторов
        tsp_route = tsp_optimizer.optimize_route(points, method='nearest_neighbor')
        tsp_metrics = calculate_route_metrics(tsp_route, points, time_matrix, time(9, 0))
        
        # Вычисляем метрики сравнения
        comparison_metrics = calculate_comparison_metrics(gnn_metrics, tsp_metrics)
        
        return {
            "success": True,
            "message": "Демонстрационное сравнение выполнено успешно",
            "demo_parameters": {
                "points_count": len(points),
                "traffic_level": traffic_level,
                "transport_mode": transport_mode,
                "vip_clients": sum(1 for p in points if p.client_level == ClientLevel.VIP),
                "standard_clients": sum(1 for p in points if p.client_level == ClientLevel.STANDARD)
            },
            "gnn_results": {
                "algorithm": "GNN (с учетом внешних факторов)",
                "route_order": gnn_route,
                "total_distance_km": gnn_metrics["total_distance"],
                "total_time_minutes": gnn_metrics["total_time"],
                "arrival_times": gnn_metrics["arrival_times"]
            },
            "tsp_results": {
                "algorithm": "TSP (без учета внешних факторов)",
                "route_order": tsp_route,
                "total_distance_km": tsp_metrics["total_distance"],
                "total_time_minutes": tsp_metrics["total_time"],
                "arrival_times": tsp_metrics["arrival_times"]
            },
            "improvement_metrics": {
                "distance_saved_km": comparison_metrics["distance_improvement"],
                "time_saved_minutes": comparison_metrics["time_improvement"],
                "distance_improvement_percent": round((comparison_metrics["distance_improvement"] / tsp_metrics["total_distance"]) * 100, 1) if tsp_metrics["total_distance"] > 0 else 0,
                "time_improvement_percent": round((comparison_metrics["time_improvement"] / tsp_metrics["total_time"]) * 100, 1) if tsp_metrics["total_time"] > 0 else 0,
                "efficiency_score": comparison_metrics["efficiency_score"]
            },
            "conclusion": generate_recommendation(comparison_metrics, {})
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при демонстрационном сравнении: {str(e)}"
        )

def calculate_route_metrics(route: List[int], points: List[RoutePoint], 
                          time_matrix: List[List[float]], start_time: time) -> Dict:
    """Вычисление метрик маршрута"""
    # Рассчитываем время прибытия
    arrival_times = []
    current_time = start_time.hour * 60 + start_time.minute
    
    for i, point_idx in enumerate(route):
        arrival_times.append(f"{int(current_time // 60):02d}:{int(current_time % 60):02d}")
        current_time += points[point_idx].stop_duration
        if i < len(route) - 1:
            next_idx = route[i + 1]
            current_time += time_matrix[point_idx][next_idx]
    
    # Рассчитываем общее расстояние и время
    total_distance = 0
    total_time = 0
    
    for i in range(len(route) - 1):
        from_idx = route[i]
        to_idx = route[i + 1]
        
        distance = data_processor.calculate_distance(
            (points[from_idx].latitude, points[from_idx].longitude),
            (points[to_idx].latitude, points[to_idx].longitude)
        )
        total_distance += distance
        total_time += time_matrix[from_idx][to_idx]
    
    # Создаем координаты маршрута
    route_coordinates = []
    for point_idx in route:
        point = points[point_idx]
        route_coordinates.append((point.latitude, point.longitude))
    
    return {
        "optimized_order": route,
        "arrival_times": arrival_times,
        "total_distance": round(total_distance, 2),
        "total_time": round(total_time, 2),
        "route_coordinates": route_coordinates,
        "success": True,
        "message": "Маршрут успешно оптимизирован"
    }

def calculate_comparison_metrics(gnn_metrics: Dict, tsp_metrics: Dict) -> Dict:
    """Вычисление метрик сравнения"""
    distance_improvement = tsp_metrics["total_distance"] - gnn_metrics["total_distance"]
    time_improvement = tsp_metrics["total_time"] - gnn_metrics["total_time"]
    
    # Процент улучшения
    distance_improvement_pct = (distance_improvement / tsp_metrics["total_distance"]) * 100 if tsp_metrics["total_distance"] > 0 else 0
    time_improvement_pct = (time_improvement / tsp_metrics["total_time"]) * 100 if tsp_metrics["total_time"] > 0 else 0
    
    # Оценка эффективности (0-1, где 1 - идеально)
    efficiency_score = 0.5  # базовая оценка
    
    if distance_improvement > 0:
        efficiency_score += 0.2
    if time_improvement > 0:
        efficiency_score += 0.2
    if distance_improvement_pct > 10:
        efficiency_score += 0.1
    if time_improvement_pct > 10:
        efficiency_score += 0.1
    
    efficiency_score = min(efficiency_score, 1.0)
    
    return {
        "distance_improvement": round(distance_improvement, 2),
        "time_improvement": round(time_improvement, 2),
        "efficiency_score": round(efficiency_score, 2)
    }

def generate_recommendation(comparison_metrics: Dict, improvement_pct: Dict) -> str:
    """Генерация рекомендации на основе метрик сравнения"""
    distance_improvement = comparison_metrics["distance_improvement"]
    time_improvement = comparison_metrics["time_improvement"]
    efficiency_score = comparison_metrics["efficiency_score"]
    
    if efficiency_score >= 0.8:
        if distance_improvement > 0 and time_improvement > 0:
            return "GNN значительно превосходит максимально упрощенный TSP по всем метрикам. Это демонстрирует преимущества интеллектуальной оптимизации над простейшими алгоритмами."
        elif distance_improvement > 0:
            return "GNN показывает лучшие результаты по расстоянию благодаря учету внешних факторов. Максимально упрощенный TSP не может конкурировать."
        elif time_improvement > 0:
            return "GNN показывает лучшие результаты по времени благодаря интеллектуальному планированию. Простейший TSP уступает по всем параметрам."
        else:
            return "GNN и максимально упрощенный TSP показывают сопоставимые результаты. GNN все равно предпочтительнее для реальных задач."
    elif efficiency_score >= 0.6:
        return "GNN показывает умеренные улучшения по сравнению с максимально упрощенным TSP. Это подтверждает важность интеллектуальных алгоритмов."
    else:
        return "Даже максимально упрощенный TSP показывает сопоставимые результаты. Это подчеркивает необходимость более сложных алгоритмов для реальных задач."