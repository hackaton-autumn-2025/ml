from fastapi import APIRouter, HTTPException
from typing import List, Dict, Tuple, Optional
from datetime import time, datetime
import json

from app.schemas.models import RoutePoint, RouteRequest, RouteResponse, DatasetInfo, RouteStatistics, ClientLevel, TransportMode, OptimizationMethod
from app.models.data_processor import DataProcessor
from app.models.improved_gnn_optimizer import ImprovedRouteOptimizer

# Создаем роутер для основных операций
router = APIRouter(prefix="/api/v1", tags=["route-optimization"])

# Инициализация компонентов
data_processor = DataProcessor()
# Загружаем обученную улучшенную модель
route_optimizer = ImprovedRouteOptimizer(model_path="models/improved_route_optimization_gnn.pth")

@router.get("/")
async def root():
    """Корневой эндпоинт API"""
    return {
        "message": "Route Optimization API",
        "version": "1.0.0",
        "endpoints": {
            "optimize_route": "/api/v1/optimize-route",
            "health": "/api/v1/health",
            "dataset_info": "/api/v1/dataset-info",
            "docs": "/docs"
        }
    }

@router.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "message": "Сервис работает корректно",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/dataset-info", response_model=DatasetInfo)
async def get_dataset_info():
    """Получение информации о загруженном датасете"""
    try:
        df = data_processor.load_dataset("data/dataset.csv")
        points = data_processor.prepare_route_points(df)
        
        vip_count = sum(1 for p in points if p.client_level == ClientLevel.VIP)
        standard_count = sum(1 for p in points if p.client_level == ClientLevel.STANDARD)
        
        return DatasetInfo(
            total_points=len(points),
            vip_clients=vip_count,
            standard_clients=standard_count,
            sample_points=[
                {
                    "id": p.id,
                    "address": p.address,
                    "client_level": p.client_level,
                    "coordinates": (p.latitude, p.longitude)
                }
                for p in points[:5]  # первые 5 точек как пример
            ]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при загрузке информации о датасете: {str(e)}"
        )

def convert_point_request_to_route_point(point_req) -> RoutePoint:
    """Конвертация PointRequest в RoutePoint"""
    return RoutePoint(
        id=point_req.id,
        address=point_req.address,
        latitude=point_req.latitude,
        longitude=point_req.longitude,
        work_start=point_req.work_start,
        work_end=point_req.work_end,
        lunch_start=point_req.lunch_start,
        lunch_end=point_req.lunch_end,
        client_level=point_req.client_level,
        stop_duration=point_req.stop_duration
    )

@router.post("/optimize-route", response_model=RouteResponse)
async def optimize_route(request: RouteRequest):
    """
    Оптимизация маршрута на основе списка точек
    
    Принимает:
    - Список точек с координатами и временными ограничениями
    - Время начала поездки
    - Уровень загруженности дорог (0-10)
    - Режим транспорта (car/walk)
    - Координаты начальной точки
    
    Возвращает:
    - Оптимизированный порядок посещения точек
    - Время прибытия в каждую точку
    - Общее расстояние и время
    - Координаты маршрута
    """
    try:
        # Конвертируем точки
        points = [convert_point_request_to_route_point(point) for point in request.points]
        
        # Парсим время начала
        start_time = request.get_start_time()
        
        # Создаем матрицу времени
        time_matrix = data_processor.create_time_matrix(
            points, 
            request.transport_mode, 
            request.traffic_level
        )
        
        # Находим ближайшую к стартовой точке
        start_point_idx = 0
        min_distance = float('inf')
        
        for i, point in enumerate(points):
            distance = data_processor.calculate_distance(
                request.start_point, 
                (point.latitude, point.longitude)
            )
            if distance < min_distance:
                min_distance = distance
                start_point_idx = i
        
        # Оптимизируем маршрут
        optimized_order = route_optimizer.optimize_route(
            points, 
            time_matrix, 
            start_point_idx, 
            method='genetic'
        )
        
        # Рассчитываем время прибытия
        arrival_times = route_optimizer.calculate_arrival_times(
            optimized_order, 
            points, 
            time_matrix, 
            start_time
        )
        
        # Рассчитываем общее расстояние и время
        total_distance = 0
        total_time = 0
        
        for i in range(len(optimized_order) - 1):
            from_idx = optimized_order[i]
            to_idx = optimized_order[i + 1]
            
            distance = data_processor.calculate_distance(
                (points[from_idx].latitude, points[from_idx].longitude),
                (points[to_idx].latitude, points[to_idx].longitude)
            )
            total_distance += distance
            
            time_travel = time_matrix[from_idx][to_idx]
            total_time += time_travel
        
        # Создаем координаты маршрута
        route_coordinates = []
        for point_idx in optimized_order:
            point = points[point_idx]
            route_coordinates.append((point.latitude, point.longitude))
        
        return RouteResponse(
            optimized_order=optimized_order,
            arrival_times=arrival_times,
            total_distance=round(total_distance, 2),
            total_time=round(total_time, 2),
            route_coordinates=route_coordinates,
            success=True,
            message="Маршрут успешно оптимизирован"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при оптимизации маршрута: {str(e)}"
        )

@router.post("/optimize-route-from-dataset")
async def optimize_route_from_dataset(
    start_time: str = "09:00",
    traffic_level: int = 3,
    transport_mode: str = "car",
    optimization_method: str = "genetic",
    max_points: int = 10
):
    """
    Оптимизация маршрута на основе загруженного датасета
    
    Параметры:
    - start_time: время начала поездки
    - traffic_level: уровень загруженности дорог (0-10)
    - transport_mode: режим транспорта (car/walk)
    - optimization_method: метод оптимизации (genetic/greedy)
    - max_points: максимальное количество точек для оптимизации
    """
    try:
        # Загружаем датасет
        df = data_processor.load_dataset("data/dataset.csv")
        
        # Ограничиваем количество точек
        if len(df) > max_points:
            df = df.head(max_points)
        
        # Подготавливаем точки
        points = data_processor.prepare_route_points(df)
        
        # Парсим время начала
        start_time_obj = datetime.strptime(start_time, '%H:%M').time()
        
        # Создаем матрицу времени
        time_matrix = data_processor.create_time_matrix(
            points, 
            transport_mode, 
            traffic_level
        )
        
        # Оптимизируем маршрут (начинаем с первой точки)
        optimized_order = route_optimizer.optimize_route(
            points, 
            time_matrix, 
            start_point_idx=0, 
            method=optimization_method
        )
        
        # Рассчитываем время прибытия
        arrival_times = route_optimizer.calculate_arrival_times(
            optimized_order, 
            points, 
            time_matrix, 
            start_time_obj
        )
        
        # Рассчитываем общее расстояние и время
        total_distance = 0
        total_time = 0
        
        for i in range(len(optimized_order) - 1):
            from_idx = optimized_order[i]
            to_idx = optimized_order[i + 1]
            
            distance = data_processor.calculate_distance(
                (points[from_idx].latitude, points[from_idx].longitude),
                (points[to_idx].latitude, points[to_idx].longitude)
            )
            total_distance += distance
            
            time_travel = time_matrix[from_idx][to_idx]
            total_time += time_travel
        
        # Создаем координаты маршрута
        route_coordinates = []
        for point_idx in optimized_order:
            point = points[point_idx]
            route_coordinates.append((point.latitude, point.longitude))
        
        # Создаем детальную информацию о точках
        route_details = []
        for i, point_idx in enumerate(optimized_order):
            point = points[point_idx]
            route_details.append({
                "order": i + 1,
                "id": point.id,
                "address": point.address,
                "coordinates": (point.latitude, point.longitude),
                "arrival_time": arrival_times[i],
                "client_level": point.client_level,
                "work_hours": f"{point.work_start}-{point.work_end}"
            })
        
        return {
            "success": True,
            "message": "Маршрут успешно оптимизирован на основе датасета",
            "optimized_order": optimized_order,
            "arrival_times": arrival_times,
            "total_distance": round(total_distance, 2),
            "total_time": round(total_time, 2),
            "route_coordinates": route_coordinates,
            "route_details": route_details,
            "statistics": RouteStatistics(
                total_points=len(points),
                vip_clients=sum(1 for p in points if p.client_level == ClientLevel.VIP),
                standard_clients=sum(1 for p in points if p.client_level == ClientLevel.STANDARD),
                optimization_method=optimization_method,
                transport_mode=transport_mode,
                traffic_level=traffic_level
            )
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при оптимизации маршрута из датасета: {str(e)}"
        )
