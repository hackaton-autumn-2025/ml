from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, time
from enum import Enum

class ClientLevel(str, Enum):
    """Перечисление уровней клиентов"""
    VIP = "VIP"
    STANDARD = "Standart"

class TransportMode(str, Enum):
    """Перечисление режимов транспорта"""
    CAR = "car"
    WALK = "walk"

class OptimizationMethod(str, Enum):
    """Перечисление методов оптимизации"""
    GENETIC = "genetic"
    GREEDY = "greedy"

class RoutePoint(BaseModel):
    """Pydantic модель для представления точки маршрута"""
    id: int = Field(..., description="Уникальный идентификатор точки", example=1)
    address: str = Field(..., description="Адрес точки", example="ул. Большая Садовая, д. 1")
    latitude: float = Field(..., ge=-90, le=90, description="Географическая широта", example=47.221532)
    longitude: float = Field(..., ge=-180, le=180, description="Географическая долгота", example=39.704423)
    work_start: str = Field(..., description="Время начала рабочего дня (HH:MM)", example="09:00")
    work_end: str = Field(..., description="Время окончания рабочего дня (HH:MM)", example="18:00")
    lunch_start: str = Field(..., description="Время начала обеда (HH:MM)", example="13:00")
    lunch_end: str = Field(..., description="Время окончания обеда (HH:MM)", example="14:00")
    client_level: ClientLevel = Field(..., description="Уровень клиента", example=ClientLevel.VIP)
    stop_duration: int = Field(default=30, ge=1, le=300, description="Время остановки в минутах", example=30)
    
    @validator('work_start', 'work_end', 'lunch_start', 'lunch_end')
    def validate_time_format(cls, v):
        """Валидация формата времени"""
        try:
            # Проверяем строгий формат HH:MM
            if len(v) != 5 or v[2] != ':':
                raise ValueError('Время должно быть в формате HH:MM')
            datetime.strptime(v, '%H:%M')
            return v
        except ValueError:
            raise ValueError('Время должно быть в формате HH:MM (например, 09:00)')
    
    @root_validator
    def validate_time_logic(cls, values):
        """Валидация логики времени"""
        work_start = datetime.strptime(values.get('work_start', '00:00'), '%H:%M').time()
        work_end = datetime.strptime(values.get('work_end', '23:59'), '%H:%M').time()
        lunch_start = datetime.strptime(values.get('lunch_start', '00:00'), '%H:%M').time()
        lunch_end = datetime.strptime(values.get('lunch_end', '23:59'), '%H:%M').time()
        
        if work_start >= work_end:
            raise ValueError('Время начала работы должно быть раньше времени окончания')
        
        if lunch_start >= lunch_end:
            raise ValueError('Время начала обеда должно быть раньше времени окончания')
        
        if not (work_start <= lunch_start <= lunch_end <= work_end):
            raise ValueError('Обеденное время должно быть в пределах рабочего времени')
        
        return values
    
    def get_work_start_time(self) -> time:
        """Получить время начала работы как объект time"""
        return datetime.strptime(self.work_start, '%H:%M').time()
    
    def get_work_end_time(self) -> time:
        """Получить время окончания работы как объект time"""
        return datetime.strptime(self.work_end, '%H:%M').time()
    
    def get_lunch_start_time(self) -> time:
        """Получить время начала обеда как объект time"""
        return datetime.strptime(self.lunch_start, '%H:%M').time()
    
    def get_lunch_end_time(self) -> time:
        """Получить время окончания обеда как объект time"""
        return datetime.strptime(self.lunch_end, '%H:%M').time()
    
    class Config:
        """Конфигурация модели"""
        use_enum_values = True
        schema_extra = {
            "example": {
                "id": 1,
                "address": "ул. Большая Садовая, д. 1",
                "latitude": 47.221532,
                "longitude": 39.704423,
                "work_start": "09:00",
                "work_end": "18:00",
                "lunch_start": "13:00",
                "lunch_end": "14:00",
                "client_level": "VIP",
                "stop_duration": 30
            }
        }

class RouteRequest(BaseModel):
    """Pydantic модель для запроса оптимизации маршрута"""
    points: List[RoutePoint] = Field(..., min_items=2, description="Список точек маршрута")
    start_time: str = Field(..., description="Время начала поездки (HH:MM)", example="09:00")
    traffic_level: int = Field(..., ge=0, le=10, description="Уровень загруженности дорог (0-10)", example=3)
    transport_mode: TransportMode = Field(..., description="Режим транспорта", example=TransportMode.CAR)
    start_point: Tuple[float, float] = Field(..., description="Координаты начальной точки", example=(47.220000, 39.700000))
    
    @validator('start_time')
    def validate_start_time(cls, v):
        """Валидация времени начала"""
        try:
            datetime.strptime(v, '%H:%M')
            return v
        except ValueError:
            raise ValueError('Время должно быть в формате HH:MM')
    
    @validator('start_point')
    def validate_start_point(cls, v):
        """Валидация координат начальной точки"""
        lat, lon = v
        if not (-90 <= lat <= 90):
            raise ValueError('Широта должна быть в диапазоне -90 до 90')
        if not (-180 <= lon <= 180):
            raise ValueError('Долгота должна быть в диапазоне -180 до 180')
        return v
    
    def get_start_time(self) -> time:
        """Получить время начала как объект time"""
        return datetime.strptime(self.start_time, '%H:%M').time()
    
    class Config:
        """Конфигурация модели"""
        use_enum_values = True
        schema_extra = {
            "example": {
                "points": [
                    {
                        "id": 1,
                        "address": "ул. Большая Садовая, д. 1",
                        "latitude": 47.221532,
                        "longitude": 39.704423,
                        "work_start": "09:00",
                        "work_end": "18:00",
                        "lunch_start": "13:00",
                        "lunch_end": "14:00",
                        "client_level": "VIP",
                        "stop_duration": 30
                    }
                ],
                "start_time": "09:00",
                "traffic_level": 3,
                "transport_mode": "car",
                "start_point": [47.220000, 39.700000]
            }
        }

class RouteResponse(BaseModel):
    """Pydantic модель для ответа с оптимизированным маршрутом"""
    optimized_order: List[int] = Field(..., description="Оптимизированный порядок посещения точек")
    arrival_times: List[str] = Field(..., description="Время прибытия в каждую точку")
    total_distance: float = Field(..., ge=0, description="Общее расстояние в км")
    total_time: float = Field(..., ge=0, description="Общее время в минутах")
    route_coordinates: List[Tuple[float, float]] = Field(..., description="Координаты маршрута")
    success: bool = Field(default=True, description="Статус успешности операции")
    message: str = Field(default="Маршрут успешно оптимизирован", description="Сообщение о результате")
    
    class Config:
        """Конфигурация модели"""
        schema_extra = {
            "example": {
                "optimized_order": [0, 2, 1, 3],
                "arrival_times": ["09:00", "09:45", "10:30", "11:15"],
                "total_distance": 15.2,
                "total_time": 135.0,
                "route_coordinates": [[47.221532, 39.704423], [47.235671, 39.689543]],
                "success": True,
                "message": "Маршрут успешно оптимизирован"
            }
        }

class DatasetInfo(BaseModel):
    """Pydantic модель для информации о датасете"""
    total_points: int = Field(..., description="Общее количество точек")
    vip_clients: int = Field(..., description="Количество VIP клиентов")
    standard_clients: int = Field(..., description="Количество стандартных клиентов")
    sample_points: List[Dict] = Field(..., description="Примеры точек из датасета")
    
    class Config:
        """Конфигурация модели"""
        schema_extra = {
            "example": {
                "total_points": 100,
                "vip_clients": 3,
                "standard_clients": 97,
                "sample_points": [
                    {
                        "id": 1,
                        "address": "ул. Большая Садовая, д. 1",
                        "client_level": "VIP",
                        "coordinates": [47.221532, 39.704423]
                    }
                ]
            }
        }

class RouteStatistics(BaseModel):
    """Pydantic модель для статистики маршрута"""
    total_points: int = Field(..., description="Общее количество точек в маршруте")
    vip_clients: int = Field(..., description="Количество VIP клиентов")
    standard_clients: int = Field(..., description="Количество стандартных клиентов")
    optimization_method: str = Field(..., description="Использованный метод оптимизации")
    transport_mode: str = Field(..., description="Режим транспорта")
    traffic_level: int = Field(..., description="Уровень загруженности дорог")
    
    class Config:
        """Конфигурация модели"""
        schema_extra = {
            "example": {
                "total_points": 10,
                "vip_clients": 2,
                "standard_clients": 8,
                "optimization_method": "genetic",
                "transport_mode": "car",
                "traffic_level": 3
            }
        }
