from app.schemas.models import RoutePoint, RouteRequest, RouteResponse
import pandas as pd
import numpy as np
from typing import List, Tuple


class DataProcessor:
    """Класс для обработки и подготовки данных"""
    
    def __init__(self):
        self.speed_car = 50  
        self.speed_walk = 5   
        
    def load_dataset(self, file_path: str = "data/dataset.csv") -> pd.DataFrame:
        """Загрузка датасета из CSV файла"""
        df = pd.read_csv(file_path)
        if 'Динамический критерий' in df.columns:
            df = df.drop('Динамический критерий', axis=1)
        return df
    
    def prepare_route_points(self, df: pd.DataFrame) -> List[RoutePoint]:
        """Подготовка точек маршрута из датафрейма"""
        points = []
        
        for _, row in df.iterrows():
            point = RoutePoint(
                id=int(row['Номер объекта']),
                address=row['Адрес объекта'],
                latitude=float(row['Географическая широта']),
                longitude=float(row['Географическая долгота']),
                work_start=row['Время начала рабочего дня'],
                work_end=row['Время окончания рабочего дня'],
                lunch_start=row['Время начала обеда'],
                lunch_end=row['Время окончания обеда'],
                client_level=row['Уровень клиента']
            )
            points.append(point)
        
        return points
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Расчет расстояния между двумя точками"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        R = 6371  
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    def calculate_travel_time(self, distance: float, transport_mode: str, traffic_level: int) -> float:
        """Расчет времени в пути с учетом транспорта и пробок"""
        base_speed = self.speed_car if transport_mode == "car" else self.speed_walk
        
        traffic_factor = 1 - (traffic_level * 0.1)
        effective_speed = base_speed * max(0.1, traffic_factor)  
        
        travel_time = distance / effective_speed  
        return travel_time * 60  
    
    def create_distance_matrix(self, points: List[RoutePoint]) -> np.ndarray:
        """Создание матрицы расстояний между всеми точками"""
        n = len(points)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    coord1 = (points[i].latitude, points[i].longitude)
                    coord2 = (points[j].latitude, points[j].longitude)
                    matrix[i][j] = self.calculate_distance(coord1, coord2)
        
        return matrix
    
    def create_time_matrix(self, points: List[RoutePoint], transport_mode: str, traffic_level: int) -> np.ndarray:
        """Создание матрицы времени между всеми точками"""
        distance_matrix = self.create_distance_matrix(points)
        n = len(points)
        time_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = distance_matrix[i][j]
                    travel_time = self.calculate_travel_time(distance, transport_mode, traffic_level)
                    
                    stop_time = points[j].stop_duration
                    time_matrix[i][j] = travel_time + stop_time
        
        return time_matrix

if __name__ == "__main__":

    processor = DataProcessor()
    df = processor.load_dataset("data/dataset.csv")
    points = processor.prepare_route_points(df)
    
    print(f"Загружено {len(points)} точек маршрута")
    print(f"Первая точка: {points[0].address}")
    print(f"VIP клиенты: {sum(1 for p in points if p.client_level == 'VIP')}")
