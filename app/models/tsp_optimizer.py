import numpy as np
from typing import List, Tuple, Dict
from datetime import time
import random

class TSPOptimizer:
    """
    Класс для оптимизации маршрутов с использованием базового алгоритма коммивояжера (TSP)
    """
    
    def __init__(self):
        """Инициализация TSP оптимизатора"""
        pass
    
    def calculate_distance_matrix(self, points) -> np.ndarray:
        """Вычисление матрицы расстояний между точками"""
        n = len(points)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = points[i].latitude, points[i].longitude
                    lat2, lon2 = points[j].latitude, points[j].longitude
                    
                    # Простая евклидова формула (не точная, но быстрая)
                    distance = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # примерное преобразование в км
                    distance_matrix[i][j] = distance
                else:
                    distance_matrix[i][j] = 0
        
        return distance_matrix
    
    def simple_nearest_neighbor_tsp(self, distance_matrix: np.ndarray, start_point: int = 0) -> List[int]:
        """Максимально простой алгоритм ближайшего соседа для TSP"""
        n = len(distance_matrix)
        visited = [False] * n
        route = [start_point]
        visited[start_point] = True
        current = start_point
        
        while len(route) < n:
            nearest = -1
            min_distance = float('inf')
            
            for i in range(n):
                if not visited[i] and distance_matrix[current][i] < min_distance:
                    min_distance = distance_matrix[current][i]
                    nearest = i
            
            if nearest != -1:
                route.append(nearest)
                visited[nearest] = True
                current = nearest
        
        return route
    
    def simple_random_tsp(self, distance_matrix: np.ndarray, start_point: int = 0) -> List[int]:
        """Максимально простой случайный алгоритм TSP"""
        n = len(distance_matrix)
        
        # Простое случайное перемешивание
        route = list(range(n))
        route.remove(start_point)
        random.shuffle(route)
        return [start_point] + route
    
    def simple_sequential_tsp(self, distance_matrix: np.ndarray, start_point: int = 0) -> List[int]:
        """Максимально простой последовательный алгоритм TSP"""
        n = len(distance_matrix)
        
        # Просто последовательный обход
        route = list(range(n))
        return route
    
    def calculate_route_distance(self, route: List[int], distance_matrix: np.ndarray) -> float:
        """Простое вычисление общего расстояния маршрута"""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
        return total_distance
    
    def optimize_route(self, points, method: str = 'nearest_neighbor') -> List[int]:
        """Основной метод оптимизации маршрута TSP (максимально простой)"""
        distance_matrix = self.calculate_distance_matrix(points)
        
        if method == 'nearest_neighbor':
            # Простой алгоритм ближайшего соседа
            route = self.simple_nearest_neighbor_tsp(distance_matrix, 0)
        elif method == 'random':
            # Простой случайный алгоритм
            route = self.simple_random_tsp(distance_matrix, 0)
        elif method == 'sequential':
            # Простой последовательный алгоритм
            route = self.simple_sequential_tsp(distance_matrix, 0)
        else:
            # По умолчанию используем простейший метод
            route = self.simple_nearest_neighbor_tsp(distance_matrix, 0)
        
        return route
    
    def calculate_arrival_times(self, route: List[int], points, time_matrix, start_time: time) -> List[str]:
        """Расчет времени прибытия в каждую точку"""
        arrival_times = []
        current_time = start_time.hour * 60 + start_time.minute  # время в минутах
        
        for i, point_idx in enumerate(route):
            arrival_times.append(f"{int(current_time // 60):02d}:{int(current_time % 60):02d}")
            
            current_time += points[point_idx].stop_duration
            
            if i < len(route) - 1:
                next_idx = route[i + 1]
                current_time += time_matrix[point_idx][next_idx]
        
        return arrival_times

if __name__ == "__main__":
    from app.schemas.models import RoutePoint, ClientLevel
    
    test_points = [
        RoutePoint(
            id=0, address="Точка A", latitude=47.221532, longitude=39.704423,
            work_start="09:00", work_end="18:00", lunch_start="13:00", lunch_end="14:00",
            client_level=ClientLevel.VIP, stop_duration=30
        ),
        RoutePoint(
            id=1, address="Точка B", latitude=47.235671, longitude=39.689543,
            work_start="09:00", work_end="18:00", lunch_start="13:00", lunch_end="14:00",
            client_level=ClientLevel.STANDARD, stop_duration=20
        ),
        RoutePoint(
            id=2, address="Точка C", latitude=47.245123, longitude=39.712345,
            work_start="09:00", work_end="18:00", lunch_start="13:00", lunch_end="14:00",
            client_level=ClientLevel.VIP, stop_duration=25
        )
    ]
    
    # Тестируем упрощенный оптимизатор
    tsp_optimizer = TSPOptimizer()
    route = tsp_optimizer.optimize_route(test_points, method='nearest_neighbor')
    
    print("🧪 Тестирование МАКСИМАЛЬНО УПРОЩЕННОГО TSP алгоритма")
    print("=" * 60)
    
    methods = ['nearest_neighbor', 'random', 'sequential']
    
    for method in methods:
        print(f"\n📈 Метод: {method}")
        route = tsp_optimizer.optimize_route(test_points, method=method)
        
        # Вычисляем расстояние
        distance_matrix = tsp_optimizer.calculate_distance_matrix(test_points)
        total_distance = tsp_optimizer.calculate_route_distance(route, distance_matrix)
        
        print(f"   Маршрут: {route}")
        print(f"   Общее расстояние: {total_distance:.2f} км")
    
    print("\n" + "=" * 60)
    print("📝 ВЫВОДЫ:")
    print("✅ Максимально упрощенный TSP алгоритм работает, но:")
    print("   - НЕ учитывает приоритет VIP клиентов")
    print("   - НЕ оптимизирует порядок посещения")
    print("   - НЕ учитывает внешние факторы")
    print("   - Использует только самые простые методы")
    print("\n🎯 Это демонстрирует необходимость интеллектуальных")
    print("   алгоритмов, таких как GNN!")