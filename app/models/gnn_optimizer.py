import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Dict
import random

class RouteOptimizationGNN(nn.Module):
    """GNN модель для оптимизации маршрутов"""
    
    def __init__(self, 
                 node_features: int = 8,  # координаты, время работы, уровень клиента, etc.
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super(RouteOptimizationGNN, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GNN слои
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(node_features, hidden_dim, heads=4, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout))
        
        # Слои для предсказания порядка посещения
        self.order_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Слои для предсказания времени прибытия
        self.time_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        """Прямой проход через GNN"""
        # Применяем GNN слои
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Предсказываем порядок посещения и время
        order_scores = self.order_predictor(x)
        time_scores = self.time_predictor(x)
        
        return order_scores, time_scores

class RouteOptimizer:
    """Класс для оптимизации маршрутов с использованием GNN"""
    
    def __init__(self, model_path: str = None):
        self.model = RouteOptimizationGNN()
        if model_path:
            self.load_model(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def create_graph_data(self, points, time_matrix, traffic_level, transport_mode):
        """Создание графа для GNN из точек маршрута"""
        n = len(points)
        
        # Создаем узлы (точки маршрута)
        node_features = []
        for point in points:
            # Нормализуем координаты
            lat_norm = (point.latitude - 47.0) / 0.5  # нормализация для Ростова-на-Дону
            lon_norm = (point.longitude - 39.0) / 0.5
            
            # Время работы (нормализованное)
            work_start_norm = point.get_work_start_time().hour / 24.0
            work_end_norm = point.get_work_end_time().hour / 24.0
            
            # Уровень клиента (VIP = 1, Standart = 0)
            client_level_norm = 1.0 if point.client_level == 'VIP' else 0.0
            
            # Время остановки
            stop_duration_norm = point.stop_duration / 60.0
            
            # Трафик и режим транспорта
            traffic_norm = traffic_level / 10.0
            transport_norm = 1.0 if transport_mode == 'car' else 0.0
            
            features = [
                lat_norm, lon_norm, work_start_norm, work_end_norm,
                client_level_norm, stop_duration_norm, traffic_norm, transport_norm
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Создаем ребра (связи между точками)
        edge_list = []
        edge_weights = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_list.append([i, j])
                    # Вес ребра = время перемещения (инвертированное для минимизации)
                    weight = 1.0 / (time_matrix[i][j] + 1e-6)
                    edge_weights.append(weight)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def optimize_route_greedy(self, points, time_matrix, start_point_idx=0):
        """Жадный алгоритм для начальной оптимизации маршрута"""
        n = len(points)
        visited = [False] * n
        route = [start_point_idx]
        visited[start_point_idx] = True
        
        current = start_point_idx
        
        while len(route) < n:
            best_next = -1
            best_time = float('inf')
            
            for next_point in range(n):
                if not visited[next_point]:
                    # Приоритет VIP клиентам, но с учетом расстояния
                    time_to_next = time_matrix[current][next_point]
                    
                    # Если клиент VIP, уменьшаем "стоимость" на 20%
                    if points[next_point].client_level == 'VIP':
                        time_to_next *= 0.8
                    
                    if time_to_next < best_time:
                        best_time = time_to_next
                        best_next = next_point
            
            if best_next != -1:
                route.append(best_next)
                visited[best_next] = True
                current = best_next
        
        return route
    
    def optimize_route_genetic(self, points, time_matrix, start_point_idx=0, 
                             population_size=50, generations=100):
        """Генетический алгоритм для оптимизации маршрута"""
        n = len(points)
        
        def create_individual():
            """Создание одной особи (маршрута)"""
            route = list(range(n))
            route.remove(start_point_idx)
            random.shuffle(route)
            return [start_point_idx] + route
        
        def fitness(individual):
            """Функция пригодности (минимизация общего времени)"""
            total_time = 0
            for i in range(len(individual) - 1):
                from_idx = individual[i]
                to_idx = individual[i + 1]
                total_time += time_matrix[from_idx][to_idx]
            
            # Штраф за нарушение временных окон
            penalty = 0
            current_time = 0  # время в минутах от начала дня
            
            for i, point_idx in enumerate(individual):
                point = points[point_idx]
                
                # Проверяем, попадаем ли в рабочее время
                arrival_hour = current_time // 60
                arrival_minute = current_time % 60
                
                work_start_minutes = point.get_work_start_time().hour * 60 + point.get_work_start_time().minute
                work_end_minutes = point.get_work_end_time().hour * 60 + point.get_work_end_time().minute
                lunch_start_minutes = point.get_lunch_start_time().hour * 60 + point.get_lunch_start_time().minute
                lunch_end_minutes = point.get_lunch_end_time().hour * 60 + point.get_lunch_end_time().minute
                
                if current_time < work_start_minutes or current_time > work_end_minutes:
                    penalty += 100  # большой штраф за работу вне времени
                elif lunch_start_minutes <= current_time <= lunch_end_minutes:
                    penalty += 50  # штраф за обеденное время
                
                current_time += point.stop_duration
                if i < len(individual) - 1:
                    next_idx = individual[i + 1]
                    current_time += time_matrix[point_idx][next_idx]
            
            return total_time + penalty
        
        def crossover(parent1, parent2):
            """Скрещивание двух родителей"""
            child = [start_point_idx]
            remaining = [x for x in parent1[1:] if x in parent2[1:]]
            random.shuffle(remaining)
            child.extend(remaining)
            return child
        
        def mutate(individual):
            """Мутация особи"""
            if len(individual) > 3:
                i, j = random.sample(range(1, len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]
            return individual
        
        # Инициализация популяции
        population = [create_individual() for _ in range(population_size)]
        
        for generation in range(generations):
            # Оценка пригодности
            fitness_scores = [fitness(ind) for ind in population]
            
            # Селекция (турнирная)
            new_population = []
            for _ in range(population_size):
                # Турнир из 3 особей
                tournament = random.sample(list(zip(population, fitness_scores)), 3)
                winner = min(tournament, key=lambda x: x[1])[0]
                new_population.append(winner.copy())
            
            # Скрещивание и мутация
            for i in range(0, population_size - 1, 2):
                if random.random() < 0.7:  # вероятность скрещивания
                    child1 = crossover(new_population[i], new_population[i + 1])
                    child2 = crossover(new_population[i + 1], new_population[i])
                    new_population[i] = child1
                    new_population[i + 1] = child2
                
                if random.random() < 0.1:  # вероятность мутации
                    new_population[i] = mutate(new_population[i])
                if random.random() < 0.1:
                    new_population[i + 1] = mutate(new_population[i + 1])
            
            population = new_population
        
        # Возвращаем лучшую особь
        fitness_scores = [fitness(ind) for ind in population]
        best_idx = np.argmin(fitness_scores)
        return population[best_idx]
    
    def optimize_route(self, points, time_matrix, start_point_idx=0, method='genetic'):
        """Основной метод оптимизации маршрута"""
        if method == 'greedy':
            return self.optimize_route_greedy(points, time_matrix, start_point_idx)
        elif method == 'genetic':
            return self.optimize_route_genetic(points, time_matrix, start_point_idx)
        else:
            raise ValueError("Метод должен быть 'greedy' или 'genetic'")
    
    def calculate_arrival_times(self, route, points, time_matrix, start_time):
        """Расчет времени прибытия в каждую точку"""
        arrival_times = []
        current_time = start_time.hour * 60 + start_time.minute  # время в минутах
        
        for i, point_idx in enumerate(route):
            arrival_times.append(f"{current_time // 60:02d}:{current_time % 60:02d}")
            
            # Добавляем время остановки
            current_time += points[point_idx].stop_duration
            
            # Добавляем время до следующей точки
            if i < len(route) - 1:
                next_idx = route[i + 1]
                current_time += time_matrix[point_idx][next_idx]
        
        return arrival_times
    
    def save_model(self, path: str):
        """Сохранение модели"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Загрузка модели"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

if __name__ == "__main__":
    # Тестирование оптимизатора
    from app.models.data_processor import DataProcessor
    
    processor = DataProcessor()
    df = processor.load_dataset("data/dataset.csv")
    points = processor.prepare_route_points(df)
    
    # Создаем матрицу времени
    time_matrix = processor.create_time_matrix(points, "car", 3)
    
    # Тестируем оптимизатор
    optimizer = RouteOptimizer()
    route = optimizer.optimize_route(points, time_matrix, method='genetic')
    
    print(f"Оптимизированный маршрут: {route}")
    print(f"Первые 5 точек: {route[:5]}")
