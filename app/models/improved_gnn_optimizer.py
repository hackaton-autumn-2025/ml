import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Dict
import random

class ImprovedRouteOptimizationGNN(nn.Module):
    """Улучшенная GNN модель для оптимизации маршрутов"""
    
    def __init__(self, 
                 node_features: int = 8,
                 hidden_dim: int = 128,  
                 num_layers: int = 4,    
                 num_heads: int = 8,     
                 dropout: float = 0.2,   
                 use_residual: bool = True,
                 use_batch_norm: bool = True):
        super(ImprovedRouteOptimizationGNN, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        
        self.input_norm = nn.BatchNorm1d(node_features) if use_batch_norm else nn.Identity()
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.convs.append(TransformerConv(
            hidden_dim, hidden_dim, 
            heads=num_heads, dropout=dropout, 
            edge_dim=4, concat=False
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity())
        
        # Промежуточные слои
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(
                hidden_dim, hidden_dim,
                heads=num_heads, dropout=dropout,
                edge_dim=4, concat=False
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity())
        
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.order_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.time_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
        self.priority_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Улучшенный прямой проход через GNN"""
        
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        
        residual = x
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if self.use_residual and i > 0 and x.size(-1) == residual.size(-1):
                x = x + residual
            
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
            
            residual = x
        
        attention_weights = self.attention_pool(x)
        attention_weights = F.softmax(attention_weights, dim=0)
        global_features = (x * attention_weights).sum(dim=0, keepdim=True)
        
        global_features = global_features.expand(x.size(0), -1)
        x_with_global = torch.cat([x, global_features], dim=-1)
        
        order_scores = self.order_predictor(x_with_global)
        time_scores = self.time_predictor(x_with_global)
        priority_scores = self.priority_predictor(x_with_global)
        
        return order_scores, time_scores, priority_scores

class ImprovedRouteOptimizer:
    """Улучшенный класс для оптимизации маршрутов"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        self.model = ImprovedRouteOptimizationGNN(
            node_features=14,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            dropout=0.2,
            use_residual=True,
            use_batch_norm=True
        )
        self.model.to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
    def create_enhanced_graph_data(self, points, time_matrix, traffic_level, transport_mode):
        """Создание улучшенного графа с дополнительными признаками"""
        n = len(points)
        
        node_features = []
        for i, point in enumerate(points):
            
            lat_norm = (point.latitude - 47.0) / 0.5
            lon_norm = (point.longitude - 39.0) / 0.5
            
            work_start_norm = point.get_work_start_time().hour / 24.0
            work_end_norm = point.get_work_end_time().hour / 24.0
            work_duration_norm = (work_end_norm - work_start_norm) % 1.0
            
            lunch_start_norm = point.get_lunch_start_time().hour / 24.0
            lunch_end_norm = point.get_lunch_end_time().hour / 24.0
            lunch_duration_norm = (lunch_end_norm - lunch_start_norm) % 1.0
            
            client_level_norm = 1.0 if point.client_level == 'VIP' else 0.0
            stop_duration_norm = point.stop_duration / 60.0
            
            traffic_norm = traffic_level / 10.0
            transport_norm = 1.0 if transport_mode == 'car' else 0.0
            
            urgency_score = self._calculate_urgency_score(point, traffic_level)
            accessibility_score = self._calculate_accessibility_score(point, time_matrix, i)
            
            features = [
                lat_norm, lon_norm, 
                work_start_norm, work_end_norm, work_duration_norm,
                lunch_start_norm, lunch_end_norm, lunch_duration_norm,
                client_level_norm, stop_duration_norm,
                traffic_norm, transport_norm,
                urgency_score, accessibility_score
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        edge_list = []
        edge_weights = []
        edge_attrs = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_list.append([i, j])
                    
                    base_weight = time_matrix[i][j]
                    
                    priority_factor = 1.0
                    if points[j].client_level == 'VIP':
                        priority_factor = 0.7
                    
                    traffic_factor = 1.0 + (traffic_level / 10.0) * 0.5
                    distance_factor = self._calculate_distance_factor(points[i], points[j])
                    
                    final_weight = base_weight * priority_factor * traffic_factor * distance_factor
                    edge_weights.append(1.0 / (final_weight + 1e-6))
                    
                    edge_attr = [
                        base_weight / 60.0, 
                        priority_factor,
                        traffic_factor,
                        distance_factor
                    ]
                    edge_attrs.append(edge_attr)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _calculate_urgency_score(self, point, traffic_level):
        """Расчет срочности посещения точки"""
        base_urgency = 1.0 if point.client_level == 'VIP' else 0.5
        
        work_end_hour = point.get_work_end_time().hour
        time_factor = 1.0 - (work_end_hour - 9) / 9.0  # от 0 до 1
        
        traffic_factor = 1.0 + (traffic_level / 10.0) * 0.3
        
        return base_urgency * time_factor * traffic_factor
    
    def _calculate_accessibility_score(self, point, time_matrix, point_idx):
        """Расчет доступности точки (среднее время до других точек)"""
        times_to_others = [time_matrix[point_idx][j] for j in range(len(time_matrix)) if j != point_idx]
        avg_time = np.mean(times_to_others) if times_to_others else 0
        return 1.0 / (avg_time / 60.0 + 1e-6) 
    
    def _calculate_distance_factor(self, point1, point2):
        """Расчет фактора расстояния между точками"""
        lat_diff = point1.latitude - point2.latitude
        lon_diff = point1.longitude - point2.longitude
        distance = np.sqrt(lat_diff**2 + lon_diff**2)
        return 1.0 / (distance + 1e-6)
    
    def optimize_route_hybrid(self, points, time_matrix, start_point_idx=0, 
                            gnn_weight=0.7, genetic_weight=0.3):
        """Гибридный метод оптимизации с использованием GNN и генетического алгоритма"""
        n = len(points)
        
        gnn_route = self._optimize_with_gnn(points, time_matrix, start_point_idx)
        
        genetic_route = self._optimize_with_genetic(points, time_matrix, start_point_idx)
        
        # Комбинируем результаты
        if random.random() < gnn_weight:
            return gnn_route
        else:
            return genetic_route
    
    def _optimize_with_gnn(self, points, time_matrix, start_point_idx=0):
        """Оптимизация с использованием GNN"""
        
        graph_data = self.create_enhanced_graph_data(points, time_matrix, 3, 'car')
        
        with torch.no_grad():
            order_scores, time_scores, priority_scores = self.model(
                graph_data.x, graph_data.edge_index, graph_data.edge_attr
            )
        
        route = [start_point_idx]
        visited = {start_point_idx}
        
        remaining_points = [(i, order_scores[i].item(), priority_scores[i].item()) 
                          for i in range(len(points)) if i != start_point_idx]
        
        remaining_points.sort(key=lambda x: x[1] + x[2] * 2.0, reverse=True)
        
        for point_idx, _, _ in remaining_points:
            route.append(point_idx)
        
        return route
    
    def _optimize_with_genetic(self, points, time_matrix, start_point_idx=0):
        """Генетический алгоритм"""
        n = len(points)
        
        def create_individual():
            route = list(range(n))
            route.remove(start_point_idx)
            random.shuffle(route)
            return [start_point_idx] + route
        
        def fitness(individual):
            total_time = 0
            for i in range(len(individual) - 1):
                total_time += time_matrix[individual[i]][individual[i + 1]]
            return total_time
        
        population = [create_individual() for _ in range(20)]
        
        for _ in range(50): 
            fitness_scores = [fitness(ind) for ind in population]
            new_population = []
            
            best_idx = np.argmin(fitness_scores)
            new_population.append(population[best_idx].copy())
            
            for _ in range(len(population) - 1):
                parent = random.choice(population)
                child = parent.copy()
                
                if len(child) > 2:
                    i, j = random.sample(range(1, len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                
                new_population.append(child)
            
            population = new_population
        
        fitness_scores = [fitness(ind) for ind in population]
        best_idx = np.argmin(fitness_scores)
        return population[best_idx]
    
    def optimize_route(self, points, time_matrix, start_point_idx=0, method='hybrid'):
        """Основной метод оптимизации маршрута (совместимость с API)"""
        if method == 'hybrid':
            return self.optimize_route_hybrid(points, time_matrix, start_point_idx)
        elif method == 'genetic':
            return self._optimize_with_genetic(points, time_matrix, start_point_idx)
        elif method == 'gnn':
            return self._optimize_with_gnn(points, time_matrix, start_point_idx)
        else:
            
            return self.optimize_route_hybrid(points, time_matrix, start_point_idx)
    
    def calculate_arrival_times(self, route, points, time_matrix, start_time):
        """Расчет времени прибытия в каждую точку"""
        arrival_times = []
        current_time = start_time.hour * 60 + start_time.minute  
        
        for i, point_idx in enumerate(route):
            hour = int(current_time // 60)
            minute = int(current_time % 60)
            arrival_times.append(f"{hour:02d}:{minute:02d}")
            
            current_time += points[point_idx].stop_duration
            
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
