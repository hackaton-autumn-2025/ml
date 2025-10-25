import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.data_processor import DataProcessor
from app.models.gnn_optimizer import RouteOptimizationGNN, RouteOptimizer

class RouteDataset(Dataset):
    """Датасет для обучения GNN модели"""
    
    def __init__(self, points_list, time_matrices, optimal_routes, features_list):
        self.points_list = points_list
        self.time_matrices = time_matrices
        self.optimal_routes = optimal_routes
        self.features_list = features_list
        
    def __len__(self):
        return len(self.points_list)
    
    def __getitem__(self, idx):
        return {
            'points': self.points_list[idx],
            'time_matrix': self.time_matrices[idx],
            'optimal_route': self.optimal_routes[idx],
            'features': self.features_list[idx]
        }

def custom_collate_fn(batch):
    """Кастомная функция для обработки батчей разного размера"""
    # Возвращаем батч как есть, без попытки конвертировать в тензоры
    return batch

class ModelTrainer:
    """Класс для обучения GNN модели"""
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Создаем директорию для моделей
        os.makedirs(model_path, exist_ok=True)
        
        # Инициализируем модель
        self.model = RouteOptimizationGNN()
        self.model.to(self.device)
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # История обучения
        self.train_losses = []
        self.val_losses = []
        
    def generate_training_data(self, num_samples: int = 1000):
        """Генерация обучающих данных"""
        data_processor = DataProcessor()
        route_optimizer = RouteOptimizer()
        
        # Загружаем базовый датасет
        df = data_processor.load_dataset("data/dataset.csv")
        base_points = data_processor.prepare_route_points(df)
        
        points_list = []
        time_matrices = []
        optimal_routes = []
        features_list = []
        
        print(f"Генерация {num_samples} обучающих примеров...")
        
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Обработано {i}/{num_samples} примеров")
            
            # Случайно выбираем подмножество точек (5-15 точек)
            num_points = np.random.randint(5, min(16, len(base_points)))
            selected_indices = np.random.choice(len(base_points), num_points, replace=False)
            selected_points = [base_points[idx] for idx in selected_indices]
            
            # Случайные параметры
            transport_mode = np.random.choice(['car', 'walk'])
            traffic_level = np.random.randint(0, 11)
            
            # Создаем матрицу времени
            time_matrix = data_processor.create_time_matrix(
                selected_points, transport_mode, traffic_level
            )
            
            # Находим оптимальный маршрут
            optimal_route = route_optimizer.optimize_route(
                selected_points, time_matrix, method='genetic'
            )
            
            # Создаем признаки для GNN
            features = self._create_features(selected_points, traffic_level, transport_mode)
            
            points_list.append(selected_points)
            time_matrices.append(time_matrix)
            optimal_routes.append(optimal_route)
            features_list.append(features)
        
        print(f"Сгенерировано {len(points_list)} обучающих примеров")
        return points_list, time_matrices, optimal_routes, features_list
    
    def _create_features(self, points, traffic_level, transport_mode):
        """Создание признаков для GNN"""
        features = []
        
        for point in points:
            # Нормализуем координаты
            lat_norm = (point.latitude - 47.0) / 0.5
            lon_norm = (point.longitude - 39.0) / 0.5
            
            # Время работы
            work_start_norm = point.get_work_start_time().hour / 24.0
            work_end_norm = point.get_work_end_time().hour / 24.0
            
            # Уровень клиента
            client_level_norm = 1.0 if point.client_level == 'VIP' else 0.0
            
            # Время остановки
            stop_duration_norm = point.stop_duration / 60.0
            
            # Трафик и транспорт
            traffic_norm = traffic_level / 10.0
            transport_norm = 1.0 if transport_mode == 'car' else 0.0
            
            feature_vector = [
                lat_norm, lon_norm, work_start_norm, work_end_norm,
                client_level_norm, stop_duration_norm, traffic_norm, transport_norm
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def create_graph_data(self, features, time_matrix):
        """Создание графа для GNN"""
        n = len(features)
        
        # Создаем узлы
        x = torch.tensor(features, dtype=torch.float)
        
        # Создаем ребра (полный граф)
        edge_list = []
        edge_weights = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_list.append([i, j])
                    # Вес ребра = инвертированное время перемещения
                    weight = 1.0 / (time_matrix[i][j] + 1e-6)
                    edge_weights.append(weight)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
    
    def train_epoch(self, dataloader):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            batch_loss = 0
            for item in batch:
                points = item['points']
                time_matrix = item['time_matrix']
                optimal_route = item['optimal_route']
                features = item['features']
                
                # Создаем граф
                graph_data = self.create_graph_data(features, time_matrix)
                
                # Перемещаем данные на устройство
                x = graph_data['x'].to(self.device)
                edge_index = graph_data['edge_index'].to(self.device)
                
                # Прямой проход
                order_scores, time_scores = self.model(x, edge_index)
                
                # Создаем целевые значения
                target_order = torch.zeros_like(order_scores)
                target_time = torch.zeros_like(time_scores)
                
                # Устанавливаем целевые значения для оптимального маршрута
                for j, point_idx in enumerate(optimal_route):
                    target_order[point_idx] = j / len(optimal_route)  # нормализованный порядок
                    target_time[point_idx] = j * 30 / 60  # примерное время в часах
                
                # Вычисляем потери
                order_loss = self.criterion(order_scores, target_order)
                time_loss = self.criterion(time_scores, target_time)
                
                loss = order_loss + time_loss
                batch_loss += loss
            
            batch_loss /= len(batch)
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch_loss = 0
                for item in batch:
                    points = item['points']
                    time_matrix = item['time_matrix']
                    optimal_route = item['optimal_route']
                    features = item['features']
                    
                    graph_data = self.create_graph_data(features, time_matrix)
                    
                    x = graph_data['x'].to(self.device)
                    edge_index = graph_data['edge_index'].to(self.device)
                    
                    order_scores, time_scores = self.model(x, edge_index)
                    
                    target_order = torch.zeros_like(order_scores)
                    target_time = torch.zeros_like(time_scores)
                    
                    for j, point_idx in enumerate(optimal_route):
                        target_order[point_idx] = j / len(optimal_route)
                        target_time[point_idx] = j * 30 / 60
                    
                    order_loss = self.criterion(order_scores, target_order)
                    time_loss = self.criterion(time_scores, target_time)
                    
                    loss = order_loss + time_loss
                    batch_loss += loss
                
                batch_loss /= len(batch)
                total_loss += batch_loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, epochs: int = 50, batch_size: int = 32):
        """Основной цикл обучения"""
        print("Генерация обучающих данных...")
        points_list, time_matrices, optimal_routes, features_list = self.generate_training_data(1000)
        
        # Разделяем на train/val
        train_points, val_points, train_matrices, val_matrices, train_routes, val_routes, train_features, val_features = train_test_split(
            points_list, time_matrices, optimal_routes, features_list,
            test_size=0.2, random_state=42
        )
        
        # Создаем датасеты
        train_dataset = RouteDataset(train_points, train_matrices, train_routes, train_features)
        val_dataset = RouteDataset(val_points, val_matrices, val_routes, val_features)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        print(f"Начинаем обучение на {epochs} эпох...")
        print(f"Устройство: {self.device}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if epoch % 5 == 0:
                print(f"Эпоха {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Сохраняем модель
        self.save_model()
        
        # Строим графики
        self.plot_training_history()
        
        print("Обучение завершено!")
    
    def save_model(self):
        """Сохранение обученной модели"""
        model_path = os.path.join(self.model_path, "route_optimization_gnn.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Сохраняем метаданные
        metadata = {
            "model_type": "RouteOptimizationGNN",
            "node_features": 8,
            "hidden_dim": 64,
            "num_layers": 3,
            "dropout": 0.1,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        
        metadata_path = os.path.join(self.model_path, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Модель сохранена в {model_path}")
    
    def plot_training_history(self):
        """Построение графиков истории обучения"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('История обучения')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses[-20:], label='Train Loss (последние 20)')
        plt.plot(self.val_losses[-20:], label='Validation Loss (последние 20)')
        plt.title('Последние 20 эпох')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'training_history.png'))
        plt.show()

def main():
    """Основная функция для запуска обучения"""
    trainer = ModelTrainer()
    trainer.train(epochs=30, batch_size=16)

if __name__ == "__main__":
    main()
