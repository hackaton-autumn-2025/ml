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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.data_processor import DataProcessor
from app.models.improved_gnn_optimizer import ImprovedRouteOptimizationGNN, ImprovedRouteOptimizer

class RouteDataset(Dataset):
    """Датасет для обучения улучшенной GNN модели"""
    
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
    return batch

class ImprovedModelTrainer:
    """Улучшенный класс для обучения GNN модели"""
    
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Создаем директорию для моделей
        os.makedirs(model_path, exist_ok=True)
        
        # Инициализируем улучшенную модель
        self.model = ImprovedRouteOptimizationGNN(
            node_features=14,  # Увеличили количество признаков
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            dropout=0.2,
            use_residual=True,
            use_batch_norm=True
        )
        self.model.to(self.device)
        
        # Улучшенный оптимизатор с weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Функция потерь с дополнительными компонентами
        self.criterion = nn.MSELoss()
        self.priority_criterion = nn.BCELoss()
        
        # Метрики для отслеживания
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 10
        
    def create_enhanced_graph_data(self, features, time_matrix):
        """Создание улучшенного графа для обучения"""
        n = len(features)
        
        # Создаем тензор признаков узлов (14 признаков)
        x = torch.tensor(features, dtype=torch.float)
        
        # Создаем ребра с дополнительными атрибутами
        edge_list = []
        edge_weights = []
        edge_attrs = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    edge_list.append([i, j])
                    
                    # Базовый вес
                    base_weight = time_matrix[i][j]
                    edge_weights.append(1.0 / (base_weight + 1e-6))
                    
                    # Дополнительные атрибуты ребра
                    edge_attr = [
                        base_weight / 60.0,  # время в часах
                        1.0,  # приоритет
                        1.0,  # трафик
                        1.0   # расстояние
                    ]
                    edge_attrs.append(edge_attr)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
    
    def train_epoch(self, dataloader):
        """Улучшенное обучение на одной эпохе"""
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
                
                # Создаем улучшенный граф
                graph_data = self.create_enhanced_graph_data(features, time_matrix)
                
                # Перемещаем данные на устройство
                x = graph_data['x'].to(self.device)
                edge_index = graph_data['edge_index'].to(self.device)
                edge_attr = graph_data['edge_attr'].to(self.device)
                
                # Прямой проход
                order_scores, time_scores, priority_scores = self.model(x, edge_index, edge_attr)
                
                # Создаем целевые значения
                target_order = torch.zeros_like(order_scores)
                target_time = torch.zeros_like(time_scores)
                target_priority = torch.zeros_like(priority_scores)
                
                # Заполняем целевые значения на основе оптимального маршрута
                for j, point_idx in enumerate(optimal_route):
                    target_order[point_idx] = j / len(optimal_route)
                    target_time[point_idx] = j / len(optimal_route)
                    
                    # Приоритет на основе уровня клиента
                    if points[point_idx].client_level == 'VIP':
                        target_priority[point_idx] = 1.0
                    else:
                        target_priority[point_idx] = 0.0
                
                # Вычисляем потери с дополнительными компонентами
                order_loss = self.criterion(order_scores, target_order)
                time_loss = self.criterion(time_scores, target_time)
                priority_loss = self.priority_criterion(priority_scores, target_priority)
                
                # Комбинированная потеря
                loss = order_loss + time_loss + 0.5 * priority_loss
                batch_loss += loss
            
            batch_loss /= len(batch)
            batch_loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += batch_loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Улучшенная валидация модели"""
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
                    
                    graph_data = self.create_enhanced_graph_data(features, time_matrix)
                    
                    x = graph_data['x'].to(self.device)
                    edge_index = graph_data['edge_index'].to(self.device)
                    edge_attr = graph_data['edge_attr'].to(self.device)
                    
                    order_scores, time_scores, priority_scores = self.model(x, edge_index, edge_attr)
                    
                    target_order = torch.zeros_like(order_scores)
                    target_time = torch.zeros_like(time_scores)
                    target_priority = torch.zeros_like(priority_scores)
                    
                    for j, point_idx in enumerate(optimal_route):
                        target_order[point_idx] = j / len(optimal_route)
                        target_time[point_idx] = j / len(optimal_route)
                        
                        if points[point_idx].client_level == 'VIP':
                            target_priority[point_idx] = 1.0
                        else:
                            target_priority[point_idx] = 0.0
                    
                    order_loss = self.criterion(order_scores, target_order)
                    time_loss = self.criterion(time_scores, target_time)
                    priority_loss = self.priority_criterion(priority_scores, target_priority)
                    
                    loss = order_loss + time_loss + 0.5 * priority_loss
                    batch_loss += loss
                
                batch_loss /= len(batch)
                total_loss += batch_loss.item()
        
        return total_loss / len(dataloader)
    
    def generate_training_data(self, num_samples: int = 1000):
        """Генерация обучающих данных"""
        from app.models.data_processor import DataProcessor
        from app.models.gnn_optimizer import RouteOptimizer
        
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
            
            # Проверяем размерность признаков
            if len(features) > 0 and len(features[0]) != 14:
                print(f"Ошибка: ожидается 14 признаков, получено {len(features[0])}")
                print(f"Признаки: {features[0]}")
            
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
            
            # Дополнительные признаки для улучшенной модели
            work_duration_norm = (work_end_norm - work_start_norm) % 1.0
            lunch_start_norm = point.get_lunch_start_time().hour / 24.0
            lunch_end_norm = point.get_lunch_end_time().hour / 24.0
            lunch_duration_norm = (lunch_end_norm - lunch_start_norm) % 1.0
            
            # Срочность и доступность
            urgency_score = self._calculate_urgency_score(point, traffic_level)
            accessibility_score = self._calculate_accessibility_score(point, points)
            
            features.append([
                lat_norm, lon_norm, 
                work_start_norm, work_end_norm, work_duration_norm,
                lunch_start_norm, lunch_end_norm, lunch_duration_norm,
                client_level_norm, stop_duration_norm,
                traffic_norm, transport_norm,
                urgency_score, accessibility_score
            ])
        
        return features
    
    def _calculate_urgency_score(self, point, traffic_level):
        """Расчет срочности посещения точки"""
        base_urgency = 1.0 if point.client_level == 'VIP' else 0.5
        work_end_hour = point.get_work_end_time().hour
        time_factor = 1.0 - (work_end_hour - 9) / 9.0
        traffic_factor = 1.0 + (traffic_level / 10.0) * 0.3
        return base_urgency * time_factor * traffic_factor
    
    def _calculate_accessibility_score(self, point, all_points):
        """Расчет доступности точки"""
        # Упрощенный расчет - можно улучшить
        return 1.0 if point.client_level == 'VIP' else 0.5
    
    def train(self, epochs: int = 50, batch_size: int = 32):
        """Улучшенный цикл обучения с early stopping"""
        print("Генерация обучающих данных...")
        
        # Генерируем больше данных для лучшего обучения
        points_list, time_matrices, optimal_routes, features_list = self.generate_training_data(
            num_samples=2000  # Увеличили количество образцов
        )
        
        # Разделение на train/validation
        train_points, val_points, train_matrices, val_matrices, train_routes, val_routes, train_features, val_features = train_test_split(
            points_list, time_matrices, optimal_routes, features_list,
            test_size=0.2, random_state=42
        )
        
        # Создаем датасеты
        train_dataset = RouteDataset(train_points, train_matrices, train_routes, train_features)
        val_dataset = RouteDataset(val_points, val_matrices, val_routes, val_features)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        print(f"Начинаем улучшенное обучение на {epochs} эпох...")
        print(f"Устройство: {self.device}")
        print(f"Размер модели: {sum(p.numel() for p in self.model.parameters()):,} параметров")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Обновляем learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Сохраняем лучшую модель
                torch.save(self.model.state_dict(), os.path.join(self.model_path, "best_model.pth"))
            else:
                self.patience_counter += 1
            
            if epoch % 5 == 0:
                print(f"Эпоха {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping на эпохе {epoch+1}")
                break
        
        # Загружаем лучшую модель
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, "best_model.pth")))
        
        # Сохраняем финальную модель
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "improved_route_optimization_gnn.pth"))
        
        # Сохраняем метаданные
        metadata = {
            "model_type": "ImprovedRouteOptimizationGNN",
            "node_features": 14,
            "hidden_dim": 128,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.2,
            "use_residual": True,
            "use_batch_norm": True,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "best_val_loss": self.best_val_loss,
            "total_parameters": sum(p.numel() for p in self.model.parameters())
        }
        
        with open(os.path.join(self.model_path, "improved_model_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Создаем улучшенные графики
        self.plot_improved_training_history()
        
        print(f"Улучшенная модель сохранена в {self.model_path}")
        print("Обучение завершено!")
    
    def plot_improved_training_history(self):
        """Создание улучшенных графиков обучения"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # График потерь
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # График learning rate
        ax2.plot(self.learning_rates, label='Learning Rate', color='green')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')
        
        # График разности потерь
        loss_diff = [abs(train - val) for train, val in zip(self.train_losses, self.val_losses)]
        ax3.plot(loss_diff, label='|Train - Val| Loss', color='purple')
        ax3.set_title('Overfitting Indicator')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.legend()
        ax3.grid(True)
        
        # График скользящего среднего
        window = min(10, len(self.train_losses) // 4)
        if window > 1:
            train_smooth = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            val_smooth = np.convolve(self.val_losses, np.ones(window)/window, mode='valid')
            
            ax4.plot(range(window-1, len(self.train_losses)), train_smooth, label=f'Train Loss (MA{window})', color='blue')
            ax4.plot(range(window-1, len(self.val_losses)), val_smooth, label=f'Val Loss (MA{window})', color='red')
            ax4.set_title('Smoothed Loss Curves')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'improved_training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Графики обучения сохранены в improved_training_history.png")

def main():
    """Основная функция для запуска улучшенного обучения"""
    trainer = ImprovedModelTrainer()
    trainer.train(epochs=50, batch_size=16)

if __name__ == "__main__":
    main()
