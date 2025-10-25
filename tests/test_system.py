import requests
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import folium
from typing import List, Tuple

class RouteOptimizationTester:
    """Класс для тестирования системы оптимизации маршрутов"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        
    def test_health(self):
        """Тест проверки состояния сервиса"""
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code == 200:
                print("✅ Сервис работает корректно")
                return True
            else:
                print(f"❌ Ошибка сервиса: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Не удается подключиться к сервису: {e}")
            return False
    
    def test_dataset_info(self):
        """Тест получения информации о датасете"""
        try:
            response = requests.get(f"{self.api_url}/dataset-info")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Информация о датасете получена:")
                print(f"   Всего точек: {data['total_points']}")
                print(f"   VIP клиентов: {data['vip_clients']}")
                print(f"   Стандартных клиентов: {data['standard_clients']}")
                return data
            else:
                print(f"❌ Ошибка получения информации о датасете: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Ошибка при запросе информации о датасете: {e}")
            return None
    
    def test_route_optimization_from_dataset(self):
        """Тест оптимизации маршрута на основе датасета"""
        try:
            params = {
                "start_time": "09:00",
                "traffic_level": 3,
                "transport_mode": "car",
                "optimization_method": "genetic",
                "max_points": 10
            }
            
            print("🔄 Тестирование оптимизации маршрута из датасета...")
            start_time = time.time()
            
            response = requests.post(f"{self.api_url}/optimize-route-from-dataset", params=params)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Маршрут оптимизирован за {processing_time:.2f} секунд")
                print(f"   Оптимизированный порядок: {data['optimized_order'][:5]}...")
                print(f"   Общее расстояние: {data['total_distance']} км")
                print(f"   Общее время: {data['total_time']} минут")
                print(f"   VIP клиентов в маршруте: {data['statistics']['vip_clients']}")
                
                return data
            else:
                print(f"❌ Ошибка оптимизации маршрута: {response.status_code}")
                print(f"   Ответ: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка при тестировании оптимизации: {e}")
            return None
    
    def test_custom_route_optimization(self):
        """Тест оптимизации пользовательского маршрута"""
        try:

            test_points = [
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
                },
                {
                    "id": 2,
                    "address": "пр. Буденновский, д. 15",
                    "latitude": 47.228945,
                    "longitude": 39.718762,
                    "work_start": "09:00",
                    "work_end": "18:00",
                    "lunch_start": "13:00",
                    "lunch_end": "14:00",
                    "client_level": "Standart",
                    "stop_duration": 20
                },
                {
                    "id": 3,
                    "address": "ул. Красноармейская, д. 67",
                    "latitude": 47.235671,
                    "longitude": 39.689543,
                    "work_start": "10:00",
                    "work_end": "19:00",
                    "lunch_start": "13:00",
                    "lunch_end": "14:00",
                    "client_level": "VIP",
                    "stop_duration": 45
                }
            ]
            
            request_data = {
                "points": test_points,
                "start_time": "09:00",
                "traffic_level": 5,
                "transport_mode": "car",
                "start_point": [47.220000, 39.700000],
                "optimization_method": "genetic"
            }
            
            print("🔄 Тестирование оптимизации пользовательского маршрута...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_url}/optimize-route",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Пользовательский маршрут оптимизирован за {processing_time:.2f} секунд")
                print(f"   Оптимизированный порядок: {data['optimized_order']}")
                print(f"   Время прибытия: {data['arrival_times']}")
                print(f"   Общее расстояние: {data['total_distance']} км")
                print(f"   Общее время: {data['total_time']} минут")
                
                return data
            else:
                print(f"❌ Ошибка оптимизации пользовательского маршрута: {response.status_code}")
                print(f"   Ответ: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка при тестировании пользовательского маршрута: {e}")
            return None
    
    def compare_optimization_methods(self):
        """Сравнение методов оптимизации"""
        methods = ["genetic", "greedy"]
        results = {}
        
        print("🔄 Сравнение методов оптимизации...")
        
        for method in methods:
            try:
                params = {
                    "start_time": "09:00",
                    "traffic_level": 3,
                    "transport_mode": "car",
                    "optimization_method": method,
                    "max_points": 8
                }
                
                start_time = time.time()
                response = requests.post(f"{self.api_url}/optimize-route-from-dataset", params=params)
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    results[method] = {
                        "total_time": data['total_time'],
                        "total_distance": data['total_distance'],
                        "processing_time": end_time - start_time,
                        "route": data['optimized_order']
                    }
                    print(f"✅ Метод {method}: время={data['total_time']:.1f}мин, расстояние={data['total_distance']:.2f}км")
                else:
                    print(f"❌ Ошибка для метода {method}")
                    
            except Exception as e:
                print(f"❌ Ошибка при тестировании метода {method}: {e}")
        
        return results
    
    def test_different_traffic_levels(self):
        """Тест при разных уровнях загруженности дорог"""
        traffic_levels = [0, 3, 5, 8, 10]
        results = {}
        
        print("🔄 Тестирование при разных уровнях загруженности...")
        
        for traffic in traffic_levels:
            try:
                params = {
                    "start_time": "09:00",
                    "traffic_level": traffic,
                    "transport_mode": "car",
                    "optimization_method": "genetic",
                    "max_points": 6
                }
                
                response = requests.post(f"{self.api_url}/optimize-route-from-dataset", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results[traffic] = {
                        "total_time": data['total_time'],
                        "total_distance": data['total_distance']
                    }
                    print(f"✅ Трафик {traffic}: время={data['total_time']:.1f}мин, расстояние={data['total_distance']:.2f}км")
                else:
                    print(f"❌ Ошибка для трафика {traffic}")
                    
            except Exception as e:
                print(f"❌ Ошибка при тестировании трафика {traffic}: {e}")
        
        return results
    
    def create_route_map(self, route_data, filename="route_map.html"):
        """Создание карты маршрута"""
        if not route_data or 'route_coordinates' not in route_data:
            print("❌ Нет данных для создания карты")
            return
        
        coordinates = route_data['route_coordinates']
        arrival_times = route_data.get('arrival_times', [])
        
        m = folium.Map(
            location=[coordinates[0][0], coordinates[0][1]],
            zoom_start=12
        )
        
        for i, (lat, lon) in enumerate(coordinates):
            color = 'red' if i == 0 else 'blue'
            arrival_time = arrival_times[i] if i < len(arrival_times) else ""
            
            folium.Marker(
                [lat, lon],
                popup=f"Точка {i+1}<br>Время прибытия: {arrival_time}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
        
        folium.PolyLine(
            coordinates,
            color="green",
            weight=3,
            opacity=0.8
        ).add_to(m)
        
        m.save(filename)
        print(f"✅ Карта маршрута сохранена в {filename}")
    
    def run_all_tests(self):
        """Запуск всех тестов"""
        print("🚀 Начинаем тестирование системы оптимизации маршрутов")
        print("=" * 60)
        
        print("\n1. Проверка состояния сервиса")
        if not self.test_health():
            print("❌ Сервис недоступен. Запустите FastAPI сервер.")
            return
        
        print("\n2. Информация о датасете")
        dataset_info = self.test_dataset_info()
        
        print("\n3. Оптимизация маршрута из датасета")
        route_data = self.test_route_optimization_from_dataset()
        
        print("\n4. Оптимизация пользовательского маршрута")
        custom_route = self.test_custom_route_optimization()
        
        print("\n5. Сравнение методов оптимизации")
        method_comparison = self.compare_optimization_methods()
        
        print("\n6. Тестирование при разных уровнях загруженности")
        traffic_results = self.test_different_traffic_levels()
        
        if route_data:
            print("\n7. Создание карты маршрута")
            self.create_route_map(route_data)
        
        print("\n" + "=" * 60)
        print("📊 ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 60)
        
        if route_data:
            print(f"✅ Система работает корректно")
            print(f"   Оптимизированный маршрут: {len(route_data['optimized_order'])} точек")
            print(f"   Общее время: {route_data['total_time']} минут")
            print(f"   Общее расстояние: {route_data['total_distance']} км")
        
        if method_comparison:
            print(f"\n📈 Сравнение методов:")
            for method, result in method_comparison.items():
                print(f"   {method}: {result['total_time']:.1f}мин, {result['processing_time']:.2f}сек")
        
        if traffic_results:
            print(f"\n🚦 Влияние загруженности дорог:")
            for traffic, result in traffic_results.items():
                print(f"   Трафик {traffic}: {result['total_time']:.1f}мин")
        
        print("\n✅ Тестирование завершено!")

def main():
    """Основная функция для запуска тестов"""
    tester = RouteOptimizationTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
