#!/usr/bin/env python3
"""
Тест API с улучшенной моделью
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Тест health check"""
    print("🔍 Тестируем health check...")
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_dataset_info():
    """Тест информации о датасете"""
    print("📊 Тестируем dataset info...")
    response = requests.get(f"{BASE_URL}/api/v1/dataset-info")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total points: {data['total_points']}")
    print(f"VIP clients: {data['vip_clients']}")
    print(f"Standard clients: {data['standard_clients']}")
    print()

def test_route_optimization():
    """Тест оптимизации маршрута"""
    print("🚀 Тестируем оптимизацию маршрута...")
    
    test_request = {
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
                "stop_duration": 30
            },
            {
                "id": 3,
                "address": "ул. Красноармейская, д. 67",
                "latitude": 47.235671,
                "longitude": 39.689543,
                "work_start": "09:00",
                "work_end": "18:00",
                "lunch_start": "13:00",
                "lunch_end": "14:00",
                "client_level": "Standart",
                "stop_duration": 30
            }
        ],
        "start_time": "09:00",
        "traffic_level": 3,
        "transport_mode": "car",
        "start_point": [47.220000, 39.700000]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/optimize-route",
        json=test_request,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Оптимизация успешна!")
        print(f"Оптимизированный порядок: {data['optimized_order']}")
        print(f"Время прибытия: {data['arrival_times']}")
        print(f"Общее расстояние: {data['total_distance']:.2f} км")
        print(f"Общее время: {data['total_time']:.2f} мин")
        print(f"Успех: {data['success']}")
        print(f"Сообщение: {data['message']}")
    else:
        print(f"❌ Ошибка: {response.text}")
    print()

def test_route_from_dataset():
    """Тест оптимизации из датасета"""
    print("📈 Тестируем оптимизацию из датасета...")
    
    params = {
        "start_time": "09:00",
        "traffic_level": 3,
        "transport_mode": "car",
        "optimization_method": "hybrid", 
        "max_points": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/optimize-route-from-dataset",
        params=params
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Оптимизация из датасета успешна!")
        print(f"Оптимизированный порядок: {data['optimized_order']}")
        print(f"Время прибытия: {data['arrival_times']}")
        print(f"Общее расстояние: {data['total_distance']:.2f} км")
        print(f"Общее время: {data['total_time']:.2f} мин")
        print(f"Метод оптимизации: {data.get('optimization_method', 'hybrid')}")
    else:
        print(f"❌ Ошибка: {response.text}")
    print()

if __name__ == "__main__":
    print("🧪 Тестирование API с улучшенной моделью\n")
    
    try:
        test_health()
        test_dataset_info()
        test_route_optimization()
        test_route_from_dataset()
        
        print("🎉 Все тесты завершены!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Ошибка подключения к API. Убедитесь, что сервер запущен на localhost:8000")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
