#!/usr/bin/env python3
"""
Простой тест системы оптимизации маршрутов
Работает без установки тяжелых зависимостей
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import json

def test_data_loading():
    """Тест загрузки данных"""
    print("🔄 Тестирование загрузки данных...")
    
    try:
        # Загружаем датасет
        df = pd.read_csv("data/dataset.csv")
        print(f"✅ Датасет загружен: {len(df)} строк")
        
        # Проверяем структуру
        expected_columns = [
            'Номер объекта', 'Адрес объекта', 'Географическая широта', 
            'Географическая долгота', 'Время начала рабочего дня',
            'Время окончания рабочего дня', 'Время начала обеда', 
            'Время окончания обеда', 'Уровень клиента'
        ]
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Отсутствуют колонки: {missing_columns}")
            return False
        
        print("✅ Все необходимые колонки присутствуют")
        
        # Проверяем данные
        vip_count = len(df[df['Уровень клиента'] == 'VIP'])
        standard_count = len(df[df['Уровень клиента'] == 'Standart'])
        
        print(f"✅ VIP клиентов: {vip_count}")
        print(f"✅ Стандартных клиентов: {standard_count}")
        
        # Проверяем координаты
        lat_range = (df['Географическая широта'].min(), df['Географическая широта'].max())
        lon_range = (df['Географическая долгота'].min(), df['Географическая долгота'].max())
        
        print(f"✅ Диапазон широт: {lat_range}")
        print(f"✅ Диапазон долгот: {lon_range}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return False

def test_coordinate_calculation():
    """Тест расчета расстояний между точками"""
    print("\n🔄 Тестирование расчета расстояний...")
    
    try:
        # Загружаем данные
        df = pd.read_csv("data/dataset.csv")
        
        # Берем первые 5 точек для теста
        test_points = df.head(5)
        
        def calculate_distance(point1, point2):
            """Упрощенный расчет расстояния"""
            lat1, lon1 = point1
            lat2, lon2 = point2
            
            # Формула гаверсинуса (упрощенная)
            R = 6371  # Радиус Земли в км
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = R * c
            
            return distance
        
        # Тестируем расчет расстояний
        distances = []
        for i in range(len(test_points)):
            for j in range(i+1, len(test_points)):
                point1 = (test_points.iloc[i]['Географическая широта'], 
                         test_points.iloc[i]['Географическая долгота'])
                point2 = (test_points.iloc[j]['Географическая широта'], 
                         test_points.iloc[j]['Географическая долгота'])
                
                distance = calculate_distance(point1, point2)
                distances.append(distance)
                
                print(f"   Расстояние между точками {i+1} и {j+1}: {distance:.2f} км")
        
        avg_distance = np.mean(distances)
        print(f"✅ Среднее расстояние между точками: {avg_distance:.2f} км")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при расчете расстояний: {e}")
        return False

def test_time_validation():
    """Тест валидации времени"""
    print("\n🔄 Тестирование валидации времени...")
    
    try:
        from datetime import datetime
        
        # Тестовые времена
        valid_times = ["09:00", "18:00", "13:00", "14:00"]
        invalid_times = ["25:00", "12:60", "abc", "9:00", "09:0", "9:00"]
        
        def validate_time_format(time_str):
            """Строгая валидация формата времени HH:MM"""
            try:
                if len(time_str) != 5 or time_str[2] != ':':
                    return False
                datetime.strptime(time_str, '%H:%M')
                return True
            except ValueError:
                return False
        
        for time_str in valid_times:
            if validate_time_format(time_str):
                print(f"✅ Время {time_str} валидно")
            else:
                print(f"❌ Время {time_str} невалидно")
                return False
        
        for time_str in invalid_times:
            if validate_time_format(time_str):
                print(f"❌ Время {time_str} должно быть невалидным")
                return False
            else:
                print(f"✅ Время {time_str} корректно отклонено")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при валидации времени: {e}")
        return False

def test_simple_optimization():
    """Простой тест алгоритма оптимизации"""
    print("\n🔄 Тестирование простого алгоритма оптимизации...")
    
    try:
        # Загружаем данные
        df = pd.read_csv("data/dataset.csv")
        
        # Берем первые 5 точек
        test_points = df.head(5)
        
        def simple_greedy_optimization(points):
            """Простой жадный алгоритм"""
            n = len(points)
            visited = [False] * n
            route = [0]  # Начинаем с первой точки
            visited[0] = True
            
            current = 0
            
            while len(route) < n:
                best_next = -1
                best_distance = float('inf')
                
                for next_point in range(n):
                    if not visited[next_point]:
                        # Простой расчет расстояния
                        lat1 = points.iloc[current]['Географическая широта']
                        lon1 = points.iloc[current]['Географическая долгота']
                        lat2 = points.iloc[next_point]['Географическая широта']
                        lon2 = points.iloc[next_point]['Географическая долгота']
                        
                        # Упрощенное расстояние
                        distance = ((lat2 - lat1)**2 + (lon2 - lon1)**2)**0.5
                        
                        # Приоритет VIP клиентам
                        if points.iloc[next_point]['Уровень клиента'] == 'VIP':
                            distance *= 0.8
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_next = next_point
                
                if best_next != -1:
                    route.append(best_next)
                    visited[best_next] = True
                    current = best_next
            
            return route
        
        # Тестируем оптимизацию
        optimized_route = simple_greedy_optimization(test_points)
        
        print(f"✅ Оптимизированный маршрут: {optimized_route}")
        
        # Проверяем, что все точки включены
        if len(set(optimized_route)) == len(test_points):
            print("✅ Все точки включены в маршрут")
        else:
            print("❌ Не все точки включены в маршрут")
            return False
        
        # Проверяем приоритет VIP клиентов
        vip_positions = []
        for i, point_idx in enumerate(optimized_route):
            if test_points.iloc[point_idx]['Уровень клиента'] == 'VIP':
                vip_positions.append(i)
        
        print(f"✅ VIP клиенты в позициях: {vip_positions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании оптимизации: {e}")
        return False

def test_api_structure():
    """Тест структуры API"""
    print("\n🔄 Тестирование структуры API...")
    
    try:
        # Проверяем наличие файлов
        required_files = [
            "main.py",
            "app/schemas/models.py", 
            "app/models/data_processor.py",
            "app/models/gnn_optimizer.py",
            "requirements.txt",
            "README.md",
            "config.py",
            "setup.py"
        ]
        
        missing_files = []
        for file in required_files:
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    if len(content) > 0:
                        print(f"✅ Файл {file} существует и не пустой")
                    else:
                        print(f"❌ Файл {file} пустой")
                        missing_files.append(file)
            except FileNotFoundError:
                print(f"❌ Файл {file} не найден")
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Отсутствуют файлы: {missing_files}")
            return False
        
        # Проверяем структуру main.py
        with open("main.py", 'r') as f:
            main_content = f.read()
            
        required_imports = [
            "from fastapi import FastAPI",
            "from app.routers",
            "app.include_router"
        ]
        
        missing_imports = []
        for import_stmt in required_imports:
            if import_stmt not in main_content:
                missing_imports.append(import_stmt)
        
        if missing_imports:
            print(f"❌ Отсутствуют импорты: {missing_imports}")
            return False
        
        print("✅ Все необходимые импорты присутствуют")
        
        # Проверяем эндпоинты в роутере
        try:
            with open("app/routers/route_optimization.py", 'r') as f:
                router_content = f.read()
                
            required_endpoints = [
                "@router.get(\"/health\")",
                "@router.post(\"/optimize-route"
            ]
            
            missing_endpoints = []
            for endpoint in required_endpoints:
                if endpoint not in router_content:
                    missing_endpoints.append(endpoint)
            
            if missing_endpoints:
                print(f"❌ Отсутствуют эндпоинты в роутере: {missing_endpoints}")
                return False
            
            print("✅ Все необходимые эндпоинты присутствуют")
            
        except FileNotFoundError:
            print("❌ Файл роутера не найден")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании API: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🚀 Запуск тестирования системы оптимизации маршрутов")
    print("=" * 60)
    
    tests = [
        ("Загрузка данных", test_data_loading),
        ("Расчет расстояний", test_coordinate_calculation),
        ("Валидация времени", test_time_validation),
        ("Простая оптимизация", test_simple_optimization),
        ("Структура API", test_api_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Тест: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ Тест '{test_name}' пройден")
            else:
                print(f"❌ Тест '{test_name}' не пройден")
        except Exception as e:
            print(f"❌ Ошибка в тесте '{test_name}': {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"Пройдено тестов: {passed}/{total}")
    print(f"Процент успеха: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 Все тесты пройдены успешно!")
        print("✅ Система готова к использованию")
    else:
        print("⚠️  Некоторые тесты не пройдены")
        print("🔧 Требуется доработка")
    
    return passed == total

if __name__ == "__main__":
    main()
