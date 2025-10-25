"""
Управление зависимостями и окружением
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

REQUIRED_DIRS = [
    "app",
    "app/schemas", 
    "app/models",
    "app/routers",
    "app/views",
    "tests",
    "educate",
    "data"
]

def check_project_structure():
    """Проверка структуры проекта"""
    missing_dirs = []
    for dir_name in REQUIRED_DIRS:
        dir_path = ROOT_DIR / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Отсутствуют директории: {missing_dirs}")
        return False
    
    print("✅ Структура проекта корректна")
    return True

def get_project_info():
    """Получение информации о проекте"""
    return {
        "root_dir": str(ROOT_DIR),
        "app_dir": str(ROOT_DIR / "app"),
        "data_dir": str(ROOT_DIR / "data"),
        "tests_dir": str(ROOT_DIR / "tests"),
        "educate_dir": str(ROOT_DIR / "educate"),
        "required_dirs": REQUIRED_DIRS
    }

if __name__ == "__main__":
    print("🔍 Проверка структуры проекта...")
    check_project_structure()
    
    print("\n📋 Информация о проекте:")
    info = get_project_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
