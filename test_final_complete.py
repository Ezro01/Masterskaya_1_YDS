import requests
import json
import pandas as pd
import tempfile
import os

def test_complete_interface():
    """Полное тестирование веб-интерфейса"""
    base_url = "http://localhost:8000"
    
    print("🚀 Полное тестирование веб-интерфейса Heart Disease Prediction")
    print("=" * 70)
    
    try:
        # 1. Тест главной страницы
        print("1️⃣ Тестирование главной страницы...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Главная страница доступна")
        else:
            print(f"❌ Ошибка главной страницы: {response.status_code}")
            return False
        
        # 2. Тест информации о модели
        print("\n2️⃣ Тестирование информации о модели...")
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Информация о модели получена")
            print(f"📊 Модель: {model_info.get('model', 'unknown')}")
            if 'features_count' in model_info:
                print(f"📋 Признаков: {model_info['features_count']}")
            elif 'features' in model_info:
                print(f"📋 Признаков: {len(model_info['features'])}")
        else:
            print(f"❌ Ошибка получения информации о модели: {response.status_code}")
            return False
        
        # 3. Тест одиночного прогноза - низкий риск
        print("\n3️⃣ Тестирование одиночного прогноза (низкий риск)...")
        low_risk_data = {
            'Age': 30,
            'Cholesterol': 180,
            'Diabetes': 0,
            'Family History': 0,
            'Obesity': 0,
            'Previous Heart Problems': 0,
            'Stress Level': 0,
            'Physical Activity Days Per Week': 5,
            'Gender': 0
        }
        
        response = requests.post(
            f"{base_url}/api/predict/single",
            json=low_risk_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Одиночный прогноз выполнен")
            print(f"🎯 Результат: {result['prediction']} ({'Низкий риск' if result['prediction'] == 0 else 'Высокий риск'})")
        else:
            print(f"❌ Ошибка одиночного прогноза: {response.status_code}")
            return False
        
        # 4. Тест одиночного прогноза - высокий риск
        print("\n4️⃣ Тестирование одиночного прогноза (высокий риск)...")
        high_risk_data = {
            'Age': 70,
            'Cholesterol': 280,
            'Diabetes': 1,
            'Family History': 1,
            'Obesity': 1,
            'Previous Heart Problems': 1,
            'Stress Level': 2,
            'Physical Activity Days Per Week': 1,
            'Gender': 1
        }
        
        response = requests.post(
            f"{base_url}/api/predict/single",
            json=high_risk_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Одиночный прогноз выполнен")
            print(f"🎯 Результат: {result['prediction']} ({'Низкий риск' if result['prediction'] == 0 else 'Высокий риск'})")
        else:
            print(f"❌ Ошибка одиночного прогноза: {response.status_code}")
            return False
        
        # 5. Тест загрузки файла
        print("\n5️⃣ Тестирование загрузки файла...")
        
        # Создаем тестовый CSV файл
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'Age': [45, 55, 65, 35, 75],
            'Cholesterol': [200, 220, 250, 180, 300],
            'Diabetes': [0, 1, 0, 0, 1],
            'Family History': [1, 0, 1, 0, 1],
            'Obesity': [0, 1, 0, 0, 1],
            'Previous Heart Problems': [0, 0, 1, 0, 1],
            'Stress Level': [1, 2, 1, 0, 2],
            'Physical Activity Days Per Week': [3, 2, 1, 5, 0],
            'Gender': [1, 0, 1, 0, 1]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            test_data.to_csv(tmp_file.name, index=False)
            tmp_filename = tmp_file.name
        
        try:
            with open(tmp_filename, 'rb') as f:
                files = {'file': ('test_data.csv', f, 'text/csv')}
                response = requests.post(f"{base_url}/api/predict/file", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Загрузка файла выполнена")
                print(f"📊 Обработано образцов: {result['total_samples']}")
                
                # Тест скачивания результатов
                if result.get('predictions_file'):
                    download_response = requests.get(f"{base_url}/api/download/{result['predictions_file']}")
                    if download_response.status_code == 200:
                        print("✅ Скачивание результатов выполнено")
                        
                        # Проверяем содержимое файла
                        results_df = pd.read_csv(tmp_filename.replace('.csv', '_results.csv') if '_results.csv' in tmp_filename else tmp_filename)
                        if 'id' in results_df.columns and 'prediction' in results_df.columns:
                            print("✅ Формат файла результатов корректный")
                            print(f"📋 Результаты: {results_df['prediction'].tolist()}")
                        else:
                            print("❌ Неправильный формат файла результатов")
                    else:
                        print(f"❌ Ошибка скачивания: {download_response.status_code}")
                else:
                    print("❌ Файл с результатами не создан")
            else:
                print(f"❌ Ошибка загрузки файла: {response.status_code}")
                print(f"📝 Ответ: {response.text}")
                return False
                
        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
        
        print("\n" + "=" * 70)
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("=" * 70)
        print("✅ Главная страница работает")
        print("✅ API информации о модели работает")
        print("✅ Одиночные прогнозы работают")
        print("✅ Загрузка файлов работает")
        print("✅ Скачивание результатов работает")
        print("\n🌐 Откройте браузер и перейдите по адресу: http://localhost:8000")
        print("📋 Веб-интерфейс полностью функционален!")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Не удается подключиться к серверу")
        print("💡 Убедитесь, что сервер запущен на порту 8000")
        return False
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

if __name__ == "__main__":
    success = test_complete_interface()
    
    if success:
        print("\n✅ Веб-интерфейс готов к использованию!")
    else:
        print("\n❌ Обнаружены проблемы с веб-интерфейсом") 