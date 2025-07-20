import requests
import json

# Тестовые данные
test_data = {
    "Age": 0.5,
    "Cholesterol": 0.5,
    "Heart rate": 0.5,
    "Diabetes": 0,
    "Family History": 0,
    "Obesity": 0,
    "Alcohol Consumption": 0,
    "Exercise Hours Per Week": 0.5,
    "Diet": 0,
    "Previous Heart Problems": 0,
    "Medication Use": 0,
    "Stress Level": 0,
    "Sedentary Hours Per Day": 0.5,
    "Income": 0.5,
    "BMI": 0.5,
    "Triglycerides": 0.5,
    "Physical Activity Days Per Week": 0,
    "Sleep Hours Per Day": 4,
    "Gender": "Female",
    "Systolic blood pressure": 0.5,
    "Diastolic blood pressure": 0.5
}

# Тестируем API
try:
    print("🧪 Тестирование API...")
    
    # Тест одиночного прогноза
    response = requests.post(
        "http://localhost:8000/api/predict/single",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ API работает!")
        print(f"Прогноз: {result['prediction']}")
        print(f"Вероятность: {result['probability']:.3f}")
        print(f"Порог: {result['threshold']:.3f}")
    else:
        print(f"❌ Ошибка API: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Ошибка подключения: {e}") 