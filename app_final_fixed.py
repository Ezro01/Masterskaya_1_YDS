import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from contextlib import asynccontextmanager
import warnings

# Отключаем предупреждения
warnings.filterwarnings('ignore')

# Глобальная переменная для предсказателя
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global predictor
    print("🚀 Запуск Heart Disease Prediction...")
    predictor = HeartDiseasePredictor()
    predictor.train_model()
    print("✅ Приложение готово!")
    yield
    # Shutdown
    print("🛑 Остановка приложения...")

# Создаем FastAPI приложение
app = FastAPI(title="Heart Disease Prediction", version="1.0", lifespan=lifespan)

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.threshold = 0.454  # Правильный порог из мастерской
        
    def train_model(self):
        """Обучение модели на основе мастерской"""
        print("📊 Загрузка данных...")
        
        # Загружаем данные
        df_train = pd.read_csv('heart_train.csv')
        print(f"📊 Загружено {len(df_train)} тренировочных образцов")
        
        # Убираем ненужные столбцы как в мастерской
        columns_to_drop = ['Unnamed: 0', 'id', 'Heart Attack Risk (Binary)', 'Smoking', 'Blood sugar', 'CK-MB', 'Troponin']
        df_clean = df_train.drop(columns=columns_to_drop, errors='ignore')
        
        # Определяем категориальные признаки (все кроме числовых)
        categorical_features = ['Diabetes', 'Family History', 'Obesity', 'Alcohol Consumption', 
                              'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level',
                              'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Gender']
        
        # Числовые признаки (все остальные)
        numerical_features = [col for col in df_clean.columns if col not in categorical_features]
        
        print(f"📋 Категориальные признаки: {len(categorical_features)}")
        print(f"📊 Числовые признаки: {len(numerical_features)}")
        
        # Создаем препроцессор с правильными настройками
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore', min_frequency=1), categorical_features)
            ])
        
        # Создаем пайплайн
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=2000,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        # Обучаем модель
        X = df_clean
        y = df_train['Heart Attack Risk (Binary)']
        
        self.model.fit(X, y)
        
        # Сохраняем имена признаков
        self.feature_names = df_clean.columns.tolist()
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        print(f"✅ Модель обучена с {len(self.feature_names)} признаками")
        print(f"🎯 Порог классификации: {self.threshold}")
        
    def predict_single(self, data_dict):
        """Предсказание для одного пациента"""
        try:
            # Создаем DataFrame
            df = pd.DataFrame([data_dict])
            
            # Убираем ненужные столбцы
            columns_to_drop = ['Unnamed: 0', 'id', 'Heart Attack Risk (Binary)', 'Smoking', 'Blood sugar', 'CK-MB', 'Troponin']
            df_clean = df.drop(columns=columns_to_drop, errors='ignore')
            
            # Преобразуем категориальные признаки в правильные типы
            for col in self.categorical_features:
                if col in df_clean.columns:
                    if col == 'Gender':
                        # Gender должен быть строкой
                        df_clean[col] = df_clean[col].astype(str)
                    elif col in ['Physical Activity Days Per Week', 'Sleep Hours Per Day']:
                        # Эти признаки должны быть числовыми
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
                    else:
                        # Остальные категориальные признаки должны быть числовыми
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
            
            # Преобразуем числовые признаки в float
            for col in self.numerical_features:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
            # Получаем вероятности
            proba = self.model.predict_proba(df_clean)[0]
            risk_probability = proba[1]  # Вероятность высокого риска
            
            # Применяем порог
            prediction = 1 if risk_probability >= self.threshold else 0
            
            return {
                'prediction': int(prediction),
                'probability': float(risk_probability),
                'threshold': self.threshold
            }
            
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            return {'prediction': 0, 'probability': 0.0, 'threshold': self.threshold}
    
    def predict_batch(self, df):
        """Предсказание для батча данных"""
        try:
            # Убираем ненужные столбцы
            columns_to_drop = ['Unnamed: 0', 'id', 'Heart Attack Risk (Binary)', 'Smoking', 'Blood sugar', 'CK-MB', 'Troponin']
            df_clean = df.drop(columns=columns_to_drop, errors='ignore')
            
            # Преобразуем категориальные признаки в правильные типы
            for col in self.categorical_features:
                if col in df_clean.columns:
                    if col == 'Gender':
                        # Gender должен быть строкой
                        df_clean[col] = df_clean[col].astype(str)
                    elif col in ['Physical Activity Days Per Week', 'Sleep Hours Per Day']:
                        # Эти признаки должны быть числовыми
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
                    else:
                        # Остальные категориальные признаки должны быть числовыми
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
            
            # Преобразуем числовые признаки в float
            for col in self.numerical_features:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
            # Получаем вероятности
            probas = self.model.predict_proba(df_clean)
            risk_probabilities = probas[:, 1]  # Вероятности высокого риска
            
            # Применяем порог
            predictions = (risk_probabilities >= self.threshold).astype(int)
            
            return predictions.tolist()
            
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            return [0] * len(df)

@app.get("/", response_class=HTMLResponse)
async def main_page():
    """Главная страница с веб-интерфейсом"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>💓 Heart Disease Prediction</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #ff6b6b, #ee5a24);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .nav {
                display: flex;
                background: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }
            
            .nav button {
                flex: 1;
                padding: 15px;
                border: none;
                background: none;
                cursor: pointer;
                font-size: 1.1em;
                transition: all 0.3s ease;
            }
            
            .nav button.active {
                background: #007bff;
                color: white;
            }
            
            .nav button:hover {
                background: #0056b3;
                color: white;
            }
            
            .content {
                padding: 30px;
                display: none;
            }
            
            .content.active {
                display: block;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
            }
            
            .form-group input, .form-group select {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s ease;
            }
            
            .form-group input:focus, .form-group select:focus {
                outline: none;
                border-color: #007bff;
            }
            
            .slider-container {
                display: flex;
                align-items: center;
                gap: 15px;
            }
            
            .slider {
                flex: 1;
                height: 6px;
                border-radius: 3px;
                background: #ddd;
                outline: none;
                -webkit-appearance: none;
            }
            
            .slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #007bff;
                cursor: pointer;
            }
            
            .slider::-moz-range-thumb {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #007bff;
                cursor: pointer;
                border: none;
            }
            
            .value-display {
                min-width: 60px;
                text-align: center;
                font-weight: bold;
                color: #007bff;
            }
            
            .btn {
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-size: 1.1em;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
                margin-top: 20px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0,123,255,0.3);
            }
            
            .result {
                margin-top: 20px;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 1.2em;
                font-weight: bold;
            }
            
            .result.low-risk {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .result.high-risk {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .file-upload {
                border: 2px dashed #ddd;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin-bottom: 20px;
                transition: border-color 0.3s ease;
            }
            
            .file-upload:hover {
                border-color: #007bff;
            }
            
            .file-upload input {
                display: none;
            }
            
            .file-upload label {
                cursor: pointer;
                color: #007bff;
                font-weight: bold;
            }
            
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            
            .info-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #007bff;
            }
            
            .info-card h3 {
                color: #007bff;
                margin-bottom: 10px;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007bff;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>💓 Heart Disease Prediction</h1>
                <p>Система прогнозирования риска сердечного приступа</p>
            </div>
            
            <div class="nav">
                <button onclick="showTab('single', this)" class="active">👤 Одиночный прогноз</button>
                <button onclick="showTab('batch', this)">📁 Загрузка файла</button>
                <button onclick="showTab('info', this)">ℹ️ Информация</button>
            </div>
            
            <div id="single" class="content active">
                <h2>Одиночный прогноз</h2>
                <form id="predictionForm">
                    <div class="form-group">
                        <label>Возраст (Age):</label>
                        <div class="slider-container">
                            <input type="range" id="Age" name="Age" min="0" max="1" step="0.01" value="0.5" class="slider">
                            <span class="value-display" id="Age-value">0.50</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Холестерин (Cholesterol):</label>
                        <div class="slider-container">
                            <input type="range" id="Cholesterol" name="Cholesterol" min="0" max="1" step="0.01" value="0.5" class="slider">
                            <span class="value-display" id="Cholesterol-value">0.50</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Частота сердечных сокращений (Heart rate):</label>
                        <div class="slider-container">
                            <input type="range" id="Heart rate" name="Heart rate" min="0" max="1" step="0.01" value="0.5" class="slider">
                            <span class="value-display" id="Heart rate-value">0.50</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Диабет (Diabetes):</label>
                        <select id="Diabetes" name="Diabetes">
                            <option value="0">Нет</option>
                            <option value="1">Да</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Семейная история (Family History):</label>
                        <select id="Family History" name="Family History">
                            <option value="0">Нет</option>
                            <option value="1">Да</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Ожирение (Obesity):</label>
                        <select id="Obesity" name="Obesity">
                            <option value="0">Нет</option>
                            <option value="1">Да</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Употребление алкоголя (Alcohol Consumption):</label>
                        <select id="Alcohol Consumption" name="Alcohol Consumption">
                            <option value="0">Нет</option>
                            <option value="1">Умеренное</option>
                            <option value="2">Высокое</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Часы упражнений в неделю (Exercise Hours Per Week):</label>
                        <div class="slider-container">
                            <input type="range" id="Exercise Hours Per Week" name="Exercise Hours Per Week" min="0" max="1" step="0.01" value="0.5" class="slider">
                            <span class="value-display" id="Exercise Hours Per Week-value">0.50</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Диета (Diet):</label>
                        <select id="Diet" name="Diet">
                            <option value="0">Плохая</option>
                            <option value="1">Средняя</option>
                            <option value="2">Хорошая</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Предыдущие проблемы с сердцем (Previous Heart Problems):</label>
                        <select id="Previous Heart Problems" name="Previous Heart Problems">
                            <option value="0">Нет</option>
                            <option value="1">Да</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Прием лекарств (Medication Use):</label>
                        <select id="Medication Use" name="Medication Use">
                            <option value="0">Нет</option>
                            <option value="1">Да</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Уровень стресса (Stress Level):</label>
                        <select id="Stress Level" name="Stress Level">
                            <option value="0">Низкий</option>
                            <option value="1">Средний</option>
                            <option value="2">Высокий</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Сидячие часы в день (Sedentary Hours Per Day):</label>
                        <div class="slider-container">
                            <input type="range" id="Sedentary Hours Per Day" name="Sedentary Hours Per Day" min="0" max="1" step="0.01" value="0.5" class="slider">
                            <span class="value-display" id="Sedentary Hours Per Day-value">0.50</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Доход (Income):</label>
                        <div class="slider-container">
                            <input type="range" id="Income" name="Income" min="0" max="1" step="0.01" value="0.5" class="slider">
                            <span class="value-display" id="Income-value">0.50</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Индекс массы тела (BMI):</label>
                        <div class="slider-container">
                            <input type="range" id="BMI" name="BMI" min="0" max="1" step="0.01" value="0.5" class="slider">
                            <span class="value-display" id="BMI-value">0.50</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Триглицериды (Triglycerides):</label>
                        <div class="slider-container">
                            <input type="range" id="Triglycerides" name="Triglycerides" min="0" max="1" step="0.01" value="0.5" class="slider">
                            <span class="value-display" id="Triglycerides-value">0.50</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Физическая активность дней в неделю (Physical Activity Days Per Week):</label>
                        <select id="Physical Activity Days Per Week" name="Physical Activity Days Per Week">
                            <option value="0">0 дней</option>
                            <option value="1">1 день</option>
                            <option value="2">2 дня</option>
                            <option value="3">3 дня</option>
                            <option value="4">4 дня</option>
                            <option value="5">5 дней</option>
                            <option value="6">6 дней</option>
                            <option value="7">7 дней</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Сон часов в день (Sleep Hours Per Day):</label>
                        <select id="Sleep Hours Per Day" name="Sleep Hours Per Day">
                            <option value="4">4 часа</option>
                            <option value="5">5 часов</option>
                            <option value="6">6 часов</option>
                            <option value="7">7 часов</option>
                            <option value="8">8 часов</option>
                            <option value="9">9 часов</option>
                            <option value="10">10 часов</option>
                            <option value="11">11 часов</option>
                            <option value="12">12 часов</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Пол (Gender):</label>
                        <select id="Gender" name="Gender">
                            <option value="Female">Женский</option>
                            <option value="Male">Мужской</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Систолическое давление (Systolic blood pressure):</label>
                        <div class="slider-container">
                            <input type="range" id="Systolic blood pressure" name="Systolic blood pressure" min="0" max="1" step="0.01" value="0.5" class="slider">
                            <span class="value-display" id="Systolic blood pressure-value">0.50</span>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Диастолическое давление (Diastolic blood pressure):</label>
                        <div class="slider-container">
                            <input type="range" id="Diastolic blood pressure" name="Diastolic blood pressure" min="0" max="1" step="0.01" value="0.5" class="slider">
                            <span class="value-display" id="Diastolic blood pressure-value">0.50</span>
                        </div>
                    </div>
                    
                    <button type="button" onclick="submitForm()" class="btn">Получить прогноз</button>
                </form>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Обработка данных...</p>
                </div>
                
                <div id="result" class="result" style="display: none;"></div>
            </div>
            
            <div id="batch" class="content">
                <h2>Загрузка файла</h2>
                <p>Загрузите CSV файл с данными для получения прогнозов</p>
                
                <div class="file-upload">
                    <input type="file" id="fileInput" accept=".csv">
                    <label for="fileInput">📁 Выберите CSV файл или перетащите сюда</label>
                </div>
                
                <button onclick="uploadFile()" class="btn">Загрузить и получить прогноз</button>
                
                <div class="loading" id="batchLoading">
                    <div class="spinner"></div>
                    <p>Обработка файла...</p>
                </div>
                
                <div id="batchResult"></div>
            </div>
            
            <div id="info" class="content">
                <h2>Информация о модели</h2>
                
                <div class="info-grid">
                    <div class="info-card">
                        <h3>🎯 Алгоритм</h3>
                        <p>Random Forest Classifier</p>
                        <p>Параметры: n_estimators=2000, max_depth=10</p>
                    </div>
                    
                    <div class="info-card">
                        <h3>📊 Порог классификации</h3>
                        <p>0.454 (из мастерской)</p>
                        <p>При вероятности ≥ 0.454 - высокий риск</p>
                    </div>
                    
                    <div class="info-card">
                        <h3>📋 Признаки</h3>
                        <p>Всего: 21 признак</p>
                        <p>Категориальные: 11</p>
                        <p>Числовые: 10 (нормализованные)</p>
                    </div>
                    
                    <div class="info-card">
                        <h3>⚠️ Важно</h3>
                        <p>Это прогноз на основе ML</p>
                        <p>Для точной диагностики обратитесь к врачу</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Показ вкладок
            function showTab(tabName, element) {
                // Скрываем все вкладки
                document.querySelectorAll('.content').forEach(content => {
                    content.classList.remove('active');
                });
                document.querySelectorAll('.nav button').forEach(btn => {
                    btn.classList.remove('active');
                });
                
                // Показываем выбранную вкладку
                document.getElementById(tabName).classList.add('active');
                element.classList.add('active');
            }
            
            // Обновление значений слайдеров
            document.querySelectorAll('.slider').forEach(slider => {
                slider.addEventListener('input', function() {
                    const value = parseFloat(this.value).toFixed(2);
                    document.getElementById(this.id + '-value').textContent = value;
                });
            });
            
            // Обработка формы
            async function submitForm() {
                const form = document.getElementById('predictionForm');
                const formData = new FormData(form);
                const data = {};
                
                for (let [key, value] of formData.entries()) {
                    data[key] = parseFloat(value) || value;
                }
                
                // Показываем загрузку
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                try {
                    const response = await fetch('/api/predict/single', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    // Скрываем загрузку
                    document.getElementById('loading').style.display = 'none';
                    
                    // Показываем результат
                    const resultDiv = document.getElementById('result');
                    resultDiv.style.display = 'block';
                    
                    if (result.prediction === 1) {
                        resultDiv.className = 'result high-risk';
                        resultDiv.innerHTML = `
                            <h3>⚠️ ВЫСОКИЙ РИСК</h3>
                            <p>Вероятность: ${(result.probability * 100).toFixed(1)}%</p>
                            <p>Порог: ${(result.threshold * 100).toFixed(1)}%</p>
                        `;
                    } else {
                        resultDiv.className = 'result low-risk';
                        resultDiv.innerHTML = `
                            <h3>✅ НИЗКИЙ РИСК</h3>
                            <p>Вероятность: ${(result.probability * 100).toFixed(1)}%</p>
                            <p>Порог: ${(result.threshold * 100).toFixed(1)}%</p>
                        `;
                    }
                    
                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    alert('Ошибка при получении прогноза: ' + error.message);
                }
            }
            
            // Загрузка файла
            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Пожалуйста, выберите файл');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                // Показываем загрузку
                document.getElementById('batchLoading').style.display = 'block';
                document.getElementById('batchResult').innerHTML = '';
                
                try {
                    const response = await fetch('/api/predict/file', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    // Скрываем загрузку
                    document.getElementById('batchLoading').style.display = 'none';
                    
                    if (result.success) {
                        // Создаем CSV данные для скачивания
                        const csvContent = "data:text/csv;charset=utf-8," + 
                            "id,prediction\n" + 
                            result.data.map(row => `${row.id},${row.prediction}`).join('\n');
                        
                        // Создаем ссылку для скачивания
                        const downloadLink = document.createElement('a');
                        downloadLink.href = encodeURI(csvContent);
                        downloadLink.download = 'predictions.csv';
                        downloadLink.className = 'btn';
                        downloadLink.textContent = '📥 Скачать результаты';
                        downloadLink.style.marginTop = '20px';
                        
                        document.getElementById('batchResult').innerHTML = `
                            <div class="result low-risk">
                                <h3>✅ Файл обработан</h3>
                                <p>Обработано образцов: ${result.samples}</p>
                                <p>Прогнозы готовы к скачиванию</p>
                            </div>
                        `;
                        
                        document.getElementById('batchResult').appendChild(downloadLink);
                    } else {
                        document.getElementById('batchResult').innerHTML = `
                            <div class="result high-risk">
                                <h3>❌ Ошибка</h3>
                                <p>${result.error}</p>
                            </div>
                        `;
                    }
                    
                } catch (error) {
                    document.getElementById('batchLoading').style.display = 'none';
                    alert('Ошибка при загрузке файла: ' + error.message);
                }
            }
            
            // Drag and drop для файлов
            const fileUpload = document.querySelector('.file-upload');
            
            fileUpload.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.style.borderColor = '#007bff';
            });
            
            fileUpload.addEventListener('dragleave', function(e) {
                e.preventDefault();
                this.style.borderColor = '#ddd';
            });
            
            fileUpload.addEventListener('drop', function(e) {
                e.preventDefault();
                this.style.borderColor = '#ddd';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    document.getElementById('fileInput').files = files;
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/api/models")
async def get_model_info():
    """Информация о модели"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return {
        "model_type": "random_forest",
        "threshold": predictor.threshold,
        "features_count": len(predictor.feature_names) if predictor.feature_names else 0,
        "categorical_features": predictor.categorical_features,
        "numerical_features": predictor.numerical_features
    }

@app.post("/api/predict/single")
async def predict_single(data: dict):
    """Одиночный прогноз"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        result = predictor.predict_single(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """Прогноз для файла"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Только CSV файлы поддерживаются")
        
        # Читаем файл
        df = pd.read_csv(file.file)
        
        if 'id' not in df.columns:
            raise HTTPException(status_code=400, detail="Файл должен содержать столбец 'id'")
        
        print(f"📊 Загружено {len(df)} образцов из файла {file.filename}")
        
        # Получаем прогнозы
        predictions = predictor.predict_batch(df)
        
        # Создаем результат
        result_df = pd.DataFrame({
            'id': df['id'],
            'prediction': predictions
        })
        
        # Возвращаем результат без сохранения файла
        return {
            "success": True,
            "samples": len(df),
            "predictions": predictions,
            "data": result_df.to_dict('records')
        }
        
    except Exception as e:
        print(f"❌ Ошибка обработки файла: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 