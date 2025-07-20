import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from contextlib import asynccontextmanager
import warnings
import joblib

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

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.threshold = 0.454  # Правильный порог из мастерской
        self.model_path = 'heart_disease_model.joblib'
        
    def train_model(self):
        """Обучение модели на основе мастерской"""
        # Сначала пытаемся загрузить существующую модель
        if self.load_model():
            print("✅ Модель загружена из файла")
            return
        
        print("📊 Обучение новой модели...")
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
        
        # Сохраняем модель
        self.save_model()
        
        print(f"✅ Модель обучена и сохранена с {len(self.feature_names)} признаками")
        print(f"🎯 Порог классификации: {self.threshold}")
    
    def save_model(self):
        """Сохранение модели и метаданных"""
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features,
                'threshold': self.threshold
            }
            joblib.dump(model_data, self.model_path)
            print(f"💾 Модель сохранена в {self.model_path}")
        except Exception as e:
            print(f"❌ Ошибка сохранения модели: {e}")
    
    def load_model(self):
        """Загрузка модели и метаданных"""
        try:
            if not os.path.exists(self.model_path):
                print(f"📁 Файл модели {self.model_path} не найден")
                return False
            
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.categorical_features = model_data['categorical_features']
            self.numerical_features = model_data['numerical_features']
            self.threshold = model_data['threshold']
            
            print(f"📁 Модель загружена из {self.model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
        
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
            
            # Убеждаемся, что все необходимые столбцы присутствуют
            missing_cols = set(self.feature_names) - set(df_clean.columns)
            for col in missing_cols:
                df_clean[col] = 0  # Добавляем недостающие столбцы с нулевыми значениями
            
            # Упорядочиваем столбцы в том же порядке, что и при обучении
            df_clean = df_clean[self.feature_names]
            
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
            
            # Убеждаемся, что все необходимые столбцы присутствуют
            missing_cols = set(self.feature_names) - set(df_clean.columns)
            for col in missing_cols:
                df_clean[col] = 0  # Добавляем недостающие столбцы с нулевыми значениями
            
            # Упорядочиваем столбцы в том же порядке, что и при обучении
            df_clean = df_clean[self.feature_names]
            
            # Получаем вероятности
            probas = self.model.predict_proba(df_clean)
            risk_probabilities = probas[:, 1]  # Вероятности высокого риска
            
            # Применяем порог
            predictions = (risk_probabilities >= self.threshold).astype(int)
            
            # Отладочная информация
            print(f"📊 Всего прогнозов: {len(predictions)}")
            print(f"📊 Прогнозов 0: {sum(predictions == 0)}")
            print(f"📊 Прогнозов 1: {sum(predictions == 1)}")
            print(f"🎯 Порог: {self.threshold}")
            print(f"📊 Примеры вероятностей: {risk_probabilities[:5]}")
            
            return predictions.tolist()
            
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            return [0] * len(df)

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    """Главная страница с веб-интерфейсом"""
    return templates.TemplateResponse("index.html", {"request": request})

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
        
        # Отладочная информация
        print(f"📊 Результаты API:")
        print(f"📊 Всего образцов: {len(df)}")
        print(f"📊 Всего прогнозов: {len(predictions)}")
        print(f"📊 Прогнозов 0: {predictions.count(0)}")
        print(f"📊 Прогнозов 1: {predictions.count(1)}")
        
        # Создаем результат
        result_df = pd.DataFrame({
            'id': df['id'],
            'prediction': predictions
        })
        
        # Проверяем результат
        print(f"📊 Размер result_df: {len(result_df)}")
        print(f"📊 Колонки result_df: {result_df.columns.tolist()}")
        print(f"📊 Первые 5 строк result_df:")
        print(result_df.head())
        
        # Проверяем, что все данные корректны
        data_records = result_df.to_dict('records')
        print(f"📊 Количество записей в data: {len(data_records)}")
        print(f"📊 Первые 5 записей в data:")
        for i, record in enumerate(data_records[:5]):
            print(f"  Запись {i}: id={record['id']}, prediction={record['prediction']} (тип: {type(record['prediction'])})")
        
        # Проверяем, что все прогнозы корректны
        zeros_count = sum(1 for record in data_records if record['prediction'] == 0)
        ones_count = sum(1 for record in data_records if record['prediction'] == 1)
        print(f"📊 В data_records: нулей={zeros_count}, единиц={ones_count}")
        
        # Проверяем, что нет None или undefined значений
        invalid_count = sum(1 for record in data_records if record['prediction'] is None or record['prediction'] == 'undefined')
        if invalid_count > 0:
            print(f"⚠️ Найдено {invalid_count} некорректных значений!")
        
        # Возвращаем результат без сохранения файла
        return {
            "success": True,
            "samples": len(df),
            "predictions": predictions,
            "data": data_records
        }
        
    except Exception as e:
        print(f"❌ Ошибка обработки файла: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 