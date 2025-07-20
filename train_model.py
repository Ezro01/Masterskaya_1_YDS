import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import warnings

# Отключаем предупреждения
warnings.filterwarnings('ignore')

def train_and_save_model():
    """Обучение и сохранение модели"""
    print("🚀 Начинаем обучение модели...")
    
    # Загружаем данные
    print("📊 Загрузка данных...")
    df_train = pd.read_csv('heart_train.csv')
    print(f"📊 Загружено {len(df_train)} тренировочных образцов")
    
    # Убираем ненужные столбцы
    columns_to_drop = ['Unnamed: 0', 'id', 'Heart Attack Risk (Binary)', 'Smoking', 'Blood sugar', 'CK-MB', 'Troponin']
    df_clean = df_train.drop(columns=columns_to_drop, errors='ignore')
    
    # Определяем категориальные признаки
    categorical_features = ['Diabetes', 'Family History', 'Obesity', 'Alcohol Consumption', 
                          'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level',
                          'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Gender']
    
    # Числовые признаки
    numerical_features = [col for col in df_clean.columns if col not in categorical_features]
    
    print(f"📋 Категориальные признаки: {len(categorical_features)}")
    print(f"📊 Числовые признаки: {len(numerical_features)}")
    
    # Создаем препроцессор
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore', min_frequency=1), categorical_features)
        ])
    
    # Создаем пайплайн
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=2000,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Обучаем модель
    print("🎯 Обучение модели...")
    X = df_clean
    y = df_train['Heart Attack Risk (Binary)']
    
    model.fit(X, y)
    
    # Сохраняем модель и метаданные
    model_data = {
        'model': model,
        'feature_names': df_clean.columns.tolist(),
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'threshold': 0.454
    }
    
    model_path = 'heart_disease_model.joblib'
    joblib.dump(model_data, model_path)
    
    print(f"✅ Модель обучена и сохранена в {model_path}")
    print(f"📊 Количество признаков: {len(df_clean.columns)}")
    print(f"🎯 Порог классификации: 0.454")
    
    # Проверяем сохраненную модель
    print("🧪 Проверка сохраненной модели...")
    loaded_data = joblib.load(model_path)
    loaded_model = loaded_data['model']
    
    # Тестовое предсказание
    test_sample = df_clean.iloc[:1]
    prediction = loaded_model.predict_proba(test_sample)[0]
    print(f"✅ Тестовое предсказание: {prediction[1]:.3f}")
    
    return True

if __name__ == "__main__":
    train_and_save_model() 