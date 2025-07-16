import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# 1. Cargar datos
print("Cargando datos...")
df = pd.read_csv('data/student_performance.csv')

# 2. Definir Features (X) y Target (y)
X = df.drop(['id_estudiante', 'rendimiento_riesgo'], axis=1)
y = df['rendimiento_riesgo']

# 3. Preprocesamiento para sklearn
# Definir qué columnas son categóricas para aplicar One-Hot Encoding
categorical_features = ['nivel_socioeconomico']
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    remainder='passthrough'
)

# 4. Dividir los datos en conjuntos de entrenamiento y prueba
print("Dividiendo los datos...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Crear el pipeline del modelo de Árbol de Decisión
print("Entrenando el modelo de Árbol de Decisión...")
model_pipeline = make_pipeline(
    preprocessor,
    DecisionTreeClassifier(random_state=42, max_depth=5)
)

# Entrenar el pipeline completo
model_pipeline.fit(X_train, y_train)

# 6. Evaluar el modelo de Árbol de Decisión
print("Evaluando el modelo de Árbol de Decisión...")
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo en el conjunto de prueba: {accuracy:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 7. Guardar el modelo entrenado de Árbol de Decisión
print("Guardando el modelo en 'models/prediction_model.joblib'...")
joblib.dump(model_pipeline, 'models/prediction_model.joblib')

# --- Entrenamiento del modelo Random Forest ---
print("Entrenando el modelo Random Forest...")
rf_pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42, n_estimators=100)
)

rf_pipeline.fit(X_train, y_train)

print("Evaluando el modelo Random Forest...")
y_pred_rf = rf_pipeline.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nPrecisión del modelo Random Forest en el conjunto de prueba: {accuracy_rf:.4f}")
print("\nReporte de Clasificación Random Forest:")
print(classification_report(y_test, y_pred_rf))

print("Guardando el modelo Random Forest en 'models/random_forest_model.joblib'...")
joblib.dump(rf_pipeline, 'models/random_forest_model.joblib')

# --- Entrenamiento del modelo de Red Neuronal con Keras ---

# Preprocesamiento para Keras
print("Preprocesando datos para Red Neuronal...")
# One-hot encode categorical feature
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = ohe.fit_transform(X[categorical_features])
# Numeric features
numeric_features = X.drop(columns=categorical_features).columns
scaler = StandardScaler()
X_num = scaler.fit_transform(X[numeric_features])
# Concatenar
X_processed = np.hstack([X_cat, X_num])

# Dividir datos para Keras
X_train_keras, X_test_keras, y_train_keras, y_test_keras = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Convertir etiquetas a categóricas para Keras
y_train_cat = to_categorical(y_train_keras)
y_test_cat = to_categorical(y_test_keras)

# Definir el modelo
print("Creando y entrenando la Red Neuronal...")
model_keras = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_keras.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

model_keras.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model_keras.fit(X_train_keras, y_train_cat, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluar el modelo
print("Evaluando la Red Neuronal...")
loss, accuracy = model_keras.evaluate(X_test_keras, y_test_cat, verbose=0)
print(f"\nPrecisión de la Red Neuronal en el conjunto de prueba: {accuracy:.4f}")

# Guardar el modelo Keras
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_keras.save(os.path.join(model_dir, 'keras_model.keras'))

print("\n¡Entrenamiento completado y modelos guardados exitosamente!")
