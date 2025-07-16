import pandas as pd
import numpy as np

# Número de estudiantes en el dataset
num_estudiantes = 1000

# Semilla para reproducibilidad
np.random.seed(42)

# Generar datos ficticios
data = {
    'id_estudiante': range(1, num_estudiantes + 1),
    'calificaciones_anteriores': np.random.uniform(4.0, 10.0, num_estudiantes).round(2),
    'asistencia_porcentaje': np.random.randint(50, 101, num_estudiantes),
    'participacion_clase': np.random.randint(1, 6, num_estudiantes),
    'horas_estudio_semanal': np.random.randint(1, 25, num_estudiantes),
    'nivel_socioeconomico': np.random.choice(['Bajo', 'Medio', 'Alto'], num_estudiantes, p=[0.3, 0.5, 0.2])
}

df = pd.DataFrame(data)

# --- Lógica para determinar el riesgo académico ---
# La probabilidad de estar en riesgo aumenta si las calificaciones son bajas,
# la asistencia es baja, la participación es baja, y las horas de estudio son pocas.
# El nivel socioeconómico bajo también añade un poco a la probabilidad.

probabilidad_riesgo = (
    (10 - df['calificaciones_anteriores']) * 0.4 +
    (100 - df['asistencia_porcentaje']) * 0.02 +
    (5 - df['participacion_clase']) * 0.1 +
    (25 - df['horas_estudio_semanal']) * 0.01
)

# Añadir factor socioeconómico
mapeo_socioeconomico = {'Bajo': 0.15, 'Medio': 0.05, 'Alto': 0}
probabilidad_riesgo += df['nivel_socioeconomico'].map(mapeo_socioeconomico)

# Normalizar la probabilidad para que esté entre 0 y 1
probabilidad_riesgo = (probabilidad_riesgo - probabilidad_riesgo.min()) / (probabilidad_riesgo.max() - probabilidad_riesgo.min())

# Asignar la etiqueta de riesgo basándose en la probabilidad
df['rendimiento_riesgo'] = (probabilidad_riesgo > np.random.uniform(0.3, 0.7, num_estudiantes)).astype(int)

# Guardar el dataset en un archivo CSV
ruta_archivo = 'data/student_performance.csv'
df.to_csv(ruta_archivo, index=False)

print(f"Dataset ficticio creado y guardado en: {ruta_archivo}")
print("\nPrimeras 5 filas del dataset:")
print(df.head())
print(f"\nDistribución de la variable objetivo 'rendimiento_riesgo':\n{df['rendimiento_riesgo'].value_counts(normalize=True)}")
