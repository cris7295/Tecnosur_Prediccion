import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataProcessor:
    def __init__(self, csv_path='data/student_performance_enhanced.csv'):
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Carga los datos del CSV mejorado"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Datos cargados exitosamente: {len(self.df)} estudiantes")
        except FileNotFoundError:
            print(f"Archivo {self.csv_path} no encontrado. Generando datos de ejemplo...")
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """Genera datos de ejemplo si no existe el archivo CSV"""
        # Listas para generar datos realistas
        nombres_m = ['Carlos', 'José', 'Diego', 'Roberto', 'Fernando', 'Andrés', 'Miguel', 'Alejandro', 
                    'Sebastián', 'Gabriel', 'Mateo', 'Nicolás', 'Santiago', 'Emilio', 'Joaquín', 
                    'Maximiliano', 'Tomás', 'Ignacio', 'Benjamín', 'Rodrigo', 'Esteban', 'Cristóbal', 
                    'Vicente', 'Agustín', 'Matías']
        
        nombres_f = ['Ana', 'María', 'Laura', 'Carmen', 'Patricia', 'Sofía', 'Valentina', 'Isabella', 
                    'Camila', 'Natalia', 'Daniela', 'Valeria', 'Mariana', 'Lucía', 'Regina', 
                    'Antonella', 'Renata', 'Julieta', 'Constanza', 'Florencia', 'Esperanza', 
                    'Amparo', 'Isidora', 'Magdalena', 'Javiera']
        
        apellidos = ['García', 'Rodríguez', 'López', 'Martínez', 'Hernández', 'González', 'Pérez', 
                    'Sánchez', 'Ramírez', 'Torres', 'Flores', 'Morales', 'Jiménez', 'Vargas', 
                    'Castro', 'Ortega', 'Ruiz', 'Mendoza', 'Herrera', 'Silva', 'Rojas', 'Guerrero', 
                    'Medina', 'Romero', 'Aguilar', 'Vega', 'Moreno', 'Castillo', 'Ramos', 'Delgado']
        
        carreras = ['Ingeniería', 'Medicina', 'Derecho', 'Psicología', 'Administración', 'Arte', 
                   'Economía', 'Arquitectura', 'Enfermería', 'Educación']
        
        niveles_socio = ['Bajo', 'Medio', 'Alto']
        estados = ['Activo', 'En Riesgo', 'Condicional']
        
        motivos_riesgo = ['Bajo rendimiento académico', 'Baja asistencia', 'Baja participación', 
                         'Pocas horas de estudio', 'Problemas económicos', 'Problemas familiares', '']
        
        data = []
        
        for i in range(1, 1001):  # Generar 1000 estudiantes
            genero = random.choice(['M', 'F'])
            nombre = random.choice(nombres_m if genero == 'M' else nombres_f)
            apellido = random.choice(apellidos)
            edad = random.randint(18, 26)
            carrera = random.choice(carreras)
            semestre = random.randint(1, 8)
            
            # Generar calificaciones correlacionadas
            base_grade = random.uniform(4.0, 10.0)
            calificaciones_anteriores = round(base_grade, 2)
            
            # Generar notas por materia con variación
            nota_matematicas = round(max(4.0, min(10.0, base_grade + random.uniform(-0.5, 0.5))), 1)
            nota_ciencias = round(max(4.0, min(10.0, base_grade + random.uniform(-0.5, 0.5))), 1)
            nota_lenguaje = round(max(4.0, min(10.0, base_grade + random.uniform(-0.5, 0.5))), 1)
            nota_historia = round(max(4.0, min(10.0, base_grade + random.uniform(-0.5, 0.5))), 1)
            promedio_general = round((nota_matematicas + nota_ciencias + nota_lenguaje + nota_historia) / 4, 1)
            
            # Otros campos
            asistencia_porcentaje = random.randint(50, 100)
            participacion_clase = random.randint(1, 5)
            horas_estudio_semanal = random.randint(1, 25)
            nivel_socioeconomico = random.choice(niveles_socio)
            
            # Determinar riesgo basado en múltiples factores
            riesgo_score = 0
            if calificaciones_anteriores < 6.0:
                riesgo_score += 2
            if asistencia_porcentaje < 70:
                riesgo_score += 2
            if participacion_clase <= 2:
                riesgo_score += 1
            if horas_estudio_semanal < 8:
                riesgo_score += 1
            
            rendimiento_riesgo = 1 if riesgo_score >= 3 else 0
            
            # Campos adicionales
            creditos_aprobados = min(semestre * 15, random.randint(semestre * 10, semestre * 20))
            creditos_totales = semestre * 20
            
            # Fecha de ingreso basada en semestre
            years_back = (semestre - 1) // 2
            months_back = ((semestre - 1) % 2) * 6
            fecha_ingreso = (datetime.now() - timedelta(days=years_back*365 + months_back*30)).strftime('%Y-%m-%d')
            
            estado_academico = 'En Riesgo' if rendimiento_riesgo == 1 else random.choice(['Activo', 'Activo', 'Activo', 'Condicional'])
            motivo = random.choice(motivos_riesgo) if rendimiento_riesgo == 1 else ''
            
            data.append({
                'id_estudiante': i,
                'nombre': nombre,
                'apellido': apellido,
                'edad': edad,
                'genero': genero,
                'carrera': carrera,
                'semestre': semestre,
                'email': f"{nombre.lower()}.{apellido.lower()}@universidad.edu",
                'telefono': f"555-{i:04d}",
                'calificaciones_anteriores': calificaciones_anteriores,
                'asistencia_porcentaje': asistencia_porcentaje,
                'participacion_clase': participacion_clase,
                'horas_estudio_semanal': horas_estudio_semanal,
                'nivel_socioeconomico': nivel_socioeconomico,
                'rendimiento_riesgo': rendimiento_riesgo,
                'nota_matematicas': nota_matematicas,
                'nota_ciencias': nota_ciencias,
                'nota_lenguaje': nota_lenguaje,
                'nota_historia': nota_historia,
                'promedio_general': promedio_general,
                'creditos_aprobados': creditos_aprobados,
                'creditos_totales': creditos_totales,
                'fecha_ingreso': fecha_ingreso,
                'estado_academico': estado_academico,
                'motivo_riesgo': motivo
            })
        
        self.df = pd.DataFrame(data)
        # Guardar el archivo generado
        self.df.to_csv(self.csv_path, index=False)
        print(f"Datos generados y guardados: {len(self.df)} estudiantes")
    
    def get_statistics(self):
        """Obtiene estadísticas generales del dataset"""
        if self.df is None:
            return {}
        
        total_estudiantes = len(self.df)
        estudiantes_riesgo = len(self.df[self.df['rendimiento_riesgo'] == 1])
        estudiantes_sin_riesgo = total_estudiantes - estudiantes_riesgo
        
        porcentaje_riesgo = (estudiantes_riesgo / total_estudiantes) * 100
        porcentaje_sin_riesgo = (estudiantes_sin_riesgo / total_estudiantes) * 100
        
        promedio_general = self.df['calificaciones_anteriores'].mean()
        promedio_asistencia = self.df['asistencia_porcentaje'].mean()
        
        stats = {
            'total_estudiantes': total_estudiantes,
            'estudiantes_riesgo': estudiantes_riesgo,
            'estudiantes_sin_riesgo': estudiantes_sin_riesgo,
            'porcentaje_riesgo': round(porcentaje_riesgo, 1),
            'porcentaje_sin_riesgo': round(porcentaje_sin_riesgo, 1),
            'promedio_general': round(promedio_general, 2),
            'promedio_asistencia': round(promedio_asistencia, 1),
            'carreras': list(self.df['carrera'].unique()),
            'semestres': sorted(list(self.df['semestre'].unique())),
            'distribución_por_carrera': self.df['carrera'].value_counts().to_dict(),
            'distribución_por_semestre': self.df['semestre'].value_counts().to_dict(),
            'distribución_riesgo_por_carrera': self.df.groupby('carrera')['rendimiento_riesgo'].mean().to_dict()
        }
        
        return stats
    
    def filter_students(self, filters=None):
        """Filtra estudiantes según criterios específicos"""
        if self.df is None:
            return pd.DataFrame()
        
        filtered_df = self.df.copy()
        
        if filters:
            if 'carrera' in filters and filters['carrera']:
                filtered_df = filtered_df[filtered_df['carrera'] == filters['carrera']]
            
            if 'semestre' in filters and filters['semestre']:
                filtered_df = filtered_df[filtered_df['semestre'] == filters['semestre']]
            
            if 'riesgo' in filters and filters['riesgo'] is not None:
                filtered_df = filtered_df[filtered_df['rendimiento_riesgo'] == filters['riesgo']]
            
            if 'estado' in filters and filters['estado']:
                filtered_df = filtered_df[filtered_df['estado_academico'] == filters['estado']]
            
            if 'busqueda' in filters and filters['busqueda']:
                search_term = filters['busqueda'].lower()
                mask = (
                    filtered_df['nombre'].str.lower().str.contains(search_term, na=False) |
                    filtered_df['apellido'].str.lower().str.contains(search_term, na=False) |
                    filtered_df['email'].str.lower().str.contains(search_term, na=False)
                )
                filtered_df = filtered_df[mask]
        
        return filtered_df
    
    def get_student_detail(self, student_id):
        """Obtiene detalles completos de un estudiante específico"""
        if self.df is None:
            return None
        
        student = self.df[self.df['id_estudiante'] == student_id]
        if student.empty:
            return None
        
        return student.iloc[0].to_dict()
    
    def get_risk_distribution(self):
        """Obtiene la distribución de riesgo para gráficos"""
        if self.df is None:
            return {}
        
        risk_counts = self.df['rendimiento_riesgo'].value_counts()
        return {
            'Sin Riesgo': risk_counts.get(0, 0),
            'En Riesgo': risk_counts.get(1, 0)
        }
    
    def get_grade_distribution_by_risk(self):
        """Obtiene distribución de calificaciones por nivel de riesgo"""
        if self.df is None:
            return pd.DataFrame()
        
        return self.df[['calificaciones_anteriores', 'rendimiento_riesgo']].copy()
    
    def validate_data(self, new_data):
        """Valida nuevos datos antes de agregarlos"""
        required_fields = [
            'nombre', 'apellido', 'edad', 'genero', 'carrera', 'semestre',
            'calificaciones_anteriores', 'asistencia_porcentaje', 'participacion_clase',
            'horas_estudio_semanal', 'nivel_socioeconomico'
        ]
        
        errors = []
        
        for field in required_fields:
            if field not in new_data or new_data[field] is None or new_data[field] == '':
                errors.append(f"Campo requerido faltante: {field}")
        
        # Validaciones específicas
        if 'edad' in new_data:
            try:
                edad = int(new_data['edad'])
                if edad < 16 or edad > 50:
                    errors.append("Edad debe estar entre 16 y 50 años")
            except (ValueError, TypeError):
                errors.append("Edad debe ser un número entero")
        
        if 'calificaciones_anteriores' in new_data:
            try:
                calif = float(new_data['calificaciones_anteriores'])
                if calif < 0 or calif > 10:
                    errors.append("Calificaciones deben estar entre 0 y 10")
            except (ValueError, TypeError):
                errors.append("Calificaciones deben ser un número")
        
        if 'asistencia_porcentaje' in new_data:
            try:
                asist = int(new_data['asistencia_porcentaje'])
                if asist < 0 or asist > 100:
                    errors.append("Asistencia debe estar entre 0 y 100%")
            except (ValueError, TypeError):
                errors.append("Asistencia debe ser un número entero")
        
        return errors
    
    def add_student(self, student_data):
        """Agrega un nuevo estudiante al dataset"""
        if self.df is None:
            return False, ["No hay datos cargados"]
        
        # Validar datos
        errors = self.validate_data(student_data)
        if errors:
            return False, errors
        
        # Generar nuevo ID
        new_id = self.df['id_estudiante'].max() + 1
        student_data['id_estudiante'] = new_id
        
        # Agregar campos faltantes con valores por defecto
        if 'email' not in student_data:
            student_data['email'] = f"{student_data['nombre'].lower()}.{student_data['apellido'].lower()}@universidad.edu"
        
        if 'telefono' not in student_data:
            student_data['telefono'] = f"555-{new_id:04d}"
        
        if 'fecha_ingreso' not in student_data:
            student_data['fecha_ingreso'] = datetime.now().strftime('%Y-%m-%d')
        
        # Agregar al DataFrame
        new_row = pd.DataFrame([student_data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Guardar cambios
        self.df.to_csv(self.csv_path, index=False)
        
        return True, ["Estudiante agregado exitosamente"]

# Instancia global del procesador
data_processor = DataProcessor()
