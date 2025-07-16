import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class SistemaRiesgoAcademico:
    def __init__(self):
        """Inicializa el sistema de lógica difusa para evaluar riesgo académico"""
        self._crear_variables()
        self._crear_funciones_membresia()
        self._crear_reglas()
        self._crear_sistema_control()
    
    def _crear_variables(self):
        """Crea las variables de entrada y salida"""
        # Variables de entrada
        self.nivel_socioeconomico = ctrl.Antecedent(np.arange(0, 11, 1), 'nivel_socioeconomico')
        self.participacion_clase = ctrl.Antecedent(np.arange(0, 11, 1), 'participacion_clase')
        self.asistencia = ctrl.Antecedent(np.arange(0, 101, 1), 'asistencia')
        self.calificaciones_anteriores = ctrl.Antecedent(np.arange(0, 11, 1), 'calificaciones_anteriores')
        
        # Variable de salida
        self.riesgo_academico = ctrl.Consequent(np.arange(0, 11, 1), 'riesgo_academico')
    
    def _crear_funciones_membresia(self):
        """Define las funciones de membresía para todas las variables"""
        # Funciones de membresía para nivel socioeconómico
        self.nivel_socioeconomico['bajo'] = fuzz.trimf(self.nivel_socioeconomico.universe, [0, 0, 4])
        self.nivel_socioeconomico['medio'] = fuzz.trimf(self.nivel_socioeconomico.universe, [2, 5, 8])
        self.nivel_socioeconomico['alto'] = fuzz.trimf(self.nivel_socioeconomico.universe, [6, 10, 10])
        
        # Funciones de membresía para participación en clase
        self.participacion_clase['baja'] = fuzz.trimf(self.participacion_clase.universe, [0, 0, 4])
        self.participacion_clase['media'] = fuzz.trimf(self.participacion_clase.universe, [2, 5, 8])
        self.participacion_clase['alta'] = fuzz.trimf(self.participacion_clase.universe, [6, 10, 10])
        
        # Funciones de membresía para asistencia
        self.asistencia['baja'] = fuzz.trimf(self.asistencia.universe, [0, 0, 60])
        self.asistencia['media'] = fuzz.trimf(self.asistencia.universe, [50, 70, 90])
        self.asistencia['alta'] = fuzz.trimf(self.asistencia.universe, [80, 100, 100])
        
        # Funciones de membresía para calificaciones anteriores
        self.calificaciones_anteriores['baja'] = fuzz.trimf(self.calificaciones_anteriores.universe, [0, 0, 5])
        self.calificaciones_anteriores['media'] = fuzz.trimf(self.calificaciones_anteriores.universe, [4, 6, 8])
        self.calificaciones_anteriores['alta'] = fuzz.trimf(self.calificaciones_anteriores.universe, [7, 10, 10])
        
        # Funciones de membresía para riesgo académico
        self.riesgo_academico['bajo'] = fuzz.trimf(self.riesgo_academico.universe, [0, 0, 4])
        self.riesgo_academico['medio'] = fuzz.trimf(self.riesgo_academico.universe, [3, 5, 7])
        self.riesgo_academico['alto'] = fuzz.trimf(self.riesgo_academico.universe, [6, 10, 10])
    
    def _crear_reglas(self):
        """Define las reglas difusas"""
        self.reglas = [
            # Regla 1: Todos los factores bajos -> Riesgo alto
            ctrl.Rule(
                self.nivel_socioeconomico['bajo'] & 
                self.participacion_clase['baja'] & 
                self.asistencia['baja'] & 
                self.calificaciones_anteriores['baja'], 
                self.riesgo_academico['alto']
            ),
            
            # Regla 2: Todos los factores medios -> Riesgo medio
            ctrl.Rule(
                self.nivel_socioeconomico['medio'] & 
                self.participacion_clase['media'] & 
                self.asistencia['media'] & 
                self.calificaciones_anteriores['media'], 
                self.riesgo_academico['medio']
            ),
            
            # Regla 3: Todos los factores altos -> Riesgo bajo
            ctrl.Rule(
                self.nivel_socioeconomico['alto'] & 
                self.participacion_clase['alta'] & 
                self.asistencia['alta'] & 
                self.calificaciones_anteriores['alta'], 
                self.riesgo_academico['bajo']
            ),
            
            # Regla 4: Asistencia baja O participación baja -> Riesgo alto
            ctrl.Rule(
                self.asistencia['baja'] | self.participacion_clase['baja'], 
                self.riesgo_academico['alto']
            ),
            
            # Regla 5: Calificaciones bajas -> Riesgo alto
            ctrl.Rule(
                self.calificaciones_anteriores['baja'], 
                self.riesgo_academico['alto']
            ),
            
            # Regla 6: Nivel socioeconómico bajo -> Riesgo medio
            ctrl.Rule(
                self.nivel_socioeconomico['bajo'], 
                self.riesgo_academico['medio']
            ),
            
            # Reglas adicionales para mayor cobertura
            ctrl.Rule(
                self.calificaciones_anteriores['alta'] & self.asistencia['alta'],
                self.riesgo_academico['bajo']
            ),
            
            ctrl.Rule(
                self.participacion_clase['alta'] & self.calificaciones_anteriores['alta'],
                self.riesgo_academico['bajo']
            ),
            
            ctrl.Rule(
                self.asistencia['media'] & self.calificaciones_anteriores['media'],
                self.riesgo_academico['medio']
            )
        ]
    
    def _crear_sistema_control(self):
        """Crea el sistema de control difuso"""
        self.sistema_control = ctrl.ControlSystem(self.reglas)
    
    def evaluar_riesgo(self, nivel_socioeconomico_val, participacion_clase_val, asistencia_val, calificaciones_anteriores_val):
        """
        Evalúa el riesgo académico usando lógica difusa.
        
        Args:
            nivel_socioeconomico_val (float): 0-10
            participacion_clase_val (float): 0-10
            asistencia_val (float): 0-100
            calificaciones_anteriores_val (float): 0-10
        
        Returns:
            float: valor de riesgo académico entre 0 y 10
        """
        try:
            # Validar y normalizar entradas
            nivel_socioeconomico_val = max(0, min(10, float(nivel_socioeconomico_val)))
            participacion_clase_val = max(0, min(10, float(participacion_clase_val)))
            asistencia_val = max(0, min(100, float(asistencia_val)))
            calificaciones_anteriores_val = max(0, min(10, float(calificaciones_anteriores_val)))
            
            print(f"Evaluando riesgo con valores: nivel={nivel_socioeconomico_val}, participacion={participacion_clase_val}, asistencia={asistencia_val}, calificaciones={calificaciones_anteriores_val}")
            
            # Crear simulador
            simulador = ctrl.ControlSystemSimulation(self.sistema_control)
            
            # Asignar valores de entrada
            simulador.input['nivel_socioeconomico'] = nivel_socioeconomico_val
            simulador.input['participacion_clase'] = participacion_clase_val
            simulador.input['asistencia'] = asistencia_val
            simulador.input['calificaciones_anteriores'] = calificaciones_anteriores_val
            
            # Ejecutar simulación
            simulador.compute()
            
            # Obtener resultado
            resultado = simulador.output['riesgo_academico']
            
            # Validar resultado
            if resultado is None or np.isnan(resultado):
                raise ValueError("El resultado es None o NaN")
            
            resultado = float(resultado)
            print(f"Resultado fuzzy calculado: {resultado}")
            return resultado
            
        except Exception as e:
            print(f"Error en lógica fuzzy: {e}")
            print("Calculando usando método heurístico...")
            
            # Método heurístico como backup
            return self._calcular_riesgo_heuristico(
                nivel_socioeconomico_val, 
                participacion_clase_val, 
                asistencia_val, 
                calificaciones_anteriores_val
            )
    
    def _calcular_riesgo_heuristico(self, nivel, participacion, asistencia, calificaciones):
        """Calcula el riesgo usando una fórmula heurística simple"""
        try:
            # Normalizar valores
            nivel = max(0, min(10, float(nivel)))
            participacion = max(0, min(10, float(participacion)))
            asistencia = max(0, min(100, float(asistencia)))
            calificaciones = max(0, min(10, float(calificaciones)))
            
            # Convertir asistencia a escala 0-10
            asistencia_normalizada = asistencia / 10
            
            # Calcular riesgo (mayor valor de entrada = menor riesgo)
            pesos = {
                'nivel': 0.2,
                'participacion': 0.3,
                'asistencia': 0.3,
                'calificaciones': 0.2
            }
            
            riesgo = (
                pesos['nivel'] * (10 - nivel) +
                pesos['participacion'] * (10 - participacion) +
                pesos['asistencia'] * (10 - asistencia_normalizada) +
                pesos['calificaciones'] * (10 - calificaciones)
            )
            
            # Asegurar que esté en el rango 0-10
            riesgo = max(0, min(10, riesgo))
            
            print(f"Resultado heurístico calculado: {riesgo}")
            return float(riesgo)
            
        except Exception as e:
            print(f"Error en cálculo heurístico: {e}")
            return 5.0  # Valor por defecto

# Crear instancia global del sistema
sistema_riesgo = SistemaRiesgoAcademico()

# Función de interfaz para mantener compatibilidad
def evaluar_riesgo(nivel_socioeconomico_val, participacion_clase_val, asistencia_val, calificaciones_anteriores_val):
    """
    Función de interfaz para evaluar riesgo académico.
    Mantiene compatibilidad con el código existente.
    """
    return sistema_riesgo.evaluar_riesgo(
        nivel_socioeconomico_val, 
        participacion_clase_val, 
        asistencia_val, 
        calificaciones_anteriores_val
    )

# Función de prueba
def test_sistema():
    """Función para probar el sistema fuzzy con diferentes valores"""
    print("=== Pruebas del Sistema Fuzzy ===")
    
    test_cases = [
        (5, 5, 50, 5),     # Caso medio
        (1, 1, 20, 2),     # Caso alto riesgo
        (10, 10, 100, 10), # Caso bajo riesgo
        (0, 0, 0, 0),      # Valores mínimos
        (3, 7, 80, 6),     # Caso mixto
    ]
    
    for i, (nivel, participacion, asistencia_val, calificaciones) in enumerate(test_cases):
        print(f"\n--- Prueba {i+1} ---")
        print(f"Inputs: nivel={nivel}, participacion={participacion}, asistencia={asistencia_val}, calificaciones={calificaciones}")
        resultado = evaluar_riesgo(nivel, participacion, asistencia_val, calificaciones)
        print(f"Resultado: {resultado}")
        
        # Interpretar resultado
        if resultado <= 3:
            interpretacion = "Riesgo BAJO"
        elif resultado <= 6:
            interpretacion = "Riesgo MEDIO"
        else:
            interpretacion = "Riesgo ALTO"
        
        print(f"Interpretación: {interpretacion}")

if __name__ == "__main__":
    test_sistema()