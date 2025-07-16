import os
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv
from src.data_processor import data_processor
import src.fuzzy_logic as fuzzy

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)

# Cargar datos del sistema académico
def get_academic_context():
    """Obtiene el contexto académico completo para la IA"""
    try:
        df = data_processor.df
        if df is None or df.empty:
            return "No hay datos académicos disponibles."
        
        # Estadísticas generales
        stats = data_processor.get_statistics()
        
        # Muestra de estudiantes (primeros 10 para contexto)
        sample_students = df.head(10).to_dict('records')
        
        # Estudiantes en riesgo
        at_risk_students = df[df['rendimiento_riesgo'] == 1].head(5).to_dict('records')
        
        context = {
            "estadisticas_generales": stats,
            "muestra_estudiantes": sample_students,
            "estudiantes_en_riesgo": at_risk_students,
            "total_estudiantes": len(df),
            "columnas_disponibles": list(df.columns),
            "descripcion_sistema": {
                "objetivo": "Sistema de predicción de riesgo académico",
                "variables_principales": [
                    "calificaciones_anteriores (4.0-10.0)",
                    "asistencia_porcentaje (0-100%)",
                    "participacion_clase (1-5)",
                    "horas_estudio_semanal (1-25)",
                    "nivel_socioeconomico (Bajo/Medio/Alto)",
                    "rendimiento_riesgo (0=Sin riesgo, 1=En riesgo)"
                ],
                "modelos_disponibles": [
                    "Árbol de Decisión",
                    "Random Forest", 
                    "Red Neuronal",
                    "Lógica Difusa"
                ]
            }
        }
        
        return json.dumps(context, indent=2, ensure_ascii=False)
    
    except Exception as e:
        return f"Error al cargar contexto académico: {str(e)}"

def find_student_by_name(search_name):
    """Busca un estudiante por nombre de manera flexible"""
    try:
        df = data_processor.df
        if df is None or df.empty:
            return None
        
        search_name = search_name.lower().strip()
        
        # Buscar por nombre completo, nombre o apellido
        for _, student in df.iterrows():
            nombre = student['nombre'].lower()
            apellido = student['apellido'].lower()
            nombre_completo = f"{nombre} {apellido}"
            
            if (search_name in nombre_completo or 
                search_name in nombre or 
                search_name in apellido or
                nombre in search_name or
                apellido in search_name):
                return student.to_dict()
        
        return None
    except Exception as e:
        print(f"Error buscando estudiante: {e}")
        return None

def generate_ai_response(prompt):
    """Genera respuesta de IA más natural y conversacional"""
    try:
        df = data_processor.df
        if df is None or df.empty:
            return "Lo siento, no tengo acceso a los datos académicos en este momento."
        
        prompt_lower = prompt.lower()
        
        # 1. Consultas sobre estudiantes específicos
        if any(word in prompt_lower for word in ['información sobre', 'datos de', 'dime sobre', 'quiero saber de']):
            # Extraer posible nombre del prompt
            words = prompt.split()
            potential_names = []
            
            for i, word in enumerate(words):
                if word.lower() in ['sobre', 'de'] and i + 1 < len(words):
                    # Tomar las siguientes 1-3 palabras como posible nombre
                    name_parts = []
                    for j in range(i + 1, min(i + 4, len(words))):
                        if words[j].replace(',', '').replace('.', '').isalpha():
                            name_parts.append(words[j].replace(',', '').replace('.', ''))
                        else:
                            break
                    if name_parts:
                        potential_names.append(' '.join(name_parts))
            
            # Buscar estudiante
            student_found = None
            for name in potential_names:
                student_found = find_student_by_name(name)
                if student_found:
                    break
            
            if student_found:
                # Calcular riesgo con lógica difusa
                calif = student_found.get('calificaciones_anteriores', 0)
                asist = student_found.get('asistencia_porcentaje', 0)
                part = student_found.get('participacion_clase', 0)
                socio = student_found.get('nivel_socioeconomico', 'Medio')
                
                mapping_nivel = {'Bajo': 3, 'Medio': 6, 'Alto': 9}
                nivel_val = mapping_nivel.get(socio, 5)
                riesgo_fuzzy = fuzzy.evaluar_riesgo(nivel_val, part * 2, asist, calif)
                
                # Respuesta natural sobre el estudiante
                nombre_completo = f"{student_found['nombre']} {student_found['apellido']}"
                estado_riesgo = "en riesgo alto" if student_found['rendimiento_riesgo'] == 1 else "sin riesgo significativo"
                
                response = f"Te cuento sobre {nombre_completo}:\n\n"
                response += f"Es estudiante de {student_found['carrera']}, actualmente en {student_found['semestre']}° semestre. "
                response += f"Su promedio general es de {student_found['promedio_general']}/10 y tiene una asistencia del {asist}%. "
                
                if student_found['rendimiento_riesgo'] == 1:
                    response += f"\n\n⚠️ Actualmente está {estado_riesgo}. "
                    if asist < 70:
                        response += "Su principal problema es la baja asistencia. "
                    if calif < 6:
                        response += "También necesita mejorar sus calificaciones. "
                    if part <= 2:
                        response += "Su participación en clase es muy limitada. "
                    
                    response += "\n\n💡 Te recomiendo que:"
                    if asist < 70:
                        response += "\n• Establezca una rutina de asistencia más consistente"
                    if calif < 6:
                        response += "\n• Busque apoyo académico o tutorías"
                    if part <= 2:
                        response += "\n• Se involucre más en las discusiones de clase"
                else:
                    response += f"\n\n✅ Está {estado_riesgo}, lo cual es excelente. Su rendimiento es estable."
                
                response += f"\n\n📊 Según nuestro análisis con lógica difusa, su nivel de riesgo es {riesgo_fuzzy:.1f}/10."
                
                return response
            else:
                return f"No encontré información sobre ese estudiante. ¿Podrías verificar el nombre? Tengo datos de {len(df)} estudiantes en el sistema."
        
        # 2. Consultas sobre estadísticas por carrera
        elif 'estadísticas por carrera' in prompt_lower or 'stats por carrera' in prompt_lower:
            stats = data_processor.get_statistics()
            response = "📚 Aquí tienes las estadísticas de riesgo por carrera:\n\n"
            
            carrera_stats = []
            for carrera in stats['carreras']:
                carrera_df = df[df['carrera'] == carrera]
                total = len(carrera_df)
                en_riesgo = len(carrera_df[carrera_df['rendimiento_riesgo'] == 1])
                porcentaje = (en_riesgo / total * 100) if total > 0 else 0
                carrera_stats.append((carrera, en_riesgo, total, porcentaje))
            
            # Ordenar por porcentaje de riesgo
            carrera_stats.sort(key=lambda x: x[3], reverse=True)
            
            for carrera, en_riesgo, total, porcentaje in carrera_stats:
                response += f"• **{carrera}**: {en_riesgo}/{total} estudiantes en riesgo ({porcentaje:.1f}%)\n"
            
            # Agregar insight
            peor_carrera = carrera_stats[0]
            mejor_carrera = carrera_stats[-1]
            response += f"\n💡 **Insight**: {peor_carrera[0]} tiene el mayor porcentaje de riesgo ({peor_carrera[3]:.1f}%), "
            response += f"mientras que {mejor_carrera[0]} tiene el menor ({mejor_carrera[3]:.1f}%)."
            
            return response
        
        # 3. Consultas sobre estudiantes en riesgo de una carrera específica
        elif 'estudiantes en riesgo' in prompt_lower and any(carrera.lower() in prompt_lower for carrera in df['carrera'].unique()):
            carrera_buscada = None
            for carrera in df['carrera'].unique():
                if carrera.lower() in prompt_lower:
                    carrera_buscada = carrera
                    break
            
            if carrera_buscada:
                estudiantes_riesgo = df[(df['carrera'] == carrera_buscada) & (df['rendimiento_riesgo'] == 1)]
                
                if len(estudiantes_riesgo) == 0:
                    return f"¡Excelente noticia! No hay estudiantes de {carrera_buscada} en riesgo alto actualmente. 🎉"
                
                response = f"📋 Estudiantes de {carrera_buscada} en riesgo alto ({len(estudiantes_riesgo)} total):\n\n"
                
                for i, (_, estudiante) in enumerate(estudiantes_riesgo.head(10).iterrows(), 1):
                    response += f"{i}. **{estudiante['nombre']} {estudiante['apellido']}** "
                    response += f"(Semestre {estudiante['semestre']}) - "
                    response += f"Promedio: {estudiante['promedio_general']}/10, "
                    response += f"Asistencia: {estudiante['asistencia_porcentaje']}%\n"
                
                if len(estudiantes_riesgo) > 10:
                    response += f"\n... y {len(estudiantes_riesgo) - 10} estudiantes más."
                
                # Agregar recomendación general
                promedio_asistencia = estudiantes_riesgo['asistencia_porcentaje'].mean()
                promedio_calificaciones = estudiantes_riesgo['calificaciones_anteriores'].mean()
                
                response += f"\n\n💡 **Patrón identificado**: "
                if promedio_asistencia < 70:
                    response += f"La asistencia promedio es baja ({promedio_asistencia:.1f}%). "
                if promedio_calificaciones < 6:
                    response += f"Las calificaciones promedio están por debajo del mínimo ({promedio_calificaciones:.1f}/10). "
                
                response += "Recomiendo intervención temprana con estos estudiantes."
                
                return response
        
        # 4. Consultas generales sobre el sistema
        elif any(word in prompt_lower for word in ['cuántos estudiantes', 'total estudiantes', 'estadísticas generales']):
            stats = data_processor.get_statistics()
            response = f"📊 **Resumen del Sistema Académico**\n\n"
            response += f"Tenemos **{stats['total_estudiantes']} estudiantes** registrados en total.\n"
            response += f"• **{stats['estudiantes_riesgo']} estudiantes** están en riesgo alto ({stats['porcentaje_riesgo']}%)\n"
            response += f"• **{stats['estudiantes_sin_riesgo']} estudiantes** están sin riesgo ({stats['porcentaje_sin_riesgo']}%)\n\n"
            response += f"📈 **Métricas clave:**\n"
            response += f"• Promedio general: **{stats['promedio_general']}/10**\n"
            response += f"• Asistencia promedio: **{stats['promedio_asistencia']}%**\n\n"
            response += f"🎓 **Carreras con más estudiantes:**\n"
            
            # Top 3 carreras
            carreras_ordenadas = sorted(stats['distribución_por_carrera'].items(), key=lambda x: x[1], reverse=True)
            for i, (carrera, cantidad) in enumerate(carreras_ordenadas[:3], 1):
                response += f"{i}. {carrera}: {cantidad} estudiantes\n"
            
            return response
        
        # 5. Respuesta por defecto más natural
        else:
            return ("¡Hola! Soy tu asistente de análisis académico. 😊\n\n"
                   "Puedo ayudarte con:\n"
                   "• Información específica de estudiantes (ej: 'información sobre Juan Pérez')\n"
                   "• Estadísticas por carrera\n"
                   "• Estudiantes en riesgo de una carrera específica\n"
                   "• Estadísticas generales del sistema\n\n"
                   "¿Qué te gustaría saber?")
    
    except Exception as e:
        print(f"Error generando respuesta: {e}")
        return f"Disculpa, hubo un error procesando tu consulta. ¿Podrías intentar de nuevo? Error: {str(e)}"

@app.route('/')
def home():
    """Ruta principal del servidor IA"""
    return jsonify({
        "message": "🤖 Servidor IA Académico - Tecnosur",
        "status": "running",
        "endpoints": {
            "/api/chat": "POST - Chat con IA",
            "/api/student-search": "POST - Buscar estudiantes",
            "/api/stats": "GET - Estadísticas del sistema"
        },
        "students_loaded": len(data_processor.df) if data_processor.df is not None else 0
    })

@app.route('/api/status')
def status():
    """Estado del servidor"""
    return jsonify({
        "status": "running",
        "students_loaded": len(data_processor.df) if data_processor.df is not None else 0,
        "server": "IA Académico Tecnosur"
    })

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Endpoint principal para consultas a la IA académica"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "Prompt vacío"}), 400
        
        print(f"✅ CONSULTA RECIBIDA: {prompt}")
        
        # Usar la nueva función de respuesta natural
        ai_response = generate_ai_response(prompt)
        
        print(f"✅ RESPUESTA GENERADA: {ai_response}")
        
        return jsonify({"respuesta": ai_response})
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

@app.route('/api/student-search', methods=['POST'])
def student_search():
    """Endpoint para buscar estudiantes específicos"""
    try:
        data = request.get_json()
        search_term = data.get('search', '').lower()
        
        df = data_processor.df
        if df is None or df.empty:
            return jsonify({"students": []})
        
        # Buscar estudiantes que coincidan
        matches = df[
            df['nombre'].str.lower().str.contains(search_term, na=False) |
            df['apellido'].str.lower().str.contains(search_term, na=False) |
            (df['nombre'] + ' ' + df['apellido']).str.lower().str.contains(search_term, na=False)
        ]
        
        students = []
        for _, student in matches.head(10).iterrows():  # Limitar a 10 resultados
            students.append({
                "id": student['id_estudiante'],
                "nombre_completo": f"{student['nombre']} {student['apellido']}",
                "carrera": student['carrera'],
                "semestre": student['semestre'],
                "riesgo": "Alto" if student['rendimiento_riesgo'] == 1 else "Bajo"
            })
        
        return jsonify({"students": students})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Endpoint para obtener estadísticas rápidas"""
    try:
        stats = data_processor.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Iniciando servidor de IA Académica...")
    print("📊 Cargando datos académicos...")
    
    # Verificar que los datos estén cargados
    if data_processor.df is not None:
        print(f"✅ Datos cargados: {len(data_processor.df)} estudiantes")
    else:
        print("⚠️ No se pudieron cargar los datos")
    
    app.run(debug=True, port=5001)
