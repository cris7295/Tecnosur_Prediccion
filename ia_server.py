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
    """Genera respuesta de IA NATURAL y conversacional (versión local mejorada siguiendo el estilo del compañero)"""
    try:
        df = data_processor.df
        if df is None or df.empty:
            return "Lo siento, no tengo acceso a los datos académicos en este momento."
        
        prompt_lower = prompt.lower()
        stats = data_processor.get_statistics()
        
        # RESPUESTAS NATURALES COMO LO HARÍA UN COORDINADOR REAL
        
        # 1. Saludos - Respuesta natural y cercana
        if any(word in prompt_lower for word in ['hola', 'buenos', 'buenas', 'saludos', 'como estas']):
            return f"¡Hola! Me alegra que me escribas. Soy el coordinador académico de la Universidad Tecnosur y estoy aquí para ayudarte con cualquier consulta sobre nuestros estudiantes. Actualmente tengo información de {stats['total_estudiantes']} estudiantes distribuidos en {len(stats['carreras'])} carreras. ¿En qué puedo ayudarte hoy?"
        
        # 2. Búsqueda de estudiantes por nombre
        elif any(word in prompt_lower for word in ['listame', 'lista', 'estudiantes', 'nombres', 'empiecen', 'empiezan']):
            # Extraer letra o criterio de búsqueda
            letter = None
            for word in prompt_lower.split():
                if len(word) == 1 and word.isalpha():
                    letter = word.upper()
                    break
            
            if not letter and 'c' in prompt_lower:
                letter = 'C'
            
            if letter:
                matching_students = df[df['nombre'].str.upper().str.startswith(letter)]
                if len(matching_students) > 0:
                    student_list = []
                    for _, student in matching_students.head(10).iterrows():
                        risk_status = "en riesgo" if student['rendimiento_riesgo'] == 1 else "sin riesgo"
                        student_list.append(f"{student['nombre']} {student['apellido']} ({student['carrera']}, {risk_status})")
                    
                    response = f"Te muestro los estudiantes cuyo nombre empieza con '{letter}':\n\n"
                    response += "\n".join([f"• {student}" for student in student_list])
                    
                    if len(matching_students) > 10:
                        response += f"\n\n(Mostrando los primeros 10 de {len(matching_students)} estudiantes encontrados)"
                    
                    response += f"\n\n¿Te interesa conocer más detalles sobre alguno de ellos?"
                    return response
                else:
                    return f"No encontré estudiantes cuyo nombre empiece con '{letter}'. ¿Quieres que busque con otra letra?"
            else:
                return "Claro, puedo ayudarte a buscar estudiantes. ¿Podrías decirme con qué letra empiezan los nombres que buscas? Por ejemplo: 'estudiantes que empiecen con A'"
        
        # 3. Comparaciones entre carreras - Respuesta natural
        elif any(word in prompt_lower for word in ['comparacion', 'compara', 'diferencia', 'vs', 'versus']):
            import random
            carreras_disponibles = stats['carreras']
            
            # Si mencionan carreras específicas, usarlas
            carreras_mencionadas = []
            for carrera in carreras_disponibles:
                if carrera.lower() in prompt_lower:
                    carreras_mencionadas.append(carrera)
            
            # Si no mencionan carreras específicas o solo una, elegir aleatoriamente
            if len(carreras_mencionadas) < 2:
                carreras_a_comparar = random.sample(carreras_disponibles, 2)
            else:
                carreras_a_comparar = carreras_mencionadas[:2]
            
            # Análisis detallado de cada carrera
            comparacion_data = []
            for carrera in carreras_a_comparar:
                carrera_df = df[df['carrera'] == carrera]
                total = len(carrera_df)
                en_riesgo = len(carrera_df[carrera_df['rendimiento_riesgo'] == 1])
                porcentaje_riesgo = (en_riesgo / total * 100) if total > 0 else 0
                promedio_calif = carrera_df['calificaciones_anteriores'].mean()
                promedio_asist = carrera_df['asistencia_porcentaje'].mean()
                promedio_part = carrera_df['participacion_clase'].mean()
                promedio_horas = carrera_df['horas_estudio_semanal'].mean()
                
                comparacion_data.append({
                    'carrera': carrera,
                    'total': total,
                    'en_riesgo': en_riesgo,
                    'porcentaje_riesgo': porcentaje_riesgo,
                    'promedio_calif': promedio_calif,
                    'promedio_asist': promedio_asist,
                    'promedio_part': promedio_part,
                    'promedio_horas': promedio_horas
                })
            
            # Generar respuesta conversacional
            c1, c2 = comparacion_data[0], comparacion_data[1]
            
            response = f"🎓 **Comparación entre {c1['carrera']} y {c2['carrera']}**\n\n"
            response += f"Te hago un análisis detallado de estas dos carreras:\n\n"
            
            # Tamaño de población
            response += f"📊 **Población estudiantil:**\n"
            response += f"• {c1['carrera']}: {c1['total']} estudiantes\n"
            response += f"• {c2['carrera']}: {c2['total']} estudiantes\n"
            if c1['total'] > c2['total']:
                response += f"→ {c1['carrera']} tiene {c1['total'] - c2['total']} estudiantes más que {c2['carrera']}\n\n"
            else:
                response += f"→ {c2['carrera']} tiene {c2['total'] - c1['total']} estudiantes más que {c1['carrera']}\n\n"
            
            # Análisis de riesgo
            response += f"⚠️ **Análisis de riesgo académico:**\n"
            response += f"• {c1['carrera']}: {c1['en_riesgo']}/{c1['total']} en riesgo ({c1['porcentaje_riesgo']:.1f}%)\n"
            response += f"• {c2['carrera']}: {c2['en_riesgo']}/{c2['total']} en riesgo ({c2['porcentaje_riesgo']:.1f}%)\n"
            
            if c1['porcentaje_riesgo'] > c2['porcentaje_riesgo']:
                diff = c1['porcentaje_riesgo'] - c2['porcentaje_riesgo']
                response += f"→ {c1['carrera']} tiene {diff:.1f}% más estudiantes en riesgo que {c2['carrera']}\n\n"
            else:
                diff = c2['porcentaje_riesgo'] - c1['porcentaje_riesgo']
                response += f"→ {c2['carrera']} tiene {diff:.1f}% más estudiantes en riesgo que {c1['carrera']}\n\n"
            
            # Rendimiento académico
            response += f"📈 **Rendimiento académico:**\n"
            response += f"• {c1['carrera']}: Promedio {c1['promedio_calif']:.1f}/10\n"
            response += f"• {c2['carrera']}: Promedio {c2['promedio_calif']:.1f}/10\n"
            
            mejor_rendimiento = c1['carrera'] if c1['promedio_calif'] > c2['promedio_calif'] else c2['carrera']
            response += f"→ {mejor_rendimiento} tiene mejor rendimiento académico\n\n"
            
            # Asistencia
            response += f"👥 **Asistencia promedio:**\n"
            response += f"• {c1['carrera']}: {c1['promedio_asist']:.1f}%\n"
            response += f"• {c2['carrera']}: {c2['promedio_asist']:.1f}%\n"
            
            mejor_asistencia = c1['carrera'] if c1['promedio_asist'] > c2['promedio_asist'] else c2['carrera']
            response += f"→ {mejor_asistencia} tiene mejor asistencia\n\n"
            
            # Conclusión inteligente
            response += f"💡 **Mi análisis:**\n"
            
            # Determinar cuál carrera está mejor
            puntos_c1 = 0
            puntos_c2 = 0
            
            if c1['promedio_calif'] > c2['promedio_calif']:
                puntos_c1 += 1
            else:
                puntos_c2 += 1
                
            if c1['promedio_asist'] > c2['promedio_asist']:
                puntos_c1 += 1
            else:
                puntos_c2 += 1
                
            if c1['porcentaje_riesgo'] < c2['porcentaje_riesgo']:  # Menos riesgo es mejor
                puntos_c1 += 1
            else:
                puntos_c2 += 1
            
            if puntos_c1 > puntos_c2:
                response += f"En general, **{c1['carrera']}** muestra mejores indicadores académicos. "
                if c1['porcentaje_riesgo'] < 30:
                    response += "Sus estudiantes están en una situación bastante estable."
                else:
                    response += "Sin embargo, aún hay oportunidades de mejora en la reducción del riesgo."
            elif puntos_c2 > puntos_c1:
                response += f"En general, **{c2['carrera']}** muestra mejores indicadores académicos. "
                if c2['porcentaje_riesgo'] < 30:
                    response += "Sus estudiantes están en una situación bastante estable."
                else:
                    response += "Sin embargo, aún hay oportunidades de mejora en la reducción del riesgo."
            else:
                response += f"Ambas carreras muestran un rendimiento muy similar. "
                response += f"Cada una tiene sus fortalezas particulares."
            
            response += f"\n\n¿Te gustaría que profundice en algún aspecto específico de estas carreras?"
            
            return response
        
        # 2. Consultas sobre estudiantes específicos
        elif any(word in prompt_lower for word in ['información sobre', 'datos de', 'dime sobre', 'quiero saber de']):
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
        
        # 3. Consultas sobre estadísticas por carrera
        elif 'estadísticas por carrera' in prompt_lower or 'stats por carrera' in prompt_lower:
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
        
        # 4. Consultas generales sobre el sistema
        elif any(word in prompt_lower for word in ['cuántos estudiantes', 'total estudiantes', 'estadísticas generales', 'resumen']):
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
        
        # 5. Respuesta más inteligente por defecto
        else:
            # Analizar el prompt para dar una respuesta más específica
            if 'hola' in prompt_lower or 'buenos' in prompt_lower or 'buenas' in prompt_lower:
                return (f"¡Hola! 👋 Soy tu asistente académico de la Universidad Tecnosur.\n\n"
                       f"Actualmente tengo información de **{stats['total_estudiantes']} estudiantes** "
                       f"distribuidos en {len(stats['carreras'])} carreras diferentes.\n\n"
                       f"Puedo ayudarte con análisis específicos como:\n"
                       f"• Comparaciones entre carreras (ej: 'compara Ingeniería con Medicina')\n"
                       f"• Información de estudiantes específicos\n"
                       f"• Estadísticas detalladas por carrera\n"
                       f"• Análisis de riesgo académico\n\n"
                       f"¿Qué te gustaría analizar hoy? 🤔")
            else:
                return (f"Hmm, no estoy seguro de cómo interpretar esa consulta. 🤔\n\n"
                       f"Soy especialista en análisis académico y puedo ayudarte con:\n\n"
                       f"📊 **Análisis disponibles:**\n"
                       f"• Comparaciones entre carreras\n"
                       f"• Estadísticas por carrera\n"
                       f"• Información de estudiantes específicos\n"
                       f"• Resumen general del sistema\n\n"
                       f"¿Podrías ser más específico sobre qué información necesitas?")
    
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
    """Endpoint principal para consultas a la IA académica - SOLO API EXTERNA"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "Prompt vacío"}), 400
        
        print(f"✅ CONSULTA RECIBIDA: {prompt}")
        
        # FORZAR uso de API externa SIEMPRE
        api_key = os.getenv('OPENROUTER_API_KEY')
        
        if not api_key or not api_key.strip():
            return jsonify({"error": "API Key no configurada"}), 500
        
        print(f"🔑 Usando API externa con key: {api_key[:20]}...")
        
        try:
            ai_response = generate_intelligent_ai_response(prompt, api_key)
            print(f"✅ RESPUESTA IA EXTERNA: {ai_response}")
            return jsonify({"respuesta": ai_response})
        except Exception as e:
            print(f"❌ Error con API externa: {e}")
            return jsonify({"error": f"Error de API externa: {str(e)}"}), 500
    
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

def generate_intelligent_ai_response(prompt, api_key):
    """Genera respuesta usando IA externa siguiendo EXACTAMENTE el método exitoso del compañero"""
    try:
        # 🔑 DEBUGGING: Verificar API Key
        print(f"🔑 API Key presente: {bool(api_key)}")
        print(f"🔑 Primeros 10 chars: {api_key[:10] if api_key else 'N/A'}")
        
        # Obtener contexto académico completo (como hace el compañero)
        df = data_processor.df
        if df is None or df.empty:
            return "Lo siento, no tengo acceso a los datos académicos en este momento."
        
        # Obtener datos reales para el contexto (muestra de estudiantes)
        sample_students = df.head(10).to_dict('records')
        at_risk_students = df[df['rendimiento_riesgo'] == 1].head(5).to_dict('records')
        stats = data_processor.get_statistics()
        
        # Crear contexto simplificado para evitar problemas de serialización
        contexto_academico = {
            "total_estudiantes": len(df),
            "estudiantes_en_riesgo": len(df[df['rendimiento_riesgo'] == 1]),
            "promedio_general": float(stats['promedio_general']),
            "promedio_asistencia": float(stats['promedio_asistencia']),
            "carreras": list(stats['carreras']),
            "descripcion_sistema": {
                "objetivo": "Sistema de predicción de riesgo académico",
                "variables_principales": [
                    "calificaciones_anteriores (4.0-10.0)",
                    "asistencia_porcentaje (0-100%)",
                    "participacion_clase (1-5)",
                    "horas_estudio_semanal (1-25)",
                    "nivel_socioeconomico (Bajo/Medio/Alto)",
                    "rendimiento_riesgo (0=Sin riesgo, 1=En riesgo)"
                ]
            }
        }
        
        # Agregar muestra de estudiantes de forma segura
        try:
            sample_data = []
            for _, student in df.head(5).iterrows():
                student_dict = {
                    "nombre": str(student['nombre']),
                    "apellido": str(student['apellido']),
                    "carrera": str(student['carrera']),
                    "semestre": int(student['semestre']),
                    "promedio": float(student.get('promedio_general', student.get('calificaciones_anteriores', 0))),
                    "asistencia": int(student['asistencia_porcentaje']),
                    "riesgo": int(student['rendimiento_riesgo'])
                }
                sample_data.append(student_dict)
            contexto_academico["muestra_estudiantes"] = sample_data
        except Exception as e:
            print(f"Error agregando muestra de estudiantes: {e}")
            contexto_academico["muestra_estudiantes"] = []
        
        # 🧠 PROMPT ENGINEERING NATURAL Y CONVERSACIONAL
        context = f"""Eres un coordinador académico amigable y cercano de la Universidad Tecnosur. Hablas de manera natural, como si fueras una persona real conversando con un colega. NO uses formato estructurado, listas con viñetas, ni emojis excesivos. Responde como lo haría un coordinador académico en una conversación normal.

DATOS ACADÉMICOS ACTUALES:
{json.dumps(contexto_academico, indent=2, ensure_ascii=False)}

INSTRUCCIONES IMPORTANTES:
- Habla de manera natural y conversacional, como una persona real
- NO uses listas con viñetas (•) ni formato estructurado
- NO uses muchos emojis, máximo 1-2 por respuesta
- Responde como si estuvieras hablando cara a cara con alguien
- Si te saludan, saluda de vuelta de manera natural y pregunta cómo puedes ayudar
- Sé directo y claro, pero mantén un tono amigable
- Cuando des números o estadísticas, hazlo de manera natural en el texto

EJEMPLOS DE CÓMO DEBES RESPONDER:
- "¡Hola! Todo bien por aquí, trabajando con los datos de nuestros estudiantes. ¿En qué puedo ayudarte hoy?"
- "Carlos sí está en riesgo académico. Su promedio es de 5.2 y solo asiste al 65% de las clases, lo cual me preocupa bastante."
- "En Medicina tenemos 89 estudiantes y 12 están en riesgo alto, eso es como un 13.5% aproximadamente."
- "María está muy bien, tiene un promedio de 8.7 en Ingeniería y su asistencia es excelente."
- "No encuentro información sobre ese estudiante en nuestros registros actuales, ¿podrías verificar el nombre?"

EJEMPLOS DE CÓMO NO DEBES RESPONDER:
- NO: "📊 Análisis General del Sistema" 
- NO: "• Total de estudiantes: 1000"
- NO: "🎓 Distribución por Carreras:"
- NO: Listas estructuradas con viñetas
- NO: Formato de reporte técnico

Recuerda: Eres una persona real hablando de manera natural, no un sistema automatizado."""

        # 📊 DEBUGGING: Verificar contexto
        print(f"📊 Contexto generado (primeros 300 chars): {context[:300]}...")
        print(f"📈 Tamaño del contexto: {len(context)} caracteres")
        
        # 📡 DEBUGGING: Preparar request
        print("📡 Enviando request a OpenRouter...")
        print(f"🎯 Modelo: deepseek/deepseek-r1:free")
        print(f"🌡️ Temperature: 0.7")
        print(f"🎛️ Max tokens: 500")
        
        # Llamar a la API con EXACTAMENTE la misma configuración del compañero
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-r1:free",
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,  # 🌡️ EXACTO como el compañero
                "max_tokens": 500    # 📏 EXACTO como el compañero
            },
            timeout=30
        )
        
        # 📥 DEBUGGING: Verificar respuesta
        print(f"📥 Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Error de API: {response.status_code}")
            print(f"📄 Respuesta completa: {response.text}")
            raise Exception(f"Error de API: {response.status_code} - {response.text}")
        
        result = response.json()
        print(f"📥 Respuesta completa de API: {json.dumps(result, indent=2)}")
        
        # Extraer respuesta EXACTAMENTE como el compañero
        if result.get('error'):
            raise Exception(result['error'].get('message', 'Error en la API de IA'))
        
        ai_response = (
            result.get('choices', [{}])[0].get('message', {}).get('content') or
            result.get('choices', [{}])[0].get('text') or
            "No se pudo generar una respuesta."
        )
        
        print(f"✅ Respuesta extraída: {ai_response}")
        
        return ai_response.strip()
        
    except Exception as e:
        print(f"❌ Error en IA externa: {e}")
        print(f"🔍 Tipo de error: {type(e).__name__}")
        raise e

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
