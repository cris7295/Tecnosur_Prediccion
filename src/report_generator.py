import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

class ReportGenerator:
    def __init__(self):
        self.report_styles = {
            'title_size': 16,
            'subtitle_size': 14,
            'text_size': 12,
            'colors': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        }
    
    def generate_student_report(self, student_data, predictions=None):
        """Genera reporte individual de estudiante"""
        report = {
            'student_info': self._format_student_info(student_data),
            'academic_summary': self._generate_academic_summary(student_data),
            'risk_analysis': self._generate_risk_analysis(student_data, predictions),
            'recommendations': self._generate_recommendations(student_data),
            'charts': self._generate_student_charts(student_data),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return report
    
    def _format_student_info(self, student_data):
        """Formatea información básica del estudiante"""
        return {
            'nombre_completo': f"{student_data.get('nombre', '')} {student_data.get('apellido', '')}",
            'id': student_data.get('id_estudiante', ''),
            'edad': student_data.get('edad', ''),
            'genero': student_data.get('genero', ''),
            'carrera': student_data.get('carrera', ''),
            'semestre': student_data.get('semestre', ''),
            'email': student_data.get('email', ''),
            'telefono': student_data.get('telefono', ''),
            'fecha_ingreso': student_data.get('fecha_ingreso', ''),
            'estado_academico': student_data.get('estado_academico', '')
        }
    
    def _generate_academic_summary(self, student_data):
        """Genera resumen académico"""
        return {
            'promedio_general': student_data.get('promedio_general', 0),
            'calificaciones_anteriores': student_data.get('calificaciones_anteriores', 0),
            'asistencia_porcentaje': student_data.get('asistencia_porcentaje', 0),
            'participacion_clase': student_data.get('participacion_clase', 0),
            'horas_estudio_semanal': student_data.get('horas_estudio_semanal', 0),
            'creditos_aprobados': student_data.get('creditos_aprobados', 0),
            'creditos_totales': student_data.get('creditos_totales', 0),
            'porcentaje_avance': round((student_data.get('creditos_aprobados', 0) / max(student_data.get('creditos_totales', 1), 1)) * 100, 1),
            'notas_por_materia': {
                'Matemáticas': student_data.get('nota_matematicas', 0),
                'Ciencias': student_data.get('nota_ciencias', 0),
                'Lenguaje': student_data.get('nota_lenguaje', 0),
                'Historia': student_data.get('nota_historia', 0)
            }
        }
    
    def _generate_risk_analysis(self, student_data, predictions=None):
        """Genera análisis de riesgo"""
        riesgo_actual = student_data.get('rendimiento_riesgo', 0)
        nivel_riesgo = 'Alto' if riesgo_actual == 1 else 'Bajo'
        
        # Factores de riesgo
        factores_riesgo = []
        
        if student_data.get('calificaciones_anteriores', 10) < 6.0:
            factores_riesgo.append('Promedio académico bajo')
        
        if student_data.get('asistencia_porcentaje', 100) < 70:
            factores_riesgo.append('Baja asistencia a clases')
        
        if student_data.get('participacion_clase', 5) <= 2:
            factores_riesgo.append('Baja participación en clase')
        
        if student_data.get('horas_estudio_semanal', 20) < 8:
            factores_riesgo.append('Pocas horas de estudio')
        
        # Factores protectores
        factores_protectores = []
        
        if student_data.get('calificaciones_anteriores', 0) >= 8.0:
            factores_protectores.append('Excelente rendimiento académico')
        
        if student_data.get('asistencia_porcentaje', 0) >= 90:
            factores_protectores.append('Excelente asistencia')
        
        if student_data.get('participacion_clase', 0) >= 4:
            factores_protectores.append('Alta participación en clase')
        
        if student_data.get('horas_estudio_semanal', 0) >= 15:
            factores_protectores.append('Dedicación al estudio')
        
        return {
            'nivel_riesgo': nivel_riesgo,
            'riesgo_numerico': riesgo_actual,
            'factores_riesgo': factores_riesgo,
            'factores_protectores': factores_protectores,
            'predicciones': predictions or {},
            'motivo_riesgo': student_data.get('motivo_riesgo', '')
        }
    
    def _generate_recommendations(self, student_data):
        """Genera recomendaciones personalizadas"""
        recommendations = []
        
        # Recomendaciones académicas
        if student_data.get('calificaciones_anteriores', 10) < 6.0:
            recommendations.append({
                'categoria': 'Académico',
                'titulo': 'Plan de Refuerzo Académico',
                'descripcion': 'Implementar sesiones de tutoría personalizada y técnicas de estudio efectivas.',
                'prioridad': 'Alta',
                'acciones': [
                    'Asignar tutor académico',
                    'Crear plan de estudio personalizado',
                    'Evaluaciones de seguimiento semanales'
                ]
            })
        
        # Recomendaciones de asistencia
        if student_data.get('asistencia_porcentaje', 100) < 70:
            recommendations.append({
                'categoria': 'Asistencia',
                'titulo': 'Mejoramiento de Asistencia',
                'descripcion': 'Establecer estrategias para mejorar la asistencia regular a clases.',
                'prioridad': 'Alta',
                'acciones': [
                    'Contacto con familia/acudiente',
                    'Identificar barreras para asistencia',
                    'Seguimiento diario de asistencia'
                ]
            })
        
        # Recomendaciones de participación
        if student_data.get('participacion_clase', 5) <= 2:
            recommendations.append({
                'categoria': 'Participación',
                'titulo': 'Fomento de Participación',
                'descripcion': 'Aplicar estrategias para aumentar la participación activa en clase.',
                'prioridad': 'Media',
                'acciones': [
                    'Técnicas de participación grupal',
                    'Presentaciones cortas',
                    'Gamificación del aprendizaje'
                ]
            })
        
        # Recomendaciones de estudio
        if student_data.get('horas_estudio_semanal', 20) < 10:
            recommendations.append({
                'categoria': 'Hábitos de Estudio',
                'titulo': 'Mejoramiento de Hábitos de Estudio',
                'descripcion': 'Desarrollar rutinas de estudio efectivas y constantes.',
                'prioridad': 'Media',
                'acciones': [
                    'Crear horario de estudio',
                    'Enseñar técnicas de concentración',
                    'Establecer metas de estudio semanales'
                ]
            })
        
        # Recomendaciones positivas
        if student_data.get('calificaciones_anteriores', 0) >= 8.0:
            recommendations.append({
                'categoria': 'Reconocimiento',
                'titulo': 'Mantener Excelencia Académica',
                'descripcion': 'Continuar con el excelente desempeño y considerar actividades de liderazgo.',
                'prioridad': 'Baja',
                'acciones': [
                    'Participar en programas de liderazgo',
                    'Mentorear a otros estudiantes',
                    'Explorar oportunidades de investigación'
                ]
            })
        
        return recommendations
    
    def _generate_student_charts(self, student_data):
        """Genera gráficos para el reporte del estudiante"""
        charts = {}
        
        # Gráfico de radar de competencias
        categories = ['Calificaciones', 'Asistencia', 'Participación', 'Horas Estudio']
        values = [
            student_data.get('calificaciones_anteriores', 0),
            student_data.get('asistencia_porcentaje', 0) / 10,  # Normalizar a escala 0-10
            student_data.get('participacion_clase', 0) * 2,     # Escalar de 1-5 a 2-10
            min(student_data.get('horas_estudio_semanal', 0) / 2.5, 10)  # Normalizar a escala 0-10
        ]
        
        charts['radar'] = {
            'categories': categories,
            'values': values,
            'title': 'Perfil Académico del Estudiante'
        }
        
        # Gráfico de notas por materia
        materias = ['Matemáticas', 'Ciencias', 'Lenguaje', 'Historia']
        notas = [
            student_data.get('nota_matematicas', 0),
            student_data.get('nota_ciencias', 0),
            student_data.get('nota_lenguaje', 0),
            student_data.get('nota_historia', 0)
        ]
        
        charts['notas_materias'] = {
            'materias': materias,
            'notas': notas,
            'title': 'Rendimiento por Materia'
        }
        
        # Progreso de créditos
        creditos_aprobados = student_data.get('creditos_aprobados', 0)
        creditos_totales = student_data.get('creditos_totales', 1)
        creditos_pendientes = creditos_totales - creditos_aprobados
        
        charts['progreso_creditos'] = {
            'aprobados': creditos_aprobados,
            'pendientes': creditos_pendientes,
            'total': creditos_totales,
            'porcentaje': round((creditos_aprobados / creditos_totales) * 100, 1),
            'title': 'Progreso Académico'
        }
        
        return charts
    
    def generate_general_report(self, students_df, statistics):
        """Genera reporte general del sistema y crea archivo PDF en memoria"""
        report = {
            'summary': self._generate_general_summary(statistics),
            'risk_distribution': self._generate_risk_distribution(students_df),
            'career_analysis': self._generate_career_analysis(students_df),
            'semester_analysis': self._generate_semester_analysis(students_df),
            'performance_trends': self._generate_performance_trends(students_df),
            'recommendations': self._generate_institutional_recommendations(statistics),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Generar PDF en memoria para descarga
        pdf_buffer = self._create_pdf_in_memory(report, students_df)
        report['pdf_buffer'] = pdf_buffer
        report['pdf_filename'] = f"reporte_general_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return report
    
    def _create_pdf_in_memory(self, report_data, students_df):
        """Crea PDF en memoria para descarga directa"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            import io
            
            # Crear buffer en memoria
            buffer = io.BytesIO()
            
            # Crear documento PDF en memoria
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Título
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Centrado
            )
            story.append(Paragraph("📊 REPORTE GENERAL DEL SISTEMA ACADÉMICO", title_style))
            story.append(Spacer(1, 20))
            
            # Fecha de generación
            story.append(Paragraph(f"<b>Fecha de Generación:</b> {report_data['timestamp']}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Resumen Ejecutivo
            story.append(Paragraph("🎯 RESUMEN EJECUTIVO", styles['Heading2']))
            summary = report_data['summary']
            
            summary_data = [
                ['Métrica', 'Valor'],
                ['Total de Estudiantes', str(summary['total_estudiantes'])],
                ['Estudiantes en Riesgo', str(summary['estudiantes_riesgo'])],
                ['Porcentaje en Riesgo', f"{summary['porcentaje_riesgo']}%"],
                ['Promedio General', f"{summary['promedio_general']:.2f}"],
                ['Promedio de Asistencia', f"{summary['promedio_asistencia']:.1f}%"],
                ['Total de Carreras', str(summary['total_carreras'])],
                ['Semestres Activos', str(summary['semestres_activos'])]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 30))
            
            # Distribución de Riesgo por Carrera
            story.append(Paragraph("📚 DISTRIBUCIÓN DE RIESGO POR CARRERA", styles['Heading2']))
            
            risk_by_career = report_data['risk_distribution']['por_carrera']
            career_data = [['Carrera', 'Total Estudiantes', 'En Riesgo', '% Riesgo']]
            
            for carrera, stats in risk_by_career.items():
                total = int(stats['count'])
                en_riesgo = int(stats['sum'])
                porcentaje = f"{stats['mean']*100:.1f}%"
                career_data.append([carrera, str(total), str(en_riesgo), porcentaje])
            
            career_table = Table(career_data)
            career_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(career_table)
            story.append(Spacer(1, 30))
            
            # Recomendaciones Institucionales
            story.append(Paragraph("💡 RECOMENDACIONES INSTITUCIONALES", styles['Heading2']))
            
            for i, rec in enumerate(report_data['recommendations'], 1):
                story.append(Paragraph(f"<b>{i}. {rec['titulo']}</b>", styles['Heading3']))
                story.append(Paragraph(f"<b>Categoría:</b> {rec['categoria']}", styles['Normal']))
                story.append(Paragraph(f"<b>Descripción:</b> {rec['descripcion']}", styles['Normal']))
                
                if 'acciones' in rec:
                    story.append(Paragraph("<b>Acciones Recomendadas:</b>", styles['Normal']))
                    for accion in rec['acciones']:
                        story.append(Paragraph(f"• {accion}", styles['Normal']))
                
                story.append(Spacer(1, 15))
            
            # Estadísticas Adicionales
            story.append(Paragraph("📈 ESTADÍSTICAS DETALLADAS", styles['Heading2']))
            
            # Top 5 estudiantes en riesgo
            students_at_risk = students_df[students_df['rendimiento_riesgo'] == 1].head(5)
            if not students_at_risk.empty:
                story.append(Paragraph("<b>Estudiantes que Requieren Atención Inmediata:</b>", styles['Heading3']))
                
                risk_students_data = [['Nombre', 'Carrera', 'Semestre', 'Promedio', 'Asistencia']]
                for _, student in students_at_risk.iterrows():
                    nombre = f"{student['nombre']} {student['apellido']}"
                    carrera = student['carrera']
                    semestre = str(student['semestre'])
                    promedio = f"{student['calificaciones_anteriores']:.2f}"
                    asistencia = f"{student['asistencia_porcentaje']:.1f}%"
                    risk_students_data.append([nombre, carrera, semestre, promedio, asistencia])
                
                risk_table = Table(risk_students_data)
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(risk_table)
            
            # Generar PDF en memoria
            doc.build(story)
            
            # Obtener contenido del buffer
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error generando PDF en memoria: {e}")
            return None
    
    def _create_pdf_report(self, report_data, students_df):
        """Crea archivo PDF físico del reporte"""
        import os
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        # Crear directorio de reportes si no existe
        reports_dir = 'reportes'
        os.makedirs(reports_dir, exist_ok=True)
        
        # Nombre del archivo con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'reporte_general_{timestamp}.pdf'
        filepath = os.path.join(reports_dir, filename)
        
        try:
            # Crear documento PDF
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Título
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Centrado
            )
            story.append(Paragraph("📊 REPORTE GENERAL DEL SISTEMA ACADÉMICO", title_style))
            story.append(Spacer(1, 20))
            
            # Fecha de generación
            story.append(Paragraph(f"<b>Fecha de Generación:</b> {report_data['timestamp']}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Resumen Ejecutivo
            story.append(Paragraph("🎯 RESUMEN EJECUTIVO", styles['Heading2']))
            summary = report_data['summary']
            
            summary_data = [
                ['Métrica', 'Valor'],
                ['Total de Estudiantes', str(summary['total_estudiantes'])],
                ['Estudiantes en Riesgo', str(summary['estudiantes_riesgo'])],
                ['Porcentaje en Riesgo', f"{summary['porcentaje_riesgo']}%"],
                ['Promedio General', f"{summary['promedio_general']:.2f}"],
                ['Promedio de Asistencia', f"{summary['promedio_asistencia']:.1f}%"],
                ['Total de Carreras', str(summary['total_carreras'])],
                ['Semestres Activos', str(summary['semestres_activos'])]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 30))
            
            # Distribución de Riesgo por Carrera
            story.append(Paragraph("📚 DISTRIBUCIÓN DE RIESGO POR CARRERA", styles['Heading2']))
            
            risk_by_career = report_data['risk_distribution']['por_carrera']
            career_data = [['Carrera', 'Total Estudiantes', 'En Riesgo', '% Riesgo']]
            
            for carrera, stats in risk_by_career.items():
                total = int(stats['count'])
                en_riesgo = int(stats['sum'])
                porcentaje = f"{stats['mean']*100:.1f}%"
                career_data.append([carrera, str(total), str(en_riesgo), porcentaje])
            
            career_table = Table(career_data)
            career_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(career_table)
            story.append(Spacer(1, 30))
            
            # Recomendaciones Institucionales
            story.append(Paragraph("💡 RECOMENDACIONES INSTITUCIONALES", styles['Heading2']))
            
            for i, rec in enumerate(report_data['recommendations'], 1):
                story.append(Paragraph(f"<b>{i}. {rec['titulo']}</b>", styles['Heading3']))
                story.append(Paragraph(f"<b>Categoría:</b> {rec['categoria']}", styles['Normal']))
                story.append(Paragraph(f"<b>Descripción:</b> {rec['descripcion']}", styles['Normal']))
                
                if 'acciones' in rec:
                    story.append(Paragraph("<b>Acciones Recomendadas:</b>", styles['Normal']))
                    for accion in rec['acciones']:
                        story.append(Paragraph(f"• {accion}", styles['Normal']))
                
                story.append(Spacer(1, 15))
            
            # Estadísticas Adicionales
            story.append(Paragraph("📈 ESTADÍSTICAS DETALLADAS", styles['Heading2']))
            
            # Top 5 estudiantes en riesgo
            students_at_risk = students_df[students_df['rendimiento_riesgo'] == 1].head(5)
            if not students_at_risk.empty:
                story.append(Paragraph("<b>Estudiantes que Requieren Atención Inmediata:</b>", styles['Heading3']))
                
                risk_students_data = [['Nombre', 'Carrera', 'Semestre', 'Promedio', 'Asistencia']]
                for _, student in students_at_risk.iterrows():
                    nombre = f"{student['nombre']} {student['apellido']}"
                    carrera = student['carrera']
                    semestre = str(student['semestre'])
                    promedio = f"{student['calificaciones_anteriores']:.2f}"
                    asistencia = f"{student['asistencia_porcentaje']:.1f}%"
                    risk_students_data.append([nombre, carrera, semestre, promedio, asistencia])
                
                risk_table = Table(risk_students_data)
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(risk_table)
            
            # Generar PDF
            doc.build(story)
            
            return filepath
            
        except ImportError:
            # Si reportlab no está disponible, crear un archivo de texto simple
            with open(filepath.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
                f.write("REPORTE GENERAL DEL SISTEMA ACADÉMICO\n")
                f.write("="*50 + "\n\n")
                f.write(f"Fecha de Generación: {report_data['timestamp']}\n\n")
                
                f.write("RESUMEN EJECUTIVO:\n")
                f.write(f"- Total de Estudiantes: {summary['total_estudiantes']}\n")
                f.write(f"- Estudiantes en Riesgo: {summary['estudiantes_riesgo']}\n")
                f.write(f"- Porcentaje en Riesgo: {summary['porcentaje_riesgo']}%\n")
                f.write(f"- Promedio General: {summary['promedio_general']:.2f}\n")
                f.write(f"- Promedio de Asistencia: {summary['promedio_asistencia']:.1f}%\n\n")
                
                f.write("RECOMENDACIONES:\n")
                for i, rec in enumerate(report_data['recommendations'], 1):
                    f.write(f"{i}. {rec['titulo']}\n")
                    f.write(f"   {rec['descripcion']}\n\n")
            
            return filepath.replace('.pdf', '.txt')
        
        except Exception as e:
            print(f"Error generando PDF: {e}")
            # Crear archivo de texto como fallback
            txt_filepath = filepath.replace('.pdf', '.txt')
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write(f"REPORTE GENERAL - {report_data['timestamp']}\n")
                f.write(f"Total Estudiantes: {summary['total_estudiantes']}\n")
                f.write(f"En Riesgo: {summary['estudiantes_riesgo']}\n")
            return txt_filepath
    
    def _generate_general_summary(self, statistics):
        """Genera resumen general"""
        return {
            'total_estudiantes': statistics.get('total_estudiantes', 0),
            'estudiantes_riesgo': statistics.get('estudiantes_riesgo', 0),
            'porcentaje_riesgo': statistics.get('porcentaje_riesgo', 0),
            'promedio_general': statistics.get('promedio_general', 0),
            'promedio_asistencia': statistics.get('promedio_asistencia', 0),
            'total_carreras': len(statistics.get('carreras', [])),
            'semestres_activos': len(statistics.get('semestres', []))
        }
    
    def _generate_risk_distribution(self, students_df):
        """Genera análisis de distribución de riesgo"""
        risk_by_career = students_df.groupby('carrera')['rendimiento_riesgo'].agg(['count', 'sum', 'mean']).round(3)
        risk_by_semester = students_df.groupby('semestre')['rendimiento_riesgo'].agg(['count', 'sum', 'mean']).round(3)
        
        return {
            'por_carrera': risk_by_career.to_dict('index'),
            'por_semestre': risk_by_semester.to_dict('index'),
            'total_riesgo': students_df['rendimiento_riesgo'].sum(),
            'porcentaje_riesgo': round(students_df['rendimiento_riesgo'].mean() * 100, 1)
        }
    
    def _generate_career_analysis(self, students_df):
        """Genera análisis por carrera"""
        career_stats = students_df.groupby('carrera').agg({
            'calificaciones_anteriores': ['mean', 'std', 'count'],
            'asistencia_porcentaje': ['mean', 'std'],
            'participacion_clase': ['mean', 'std'],
            'rendimiento_riesgo': ['sum', 'mean']
        }).round(2)
        
        return career_stats.to_dict('index')
    
    def _generate_semester_analysis(self, students_df):
        """Genera análisis por semestre"""
        semester_stats = students_df.groupby('semestre').agg({
            'calificaciones_anteriores': ['mean', 'std', 'count'],
            'asistencia_porcentaje': ['mean', 'std'],
            'participacion_clase': ['mean', 'std'],
            'rendimiento_riesgo': ['sum', 'mean']
        }).round(2)
        
        return semester_stats.to_dict('index')
    
    def _generate_performance_trends(self, students_df):
        """Genera análisis de tendencias de rendimiento"""
        # Correlaciones entre variables
        numeric_cols = ['calificaciones_anteriores', 'asistencia_porcentaje', 'participacion_clase', 
                       'horas_estudio_semanal', 'rendimiento_riesgo']
        
        correlations = students_df[numeric_cols].corr().round(3)
        
        # Identificar estudiantes de alto y bajo rendimiento
        high_performers = students_df[students_df['calificaciones_anteriores'] >= 8.5]
        low_performers = students_df[students_df['calificaciones_anteriores'] < 6.0]
        
        return {
            'correlaciones': correlations.to_dict(),
            'alto_rendimiento': {
                'count': len(high_performers),
                'caracteristicas': {
                    'asistencia_promedio': high_performers['asistencia_porcentaje'].mean(),
                    'participacion_promedio': high_performers['participacion_clase'].mean(),
                    'horas_estudio_promedio': high_performers['horas_estudio_semanal'].mean()
                }
            },
            'bajo_rendimiento': {
                'count': len(low_performers),
                'caracteristicas': {
                    'asistencia_promedio': low_performers['asistencia_porcentaje'].mean(),
                    'participacion_promedio': low_performers['participacion_clase'].mean(),
                    'horas_estudio_promedio': low_performers['horas_estudio_semanal'].mean()
                }
            }
        }
    
    def _generate_institutional_recommendations(self, statistics):
        """Genera recomendaciones institucionales"""
        recommendations = []
        
        porcentaje_riesgo = statistics.get('porcentaje_riesgo', 0)
        
        if porcentaje_riesgo > 30:
            recommendations.append({
                'categoria': 'Crítico',
                'titulo': 'Alto Porcentaje de Estudiantes en Riesgo',
                'descripcion': f'El {porcentaje_riesgo}% de estudiantes están en riesgo académico.',
                'acciones': [
                    'Implementar programa de intervención temprana',
                    'Reforzar sistema de tutorías',
                    'Revisar metodologías de enseñanza'
                ]
            })
        elif porcentaje_riesgo > 20:
            recommendations.append({
                'categoria': 'Atención',
                'titulo': 'Porcentaje Moderado de Riesgo',
                'descripcion': f'El {porcentaje_riesgo}% de estudiantes requieren atención.',
                'acciones': [
                    'Fortalecer seguimiento académico',
                    'Capacitar docentes en detección temprana',
                    'Mejorar canales de comunicación'
                ]
            })
        
        promedio_general = statistics.get('promedio_general', 0)
        if promedio_general < 7.0:
            recommendations.append({
                'categoria': 'Académico',
                'titulo': 'Promedio General Bajo',
                'descripcion': f'El promedio institucional es {promedio_general}.',
                'acciones': [
                    'Revisar estándares académicos',
                    'Implementar programas de nivelación',
                    'Fortalecer recursos de apoyo académico'
                ]
            })
        
        return recommendations
    
    def export_to_html(self, report_data, report_type='student'):
        """Exporta reporte a HTML"""
        if report_type == 'student':
            return self._generate_student_html(report_data)
        else:
            return self._generate_general_html(report_data)
    
    def _generate_student_html(self, report_data):
        """Genera HTML para reporte de estudiante"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte Académico - {report_data['student_info']['nombre_completo']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #3498db; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .risk-high {{ background-color: #ffebee; border-left: 5px solid #f44336; }}
                .risk-low {{ background-color: #e8f5e8; border-left: 5px solid #4caf50; }}
                .recommendation {{ background-color: #fff3e0; padding: 10px; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Reporte Académico Individual</h1>
                <h2>{report_data['student_info']['nombre_completo']}</h2>
                <p>Generado el: {report_data['timestamp']}</p>
            </div>
            
            <div class="section">
                <h3>Información del Estudiante</h3>
                <table>
                    <tr><td><strong>ID:</strong></td><td>{report_data['student_info']['id']}</td></tr>
                    <tr><td><strong>Carrera:</strong></td><td>{report_data['student_info']['carrera']}</td></tr>
                    <tr><td><strong>Semestre:</strong></td><td>{report_data['student_info']['semestre']}</td></tr>
                    <tr><td><strong>Estado:</strong></td><td>{report_data['student_info']['estado_academico']}</td></tr>
                </table>
            </div>
            
            <div class="section {'risk-high' if report_data['risk_analysis']['nivel_riesgo'] == 'Alto' else 'risk-low'}">
                <h3>Análisis de Riesgo: {report_data['risk_analysis']['nivel_riesgo']}</h3>
                <p><strong>Factores de Riesgo:</strong></p>
                <ul>
                    {''.join([f'<li>{factor}</li>' for factor in report_data['risk_analysis']['factores_riesgo']])}
                </ul>
                <p><strong>Factores Protectores:</strong></p>
                <ul>
                    {''.join([f'<li>{factor}</li>' for factor in report_data['risk_analysis']['factores_protectores']])}
                </ul>
            </div>
            
            <div class="section">
                <h3>Resumen Académico</h3>
                <table>
                    <tr><td><strong>Promedio General:</strong></td><td>{report_data['academic_summary']['promedio_general']}</td></tr>
                    <tr><td><strong>Asistencia:</strong></td><td>{report_data['academic_summary']['asistencia_porcentaje']}%</td></tr>
                    <tr><td><strong>Participación:</strong></td><td>{report_data['academic_summary']['participacion_clase']}/5</td></tr>
                    <tr><td><strong>Horas de Estudio:</strong></td><td>{report_data['academic_summary']['horas_estudio_semanal']} hrs/semana</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h3>Recomendaciones</h3>
                {''.join([f'<div class="recommendation"><h4>{rec["titulo"]}</h4><p>{rec["descripcion"]}</p></div>' for rec in report_data['recommendations']])}
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_general_html(self, report_data):
        """Genera HTML para reporte general"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte General del Sistema</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .stats {{ display: flex; justify-content: space-around; }}
                .stat-box {{ text-align: center; padding: 20px; background-color: #ecf0f1; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Reporte General del Sistema</h1>
                <p>Generado el: {report_data['timestamp']}</p>
            </div>
            
            <div class="section">
                <h3>Resumen Ejecutivo</h3>
                <div class="stats">
                    <div class="stat-box">
                        <h4>{report_data['summary']['total_estudiantes']}</h4>
                        <p>Total Estudiantes</p>
                    </div>
                    <div class="stat-box">
                        <h4>{report_data['summary']['estudiantes_riesgo']}</h4>
                        <p>En Riesgo</p>
                    </div>
                    <div class="stat-box">
                        <h4>{report_data['summary']['porcentaje_riesgo']}%</h4>
                        <p>% en Riesgo</p>
                    </div>
                    <div class="stat-box">
                        <h4>{report_data['summary']['promedio_general']}</h4>
                        <p>Promedio General</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h3>Recomendaciones Institucionales</h3>
                {''.join([f'<div><h4>{rec["titulo"]}</h4><p>{rec["descripcion"]}</p></div>' for rec in report_data['recommendations']])}
            </div>
        </body>
        </html>
        """
        
        return html_template

# Instancia global del generador de reportes
report_generator = ReportGenerator()
