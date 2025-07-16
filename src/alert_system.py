import pandas as pd
from datetime import datetime, timedelta
import json
import os

class AlertSystem:
    def __init__(self):
        self.alerts_file = 'data/alerts.json'
        self.alerts = self.load_alerts()
        self.alert_types = {
            'BAJO_RENDIMIENTO': 'Bajo Rendimiento Académico',
            'BAJA_ASISTENCIA': 'Baja Asistencia',
            'BAJA_PARTICIPACION': 'Baja Participación',
            'RIESGO_ALTO': 'Estudiante en Riesgo Alto',
            'NUEVO_ARCHIVO': 'Nuevo Archivo Subido',
            'CAMBIO_ESTADO': 'Cambio de Estado Académico'
        }
    
    def load_alerts(self):
        """Carga las alertas desde el archivo JSON"""
        try:
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            print(f"Error cargando alertas: {e}")
            return []
    
    def save_alerts(self):
        """Guarda las alertas en el archivo JSON"""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.alerts_file), exist_ok=True)
            
            with open(self.alerts_file, 'w', encoding='utf-8') as f:
                json.dump(self.alerts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error guardando alertas: {e}")
    
    def create_alert(self, alert_type, student_id, student_name, message, priority='MEDIUM'):
        """Crea una nueva alerta"""
        alert = {
            'id': len(self.alerts) + 1,
            'type': alert_type,
            'type_name': self.alert_types.get(alert_type, alert_type),
            'student_id': student_id,
            'student_name': student_name,
            'message': message,
            'priority': priority,  # LOW, MEDIUM, HIGH, CRITICAL
            'timestamp': datetime.now().isoformat(),
            'read': False,
            'resolved': False
        }
        
        self.alerts.append(alert)
        self.save_alerts()
        
        return alert
    
    def check_student_alerts(self, student_data):
        """Verifica si un estudiante necesita alertas"""
        alerts_created = []
        
        student_id = student_data.get('id_estudiante')
        student_name = f"{student_data.get('nombre', '')} {student_data.get('apellido', '')}"
        
        # Verificar bajo rendimiento
        if student_data.get('calificaciones_anteriores', 10) < 5.0:
            alert = self.create_alert(
                'BAJO_RENDIMIENTO',
                student_id,
                student_name,
                f"El estudiante {student_name} tiene un promedio de {student_data.get('calificaciones_anteriores')}, por debajo del mínimo requerido.",
                'HIGH'
            )
            alerts_created.append(alert)
        
        # Verificar baja asistencia
        if student_data.get('asistencia_porcentaje', 100) < 60:
            alert = self.create_alert(
                'BAJA_ASISTENCIA',
                student_id,
                student_name,
                f"El estudiante {student_name} tiene una asistencia del {student_data.get('asistencia_porcentaje')}%, por debajo del 60% mínimo.",
                'HIGH'
            )
            alerts_created.append(alert)
        
        # Verificar baja participación
        if student_data.get('participacion_clase', 5) <= 2:
            alert = self.create_alert(
                'BAJA_PARTICIPACION',
                student_id,
                student_name,
                f"El estudiante {student_name} tiene una participación muy baja ({student_data.get('participacion_clase')}/5).",
                'MEDIUM'
            )
            alerts_created.append(alert)
        
        # Verificar riesgo alto
        if student_data.get('rendimiento_riesgo', 0) == 1:
            alert = self.create_alert(
                'RIESGO_ALTO',
                student_id,
                student_name,
                f"El estudiante {student_name} ha sido clasificado como de ALTO RIESGO académico.",
                'CRITICAL'
            )
            alerts_created.append(alert)
        
        return alerts_created
    
    def bulk_check_alerts(self, students_df):
        """Verifica alertas para múltiples estudiantes"""
        all_alerts = []
        
        for _, student in students_df.iterrows():
            student_alerts = self.check_student_alerts(student.to_dict())
            all_alerts.extend(student_alerts)
        
        return all_alerts
    
    def get_recent_alerts(self, days=7, limit=50):
        """Obtiene alertas recientes"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_date
        ]
        
        # Ordenar por timestamp descendente
        recent_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return recent_alerts[:limit]
    
    def get_unread_alerts(self):
        """Obtiene alertas no leídas"""
        return [alert for alert in self.alerts if not alert['read']]
    
    def get_alerts_by_priority(self, priority):
        """Obtiene alertas por prioridad"""
        return [alert for alert in self.alerts if alert['priority'] == priority]
    
    def get_alerts_by_student(self, student_id):
        """Obtiene alertas de un estudiante específico"""
        return [alert for alert in self.alerts if alert['student_id'] == student_id]
    
    def mark_alert_read(self, alert_id):
        """Marca una alerta como leída"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['read'] = True
                self.save_alerts()
                return True
        return False
    
    def mark_alert_resolved(self, alert_id):
        """Marca una alerta como resuelta"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['resolved'] = True
                alert['read'] = True
                self.save_alerts()
                return True
        return False
    
    def get_alert_statistics(self):
        """Obtiene estadísticas de alertas"""
        total_alerts = len(self.alerts)
        unread_alerts = len(self.get_unread_alerts())
        resolved_alerts = len([a for a in self.alerts if a['resolved']])
        
        priority_counts = {}
        for priority in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            priority_counts[priority] = len(self.get_alerts_by_priority(priority))
        
        type_counts = {}
        for alert in self.alerts:
            alert_type = alert['type']
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'unread_alerts': unread_alerts,
            'resolved_alerts': resolved_alerts,
            'pending_alerts': total_alerts - resolved_alerts,
            'priority_distribution': priority_counts,
            'type_distribution': type_counts
        }
    
    def create_file_upload_alert(self, filename, records_count):
        """Crea alerta por subida de archivo"""
        alert = self.create_alert(
            'NUEVO_ARCHIVO',
            None,
            'Sistema',
            f"Se ha subido un nuevo archivo: {filename} con {records_count} registros.",
            'MEDIUM'
        )
        return alert
    
    def simulate_email_alert(self, alert):
        """Simula el envío de alerta por email"""
        email_content = {
            'to': 'coordinador@universidad.edu',
            'subject': f"[ALERTA ACADÉMICA] {alert['type_name']}",
            'body': f"""
            Estimado Coordinador Académico,
            
            Se ha generado una nueva alerta en el sistema:
            
            Tipo: {alert['type_name']}
            Estudiante: {alert['student_name']}
            Mensaje: {alert['message']}
            Prioridad: {alert['priority']}
            Fecha: {alert['timestamp']}
            
            Por favor, revise el sistema para más detalles.
            
            Saludos,
            Sistema de Alertas Académicas
            """,
            'timestamp': datetime.now().isoformat(),
            'sent': True
        }
        
        # En un sistema real, aquí se enviaría el email
        print(f"📧 Email simulado enviado: {email_content['subject']}")
        
        return email_content
    
    def get_dashboard_alerts(self):
        """Obtiene alertas para mostrar en el dashboard"""
        recent_alerts = self.get_recent_alerts(days=30, limit=10)
        critical_alerts = [a for a in recent_alerts if a['priority'] == 'CRITICAL']
        high_alerts = [a for a in recent_alerts if a['priority'] == 'HIGH']
        
        return {
            'recent': recent_alerts,
            'critical': critical_alerts,
            'high_priority': high_alerts,
            'unread_count': len(self.get_unread_alerts())
        }
    
    def generate_alert_recommendations(self, student_data):
        """Genera recomendaciones basadas en las alertas del estudiante"""
        recommendations = []
        
        if student_data.get('calificaciones_anteriores', 10) < 5.0:
            recommendations.append({
                'type': 'ACADEMICO',
                'title': 'Refuerzo Académico',
                'description': 'Se recomienda tutoría personalizada y plan de mejoramiento académico.',
                'priority': 'HIGH'
            })
        
        if student_data.get('asistencia_porcentaje', 100) < 60:
            recommendations.append({
                'type': 'ASISTENCIA',
                'title': 'Plan de Asistencia',
                'description': 'Implementar seguimiento diario de asistencia y contacto con familia.',
                'priority': 'HIGH'
            })
        
        if student_data.get('participacion_clase', 5) <= 2:
            recommendations.append({
                'type': 'PARTICIPACION',
                'title': 'Estrategias de Participación',
                'description': 'Aplicar técnicas para fomentar la participación activa en clase.',
                'priority': 'MEDIUM'
            })
        
        if student_data.get('horas_estudio_semanal', 20) < 8:
            recommendations.append({
                'type': 'ESTUDIO',
                'title': 'Técnicas de Estudio',
                'description': 'Enseñar métodos de estudio efectivos y crear horario de estudio.',
                'priority': 'MEDIUM'
            })
        
        return recommendations

# Instancia global del sistema de alertas
alert_system = AlertSystem()
