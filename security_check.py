#!/usr/bin/env python3
"""
🔒 Script de Verificación de Seguridad - Tecnosur Predicción
Verifica que no haya secretos expuestos en el proyecto
"""

import os
import re
import sys
from pathlib import Path

class SecurityChecker:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues_found = []
        
        # Patrones de secretos comunes
        self.secret_patterns = {
            'OpenRouter API Key': r'sk-or-v1-[a-zA-Z0-9]{20,}',
            'OpenAI API Key': r'sk-[a-zA-Z0-9]{48}',
            'Generic API Key': r'api[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9]{20,}',
            'Bearer Token': r'Bearer\s+[a-zA-Z0-9\-_]{20,}',
            'Password': r'password["\s]*[:=]["\s]*["\'][^"\']{8,}["\']',
            'Secret': r'secret["\s]*[:=]["\s]*["\'][^"\']{8,}["\']',
            'Token': r'token["\s]*[:=]["\s]*["\'][^"\']{20,}["\']',
        }
        
        # Archivos a excluir
        self.exclude_patterns = [
            r'\.git/',
            r'__pycache__/',
            r'\.env',
            r'node_modules/',
            r'\.pyc$',
            r'security_check\.py$',
            r'SECURITY_CHECKLIST\.md$',
            r'\.pre-commit-config\.yaml$'
        ]
        
        # Extensiones de archivos a verificar
        self.include_extensions = ['.py', '.js', '.json', '.yaml', '.yml', '.md', '.txt', '.sh']

    def should_exclude_file(self, file_path):
        """Verifica si un archivo debe ser excluido de la verificación"""
        file_str = str(file_path)
        return any(re.search(pattern, file_str) for pattern in self.exclude_patterns)

    def check_file_for_secrets(self, file_path):
        """Verifica un archivo específico en busca de secretos"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for secret_type, pattern in self.secret_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.issues_found.append({
                        'file': str(file_path),
                        'line': line_num,
                        'type': secret_type,
                        'match': match.group()[:50] + '...' if len(match.group()) > 50 else match.group()
                    })
                    
        except Exception as e:
            print(f"⚠️ Error leyendo {file_path}: {e}")

    def check_env_file_security(self):
        """Verifica la configuración del archivo .env"""
        env_file = self.project_root / '.env'
        gitignore_file = self.project_root / '.gitignore'
        
        # Verificar que .env existe
        if not env_file.exists():
            self.issues_found.append({
                'file': '.env',
                'line': 0,
                'type': 'Missing File',
                'match': 'Archivo .env no encontrado'
            })
        
        # Verificar que .env está en .gitignore
        if gitignore_file.exists():
            with open(gitignore_file, 'r') as f:
                gitignore_content = f.read()
                if '.env' not in gitignore_content:
                    self.issues_found.append({
                        'file': '.gitignore',
                        'line': 0,
                        'type': 'Missing Protection',
                        'match': '.env no está en .gitignore'
                    })

    def check_code_for_env_usage(self):
        """Verifica que el código use variables de entorno correctamente"""
        python_files = list(self.project_root.rglob('*.py'))
        
        for py_file in python_files:
            if self.should_exclude_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Buscar uso correcto de os.getenv
                if 'OPENROUTER_API_KEY' in content:
                    if 'os.getenv(' not in content and 'os.environ.get(' not in content:
                        self.issues_found.append({
                            'file': str(py_file),
                            'line': 0,
                            'type': 'Incorrect Usage',
                            'match': 'API key referenciada sin usar os.getenv()'
                        })
                        
            except Exception as e:
                print(f"⚠️ Error verificando {py_file}: {e}")

    def scan_project(self):
        """Escanea todo el proyecto en busca de problemas de seguridad"""
        print("🔍 Iniciando verificación de seguridad...")
        print(f"📁 Directorio: {self.project_root}")
        
        # Verificar configuración de .env
        self.check_env_file_security()
        
        # Verificar uso correcto de variables de entorno
        self.check_code_for_env_usage()
        
        # Escanear archivos en busca de secretos
        files_scanned = 0
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file() and not self.should_exclude_file(file_path):
                if file_path.suffix in self.include_extensions:
                    self.check_file_for_secrets(file_path)
                    files_scanned += 1
        
        print(f"📊 Archivos escaneados: {files_scanned}")
        return len(self.issues_found) == 0

    def generate_report(self):
        """Genera un reporte de los problemas encontrados"""
        if not self.issues_found:
            print("\n✅ ¡EXCELENTE! No se encontraron problemas de seguridad.")
            print("🔒 El proyecto está configurado correctamente.")
            return True
        
        print(f"\n🚨 Se encontraron {len(self.issues_found)} problemas de seguridad:")
        print("=" * 60)
        
        for i, issue in enumerate(self.issues_found, 1):
            print(f"\n{i}. 📁 Archivo: {issue['file']}")
            print(f"   📍 Línea: {issue['line']}")
            print(f"   🏷️ Tipo: {issue['type']}")
            print(f"   🔍 Encontrado: {issue['match']}")
        
        print("\n" + "=" * 60)
        print("🛠️ ACCIONES RECOMENDADAS:")
        print("1. Mover cualquier secreto hardcodeado al archivo .env")
        print("2. Asegurar que .env esté en .gitignore")
        print("3. Usar os.getenv() para acceder a variables de entorno")
        print("4. Rotar cualquier clave que haya sido expuesta")
        
        return False

def main():
    """Función principal"""
    print("🔒 VERIFICADOR DE SEGURIDAD - TECNOSUR PREDICCIÓN")
    print("=" * 50)
    
    checker = SecurityChecker()
    is_secure = checker.scan_project()
    success = checker.generate_report()
    
    if success:
        print("\n🎉 Verificación completada exitosamente!")
        sys.exit(0)
    else:
        print("\n⚠️ Se encontraron problemas de seguridad que requieren atención.")
        sys.exit(1)

if __name__ == "__main__":
    main()
