# 🔑 INSTRUCCIONES PARA CONFIGURAR NUEVA API KEY

## 🚨 SITUACIÓN ACTUAL
- ✅ Archivo `.env` eliminado del tracking de Git
- ✅ Historial de Git limpiado (API key anterior eliminada)
- ✅ Repositorio actualizado sin secretos expuestos
- ✅ Herramientas de seguridad implementadas

## 📋 PASOS PARA CONFIGURAR NUEVA API KEY

### 1. 🔑 Generar Nueva API Key
1. Ve a [OpenRouter Dashboard](https://openrouter.ai/keys)
2. Genera una nueva API key
3. **COPIA la key inmediatamente** (solo se muestra una vez)

### 2. 📝 Configurar en el Archivo .env
El archivo `.env` ya existe en tu proyecto local. Ábrelo y actualiza:

```bash
# API Keys - NUNCA subir al repositorio
OPENROUTER_API_KEY=tu_nueva_api_key_aqui
```

**IMPORTANTE**: Reemplaza `tu_nueva_api_key_aqui` con tu nueva API key real.

### 3. ✅ Verificar Configuración
Ejecuta el verificador de seguridad:
```bash
python security_check.py
```

### 4. 🧪 Probar Funcionalidad
Inicia el servidor IA:
```bash
python ia_server.py
```

Deberías ver:
```
🔑 API Key presente: True
🔑 Primeros 10 chars: sk-or-v1-a
```

## 🛡️ MEDIDAS DE SEGURIDAD IMPLEMENTADAS

### ✅ Protecciones Activas
- **`.gitignore`** actualizado - `.env` protegido
- **Git tracking** eliminado - `.env` ya no se rastrea
- **Historial limpio** - API keys anteriores eliminadas
- **Verificador automático** - `security_check.py`
- **Pre-commit hooks** - Prevención automática

### 🔍 Verificaciones Disponibles
```bash
# Verificar seguridad completa
python security_check.py

# Verificar que .env no esté en Git
git ls-files | findstr ".env"
# (No debe devolver nada)

# Verificar estado de Git
git status
# (.env debe aparecer como "untracked" si existe)
```

## 🚀 PRÓXIMOS PASOS

### Inmediatos (Ahora)
1. **Generar nueva API key** en OpenRouter
2. **Actualizar archivo .env** con la nueva key
3. **Probar funcionalidad** del servidor IA
4. **Ejecutar verificador** de seguridad

### Preventivos (Futuro)
1. **Instalar pre-commit hooks**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. **Ejecutar verificaciones semanales**:
   ```bash
   python security_check.py
   ```

3. **Rotar API keys mensualmente** (buena práctica)

## 🆘 EN CASO DE NUEVA EXPOSICIÓN

Si GitHub detecta otra exposición:

1. **INMEDIATO**: Rotar API key en OpenRouter
2. **VERIFICAR**: `git ls-files | findstr ".env"`
3. **LIMPIAR**: Si aparece, ejecutar `git rm --cached .env`
4. **COMMIT**: `git commit -m "Remove .env from tracking"`
5. **FORZAR PUSH**: `git push origin rama --force`

## 📞 CONTACTO DE EMERGENCIA

- **OpenRouter Support**: [Documentación](https://openrouter.ai/docs)
- **GitHub Security**: Settings > Security > Secret scanning alerts

---

**🔒 ESTADO ACTUAL**: SEGURO ✅  
**📅 Última Actualización**: $(date)  
**🎯 Próxima Verificación**: Semanal  

**⚠️ RECORDATORIO**: El archivo `.env` debe existir localmente pero NUNCA debe subirse al repositorio.
