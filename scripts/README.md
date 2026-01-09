# 🛠️ Scripts Utilitarios de OpenTopologyC

Esta carpeta contiene scripts de utilidad para facilitar tareas comunes.

## 📋 Scripts Disponibles

### 🚀 Quick Train
**Archivo:** `quick_train.py`

Entrena un modelo rápidamente con configuración simplificada.

```bash
# Edita los parámetros en el script primero
python scripts/quick_train.py
```

**Configuración:**
- Define `DUMP_DIR`: ruta a tus archivos DUMP
- Ajusta parámetros del material (a0, lattice_type, etc.)
- Configura Random Forest (n_estimators, max_depth)

---

### 🔍 Validate System
**Archivo:** `validate_system.py`

Valida que el sistema esté correctamente configurado y todas las dependencias instaladas.

```bash
python scripts/validate_system.py
```

**Verifica:**
- Dependencias de Python (numpy, scipy, sklearn, etc.)
- Configuración de Qt (PyQt5/PySide6)
- Backends de matplotlib
- Estructura de directorios

---

### 🔧 Fix Qt Backend
**Archivo:** `fix_qt_backend.py`

Repara automáticamente conflictos entre PySide6 y PyQt5.

```bash
python scripts/fix_qt_backend.py
```

**Soluciona:**
- Conflictos de backends Qt
- Configuración de matplotlib
- Variables de entorno Qt

---

### 🧪 Check Qt Versions
**Archivo:** `check_qt_versions.py`

Verifica las versiones de Qt instaladas y detecta conflictos.

```bash
python scripts/check_qt_versions.py
```

**Muestra:**
- Versiones de PyQt5 y PySide6
- Backend activo de matplotlib
- Recomendaciones de corrección

---

## 📝 Uso desde la Raíz del Proyecto

Todos los scripts están diseñados para ejecutarse desde la raíz del proyecto:

```bash
cd /path/to/opentopologyc-navidad
python scripts/quick_train.py
python scripts/validate_system.py
python scripts/fix_qt_backend.py
python scripts/check_qt_versions.py
```

## 🔙 Volver

Regresa al [README principal](../README.md) del proyecto.
