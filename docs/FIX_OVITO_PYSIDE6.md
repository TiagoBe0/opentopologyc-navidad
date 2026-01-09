# 🔧 Solución: Error OVITO - ModuleNotFoundError: No module named 'shiboken6'

## 🚨 Problema

Al ejecutar `main_qt.py` en tu PC de oficina aparece este error:

```
ModuleNotFoundError: No module named 'shiboken6'
```

**Causa:** OVITO requiere PySide6/shiboken6, pero solo tienes PyQt5 instalado.

---

## ✅ Solución Rápida (Recomendada)

### **Paso 1: Actualizar el repositorio**

```bash
cd ~/Documentos/software-final-vacancias-navidad/opentopologyc-navidad
git pull origin claude/integrate-wigner-seitz-WvANw
```

### **Paso 2: Ejecutar el script de reparación**

```bash
python scripts/fix_ovito_pyside6.py
```

Este script automáticamente:
- ✅ Detecta qué paquetes faltan
- ✅ Instala PySide6 y shiboken6 (requeridos por OVITO)
- ✅ Mantiene PyQt5 (para la aplicación)
- ✅ Configura todo para que coexistan sin conflictos

### **Paso 3: Reiniciar Python y probar**

```bash
# Cerrar todas las sesiones de Python/IPython/Jupyter
# Luego:
python main_qt.py
```

---

## 🛠️ Solución Manual (si el script falla)

```bash
# 1. Instalar PySide6 (requerido por OVITO)
pip install PySide6

# 2. Instalar shiboken6 (requerido por OVITO)
pip install shiboken6

# 3. Verificar que PyQt5 sigue instalado
pip install PyQt5

# 4. Reiniciar Python
python main_qt.py
```

---

## 🔍 Verificar que todo funciona

```bash
# Verificar que ambos backends están instalados
python -c "import PyQt5; print('✓ PyQt5:', PyQt5.QtCore.PYQT_VERSION_STR)"
python -c "import PySide6; print('✓ PySide6:', PySide6.__version__)"
python -c "import shiboken6; print('✓ shiboken6 instalado')"
python -c "import ovito; print('✓ OVITO:', ovito.__version__)"
```

Deberías ver ✓ en todos.

---

## 📚 ¿Por qué funciona ahora?

### **Antes:**
- OVITO se importaba al inicio → Error si faltaba shiboken6 → App no iniciaba

### **Ahora:**
- OVITO se importa solo cuando se necesita (lazy loading)
- La app inicia sin problemas
- OVITO solo se carga cuando usas funciones que lo requieren

### **Configuración final:**
- **PyQt5:** Para la GUI principal y matplotlib ✓
- **PySide6:** Para OVITO ✓
- **Ambos coexisten sin conflictos** ✓

---

## 🎯 Funcionalidades que requieren OVITO

Si no tienes OVITO instalado, la aplicación seguirá funcionando **excepto** estas funciones:

- ❌ Extracción de superficie con `ConstructSurfaceModifier`
- ❌ Filtrado por distancia a superficie
- ✅ **TODO LO DEMÁS funciona sin OVITO**

---

## ⚠️ Si el problema persiste

### 1. Limpiar cachés de Python

```bash
python -m pip cache purge
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

### 2. Reinstalar todos los paquetes

```bash
pip uninstall PySide6 shiboken6 PyQt5 matplotlib -y
pip install PyQt5 PySide6 shiboken6 matplotlib
```

### 3. Verificar conflictos de entorno

```bash
# Verificar que no haya múltiples instalaciones de Python
which python
python --version

# Verificar que pip instala en el Python correcto
which pip
pip --version
```

### 4. Usar un entorno virtual limpio

```bash
# Crear entorno virtual
conda create -n opentopologyc python=3.9 -y
conda activate opentopologyc

# Instalar dependencias
pip install -r requirements_qt.txt
pip install PySide6 shiboken6

# Probar
python main_qt.py
```

---

## 📞 Soporte Adicional

Si ninguna solución funciona, ejecuta este script de diagnóstico:

```bash
python scripts/validate_system.py
```

Copia la salida y revísala para identificar el problema específico.

---

## 🎉 Resumen

```bash
# 1. Actualizar código
git pull origin claude/integrate-wigner-seitz-WvANw

# 2. Reparar dependencias
python scripts/fix_ovito_pyside6.py

# 3. Reiniciar Python

# 4. Ejecutar aplicación
python main_qt.py

# ✅ Listo!
```

---

## 📝 Diferencias entre PC de Casa y Oficina

| Aspecto | PC Casa | PC Oficina |
|---------|---------|------------|
| PyQt5 | ✓ Instalado | ✓ Instalado |
| PySide6 | ✓ Instalado | ✗ **Faltaba** |
| shiboken6 | ✓ Instalado | ✗ **Faltaba** |
| OVITO | ✓ Funciona | ✗ No podía importar |

**Solución:** Instalar PySide6 + shiboken6 en PC de oficina.

---

Última actualización: 2026-01-09
