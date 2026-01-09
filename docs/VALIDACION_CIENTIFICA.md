# 🔬 Validación Científica - Guía Paso a Paso

**Objetivo:** Entrenar y validar un modelo de predicción de vacancias atómicas con tus datos reales.

---

## 📋 Pre-requisitos

Antes de comenzar, asegúrate de tener:

✅ **Datos de entrenamiento:**
- Carpeta con dumps LAMMPS (ej: `db_test_pequeña/`)
- Mínimo 50-100 muestras para entrenamiento decente
- Cada dump debe corresponder a una configuración con cierto número de vacancias

✅ **Parámetros del material:**
- Tipo de red cristalina (fcc, bcc, hcp, etc.)
- Parámetro de red `a0` en Å
- Número de átomos en cristal perfecto

✅ **Entorno configurado:**
```bash
pip install -r requirements_qt.txt
```

---

## 🚀 Método Rápido: quick_train.py

### Paso 1: Configurar Script

Edita `quick_train.py` con tus parámetros:

```python
# Línea 23: Ruta a tus dumps
DUMP_DIR = "/home/santi-simaf/Documentos/.../db_test_pequeña"

# Línea 27-31: Parámetros del material
MATERIAL = {
    "total_atoms": 16384,    # ← Tu número de átomos
    "a0": 3.532,             # ← Tu parámetro de red
    "lattice_type": "fcc"    # ← Tu tipo de red
}
```

### Paso 2: Ejecutar

```bash
python quick_train.py
```

### Paso 3: Interpretar Resultados

El script mostrará:

```
============================================================
RESULTADOS
============================================================

Accuracy: 0.8542

Classification Report:
              precision    recall  f1-score   support
           0       0.85      0.82      0.84       100
           1       0.88      0.90      0.89       120

Confusion Matrix:
[[82  18]
 [12 108]]

Top 10 Features Más Importantes:
  1. hull_volume: 0.2341
  2. grid_count: 0.1892
  ...
```

**Interpretar:**

| Accuracy | Significado | Acción |
|----------|-------------|--------|
| **> 90%** | 🟢 Excelente | Listo para producción |
| **70-90%** | 🟡 Aceptable | Funcional, puede mejorar |
| **< 70%** | 🔴 Bajo | Necesita mejoras |

**Si accuracy < 70%:**
- Agregar más muestras de entrenamiento
- Verificar que labels sean correctos
- Experimentar con `n_estimators=200`

---

## 📊 Método Interactivo: validate_system.py

Para más control y opciones:

```bash
python validate_system.py
```

**Menú:**
```
1. Extraer features de dumps
2. Entrenar modelo
3. Analizar resultados
4. Información de predicciones
5. Gestión de modelos (ModelManager)
6. Validación completa (1→2→3)
0. Salir
```

---

## 🎯 Validación Científica

### 1. Verificar Accuracy

```python
# Objetivo mínimo: 70%
# Objetivo deseable: >85%

if accuracy >= 0.85:
    # Modelo preciso ✓
elif accuracy >= 0.70:
    # Modelo funcional, puede mejorar
else:
    # Modelo necesita mejoras
```

### 2. Analizar Confusion Matrix

```
Confusion Matrix:
[[82  18]    ← Clase 0: 82 correctas, 18 errores
 [12 108]]   ← Clase 1: 108 correctas, 12 errores
```

**Verificar:**
- Diagonal principal alta (predicciones correctas)
- Fuera de diagonal bajo (errores)
- Sin bias hacia una clase

### 3. Feature Importance

```
Top 10 Features Más Importantes:
  1. hull_volume: 0.2341
  2. grid_count: 0.1892
  3. radial_mean: 0.1456
```

**Verificar:**
- Features importantes tengan sentido físico
- `hull_volume`, `grid_count` suelen ser importantes para vacancias
- Si un feature tiene importancia ~0, puede eliminarse

### 4. Validar Físicamente

**Test crítico:** ¿Las predicciones tienen sentido?

```python
# Predecir en dump conocido
dump_con_10_vacancias.dump
→ Predicción: 9.8 vacancias

# Si predicción está muy lejos (ej: 2 vacancias cuando son 10)
# → Modelo tiene problemas, revisar datos
```

---

## 🔧 Troubleshooting

### Columna Target (Vacancias)

**✅ Auto-detección implementada:** El sistema ahora detecta automáticamente la columna target.

**Columnas detectadas automáticamente (en orden de prioridad):**
1. `n_vacancies`
2. `label`
3. `target`
4. `vacancies`
5. `y`
6. `class`

**Opción 1: Usar auto-detección (Recomendado)**

Si tu CSV tiene alguna de estas columnas, el sistema la detectará automáticamente:

```python
# quick_train.py o validate_system.py
pipeline = TrainingPipeline(
    csv_file="dataset_features.csv",
    # ... otros parámetros ...
    target_column=None  # Auto-detectar
)
```

**Opción 2: Especificar columna manualmente**

Si tu columna tiene un nombre diferente:

```python
# En quick_train.py, cambiar:
TRAINING = {
    # ...
    "target_column": "mi_columna_vacancias"  # Tu nombre personalizado
}
```

**Opción 3: Desde GUI**

1. Abrir ventana "Entrenamiento"
2. En "Parámetros del modelo", ver campo "Columna target"
3. Dejar vacío para auto-detectar, o escribir nombre de tu columna

**Opción 4: Renombrar columna en CSV**

Si prefieres renombrar tu columna:

```python
import pandas as pd

df = pd.read_csv("dataset_features.csv")
df['n_vacancies'] = df['tu_columna_original']
df.to_csv("dataset_features.csv", index=False)
```

### Error: "Accuracy muy bajo (< 50%)"

**Causas posibles:**

1. **Datos insuficientes:**
   - Necesitas mín. 50-100 muestras
   - Solución: Recolectar más dumps

2. **Labels incorrectos:**
   - Verificar que `n_vacancies` sea correcto
   - Solución: Validar labels manualmente

3. **Features no informativos:**
   - Algunos features no ayudan a distinguir clases
   - Solución: Revisar feature importance

4. **Problema de la tarea:**
   - Tal vez la tarea es muy difícil
   - Solución: Simplificar (ej: clasificar vacancies en rangos)

### Error: "Segmentation Fault"

**Ya está solucionado** en la última versión.

Si persiste:
```bash
git pull origin claude/integrate-gui-windows-D2Jbi
```

---

## 📈 Mejoras Incrementales

### Experimento 1: Más Estimators

```python
# En quick_train.py, línea 44:
"n_estimators": 200,  # Cambiar de 100 a 200
```

**Efecto:** Modelo más robusto, puede mejorar accuracy 1-3%

### Experimento 2: Max Depth Limitado

```python
# En quick_train.py, línea 45:
"max_depth": 10,  # Cambiar de None a 10
```

**Efecto:** Previene overfitting, mejor generalización

### Experimento 3: Versionar Modelos

```python
# Entrenar versión 1.0 con config básica
TRAINING["model_version"] = "1.0"

# Entrenar versión 2.0 con más estimators
TRAINING["n_estimators"] = 200
TRAINING["model_version"] = "2.0"
```

Luego comparar:
```python
from core.model_manager import ModelManager

manager = ModelManager()
manager.print_summary()
best = manager.compare_models()
```

---

## 🎓 Criterios de Éxito

Tu modelo está **listo para producción** si:

✅ **Accuracy > 85%** en test set
✅ **Confusion matrix balanceada** (sin bias extremo)
✅ **Features importantes** tienen sentido físico
✅ **Predicciones validadas** en dumps conocidos
✅ **Reproducible** con ModelManager versionado

---

## 📚 Ejemplo Completo

```bash
# 1. Configurar quick_train.py
nano quick_train.py  # Editar DUMP_DIR, MATERIAL

# 2. Entrenar
python quick_train.py

# 3. Ver resultados
# (Se muestran automáticamente)

# 4. Si accuracy > 70%, hacer predicciones
python main_qt.py
# → GUI → Predicción → Cargar modelo → Predecir

# 5. Gestionar modelos
python
>>> from core.model_manager import ModelManager
>>> manager = ModelManager()
>>> manager.print_summary()
```

---

## 🚀 Próximos Pasos

Después de validar con éxito:

1. **Entrenar versiones mejoradas:**
   - Experimentar con hiperparámetros
   - Probar XGBoost / LightGBM (Fase 2)

2. **Análisis avanzado:**
   - Cross-validation
   - Learning curves
   - Feature engineering

3. **Producción:**
   - Integrar en pipeline científico
   - Automatizar predicciones
   - Publicar resultados

---

## 📞 Soporte

**Documentos relacionados:**
- `ANALISIS_Y_RECOMENDACIONES.md` - Roadmap completo
- `QUICK_WINS_COMPLETADO.md` - Features implementadas
- `README_QT.md` - Uso de la GUI

**Debugging:**
- Ver logs: `cat opentopologyc.log`
- Buscar errores: `grep ERROR opentopologyc.log`

---

**Fecha:** 2026-01-07
**Versión:** 1.0
**Estado:** ✅ Listo para usar
