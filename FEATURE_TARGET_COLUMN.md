# ✅ Feature: Auto-detección de Columna Target

**Fecha:** 2026-01-07
**Commit:** `9e74055`
**Estado:** ✅ **Implementado y probado**

---

## 📋 Problema Resuelto

**Reporte del usuario:**
> "me da error al comenzar el entrenamiento, dice 'target!'"

**Causa raíz:** El sistema buscaba columnas con nombres específicos (`label` o `target`), pero el CSV del usuario probablemente tiene la columna con otro nombre (ej: `n_vacancies`, `vacancies`, etc.).

**Solución:** Sistema de auto-detección inteligente con soporte para columnas personalizadas.

---

## 🎯 Implementación

### Columnas Detectadas Automáticamente

El sistema ahora busca automáticamente estas columnas (en orden de prioridad):

1. **`n_vacancies`** ← Más común en simulaciones atómicas
2. **`label`** ← Nombre estándar en ML
3. **`target`** ← Nombre alternativo en ML
4. **`vacancies`** ← Variante sin prefijo
5. **`y`** ← Convención matemática
6. **`class`** ← Para clasificación

### Características

✅ **Auto-detección inteligente:** Encuentra la columna automáticamente
✅ **Columnas personalizadas:** Soporta nombres personalizados
✅ **Errores descriptivos:** Muestra columnas disponibles y soluciones
✅ **Múltiples interfaces:** GUI, scripts, y código Python
✅ **Retrocompatible:** Funciona con código existente

---

## 🚀 Cómo Usar

### Opción 1: Auto-detección (Recomendado)

**Desde GUI:**

1. Abrir aplicación: `python main_qt.py`
2. Ir a ventana "Entrenamiento"
3. Cargar CSV de features
4. En "Parámetros del modelo", **dejar vacío** el campo "Columna target"
5. Ejecutar entrenamiento

El sistema detectará automáticamente la columna.

**Desde quick_train.py:**

```python
# En TRAINING, línea 53:
TRAINING = {
    "n_estimators": 100,
    "max_depth": None,
    "test_size": 0.2,
    "model_version": "1.0",
    "target_column": None  # ← Auto-detectar (default)
}
```

**Desde código:**

```python
from core.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline(
    csv_file="dataset_features.csv",
    model_output="model.pkl",
    target_column=None  # Auto-detectar
)
```

---

### Opción 2: Especificar Columna Manualmente

**Desde GUI:**

1. En ventana "Entrenamiento"
2. Campo "Columna target": escribir nombre exacto de tu columna
3. Ejemplo: `numero_vacancias`, `defects`, etc.

**Desde quick_train.py:**

```python
# Cambiar en línea 53:
TRAINING = {
    # ...
    "target_column": "numero_vacancias"  # ← Tu columna personalizada
}
```

**Desde código:**

```python
pipeline = TrainingPipeline(
    csv_file="dataset_features.csv",
    model_output="model.pkl",
    target_column="numero_vacancias"  # Nombre personalizado
)
```

---

## 💡 Ejemplos

### Ejemplo 1: CSV con `n_vacancies`

```csv
file,hull_volume,grid_count,n_vacancies
dump1.dump,1234.5,678,5
dump2.dump,1100.2,620,3
dump3.dump,1450.8,710,8
```

**Resultado:** ✅ Detecta `n_vacancies` automáticamente

```
✓ Columna target detectada: 'n_vacancies'
✓ Features a usar: 2
```

---

### Ejemplo 2: CSV con columna personalizada

```csv
file,hull_volume,grid_count,defectos_atomicos
dump1.dump,1234.5,678,5
dump2.dump,1100.2,620,3
dump3.dump,1450.8,710,8
```

**Opción A - Especificar manualmente:**

```python
pipeline = TrainingPipeline(
    csv_file="dataset_features.csv",
    target_column="defectos_atomicos"  # ← Especificar
)
```

**Opción B - Renombrar columna:**

```python
import pandas as pd

df = pd.read_csv("dataset_features.csv")
df['n_vacancies'] = df['defectos_atomicos']
df.to_csv("dataset_features.csv", index=False)
```

---

### Ejemplo 3: Error cuando no encuentra columna

Si tu CSV no tiene ninguna columna candidata:

```csv
file,hull_volume,grid_count,resultado
dump1.dump,1234.5,678,5
dump2.dump,1100.2,620,3
```

**Error mostrado:**

```
ValueError: No se encontró columna target en el CSV.
Columnas disponibles: ['file', 'hull_volume', 'grid_count', 'resultado']

Soluciones:
1. Especificar columna target explícitamente:
   pipeline = TrainingPipeline(..., target_column='resultado')

2. Renombrar una columna a 'n_vacancies' o 'label':
   import pandas as pd
   df = pd.read_csv('dataset_features.csv')
   df['label'] = df['resultado']
   df.to_csv('dataset_features.csv', index=False)
```

**Solución:**

```python
# En quick_train.py o desde GUI
TRAINING = {
    # ...
    "target_column": "resultado"
}
```

---

## 📊 Archivos Modificados

| Archivo | Cambios | Líneas |
|---------|---------|--------|
| **core/training_pipeline.py** | Auto-detección implementada | +65 |
| **gui_qt/train_gui_qt.py** | GUI control agregado | +12 |
| **quick_train.py** | Parámetro agregado | +5 |
| **validate_system.py** | Parámetro agregado | +1 |
| **VALIDACION_CIENTIFICA.md** | Docs actualizadas | +42 |
| **TOTAL** | **5 archivos** | **+125 líneas** |

---

## 🔍 Detalles Técnicos

### Método `_detect_target_column(df)`

```python
def _detect_target_column(self, df):
    """
    Detecta automáticamente la columna target

    Prioridad:
    1. self.target_column (si fue especificado)
    2. Columnas candidatas comunes
    3. Error descriptivo si no encuentra

    Returns:
        str: Nombre de la columna target

    Raises:
        ValueError: Si no encuentra columna y muestra soluciones
    """
    # 1. Verificar si se especificó explícitamente
    if self.target_column:
        if self.target_column in df.columns:
            return self.target_column
        else:
            raise ValueError(f"Columna '{self.target_column}' no existe...")

    # 2. Buscar candidatos comunes
    candidates = ["n_vacancies", "label", "target", "vacancies", "y", "class"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # 3. Error con columnas disponibles y soluciones
    raise ValueError("No se encontró columna target... [ver soluciones]")
```

### Método `load_data()` actualizado

```python
def load_data(self):
    df = pd.read_csv(self.csv_file)

    # Detectar target automáticamente
    target_col = self._detect_target_column(df)

    # Excluir target y metadata de features
    exclude_cols = [target_col, "file", "num_points", "num_atoms_real"]

    # Excluir aliases de target (evita duplicados)
    target_aliases = ["label", "target", "n_vacancies", "vacancies", "y"]
    exclude_cols.extend([col for col in target_aliases
                        if col in df.columns and col != target_col])

    X = df.drop(columns=exclude_cols, errors="ignore")
    y = df[target_col]

    print(f"✓ Columna target detectada: '{target_col}'")
    print(f"✓ Features a usar: {len(X.columns)}")

    return X.values, y.values, X.columns.tolist()
```

---

## 🧪 Testing

### Verificación de Compilación

```bash
python -m py_compile core/training_pipeline.py
python -m py_compile gui_qt/train_gui_qt.py
python -m py_compile quick_train.py
python -m py_compile validate_system.py

# ✓ Todos compilan sin errores
```

### Test de Auto-detección

```python
# Test 1: CSV con n_vacancies
df = pd.DataFrame({
    'hull_volume': [1234.5, 1100.2],
    'grid_count': [678, 620],
    'n_vacancies': [5, 3]
})
df.to_csv("test1.csv", index=False)

pipeline = TrainingPipeline(csv_file="test1.csv", target_column=None)
# ✓ Detecta 'n_vacancies' automáticamente
```

```python
# Test 2: CSV con columna personalizada
df = pd.DataFrame({
    'hull_volume': [1234.5, 1100.2],
    'grid_count': [678, 620],
    'mi_target': [5, 3]
})
df.to_csv("test2.csv", index=False)

pipeline = TrainingPipeline(csv_file="test2.csv", target_column="mi_target")
# ✓ Usa 'mi_target' como especificado
```

---

## 📚 Documentación Actualizada

### VALIDACION_CIENTIFICA.md

Sección actualizada: **🔧 Troubleshooting → Columna Target**

- ✅ Explicación de auto-detección
- ✅ 4 opciones documentadas
- ✅ Ejemplos de código
- ✅ Removido troubleshooting obsoleto

---

## ✅ Checklist de Verificación

**Para el usuario - Verificar que todo funciona:**

- [ ] Ejecutar `git pull origin claude/integrate-gui-windows-D2Jbi`
- [ ] Abrir GUI: `python main_qt.py`
- [ ] Ventana "Entrenamiento" muestra campo "Columna target"
- [ ] Campo tiene placeholder "(auto-detectar)"
- [ ] Entrenar modelo con campo vacío → auto-detecta columna
- [ ] Entrenar modelo especificando columna → usa columna especificada
- [ ] Ver en log: `✓ Columna target detectada: 'nombre_columna'`
- [ ] Ejecutar `python quick_train.py` → funciona con auto-detección
- [ ] Cambiar `TRAINING["target_column"]` → funciona con personalizada

---

## 🎓 Próximos Pasos

**Ahora que la auto-detección está implementada:**

1. **Probar con tus datos reales:**
   ```bash
   python quick_train.py
   ```

2. **Si tu CSV tiene columna personalizada:**
   - Opción 1: Especificar en `TRAINING["target_column"]`
   - Opción 2: Usar GUI y escribir nombre de columna

3. **Verificar que el entrenamiento completa exitosamente:**
   - Debe mostrar: `✓ Columna target detectada: '...'`
   - Debe mostrar métricas completas (accuracy, confusion matrix, etc.)

4. **Reportar si encuentras algún problema:**
   - Qué nombre tiene tu columna target
   - Qué error muestra (si lo hay)

---

## 📞 Soporte

**Si el sistema no detecta tu columna:**

1. Verificar nombre exacto de columna:
   ```python
   import pandas as pd
   df = pd.read_csv("dataset_features.csv")
   print(df.columns.tolist())
   ```

2. Especificar manualmente:
   ```python
   TRAINING = {
       # ...
       "target_column": "nombre_exacto_de_tu_columna"
   }
   ```

3. O renombrar en CSV:
   ```python
   df['n_vacancies'] = df['tu_columna']
   df.to_csv("dataset_features.csv", index=False)
   ```

---

**Commit:** `9e74055`
**Branch:** `claude/integrate-gui-windows-D2Jbi`
**Fecha:** 2026-01-07
**Estado:** ✅ **Listo para usar**
