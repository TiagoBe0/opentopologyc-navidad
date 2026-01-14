# 📦 Models Directory

Esta carpeta contiene los modelos de Machine Learning entrenados para predecir vacancias en nanomateriales.

## 🎯 Propósito

Los modelos guardados aquí son utilizados automáticamente por las interfaces de predicción (tanto Qt como Tkinter) para facilitar su uso sin tener que navegar por el sistema de archivos.

## 💾 Formato de Modelos

Los modelos se guardan en formato:
- **`.joblib`** (recomendado) - Formato eficiente de scikit-learn
- **`.pkl`** - Formato pickle estándar de Python

## 🔄 Flujo de Trabajo

### 1. Entrenar un Modelo

Al entrenar un modelo usando la interfaz de entrenamiento:
- **Qt GUI**: Se sugiere automáticamente guardar en `models/` con un nombre descriptivo
- **Tkinter GUI**: La carpeta por defecto es `models/`

Ejemplo de nombre generado: `vacancy_model_20260114_153045.joblib`

### 2. Usar el Modelo para Predicción

En la interfaz de predicción:
- **Selector automático**: Se muestra una lista desplegable con todos los modelos disponibles
- **Modelos ordenados**: Los más recientes aparecen primero
- **Botón refrescar**: Actualiza la lista si agregas modelos nuevamente
- **Carga manual**: También puedes cargar un modelo desde cualquier ubicación

## 📝 Convenciones de Nombres (Sugeridas)

```
vacancy_model_YYYYMMDD_HHMMSS.joblib    # Timestamp para orden cronológico
vacancy_model_v1.0.joblib                # Versionado semántico
vacancy_model_gold_100epochs.joblib     # Descriptivo con parámetros
vacancy_model_production.joblib         # Modelo en producción
```

## 🗑️ Limpieza

Esta carpeta puede crecer con el tiempo. Se recomienda:
- Mantener solo los modelos que estés usando activamente
- Eliminar modelos antiguos o experimentales
- Hacer backup de modelos importantes antes de eliminarlos

## ⚙️ Contenido del Modelo

Cada archivo `.joblib` o `.pkl` puede contener:
- **Modelo entrenado** (RandomForest, XGBoost, etc.)
- **Metadatos** (fecha, parámetros, métricas)
- **Información de features** (nombres, importancias)

## 🔍 Verificación

Para verificar un modelo desde Python:

```python
import joblib

# Cargar modelo
model = joblib.load("models/vacancy_model_20260114_153045.joblib")

# Ver tipo
print(type(model))  # sklearn.ensemble.RandomForestRegressor

# Hacer predicción de prueba
# prediction = model.predict(X_test)
```

## 📊 Mejores Prácticas

1. **Nombra descriptivamente**: Incluye fecha, versión o propósito
2. **Documenta**: Mantén notas sobre qué dataset se usó
3. **Versiona**: Si mejoras un modelo, crea una nueva versión
4. **Respalda**: Los modelos importantes deberían tener backup
5. **Limpia**: Elimina modelos obsoletos regularmente

---

**Nota**: Esta carpeta se crea automáticamente si no existe cuando usas las interfaces de entrenamiento o predicción.
