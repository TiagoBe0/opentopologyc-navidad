# 🎯 Guía de Ajuste Fino por Cluster

## Problema que Resuelve

Algunos modelos de ML funcionan bien para **defectos pequeños** (pocas vacancias) pero erran en **clusters grandes**, y viceversa. Esta funcionalidad permite usar **diferentes modelos para diferentes clusters** en una misma predicción.

## 📋 ¿Cuándo Usar Ajuste Fino?

- ✅ Cuando tienes múltiples modelos entrenados con diferentes datasets
- ✅ Cuando observas que un modelo sobre/subestima clusters grandes o pequeños
- ✅ Cuando quieres experimentar combinaciones de modelos
- ✅ Cuando necesitas máxima precisión en predicciones

## 🔄 Flujo de Trabajo

### 1. Predicción Normal (Pasos 1-3)

```
PASO 1: Alpha Shape → Filtrar superficie
PASO 2: Clustering → Separar defectos
PASO 3: Predicción → Obtener vacancias por cluster
```

**Resultado:** Predicción inicial con un solo modelo

### 2. Ajuste Fino (PASO 4) - ¡NUEVO!

Después del Paso 3, automáticamente aparece:

```
⚙️ PASO 4: Ajuste Fino por Cluster (Opcional)

Predicción Total Actual: 245.3 vacancias

┌─────────────────────────────────────────────────┐
│ Cluster │ Átomos │ Predicción │ Modelo      │ 🔄 │
├─────────────────────────────────────────────────┤
│    0    │  1234  │   45.2     │ modelo_v1   │ 🔄 │
│    1    │   523  │   12.8     │ modelo_v1   │ 🔄 │
│    2    │  2891  │  187.3     │ modelo_v1   │ 🔄 │
└─────────────────────────────────────────────────┘

Modelo alternativo: [ modelo_v2_large_defects ▼ ]

[ 🔄 Re-predecir Cluster Seleccionado ]
```

## 🎮 Cómo Usar

### Paso a Paso

1. **Seleccionar Cluster**
   - Click en la fila del cluster que quieres ajustar
   - Ejemplo: Cluster 2 tiene muchos átomos (cluster grande)

2. **Elegir Modelo Alternativo**
   - Usar el dropdown "Modelo alternativo"
   - Ejemplo: Seleccionar `modelo_v2_large_defects.joblib`

3. **Re-predecir**
   - Click en "🔄 Re-predecir Cluster Seleccionado"
   - Ver cambio en predicción y total

4. **Repetir si Necesario**
   - Puedes re-predecir otros clusters
   - El total se actualiza automáticamente

### Ejemplo Práctico

**Situación:**
- Cluster 0 (pequeño, 500 átomos) → Modelo general: 45 vacancias ❌ (parece mucho)
- Cluster 1 (grande, 3000 átomos) → Modelo general: 120 vacancias ❌ (parece poco)

**Solución con Ajuste Fino:**
1. Seleccionar Cluster 0 → Elegir `modelo_small_defects.joblib` → Re-predecir
   - Nueva predicción: 28 vacancias ✅
2. Seleccionar Cluster 1 → Elegir `modelo_large_defects.joblib` → Re-predecir
   - Nueva predicción: 185 vacancias ✅
3. Total actualizado: 213 vacancias (antes era 165)

## 💡 Estrategias de Uso

### Estrategia 1: Por Tamaño de Cluster

```python
Clusters pequeños (< 1000 átomos) → modelo_small.joblib
Clusters medianos (1000-3000)      → modelo_medium.joblib
Clusters grandes (> 3000 átomos)   → modelo_large.joblib
```

### Estrategia 2: Por Área de Superficie

Calculada con Alpha Shape:
```python
Baja superficie (< 500 Ų)   → modelo_point_defects.joblib
Alta superficie (> 500 Ų)   → modelo_extended_defects.joblib
```

### Estrategia 3: Iterativa

1. Hacer predicción inicial
2. Identificar clusters con mayor error
3. Re-predecir solo esos clusters con modelos alternativos
4. Evaluar mejora en error total

## 🔧 Detalles Técnicos

### ¿Qué se Guarda por Cluster?

```python
{
    'cluster_id': 0,
    'n_atoms': 1234,
    'prediction': 45.2,
    'model_name': 'modelo_v1.joblib',
    'features': {...},  # Features extraídas
    'positions': [...]  # Posiciones atómicas
}
```

### ¿Cómo Funciona la Re-predicción?

1. Se reutilizan las **mismas features** ya calculadas
2. Se carga el **modelo alternativo** seleccionado
3. Se hace `model.predict(features)` con el nuevo modelo
4. Se actualiza la tabla y el total

**Ventaja:** No necesita recalcular features (instantáneo)

### Normalización

- ✅ Se usa la misma normalización que en entrenamiento
- ✅ Se preserva el `box_size` de referencia
- ✅ Garantiza compatibilidad con cualquier modelo

## 📊 Visualización

El total se actualiza en tiempo real:

```
Antes:  245.3 vacancias
         ↓ (re-predecir cluster 2)
Después: 212.1 vacancias
         Δ = -33.2 vacancias
```

Status feedback:
```
✓ Cluster 2: 187.3 → 154.1 (-33.2)
```

## ⚠️ Consideraciones

### Cuando NO Usar

- ❌ Si solo tienes un modelo entrenado
- ❌ Si el clustering no tiene sentido físico
- ❌ Si los modelos fueron entrenados con diferentes features

### Mejores Prácticas

1. **Entrena modelos especializados** antes de predecir
   - Modelo para defectos pequeños
   - Modelo para defectos grandes

2. **Documenta qué modelo usaste** para cada cluster
   - La tabla muestra el modelo actual por cluster

3. **Valida con datos conocidos** antes de producción
   - Prueba la estrategia con muestras de validación

4. **Considera área de superficie** además de número de átomos
   - Clusters grandes pueden ser compactos o extendidos

## 🎓 Entrenamiento de Modelos Especializados

### Crear Dataset Pequeño

```python
# Filtrar solo dumps con pocas vacancias
df_small = df[df['n_vacancies'] < 50]
# Entrenar modelo
model_small.fit(X_small, y_small)
# Guardar en models/
joblib.dump(model_small, 'models/vacancy_small_defects.joblib')
```

### Crear Dataset Grande

```python
# Filtrar solo dumps con muchas vacancias
df_large = df[df['n_vacancies'] > 100]
# Entrenar modelo
model_large.fit(X_large, y_large)
# Guardar en models/
joblib.dump(model_large, 'models/vacancy_large_defects.joblib')
```

Ahora en predicción podrás elegir entre ambos modelos según el cluster.

## 📈 Métricas de Éxito

Compara:
- **Error con modelo único:** |pred_total - real| = 45 vacancias
- **Error con ajuste fino:** |pred_total_ajustado - real| = 12 vacancias

**Mejora:** 73% reducción en error ✅

## 🚀 Ejemplo Completo

```
1. Cargar dump → 4325 átomos reales
2. Alpha Shape → 3892 átomos superficiales
3. Clustering → 5 clusters detectados:
   - Cluster 0: 450 átomos (pequeño)
   - Cluster 1: 892 átomos (mediano)
   - Cluster 2: 2100 átomos (grande)
   - Cluster 3: 280 átomos (pequeño)
   - Cluster 4: 170 átomos (muy pequeño)

4. Predicción inicial (modelo_general.joblib):
   - Total: 234 vacancias

5. Ajuste Fino:
   - Cluster 0 → usar modelo_small → 18 vac (antes 22)
   - Cluster 2 → usar modelo_large → 156 vac (antes 128)
   - Cluster 4 → usar modelo_small → 7 vac (antes 11)

6. Total ajustado: 258 vacancias
7. Total real: 265 vacancias
8. Error: 7 vacancias (vs 31 sin ajuste)
```

## 📚 Referencias

- Ver `models/README.md` para gestión de modelos
- Ver código en `opentopologyc/gui_qt/prediction_gui_qt.py` líneas 920+

---

**Última actualización:** 2026-01-14
**Versión:** 1.0
**Autor:** OpenTopologyC Team
