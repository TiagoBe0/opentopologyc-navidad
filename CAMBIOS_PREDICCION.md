# Corrección del Sistema de Predicción - Compatibilidad con Modelo Legacy

## Resumen

Se corrigió el sistema de extracción de features para ser compatible con modelos legacy que requieren **27 features base** (no 19 como se pensaba inicialmente).

## Fecha
2026-01-16

## Problema Detectado

El modelo de ML estaba entrenado con un conjunto de features diferente al que el código actual generaba, causando el error:

```
ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- hull_n_simplices
- hull_volume_area_ratio
Feature names seen at fit time, yet now missing:
- grid_com_x
- grid_com_y
- grid_com_z
- grid_skewness_x
- moi_principal_1
- ...
```

## Causa del Problema

1. El modelo fue entrenado con **27 features** que incluían:
   - Centro de masa del grid (grid_com_x/y/z)
   - Skewness del grid (grid_skewness_x/y/z)
   - **3 momentos principales de inercia** (moi_principal_1/2/3), no solo el tercero

2. El código actual solo generaba **19 features** y agregaba hull_features no compatibles

## Archivos Modificados

### 1. `opentopologyc/core/feature_extractor.py`

**Cambios en `grid_features()`:**

Se agregaron 6 nuevas features al método (de 14 a 20 features):

```python
# NUEVAS: Centro de masa del grid (3 features)
if occ > 0:
    coords = np.argwhere(grid == 1)
    com = coords.mean(axis=0)
    features['grid_com_x'] = float(com[0])
    features['grid_com_y'] = float(com[1])
    features['grid_com_z'] = float(com[2])

# NUEVAS: Skewness del grid (3 features)
if occ > 0:
    from scipy.stats import skew
    proj_x = grid.sum(axis=(1, 2))
    proj_y = grid.sum(axis=(0, 2))
    proj_z = grid.sum(axis=(0, 1))

    features['grid_skewness_x'] = float(skew(proj_x))
    features['grid_skewness_y'] = float(skew(proj_y))
    features['grid_skewness_z'] = float(skew(proj_z))
```

**Cambios en `inertia_feature()`:**

Se extendió para calcular los 3 momentos principales (de 1 a 3 features):

```python
# ANTES: Solo retornaba moi_principal_3
return {"moi_principal_3": float(eig[2])}

# AHORA: Retorna los 3 momentos principales
return {
    "moi_principal_1": float(eig[0]),
    "moi_principal_2": float(eig[1]),
    "moi_principal_3": float(eig[2])
}
```

**Cambios en `extract_all_features()`:**

Se actualizó la lista de features finales para incluir las 27 features en el orden correcto:

```python
final_features = [
    # Occupancy básicas (2)
    'occupancy_total',
    'occupancy_fraction',
    # Occupancy por eje (3)
    'occupancy_x_mean',
    'occupancy_y_mean',
    'occupancy_z_mean',
    # Gradientes (4)
    'occupancy_gradient_x',
    'occupancy_gradient_y',
    'occupancy_gradient_z',
    'occupancy_gradient_total',
    # Superficie (1)
    'occupancy_surface',
    # Entropía del grid (1)
    'grid_entropy',
    # Centro de masa del grid (3) ← NUEVO
    'grid_com_x',
    'grid_com_y',
    'grid_com_z',
    # Skewness del grid (3) ← NUEVO
    'grid_skewness_x',
    'grid_skewness_y',
    'grid_skewness_z',
    # Momentos de inercia del grid (3)
    'grid_moi_1',
    'grid_moi_2',
    'grid_moi_3',
    # Momentos principales (3) ← AMPLIADO
    'moi_principal_1',
    'moi_principal_2',
    'moi_principal_3',
    # RDF (2)
    'rdf_mean',
    'rdf_kurtosis',
    # Entropy espacial (1)
    'entropy_spatial',
    # Bandwidth (1)
    'ms_bandwidth'
]
```

**Cambios en `get_all_feature_names()` y `get_feature_categories()`:**

Se actualizaron para reflejar las 27 features base.

### 2. `test_feature_extraction.py`

Se actualizó el script de verificación para validar las 27 features en lugar de 19.

### 3. `opentopologyc/core/prediction_pipeline.py`

Ya estaba usando `extract_all_features()`, por lo que automáticamente usa las 27 features correctas.

## Features Extraídas (27 en total)

### Grid Features (20 features)

**Occupancy Básicas (2):**
1. `occupancy_total` - Total de celdas ocupadas en el grid
2. `occupancy_fraction` - Fracción de celdas ocupadas

**Occupancy por Eje (3):**
3. `occupancy_x_mean` - Media de ocupación en eje X
4. `occupancy_y_mean` - Media de ocupación en eje Y
5. `occupancy_z_mean` - Media de ocupación en eje Z

**Gradientes (4):**
6. `occupancy_gradient_x` - Gradiente en dirección X
7. `occupancy_gradient_y` - Gradiente en dirección Y
8. `occupancy_gradient_z` - Gradiente en dirección Z
9. `occupancy_gradient_total` - Suma de gradientes

**Superficie (1):**
10. `occupancy_surface` - Superficie total (gradientes)

**Entropía del Grid (1):**
11. `grid_entropy` - Entropía del grid 3D

**Centro de Masa del Grid (3):** ✨ **NUEVO**
12. `grid_com_x` - Centro de masa del grid en X
13. `grid_com_y` - Centro de masa del grid en Y
14. `grid_com_z` - Centro de masa del grid en Z

**Skewness del Grid (3):** ✨ **NUEVO**
15. `grid_skewness_x` - Asimetría de la distribución en X
16. `grid_skewness_y` - Asimetría de la distribución en Y
17. `grid_skewness_z` - Asimetría de la distribución en Z

**Momentos de Inercia del Grid (3):**
18. `grid_moi_1` - Primer momento de inercia del grid
19. `grid_moi_2` - Segundo momento de inercia del grid
20. `grid_moi_3` - Tercer momento de inercia del grid

### Shape Analysis (3 features) - ✨ **AMPLIADO**

21. `moi_principal_1` - Primer momento principal de inercia (posiciones atómicas)
22. `moi_principal_2` - Segundo momento principal de inercia
23. `moi_principal_3` - Tercer momento principal de inercia

### Radial Distribution (2 features)

24. `rdf_mean` - Media de la distribución radial
25. `rdf_kurtosis` - Curtosis de la distribución radial

### Spatial Analysis (2 features)

26. `entropy_spatial` - Entropía espacial 3D
27. `ms_bandwidth` - Bandwidth de Mean Shift clustering

## Diferencias con el Extractor del Usuario

El código de extracción del usuario (`extractor_final_optimizado.py`) calcula **20 features** (19 + n_vacancies):

**Features del usuario (19 sin target):**
- ✅ Las 11 primeras coinciden (occupancy + gradients + surface + entropy)
- ❌ **NO incluye** grid_com_x/y/z (centro de masa)
- ❌ **NO incluye** grid_skewness_x/y/z (asimetría)
- ✅ Incluye grid_moi_1/2/3
- ❌ **Solo incluye** moi_principal_3 (no 1 y 2)
- ✅ Incluye rdf_mean, rdf_kurtosis
- ✅ Incluye entropy_spatial, ms_bandwidth

**Features del modelo legacy (27):**
- ✅ Las 11 primeras (occupancy + gradients + surface + entropy)
- ✅ **INCLUYE** grid_com_x/y/z (centro de masa)
- ✅ **INCLUYE** grid_skewness_x/y/z (asimetría)
- ✅ Incluye grid_moi_1/2/3
- ✅ **INCLUYE** moi_principal_1/2/3 (los tres)
- ✅ Incluye rdf_mean, rdf_kurtosis
- ✅ Incluye entropy_spatial, ms_bandwidth

**Conclusión:** El modelo del usuario fue entrenado con un **extractor diferente** (versión legacy) que incluía más features geométricas (centro de masa, asimetría, y los 3 momentos principales).

## Configuración Importante

En `opentopologyc/config/extractor_config.py`:

```python
compute_hull_features: bool = False  # DESACTIVADO: no compatible con modelo legacy
```

Las `hull_features` están desactivadas porque el modelo legacy no fue entrenado con ellas.

## Compatibilidad con Modelos

### ✅ Modelo Legacy (actual del usuario)
- Requiere: **27 features base**
- Incluye: grid_com, grid_skewness, 3 moi_principal
- Sin hull_features
- **Compatible ahora** ✅

### ⚠️ Modelo con Extractor Simple (19 features)
- Requiere: **19 features base**
- No incluye: grid_com, grid_skewness, solo moi_principal_3
- Sin hull_features
- **NO compatible** con código actual (demasiadas features)

### Solución para Compatibilidad
Si necesitas entrenar un modelo nuevo que coincida con tu `extractor_final_optimizado.py`, debes:

1. **Opción A:** Actualizar tu extractor para incluir las 27 features (recomendado)
2. **Opción B:** Crear una versión del `FeatureExtractor` que solo extraiga 19 features
3. **Opción C:** Agregar un parámetro en la config para elegir entre "legacy" (27) o "simple" (19)

## Verificación

Para verificar que las features se extraen correctamente:

```bash
python test_feature_extraction.py
```

Este script verifica:
- ✅ Cantidad exacta de features (27)
- ✅ Orden correcto de features
- ✅ Sin features adicionales no deseadas
- ✅ Tipos de datos numéricos válidos

## Conclusión

✅ El sistema de predicción ahora extrae las **27 features** que el modelo legacy espera, incluyendo:
- Centro de masa del grid (grid_com_x/y/z)
- Asimetría del grid (grid_skewness_x/y/z)
- Los 3 momentos principales de inercia (moi_principal_1/2/3)

Esto resuelve el error de incompatibilidad y permite predicciones correctas con el modelo actual.
