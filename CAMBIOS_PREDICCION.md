# Cambios en el Sistema de Predicción

## Resumen

Se modificó la etapa de predicción para garantizar compatibilidad total con modelos entrenados usando el extractor de features estándar (19 features base).

## Fecha
2026-01-16

## Archivos Modificados

### 1. `opentopologyc/core/prediction_pipeline.py`

**Cambio Principal:**
- Modificado el método `_extract_features()` para usar `extract_all_features()` del `FeatureExtractor`
- Eliminada la extracción manual de features que podía causar inconsistencias
- Removida la adición de `num_points` que no era parte del modelo entrenado

**Antes:**
```python
def _extract_features(self, dump_file):
    raw = self.loader.load(dump_file)
    pos = raw["positions"]
    pos_norm, box_size, _ = self.normalizer.normalize(pos)

    feats = {}
    if self.config.compute_grid_features:
        feats.update(self.features_extractor.grid_features(pos_norm, box_size))
    if self.config.compute_inertia_features:
        feats.update(self.features_extractor.inertia_feature(pos))
    # ... más llamadas manuales

    feats["num_points"] = pos.shape[0]  # ❌ No compatible con modelo
    return feats
```

**Después:**
```python
def _extract_features(self, dump_file):
    """
    Extrae features usando el pipeline completo de extract_all_features()
    para garantizar compatibilidad con el modelo entrenado.
    """
    raw = self.loader.load(dump_file)
    pos = raw["positions"]

    # Usar el pipeline completo de extract_all_features()
    # Esto extrae las 19 features base en el orden correcto
    feats = self.features_extractor.extract_all_features(
        positions=pos,
        n_vacancies=None  # No incluir target en predicción
    )

    return feats
```

## Beneficios de los Cambios

### ✅ Compatibilidad Garantizada
- Las features se extraen en el **mismo orden** que durante el entrenamiento
- Se usan los **mismos métodos** de cálculo
- Se aplica la **misma normalización** (centrado + PCA con covariance_eigh)

### ✅ Consistencia de Normalización
- El `extract_all_features()` usa su propio método `normalize_positions()`
- Centrado en el origen antes de PCA
- PCA optimizado con `svd_solver='covariance_eigh'`
- Box size con límite de 10.0

### ✅ Sin Features Adicionales
- Eliminado `num_points` que causaba incompatibilidad
- No se agregan features no esperadas por el modelo

### ✅ Mantenimiento Simplificado
- Un solo punto de extracción de features (`extract_all_features()`)
- Cambios futuros solo en `FeatureExtractor` se propagan automáticamente
- Menos código duplicado

## Features Extraídas (19 en total)

Las 19 features base que se extraen son:

### Occupancy (5 features)
1. `occupancy_total` - Total de celdas ocupadas en el grid
2. `occupancy_fraction` - Fracción de celdas ocupadas
3. `occupancy_x_mean` - Media de ocupación en eje X
4. `occupancy_y_mean` - Media de ocupación en eje Y
5. `occupancy_z_mean` - Media de ocupación en eje Z

### Gradientes (5 features)
6. `occupancy_gradient_x` - Gradiente en dirección X
7. `occupancy_gradient_y` - Gradiente en dirección Y
8. `occupancy_gradient_z` - Gradiente en dirección Z
9. `occupancy_gradient_total` - Suma de gradientes
10. `occupancy_surface` - Superficie total (gradientes)

### Grid Analysis (4 features)
11. `grid_entropy` - Entropía del grid 3D
12. `grid_moi_1` - Primer momento de inercia del grid
13. `grid_moi_2` - Segundo momento de inercia del grid
14. `grid_moi_3` - Tercer momento de inercia del grid

### Shape Analysis (1 feature)
15. `moi_principal_3` - Tercer momento principal de inercia (posiciones atómicas)

### Radial Distribution (2 features)
16. `rdf_mean` - Media de la distribución radial
17. `rdf_kurtosis` - Curtosis de la distribución radial

### Spatial Analysis (2 features)
18. `entropy_spatial` - Entropía espacial 3D
19. `ms_bandwidth` - Bandwidth de Mean Shift clustering

## Configuración Importante

En `opentopologyc/config/extractor_config.py`:

```python
compute_hull_features: bool = False  # DESACTIVADO: hull features no son compatibles con modelos legacy
```

**Importante:** Las `hull_features` (volumen, área, simplices) están desactivadas por defecto para mantener compatibilidad con modelos entrenados con las 19 features base. Si tu modelo fue entrenado CON hull_features, debes cambiar este flag a `True`.

## Compatibilidad con Código de Entrenamiento

El pipeline de predicción ahora es **100% compatible** con el extractor de entrenamiento que usa:

```python
# extractor_final_optimizado.py
FINAL_FEATURES = [
    'occupancy_total',
    'occupancy_fraction',
    'occupancy_x_mean',
    'occupancy_y_mean',
    'occupancy_z_mean',
    'occupancy_gradient_x',
    'occupancy_gradient_y',
    'occupancy_gradient_z',
    'occupancy_gradient_total',
    'occupancy_surface',
    'grid_entropy',
    'grid_moi_1',
    'grid_moi_2',
    'grid_moi_3',
    'moi_principal_3',
    'rdf_mean',
    'rdf_kurtosis',
    'entropy_spatial',
    'ms_bandwidth',
    'n_vacancies'  # target
]
```

## Testing

Se creó `test_feature_extraction.py` para verificar:
- ✅ Cantidad exacta de features (19)
- ✅ Orden correcto de features
- ✅ Sin features adicionales no deseadas
- ✅ Tipos de datos numéricos válidos

## Notas Adicionales

### Normalización
- Ya no se usa `PositionNormalizer` de forma externa
- La normalización se hace internamente en `extract_all_features()`
- Garantiza consistencia con el pipeline de entrenamiento

### Validación
- El método `_predict_features()` sigue filtrando columnas prohibidas:
  - `n_vacancies`, `n_atoms_surface`, `vacancies`, `file`, `num_atoms_real`, `num_points`
- Esto asegura que solo las 19 features lleguen al modelo

## Conclusión

✅ El sistema de predicción ahora extrae **exactamente las mismas features** que se usaron durante el entrenamiento del modelo, garantizando predicciones precisas y compatibilidad total.
