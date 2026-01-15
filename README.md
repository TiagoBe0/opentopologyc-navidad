# OpenTopologyC Kit-Tools

**Versión simplificada enfocada en predicción de vacancias**

## 📋 Descripción

OpenTopologyC Kit-Tools es una versión optimizada que incluye **únicamente** las herramientas necesarias para la predicción de vacancias en estructuras cristalinas a partir de dumps LAMMPS.

Esta versión **NO incluye**:
- ❌ Entrenamiento de modelos ML
- ❌ Extracción masiva de features
- ❌ Pipeline completo de procesamiento

Esta versión **SÍ incluye**:
- ✅ **Alpha Shape filtering** - Filtrado de átomos superficiales
- ✅ **Clustering** - Múltiples algoritmos (KMeans, MeanShift, HDBSCAN, Hierarchical)
- ✅ **Predicción** - Usando modelos ML pre-entrenados
- ✅ **Visualizador 3D** - Visualización interactiva con matplotlib

## 🚀 Inicio Rápido

### Requisitos

```bash
python >= 3.9
pip install -r requirements_qt.txt
```

### Instalación

```bash
# Clonar repositorio
git clone <repo-url>
cd opentopologyc-navidad

# Instalar dependencias
pip install -r requirements_qt.txt
```

### Ejecutar

```bash
# Opción 1: Launcher principal
python main.py

# Opción 2: Directamente desde scripts
python scripts/main_qt.py
```

## 🔧 Flujo de Trabajo

### Paso 1: Alpha Shape (Opcional)
Filtra átomos superficiales usando el algoritmo Alpha Shape con ghost particles.

**Parámetros:**
- `probe_radius`: Radio de sonda (Å) - típicamente 2.0-3.0
- `ghost_layers`: Número de capas fantasma - típicamente 2-3
- `lattice_param` (a0): Parámetro de red del material

### Paso 2: Clustering (Opcional)
Agrupa átomos en clusters usando diferentes algoritmos.

**Algoritmos disponibles:**
- **KMeans** - Particionado en K clusters
- **MeanShift** - Clustering por densidad
- **HDBSCAN** - Clustering jerárquico por densidad
- **Hierarchical** - Clustering jerárquico mejorado

### Paso 3: Predicción
Predice vacancias usando un modelo ML pre-entrenado.

**Opciones:**
- **Sin clustering**: Predicción directa sobre todos los átomos
- **Con clustering**: Predicción cluster por cluster, luego suma

**Parámetros del material:**
- `a0`: Parámetro de red (Å)
- `lattice_type`: Tipo de celda (fcc, bcc, hcp, diamond, sc)
- `total_atoms`: Número total de átomos en cristal perfecto

## 📂 Estructura del Proyecto

```
opentopologyc-navidad/
├── main.py                      # Punto de entrada
├── scripts/
│   └── main_qt.py               # Launcher Qt
├── opentopologyc/
│   ├── core/                    # Lógica principal
│   │   ├── alpha_shape_filter.py    # Alpha Shape
│   │   ├── clustering_engine.py     # Clustering
│   │   ├── prediction_pipeline.py   # Pipeline de predicción
│   │   ├── feature_extractor.py     # Extracción de features
│   │   ├── model_manager.py         # Gestión de modelos
│   │   ├── loader.py                # Carga de dumps
│   │   └── ...
│   ├── gui_qt/                  # Interfaz gráfica Qt
│   │   ├── prediction_gui_qt.py     # GUI principal
│   │   ├── visualizer_3d_qt.py      # Visualizador 3D
│   │   └── base_window.py           # Base para ventanas
│   └── config/
│       └── extractor_config.py      # Configuración
├── models/                      # Modelos ML pre-entrenados
└── requirements_qt.txt          # Dependencias
```

## 🎯 Uso de Modelos Pre-entrenados

Los modelos deben estar en formato `.pkl` (joblib) y colocarse en:
```
models/
└── vacancy_rf_v1.0/
    ├── model.pkl           # Modelo Random Forest
    └── metadata.json       # Metadatos (opcional)
```

La GUI detectará automáticamente modelos en el directorio `models/`.

## 🔬 Features Extraídas para Predicción

El sistema extrae automáticamente estas features durante la predicción:

| Categoría | Features |
|-----------|----------|
| **Grid** | 20 features topológicos en grid 3D |
| **Hull** | Volumen, área, ratio del convex hull |
| **Inercia** | Momentos de inercia principales |
| **Radial** | Distribución radial de átomos |
| **Entropía** | Entropía espacial |
| **Clustering** | Bandwidth de densidad |

## 📊 Formato de Dumps LAMMPS

Los dumps deben seguir este formato:

```
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
16384
ITEM: BOX BOUNDS pp pp pp
0.0 70.64
0.0 70.64
0.0 70.64
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 1.766 1.766 0.0
...
```

**Columnas requeridas:**
- `id`: ID del átomo
- `type`: Tipo de átomo
- `x y z`: Coordenadas

## 🐛 Solución de Problemas

### Error: "No module named 'opentopologyc'"
```bash
# Asegúrate de ejecutar desde el directorio raíz
python main.py
# O agrega el path manualmente
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Error: "Cannot load model"
- Verifica que el archivo `.pkl` existe
- Verifica que el modelo fue entrenado con scikit-learn compatible
- Usa `model_manager.py` para validar el modelo

### Visualización 3D no funciona
- Verifica que matplotlib usa backend QtAgg
- Reinstala PySide6: `pip install --upgrade PySide6`

## 📝 Licencia

[Especificar licencia]

## 👥 Contribuciones

Esta es la versión kit-tools (solo predicción). Para entrenamiento y extracción, ver rama principal.

## 🔗 Enlaces

- Rama principal: [opentopologyc](https://github.com/...)
- Documentación completa: [docs/](docs/)
