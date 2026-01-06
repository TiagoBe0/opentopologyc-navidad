# OpenTopologyC - Versión Qt

## 🚀 Ejecutar la Aplicación

### Opción 1: Script Principal (Recomendado)
```bash
python main_qt.py
```

### Opción 2: Directamente desde gui_qt
```bash
python gui_qt/main_window.py
```

### Opción 3: Desde cualquier directorio
```bash
cd /ruta/a/opentopologyc-navidad
python -m main_qt
```

## 📦 Dependencias

Asegúrate de tener instaladas las siguientes librerías:

```bash
pip install PyQt5 numpy scipy matplotlib scikit-learn joblib ovito pandas
```

## 🎯 Módulos Disponibles

### 1. 🔬 Extractor de Features
- Procesa archivos dump de LAMMPS
- Extrae características topológicas
- Genera dataset para entrenamiento

### 2. 🤖 Entrenamiento
- Entrena modelos Random Forest
- Cross-validation
- Guarda modelos entrenados

### 3. 🎯 Predicción + Visualizador 3D
- Carga modelo entrenado
- Aplica Alpha Shape con Ghost Particles
- Clustering de nanoporos (KMeans, MeanShift, Aglomerativo, HDBSCAN)
- Predicción de vacancias
- **Visualización 3D interactiva** con etapas múltiples:
  - Etapa 1: Dump original
  - Etapa 2: Alpha Shape (átomos superficiales)
  - Etapa 3: Clustering (todos los clusters coloreados)
  - Etapa 4: Cluster seleccionado

## 🔧 Estructura del Proyecto

```
opentopologyc-navidad/
├── main_qt.py           # Punto de entrada Qt (usar este)
├── main.py              # Punto de entrada Tkinter (legacy)
├── gui_qt/              # Interfaces Qt
│   ├── main_window.py
│   ├── prediction_gui_qt.py
│   ├── train_gui_qt.py
│   ├── extractor_gui_qt.py
│   └── visualizer_3d_qt.py
├── gui/                 # Interfaces Tkinter (legacy)
├── core/                # Lógica del pipeline
│   ├── prediction_pipeline.py
│   ├── clustering_engine.py
│   ├── alpha_shape_filter.py
│   └── ...
└── config/              # Configuraciones
```

## ⚠️ Problemas Comunes

### Error: "No module named 'gui_qt'"
**Solución:** Ejecuta desde el directorio raíz usando `python main_qt.py`

### Error: "No module named 'PyQt5'"
**Solución:**
```bash
pip install PyQt5
```

### Error: "invalid literal for int()"
**Solución:** Este error ya fue corregido. Haz pull de la rama:
```bash
git pull origin claude/integrate-gui-windows-D2Jbi
```

## 📊 Flujo de Trabajo Típico

1. **Extraer Features:**
   - Abre "🔬 Extractor de Features"
   - Selecciona carpeta con dumps
   - Configura parámetros
   - Ejecuta extracción → genera `dataset_features.csv`

2. **Entrenar Modelo:**
   - Abre "🤖 Entrenamiento"
   - Carga `dataset_features.csv`
   - Entrena modelo
   - Guarda modelo → `model_rf.joblib`

3. **Predecir con Visualización:**
   - Abre "🎯 Predicción + Visualizador"
   - Carga dump de prueba
   - Carga modelo entrenado
   - Configura Alpha Shape y Clustering
   - Ejecuta predicción
   - **Explora etapas en el visualizador 3D**

## 🎨 Controles del Visualizador 3D

- **Selector de Etapas:** Dropdown para cambiar entre las 4 etapas del pipeline
- **Ejes/Grid:** Checkboxes para mostrar/ocultar
- **Tamaño:** Slider para ajustar tamaño de átomos
- **Alpha:** Slider para ajustar transparencia
- **Rotación:** Click y arrastra en el gráfico 3D
- **Zoom:** Scroll del mouse

## 📝 Notas

- Los archivos temporales se guardan en la misma carpeta del dump de entrada
- El visualizador carga automáticamente todas las etapas después de la predicción
- Clustering es opcional (desactivar si solo quieres Alpha Shape)
