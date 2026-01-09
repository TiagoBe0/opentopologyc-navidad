# OpenTopologyC Navidad 🎄

Sistema inteligente para detección de vacancias en simulaciones atomísticas usando Machine Learning y análisis topológico.

## 🚀 Inicio Rápido

### Interfaz Gráfica Qt (Recomendado)
```bash
python main_qt.py
```

### Interfaz Gráfica Tkinter (Alternativa)
```bash
python main.py
```

### Entrenamiento Rápido
```bash
# Edita parámetros en scripts/quick_train.py primero
python scripts/quick_train.py
```

---

## 📂 Estructura del Proyecto

```
opentopologyc-navidad/
├── main.py                    # Entrada principal (Tkinter)
├── main_qt.py                 # Entrada principal (Qt)
├── requirements_qt.txt        # Dependencias
│
├── 📚 docs/                   # Documentación
│   ├── README_QT.md           # Guía interfaz Qt
│   ├── VALIDACION_CIENTIFICA.md
│   └── ...
│
├── 🛠️ scripts/                # Scripts utilitarios
│   ├── quick_train.py         # Entrenamiento rápido
│   ├── validate_system.py     # Validar sistema
│   ├── fix_qt_backend.py      # Reparar Qt
│   └── check_qt_versions.py   # Verificar Qt
│
├── 💎 core/                   # Lógica del negocio
│   ├── wigner_seitz.py        # Algoritmo Wigner-Seitz
│   ├── feature_extractor.py   # Extracción de features
│   ├── training_pipeline.py   # Pipeline de entrenamiento
│   └── ...
│
├── 🖼️ gui/                    # Interfaces Tkinter
│   ├── main_gui.py
│   ├── prediction_gui.py
│   └── ...
│
├── 🎨 gui_qt/                 # Interfaces Qt
│   ├── visualizer_3d_qt.py    # Visualizador 3D
│   ├── prediction_gui_qt.py
│   └── ...
│
├── ⚙️ config/                 # Configuración
│   └── extractor_config.py
│
└── 🧪 tests/                  # Tests unitarios
    └── test_wigner_seitz.py
```

---

## 🔬 Características Principales

### 🎯 Detección de Vacancias
- **Machine Learning:** Random Forest para predicción
- **Wigner-Seitz:** Método tradicional de física de materiales
- **Comparación:** Ambos métodos lado a lado

### 📊 Visualización 3D
- Visualizador interactivo con OVITO
- Mapas de calor de defectos
- Rotación y zoom en tiempo real

### 🧮 Análisis Topológico
- Alpha shapes
- Clustering espacial
- Features geométricos avanzados

---

## 📖 Documentación

- **[Guía Qt](docs/README_QT.md)** - Uso de la interfaz Qt
- **[Validación Científica](docs/VALIDACION_CIENTIFICA.md)** - Protocolo de validación
- **[Scripts Utilitarios](scripts/README.md)** - Guía de scripts
- **[Documentación Completa](docs/)** - Toda la documentación

---

## 🛠️ Instalación

### Dependencias
```bash
pip install -r requirements_qt.txt
```

### Verificar Sistema
```bash
python scripts/validate_system.py
```

### Reparar Qt (si hay problemas)
```bash
python scripts/fix_qt_backend.py
```

---

## 🎓 Uso

### 1. Extracción de Features
```python
from core.feature_extractor import FeatureExtractor
from config.extractor_config import ExtractorConfig

config = ExtractorConfig(
    input_dir="path/to/dumps",
    probe_radius=2.0,
    a0=3.532,
    lattice_type="fcc"
)

extractor = FeatureExtractor(config)
features = extractor.extract_all_features(positions)
```

### 2. Análisis Wigner-Seitz
```python
from core.wigner_seitz import count_vacancies_wigner_seitz

results = count_vacancies_wigner_seitz(
    reference_file="perfect.dump",
    defective_file="defective.dump",
    use_pbc=True,
    use_affine=False
)

print(f"Vacancias: {results['n_vacancies']}")
print(f"Concentración: {results['vacancy_concentration']*100:.3f}%")
```

### 3. Entrenamiento de Modelo
```bash
# Edita parámetros en scripts/quick_train.py
python scripts/quick_train.py
```

---

## 🧪 Tests

```bash
# Ejecutar todos los tests
python -m unittest discover tests

# Test específico de Wigner-Seitz
python tests/test_wigner_seitz.py
```

---

## 📝 Correcciones Recientes

### ✅ Wigner-Seitz (Última actualización)
- Bug de coordenadas escaladas corregido
- Validación de estructuras mejorada
- Detección de intersticiales con umbral de distancia
- Manejo robusto de errores
- Suite completa de tests unitarios

---

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo licencia MIT.

---

## 👥 Autores

OpenTopologyC Team - Detección de vacancias con ML y análisis topológico

---

## 🙏 Agradecimientos

- OVITO para visualización 3D
- Scikit-learn para Machine Learning
- PyQt5 para interfaces gráficas
- SciPy para análisis científico
