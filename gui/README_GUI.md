# OpenTopologyC - Guía de Uso de las Interfaces Gráficas

## Descripción General

OpenTopologyC incluye tres interfaces gráficas integradas:

1. **GUI Principal (`main_gui.py`)**: Ventana de inicio que permite acceder a las otras dos interfaces
2. **Extractor de Features (`gui_extractor.py`)**: Configuración y ejecución de extracción de características topológicas
3. **Entrenamiento de Modelos (`train_gui.py`)**: Configuración y ejecución del entrenamiento de modelos Random Forest

## Inicio Rápido

### Ejecutar la aplicación

Desde la raíz del proyecto:

```bash
python3 main.py
```

O directamente:

```bash
python3 gui/main_gui.py
```

## Interfaces Disponibles

### 1. GUI Principal (Main GUI)

La ventana principal ofrece dos opciones:

- **🔬 Extractor de Features**: Abre la interfaz para extraer características topológicas de archivos dump
- **🤖 Entrenamiento de Modelos**: Abre la interfaz para entrenar modelos de predicción

**Atajos de teclado:**
- `F1`: Abrir Extractor de Features
- `F2`: Abrir Entrenamiento de Modelos
- `Escape`: Salir de la aplicación

### 2. Extractor de Features

Permite configurar y ejecutar el pipeline de extracción de características.

#### Parámetros configurables:

**Directorio de Datos:**
- Directorio de dumps: Carpeta con archivos dump a procesar

**Parámetros de Extracción:**
- Radio de sonda: Radio usado para cálculos de superficie (default: 2.0)
- Surface distance: Habilitar cálculo de distancia a superficie
- Valor surface distance: Distancia para el cálculo (default: 4.0)

**Parámetros del Material:**
- Átomos totales: Número total de átomos en el sistema (default: 16384)
- Parámetro de red (a0): Parámetro de red del material (default: 3.532)
- Tipo de red: fcc, bcc, hcp, sc, diamond (default: fcc)

**Features a Calcular:**
- Grid features: Características de grilla espacial
- Hull (Convex Hull): Características del casco convexo
- Inertia moments: Momentos de inercia
- Radial features: Características radiales
- Entropy: Entropía del sistema
- Clustering / Bandwidth: Características de clustering

#### Flujo de trabajo:

1. Seleccionar directorio de dumps
2. Configurar parámetros
3. Seleccionar features a calcular
4. **💾 Crear Configuración**: Guarda la configuración en `config_extractor.json`
5. **🚀 Run Pipeline**: Ejecuta el pipeline de extracción

**Atajos de teclado:**
- `Escape`: Cerrar ventana y volver a Main GUI
- `Enter`: Ejecutar pipeline (si está habilitado)

### 3. Entrenamiento de Modelos

Permite entrenar modelos Random Forest para predecir vacancias.

#### Parámetros configurables:

**Datos de Entrada:**
- Dataset CSV: Archivo CSV con features extraídas
- Directorio de salida: Donde se guardarán los modelos y gráficos (default: modelos_entrenados)

**Parámetros del Modelo:**
- Tamaño del test set (%): Porcentaje de datos para test (default: 20%)
- Random state: Semilla para reproducibilidad (default: 42)
- Top features a mostrar: Número de features más importantes a mostrar (default: 20)

**Configuración del modelo:**
- Random Forest con 200 árboles
- max_features='sqrt'
- Imputación de valores faltantes (mediana)
- Escalado de features (StandardScaler)
- Out-of-bag score habilitado

#### Flujo de trabajo:

1. Seleccionar dataset CSV (archivo de features extraídas)
2. Configurar directorio de salida
3. Ajustar parámetros del modelo
4. **🎯 Entrenar Modelo**: Inicia el entrenamiento
5. Ver resultados en la consola de salida
6. **📂 Cargar Modelo**: Cargar un modelo previamente entrenado

**Salidas generadas:**
- Modelo entrenado (.joblib)
- Gráficos de importancia de features
- Gráficos de métricas de evaluación
- Logs del entrenamiento

**Atajos de teclado:**
- `Escape`: Cerrar ventana y volver a Main GUI
- `Ctrl+O`: Seleccionar dataset CSV
- `Ctrl+S`: Seleccionar directorio de salida
- `F5`: Entrenar modelo (si no está ejecutándose)

## Características de Integración

### Gestión de Ventanas

- La ventana principal (`MainGUI`) oculta temporalmente cuando se abre una ventana secundaria
- Al cerrar una ventana secundaria (Extractor o Training), la ventana principal vuelve a aparecer
- El botón "Salir" en ventanas secundarias cierra solo esa ventana y regresa a la principal
- El botón "Salir" en la ventana principal cierra toda la aplicación

### Ejecución en Hilos Separados

Tanto el Extractor como el Training ejecutan sus procesos en hilos separados para:
- Mantener la interfaz responsiva durante la ejecución
- Permitir ver el progreso en tiempo real
- Evitar bloqueos de la UI

### Consola de Salida

La GUI de Training incluye una consola integrada con:
- Mensajes codificados por colores (INFO, WARNING, ERROR, SUCCESS)
- Capacidad de guardar logs
- Botón para limpiar consola

## Solución de Problemas

### Error: "No module named 'tkinter'"

Tkinter debe estar instalado en tu sistema:

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (con Homebrew)
brew install python-tk

# Windows
# Tkinter viene incluido con Python
```

### Error: "No se pudo importar ExtractorPipeline"

Asegúrate de que el módulo `core.pipeline` esté disponible:

```bash
ls core/pipeline.py
```

### Error: "No module named 'train_step'"

Verifica que el archivo `core/train_step.py` existe y define la clase `RandomForestTrainer`.

## Estructura de Archivos

```
opentopologyc-navidad/
├── main.py                  # Punto de entrada principal
├── gui/
│   ├── main_gui.py          # Interfaz principal
│   ├── gui_extractor.py     # Interfaz de extracción
│   ├── train_gui.py         # Interfaz de entrenamiento
│   └── README_GUI.md        # Este archivo
├── core/
│   ├── pipeline.py          # Pipeline de extracción
│   └── train_step.py        # Clase de entrenamiento
└── config/
    └── extractor_config.py  # Configuración del extractor
```

## Notas Técnicas

### Cambios Realizados para la Integración

1. **Importaciones corregidas:**
   - `train_gui.py` ahora importa correctamente desde `core.train_step`

2. **Gestión de ventanas mejorada:**
   - Las ventanas secundarias usan `destroy()` en lugar de `quit()`
   - La ventana principal usa `withdraw()` y `deiconify()` para gestionar la visibilidad

3. **main.py actualizado:**
   - Ahora lanza `MainGUI` en lugar de `ExtractorGUI` directamente

### Recomendaciones

- Crear la configuración del extractor antes de ejecutar el pipeline
- Guardar logs importantes de entrenamientos usando el botón "💾 Guardar Logs"
- Usar nombres descriptivos para los directorios de salida de modelos
- Verificar métricas del modelo en la consola antes de usar el modelo en producción

## Contacto y Soporte

Para reportar problemas o sugerir mejoras, contacta al equipo de desarrollo.
