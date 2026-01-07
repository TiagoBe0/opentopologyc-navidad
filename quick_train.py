#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick Train - Entrena un modelo rápidamente

Script simple para entrenar un modelo con tus datos.
Ajusta los parámetros según tu material y ejecuta.

Uso:
    python quick_train.py
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from core.pipeline import ExtractorPipeline
from core.training_pipeline import TrainingPipeline
from config.extractor_config import ExtractorConfig


# ============================================================
# CONFIGURACIÓN - AJUSTA ESTOS PARÁMETROS
# ============================================================

# Ruta a tu carpeta con dumps de entrenamiento
# Ejemplo: "/home/santi-simaf/Documentos/software-final-vacancias-navidad/db_test_pequeña"
DUMP_DIR = "/ruta/a/tu/carpeta/con/dumps"  # ← CAMBIAR ESTO

# Parámetros del material
MATERIAL = {
    "total_atoms": 16384,    # Átomos en cristal perfecto
    "a0": 3.532,             # Parámetro de red (Å)
                             # Cu: 3.532, Al: 4.05, Au: 4.08, Ag: 4.09
    "lattice_type": "fcc"    # Tipo de red: fcc, bcc, hcp, diamond, sc
}

# Parámetros de extracción
EXTRACTOR = {
    "probe_radius": 2.0,     # Radio de sonda OVITO (Å)
}

# Parámetros de entrenamiento
TRAINING = {
    "n_estimators": 100,     # Número de árboles en Random Forest
    "max_depth": None,       # Profundidad máxima (None = sin límite)
    "test_size": 0.2,        # 20% para test
    "model_version": "1.0",  # Versión del modelo
    "target_column": None    # Columna target (None = auto-detectar)
                             # Opciones comunes: "n_vacancies", "label", "target"
}

# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

def main():
    """Ejecuta extracción y entrenamiento"""

    print("\n" + "="*80)
    print("  QUICK TRAIN - OpenTopologyC")
    print("="*80)

    # Verificar que la carpeta existe
    if not Path(DUMP_DIR).exists():
        print(f"\n✗ ERROR: Carpeta no encontrada: {DUMP_DIR}")
        print("\nAjusta la variable DUMP_DIR en este script con la ruta correcta.")
        print("Ejemplo: DUMP_DIR = '/home/usuario/mis_dumps'")
        return

    print(f"\n✓ Carpeta encontrada: {DUMP_DIR}")

    # ========================================
    # PASO 1: EXTRAER FEATURES
    # ========================================
    print("\n" + "-"*80)
    print("PASO 1: EXTRAYENDO FEATURES")
    print("-"*80)

    config = ExtractorConfig(
        input_dir=DUMP_DIR,
        probe_radius=EXTRACTOR["probe_radius"],
        total_atoms=MATERIAL["total_atoms"],
        a0=MATERIAL["a0"],
        lattice_type=MATERIAL["lattice_type"],
        compute_grid_features=True,
        compute_hull_features=True,
        compute_inertia_features=True,
        compute_radial_features=True,
        compute_entropy_features=True,
        compute_clustering_features=True
    )

    print(f"\nMaterial: {MATERIAL['lattice_type'].upper()}")
    print(f"  a0 = {MATERIAL['a0']} Å")
    print(f"  Átomos perfectos = {MATERIAL['total_atoms']}")
    print(f"  Probe radius = {EXTRACTOR['probe_radius']} Å")

    print("\nProcesando dumps...")
    pipeline = ExtractorPipeline(config)
    df = pipeline.run()

    if df is None or len(df) == 0:
        print("\n✗ ERROR: No se pudieron extraer features")
        print("Verifica que:")
        print("  - La carpeta contiene dumps LAMMPS válidos")
        print("  - Los dumps tienen formato correcto")
        return

    csv_file = f"{DUMP_DIR}/dataset_features.csv"
    print(f"\n✓ Features extraídos exitosamente!")
    print(f"  Muestras: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    print(f"  CSV: {csv_file}")

    # ========================================
    # PASO 2: ENTRENAR MODELO
    # ========================================
    print("\n" + "-"*80)
    print("PASO 2: ENTRENANDO MODELO")
    print("-"*80)

    # Nota: TrainingPipeline ahora detecta automáticamente la columna target
    # Busca: n_vacancies, label, target, vacancies, y, class
    # O puedes especificar TRAINING["target_column"] = "tu_columna"

    training_pipeline = TrainingPipeline(
        csv_file=csv_file,
        model_output="model_vacancy.pkl",
        n_estimators=TRAINING["n_estimators"],
        max_depth=TRAINING["max_depth"],
        test_size=TRAINING["test_size"],
        use_model_manager=True,
        model_name="vacancy_rf",
        model_version=TRAINING["model_version"],
        target_column=TRAINING["target_column"]
    )

    print(f"\nRandom Forest:")
    print(f"  n_estimators = {TRAINING['n_estimators']}")
    print(f"  max_depth = {TRAINING['max_depth']}")
    print(f"  test_size = {TRAINING['test_size']}")

    print("\nEntrenando...")
    result = training_pipeline.train()

    # ========================================
    # PASO 3: RESULTADOS
    # ========================================
    print("\n" + "="*80)
    print("  VALIDACIÓN DE RESULTADOS")
    print("="*80)

    accuracy = result['accuracy']

    print(f"\n✓ ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

    if accuracy >= 0.9:
        print("   → EXCELENTE: Modelo muy preciso")
    elif accuracy >= 0.7:
        print("   → ACEPTABLE: Modelo funcional")
    else:
        print("   → BAJO: Modelo necesita mejoras")
        print("\n   Recomendaciones:")
        print("   - Agregar más muestras de entrenamiento")
        print("   - Verificar calidad de los datos")
        print("   - Experimentar con n_estimators más alto (200, 500)")

    print(f"\n✓ Modelos guardados:")
    print(f"   - Legacy: {result['model_path']}")
    print(f"   - Versionado: {result['model_dir']}")

    print("\n" + "="*80)
    print("  ✓ ENTRENAMIENTO COMPLETADO")
    print("="*80)

    print("\nPróximos pasos:")
    print("  1. Revisar confusion matrix y feature importance arriba")
    print("  2. Hacer predicciones con: python main_qt.py")
    print("  3. Comparar modelos con: ModelManager().print_summary()")

    print("\n✓ Para hacer predicciones:")
    print("   python main_qt.py")
    print("   → Ir a 'Predicción'")
    print("   → Cargar dump nuevo")
    print("   → Cargar modelo: model_vacancy.pkl")
    print("   → Configurar material en 'Parámetros del Material'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Programa interrumpido")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
