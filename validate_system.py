#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de Validación Científica - OpenTopologyC

Este script te guía paso a paso para:
1. Extraer features de tus dumps
2. Entrenar modelo con métricas completas
3. Validar resultados científicamente
4. Hacer predicciones con parámetros configurables

Uso:
    python validate_system.py
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from core.pipeline import ExtractorPipeline
from core.training_pipeline import TrainingPipeline
from core.prediction_pipeline import PredictionPipeline
from core.model_manager import ModelManager
from config.extractor_config import ExtractorConfig


def print_section(title):
    """Imprime sección decorada"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def step1_extract_features():
    """
    PASO 1: Extraer features de dumps LAMMPS

    Necesitas una carpeta con dumps de entrenamiento donde:
    - Cada dump tiene un nombre que indica las vacancias (ej: 1.2_vac, 3.6_vac)
    - O tienes un CSV con labels
    """
    print_section("PASO 1: EXTRACCIÓN DE FEATURES")

    # Configurar extractor
    print("Configurando extractor...")

    # AJUSTA ESTOS PARÁMETROS SEGÚN TU MATERIAL
    config = ExtractorConfig(
        input_dir="",  # ← LLENAR: Ruta a tu carpeta con dumps
        probe_radius=2.0,           # Radio de sonda OVITO (Å)
        total_atoms=16384,          # Átomos en cristal perfecto
        a0=3.532,                   # Parámetro de red (Å) - Cu: 3.532, Al: 4.05
        lattice_type="fcc",         # Tipo de red

        # Features a computar
        compute_grid_features=True,
        compute_hull_features=True,
        compute_inertia_features=True,
        compute_radial_features=True,
        compute_entropy_features=True,
        compute_clustering_features=True
    )

    print(f"  Directorio: {config.input_dir}")
    print(f"  Material: {config.lattice_type}, a0={config.a0} Å")
    print(f"  Átomos perfectos: {config.total_atoms}")

    # Extraer features
    print("\nExtrayendo features...")
    pipeline = ExtractorPipeline(config)
    df = pipeline.run()

    if df is not None:
        output_csv = f"{config.input_dir}/dataset_features.csv"
        print(f"\n✓ Features extraídos exitosamente!")
        print(f"  Muestras: {len(df)}")
        print(f"  Features: {len(df.columns)}")
        print(f"  CSV guardado: {output_csv}")
        return output_csv
    else:
        print("\n✗ Error al extraer features")
        return None


def step2_train_model(csv_file):
    """
    PASO 2: Entrenar modelo con métricas completas

    Usa el CSV generado en el Paso 1 para entrenar un modelo
    Random Forest con todas las métricas de evaluación.
    """
    print_section("PASO 2: ENTRENAMIENTO DE MODELO")

    if not csv_file or not Path(csv_file).exists():
        print(f"✗ CSV no encontrado: {csv_file}")
        print("\nPrimero ejecuta el Paso 1 para generar el CSV de features")
        return None

    print(f"Dataset: {csv_file}")

    # Configurar training pipeline
    print("\nConfigurando entrenamiento...")

    pipeline = TrainingPipeline(
        csv_file=csv_file,
        model_output="model_vacancy.pkl",  # Modelo legacy
        n_estimators=100,                  # Número de árboles
        max_depth=None,                    # Profundidad (None = sin límite)
        test_size=0.2,                     # 20% para test
        random_state=42,

        # NUEVO: ModelManager integrado
        use_model_manager=True,
        model_name="vacancy_rf",
        model_version="1.0",
        target_column=None                 # Auto-detectar columna target
    )

    print(f"  n_estimators: {pipeline.n_estimators}")
    print(f"  max_depth: {pipeline.max_depth}")
    print(f"  test_size: {pipeline.test_size}")

    # Entrenar
    print("\nEntrenando modelo...")
    print("(Esto mostrará métricas completas: accuracy, confusion matrix, feature importance)")

    result = pipeline.train()

    # Resumen
    print_section("RESUMEN DE ENTRENAMIENTO")
    print(f"✓ Accuracy: {result['accuracy']:.4f}")
    print(f"✓ Modelo legacy: {result['model_path']}")
    print(f"✓ Modelo versionado: {result['model_dir']}")

    return result


def step3_analyze_results(result):
    """
    PASO 3: Analizar resultados científicamente

    Verifica que el modelo tenga buen rendimiento:
    - Accuracy > 70% (mínimo aceptable)
    - Confusion matrix sin bias extremo
    - Features importantes tienen sentido físico
    """
    print_section("PASO 3: ANÁLISIS DE RESULTADOS")

    if not result:
        print("✗ No hay resultados para analizar")
        return

    accuracy = result['accuracy']

    # Verificar accuracy
    print("1. ACCURACY")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    if accuracy >= 0.9:
        print("   ✓ EXCELENTE: Accuracy > 90%")
    elif accuracy >= 0.7:
        print("   ✓ ACEPTABLE: Accuracy entre 70-90%")
    else:
        print("   ⚠ BAJO: Accuracy < 70%")
        print("   Recomendaciones:")
        print("   - Verificar calidad de datos")
        print("   - Agregar más muestras de entrenamiento")
        print("   - Experimentar con más features")
        print("   - Probar otros algoritmos (XGBoost, SVM)")

    # Confusion matrix
    print("\n2. CONFUSION MATRIX")
    cm = result.get('confusion_matrix')
    if cm:
        print("   (Ver output del entrenamiento arriba)")
        print("   Verifica que:")
        print("   - Diagonal principal tenga valores altos (predicciones correctas)")
        print("   - Valores fuera de diagonal sean bajos (errores)")

    # Feature importance
    print("\n3. FEATURE IMPORTANCE")
    importances = result.get('feature_importances')
    feature_names = result.get('feature_names')

    if importances and feature_names:
        print("   Top 5 features más importantes:")
        indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        for i in range(min(5, len(indices))):
            idx = indices[i]
            print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

        print("\n   Verifica que:")
        print("   - Features importantes tengan sentido físico")
        print("   - ej: hull_volume, grid_count, radial_mean")

    # Recomendaciones
    print("\n4. PRÓXIMOS PASOS")
    if accuracy >= 0.7:
        print("   ✓ Modelo aceptable, puedes:")
        print("   - Hacer predicciones con datos nuevos")
        print("   - Experimentar con hiperparámetros")
        print("   - Comparar con otros algoritmos")
    else:
        print("   ⚠ Modelo necesita mejoras:")
        print("   - Recolectar más datos de entrenamiento")
        print("   - Verificar que labels sean correctos")
        print("   - Revisar que features se calculen correctamente")


def step4_make_predictions():
    """
    PASO 4: Hacer predicciones con parámetros configurables

    Usa el modelo entrenado para predecir vacancias en dumps nuevos.
    """
    print_section("PASO 4: HACER PREDICCIONES")

    print("Para hacer predicciones:")
    print("\n1. DESDE GUI (Recomendado):")
    print("   python main_qt.py")
    print("   - Ir a 'Predicción'")
    print("   - Cargar dump nuevo")
    print("   - Cargar modelo entrenado (model_vacancy.pkl)")
    print("   - Configurar 'Parámetros del Material':")
    print("     * Total atoms: 16384")
    print("     * a0: 3.532 Å (Cu) o 4.05 Å (Al)")
    print("     * Tipo de red: fcc, bcc, etc.")
    print("   - Ver visualización 3D de resultados")

    print("\n2. DESDE CÓDIGO:")
    print("   Ver ejemplo en: validate_predictions.py")


def step5_compare_models():
    """
    PASO 5: Comparar versiones de modelos

    Usa ModelManager para comparar diferentes versiones
    """
    print_section("PASO 5: GESTIÓN DE MODELOS")

    manager = ModelManager()

    print("Modelos disponibles:")
    models = manager.list_models()

    if not models:
        print("  (No hay modelos versionados aún)")
        print("\n  Entrena un modelo con use_model_manager=True")
    else:
        manager.print_summary()

        print("\nComparar modelos por accuracy:")
        best = manager.compare_models(metric='accuracy')
        if best:
            print(f"  Mejor modelo: {best[0]['name']} v{best[0]['version']}")
            print(f"  Accuracy: {best[0]['accuracy']:.4f}")


def main():
    """Ejecuta validación completa del sistema"""
    print("\n" + "="*80)
    print("  VALIDACIÓN CIENTÍFICA - OpenTopologyC")
    print("  Sistema de Predicción de Vacancias Atómicas")
    print("="*80)

    print("\nEste script te guiará paso a paso para validar el sistema.")
    print("Asegúrate de tener:")
    print("  ✓ Carpeta con dumps LAMMPS de entrenamiento")
    print("  ✓ Conocer parámetros del material (a0, lattice_type, total_atoms)")

    while True:
        print("\n" + "-"*80)
        print("MENÚ:")
        print("  1. Extraer features de dumps")
        print("  2. Entrenar modelo")
        print("  3. Analizar resultados")
        print("  4. Información de predicciones")
        print("  5. Gestión de modelos (ModelManager)")
        print("  6. Validación completa (1→2→3)")
        print("  0. Salir")
        print("-"*80)

        choice = input("\nSelecciona una opción: ").strip()

        if choice == "1":
            csv_file = step1_extract_features()
            if csv_file:
                print(f"\n✓ CSV generado: {csv_file}")
                print("  Siguiente paso: Opción 2 (Entrenar modelo)")

        elif choice == "2":
            csv_file = input("\nRuta al CSV de features: ").strip()
            result = step2_train_model(csv_file)
            if result:
                print("\n  Siguiente paso: Opción 3 (Analizar resultados)")

        elif choice == "3":
            # Cargar último resultado (simplificado)
            print("\nCarga el resultado del entrenamiento más reciente")
            print("(O revisa el output del Paso 2)")
            result = {}  # Placeholder
            step3_analyze_results(result)

        elif choice == "4":
            step4_make_predictions()

        elif choice == "5":
            step5_compare_models()

        elif choice == "6":
            # Validación completa
            print_section("VALIDACIÓN COMPLETA")
            csv_file = step1_extract_features()
            if csv_file:
                result = step2_train_model(csv_file)
                if result:
                    step3_analyze_results(result)

        elif choice == "0":
            print("\n¡Hasta luego!")
            break

        else:
            print("\n✗ Opción inválida")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
