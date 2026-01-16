#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de verificación: extracción de features para predicción

Verifica que:
1. Se extraen exactamente las 19 features base
2. El orden coincide con el modelo entrenado
3. No hay features adicionales no deseadas
"""

import sys
import numpy as np
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from opentopologyc.config.extractor_config import ExtractorConfig
from opentopologyc.core.feature_extractor import FeatureExtractor

# Features esperadas del modelo legacy (27 features base)
EXPECTED_FEATURES = [
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
    # Centro de masa del grid (3)
    'grid_com_x',
    'grid_com_y',
    'grid_com_z',
    # Skewness del grid (3)
    'grid_skewness_x',
    'grid_skewness_y',
    'grid_skewness_z',
    # Momentos de inercia del grid (3)
    'grid_moi_1',
    'grid_moi_2',
    'grid_moi_3',
    # Momentos principales (3)
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


def test_feature_extraction():
    """
    Prueba la extracción de features con datos sintéticos
    """
    print("="*80)
    print("VERIFICACIÓN DE EXTRACCIÓN DE FEATURES")
    print("="*80)

    # Crear configuración
    config = ExtractorConfig(
        input_dir=".",  # dummy
        probe_radius=2.0,
        total_atoms=16384,
        a0=3.532,
        compute_grid_features=True,
        compute_hull_features=False,  # IMPORTANTE: False para compatibilidad
        compute_inertia_features=True,
        compute_radial_features=True,
        compute_entropy_features=True,
        compute_clustering_features=True
    )

    # Crear extractor
    extractor = FeatureExtractor(config)

    # Crear posiciones sintéticas (simulando átomos de superficie)
    np.random.seed(42)
    n_atoms = 1000
    positions = np.random.randn(n_atoms, 3) * 5.0  # Cluster centrado en origen

    print(f"\n✓ Configuración creada")
    print(f"✓ Posiciones sintéticas: {n_atoms} átomos")

    # Extraer features
    print(f"\n{'-'*80}")
    print("Extrayendo features...")
    print(f"{'-'*80}")

    features = extractor.extract_all_features(
        positions=positions,
        n_vacancies=None  # No incluir target en predicción
    )

    # Obtener lista de features extraídas
    extracted_features = list(features.keys())

    print(f"\n✓ Features extraídas: {len(extracted_features)}")
    print(f"\nFeatures obtenidas:")
    for i, feat in enumerate(extracted_features, 1):
        value = features[feat]
        print(f"  {i:2d}. {feat:30s} = {value:.6f}")

    # Verificaciones
    print(f"\n{'='*80}")
    print("VERIFICACIONES")
    print(f"{'='*80}")

    # 1. Verificar cantidad
    print(f"\n1. Cantidad de features:")
    print(f"   Esperadas: {len(EXPECTED_FEATURES)}")
    print(f"   Obtenidas: {len(extracted_features)}")

    if len(extracted_features) == len(EXPECTED_FEATURES):
        print(f"   ✓ CORRECTO: Cantidad coincide")
    else:
        print(f"   ✗ ERROR: Cantidad no coincide")
        return False

    # 2. Verificar orden
    print(f"\n2. Orden de features:")
    order_ok = True
    for i, (expected, actual) in enumerate(zip(EXPECTED_FEATURES, extracted_features), 1):
        if expected == actual:
            print(f"   ✓ Posición {i:2d}: {expected:30s}")
        else:
            print(f"   ✗ Posición {i:2d}: esperado '{expected}', obtenido '{actual}'")
            order_ok = False

    if order_ok:
        print(f"\n   ✓ CORRECTO: Orden coincide exactamente")
    else:
        print(f"\n   ✗ ERROR: Orden no coincide")
        return False

    # 3. Verificar que no haya features adicionales
    print(f"\n3. Features adicionales:")
    extra_features = set(extracted_features) - set(EXPECTED_FEATURES)
    missing_features = set(EXPECTED_FEATURES) - set(extracted_features)

    if extra_features:
        print(f"   ✗ ERROR: Features extras no deseadas: {extra_features}")
        return False
    else:
        print(f"   ✓ CORRECTO: No hay features adicionales")

    if missing_features:
        print(f"   ✗ ERROR: Features faltantes: {missing_features}")
        return False
    else:
        print(f"   ✓ CORRECTO: No faltan features")

    # 4. Verificar tipos de datos
    print(f"\n4. Tipos de datos:")
    types_ok = True
    for feat, value in features.items():
        if not isinstance(value, (int, float, np.number)):
            print(f"   ✗ ERROR: {feat} tiene tipo {type(value)}, esperado numérico")
            types_ok = False

    if types_ok:
        print(f"   ✓ CORRECTO: Todos los valores son numéricos")
    else:
        return False

    # 5. Verificar que no haya NaN (con datos sintéticos no debería haberlos)
    print(f"\n5. Valores NaN:")
    nan_count = sum(1 for v in features.values() if np.isnan(v))
    if nan_count == 0:
        print(f"   ✓ CORRECTO: No hay valores NaN")
    else:
        print(f"   ⚠ WARNING: {nan_count} features con valores NaN")
        for feat, value in features.items():
            if np.isnan(value):
                print(f"      - {feat}")

    # Resumen final
    print(f"\n{'='*80}")
    print("✅ VERIFICACIÓN EXITOSA")
    print("="*80)
    print("\nLa extracción de features es compatible con el modelo legacy:")
    print(f"  • 27 features base en el orden correcto")
    print(f"  • 20 grid features (occupancy, gradients, COM, skewness, MOI)")
    print(f"  • 3 momentos principales de inercia")
    print(f"  • 2 RDF features")
    print(f"  • 2 features espaciales (entropy, bandwidth)")
    print(f"  • Sin features adicionales (num_points, etc.)")
    print(f"  • Sin hull_features (compatibilidad con modelos legacy)")
    print(f"  • Valores numéricos válidos")
    print()

    return True


if __name__ == "__main__":
    try:
        success = test_feature_extraction()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR DURANTE LA VERIFICACIÓN:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
