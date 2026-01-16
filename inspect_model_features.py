#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para inspeccionar las features que espera un modelo entrenado
"""

import sys
import joblib
from pathlib import Path
import argparse

def inspect_model(model_path):
    """
    Inspecciona un modelo de sklearn y muestra las features que espera
    """
    print("="*80)
    print("INSPECTOR DE FEATURES DEL MODELO")
    print("="*80)
    print(f"\nModelo: {model_path}")

    try:
        # Cargar modelo
        print("\n1. Cargando modelo...")
        model = joblib.load(model_path)
        print(f"   ✓ Tipo de modelo: {type(model).__name__}")

        # Obtener feature names si están disponibles
        print("\n2. Features esperadas por el modelo:")

        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
            print(f"   ✓ Total de features: {len(features)}")
            print(f"\n   Lista de features:")
            for i, feat in enumerate(features, 1):
                print(f"      {i:2d}. {feat}")

            # Guardar a archivo
            output_file = Path(model_path).parent / "model_features.txt"
            with open(output_file, 'w') as f:
                f.write("Features esperadas por el modelo:\n")
                f.write("="*80 + "\n\n")
                for i, feat in enumerate(features, 1):
                    f.write(f"{i:2d}. {feat}\n")

            print(f"\n   ✓ Lista guardada en: {output_file}")

            # Categorizar features
            print(f"\n3. Categorización de features:")
            categories = {}
            for feat in features:
                if feat.startswith('occupancy_'):
                    cat = 'occupancy'
                elif feat.startswith('grid_'):
                    cat = 'grid'
                elif feat.startswith('moi_'):
                    cat = 'moi'
                elif feat.startswith('rdf_'):
                    cat = 'rdf'
                elif feat.startswith('hull_'):
                    cat = 'hull'
                elif feat.startswith('entropy_'):
                    cat = 'entropy'
                elif feat in ['ms_bandwidth', 'num_points']:
                    cat = 'other'
                else:
                    cat = 'unknown'

                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(feat)

            for cat, feats in sorted(categories.items()):
                print(f"\n   {cat.upper()} ({len(feats)} features):")
                for feat in feats:
                    print(f"      • {feat}")

        else:
            print("   ⚠️  El modelo no tiene feature_names_in_ guardado")
            print("   Esto puede pasar con modelos entrenados en sklearn < 1.0")

            # Intentar obtener n_features
            if hasattr(model, 'n_features_in_'):
                print(f"   ℹ️  Número de features: {model.n_features_in_}")
            else:
                print("   ⚠️  No se puede determinar el número de features")

        # Información adicional del modelo
        print(f"\n4. Información adicional:")
        if hasattr(model, 'n_estimators'):
            print(f"   • n_estimators: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"   • max_depth: {model.max_depth}")
        if hasattr(model, 'random_state'):
            print(f"   • random_state: {model.random_state}")

        print("\n" + "="*80)
        print("✅ Inspección completada")
        print("="*80 + "\n")

        return features if hasattr(model, 'feature_names_in_') else None

    except Exception as e:
        print(f"\n❌ Error al inspeccionar modelo: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspecciona las features que espera un modelo de ML"
    )
    parser.add_argument(
        'model_path',
        help='Ruta al archivo del modelo (.pkl o .joblib)'
    )

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"❌ Error: No se encuentra el archivo {args.model_path}")
        sys.exit(1)

    features = inspect_model(args.model_path)

    if features is not None:
        sys.exit(0)
    else:
        sys.exit(1)
