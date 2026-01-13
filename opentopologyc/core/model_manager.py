#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ModelManager - Gestión de modelos ML con versionado y metadata

Organiza modelos en estructura:
    models/
    ├── vacancy_rf_v1.0/
    │   ├── model.pkl
    │   ├── metadata.json
    │   └── scaler.pkl (opcional)
    └── vacancy_rf_v2.0/
        └── ...
"""

import json
from pathlib import Path
from datetime import datetime
import joblib
from typing import Dict, List, Optional, Any


class ModelManager:
    """Gestión centralizada de modelos ML con versionado"""

    def __init__(self, base_dir: str = "models"):
        """
        Args:
            base_dir: Directorio base para guardar modelos
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def save_model(
        self,
        model: Any,
        name: str,
        version: str,
        metadata: Dict[str, Any],
        scaler: Optional[Any] = None
    ) -> str:
        """
        Guarda modelo con metadata y opcionalmente un scaler

        Args:
            model: Modelo scikit-learn entrenado
            name: Nombre del modelo (ej: "vacancy_rf")
            version: Versión (ej: "1.0")
            metadata: Dict con info adicional:
                - accuracy: float
                - n_estimators: int (para RF)
                - max_depth: int (para RF)
                - dataset: str (ruta al CSV de entrenamiento)
                - features: List[str] (nombres de features)
                - etc.
            scaler: Opcional, StandardScaler u otro preprocessor

        Returns:
            Ruta al directorio del modelo
        """
        # Crear directorio del modelo
        model_dir = self.base_dir / f"{name}_v{version}"
        model_dir.mkdir(exist_ok=True)

        # Guardar modelo
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)

        # Guardar scaler si existe
        if scaler is not None:
            scaler_path = model_dir / "scaler.pkl"
            joblib.dump(scaler, scaler_path)

        # Preparar metadata completa
        metadata_full = {
            "name": name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_path),
            "has_scaler": scaler is not None,
            **metadata
        }

        # Guardar metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_full, f, indent=2, ensure_ascii=False)

        print(f"✓ Modelo guardado: {model_dir}")
        print(f"  - Accuracy: {metadata.get('accuracy', 'N/A')}")
        print(f"  - Features: {len(metadata.get('features', []))} features")

        return str(model_dir)

    def load_model(self, name: str, version: str) -> tuple:
        """
        Carga modelo y metadata

        Args:
            name: Nombre del modelo
            version: Versión

        Returns:
            Tuple (model, metadata, scaler)
            scaler será None si no existe
        """
        model_dir = self.base_dir / f"{name}_v{version}"

        if not model_dir.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_dir}")

        # Cargar modelo
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Archivo model.pkl no encontrado en {model_dir}")

        model = joblib.load(model_path)

        # Cargar metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Cargar scaler si existe
        scaler = None
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        return model, metadata, scaler

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lista todos los modelos disponibles con su metadata

        Returns:
            Lista de dicts con metadata de cada modelo
        """
        models = []

        for model_dir in sorted(self.base_dir.iterdir()):
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        models.append(metadata)

        # Ordenar por fecha de creación (más reciente primero)
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return models

    def get_latest_version(self, name: str) -> Optional[str]:
        """
        Obtiene la versión más reciente de un modelo

        Args:
            name: Nombre del modelo

        Returns:
            Versión como string (ej: "2.0") o None si no existe
        """
        models = self.list_models()
        matching = [m for m in models if m.get('name') == name]

        if not matching:
            return None

        # Asumir que están ordenados por fecha
        return matching[0].get('version')

    def delete_model(self, name: str, version: str) -> bool:
        """
        Elimina un modelo

        Args:
            name: Nombre del modelo
            version: Versión

        Returns:
            True si se eliminó, False si no existía
        """
        model_dir = self.base_dir / f"{name}_v{version}"

        if not model_dir.exists():
            return False

        # Eliminar archivos
        import shutil
        shutil.rmtree(model_dir)

        print(f"✓ Modelo eliminado: {model_dir}")
        return True

    def compare_models(self, metric: str = 'accuracy') -> List[Dict[str, Any]]:
        """
        Compara modelos según una métrica

        Args:
            metric: Métrica a usar para comparación (default: 'accuracy')

        Returns:
            Lista de modelos ordenados por la métrica (mejor primero)
        """
        models = self.list_models()

        # Filtrar modelos que tienen la métrica
        models_with_metric = [m for m in models if metric in m]

        # Ordenar por métrica (descendente)
        models_with_metric.sort(key=lambda x: x[metric], reverse=True)

        return models_with_metric

    def print_summary(self):
        """Imprime resumen de todos los modelos"""
        models = self.list_models()

        if not models:
            print("No hay modelos guardados.")
            return

        print(f"\n{'='*80}")
        print(f"MODELOS DISPONIBLES ({len(models)})")
        print(f"{'='*80}\n")

        for i, model in enumerate(models, 1):
            print(f"{i}. {model.get('name', 'Unknown')} v{model.get('version', '?')}")
            print(f"   Creado: {model.get('created_at', 'N/A')}")
            print(f"   Accuracy: {model.get('accuracy', 'N/A'):.4f}" if isinstance(model.get('accuracy'), float) else f"   Accuracy: N/A")
            print(f"   Dataset: {model.get('dataset', 'N/A')}")
            print(f"   Features: {len(model.get('features', []))} features")
            print()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear manager
    manager = ModelManager()

    # Ejemplo: guardar modelo ficticio
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    # Entrenar modelo dummy
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)

    # Guardar con metadata
    manager.save_model(
        model=model,
        name="vacancy_rf",
        version="1.0",
        metadata={
            "accuracy": 0.85,
            "n_estimators": 50,
            "max_depth": None,
            "dataset": "data/train_v1.csv",
            "features": [f"feature_{i}" for i in range(10)],
            "notes": "Modelo baseline con features básicos"
        }
    )

    # Listar modelos
    manager.print_summary()

    # Cargar modelo
    loaded_model, metadata, scaler = manager.load_model("vacancy_rf", "1.0")
    print(f"\n✓ Modelo cargado: {metadata['name']} v{metadata['version']}")
