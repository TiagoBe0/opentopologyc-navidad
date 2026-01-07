import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from core.model_manager import ModelManager


class TrainingPipeline:
    def __init__(
        self,
        csv_file,
        model_output,
        n_estimators=100,
        max_depth=None,
        test_size=0.2,
        random_state=42,
        use_model_manager=True,
        model_name="vacancy_rf",
        model_version="1.0"
    ):
        self.csv_file = csv_file
        self.model_output = model_output
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state
        self.use_model_manager = use_model_manager
        self.model_name = model_name
        self.model_version = model_version
        self.model_manager = ModelManager() if use_model_manager else None

    # ======================================================
    # DATA
    # ======================================================
    def load_data(self):
        import pandas as pd

        df = pd.read_csv(self.csv_file)

        # Separar features y target
        X = df.drop(columns=["label", "target", "file", "n_vacancies"], errors="ignore")
        y = df["label"] if "label" in df else df["target"]

        # Obtener nombres de features
        feature_names = X.columns.tolist()

        return X.values, y.values, feature_names

    # ======================================================
    # TRAIN
    # ======================================================
    def train(self, progress_callback=None):
        """
        Entrena modelo con métricas completas

        Args:
            progress_callback: Función callback(step, total)

        Returns:
            Dict con accuracy, métricas, y model_path
        """

        # ---------- STEP 1: LOAD ----------
        if progress_callback:
            progress_callback(1, 6)

        X, y, feature_names = self.load_data()

        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO DE MODELO")
        print(f"{'='*60}")
        print(f"Dataset: {self.csv_file}")
        print(f"Muestras: {len(X)}")
        print(f"Features: {len(feature_names)}")
        print(f"Clases únicas: {np.unique(y)}")

        # ---------- STEP 2: SPLIT ----------
        if progress_callback:
            progress_callback(2, 6)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Mantener proporción de clases
        )

        print(f"\nSplit:")
        print(f"  Train: {len(X_train)} muestras")
        print(f"  Test:  {len(X_test)} muestras")

        # ---------- STEP 3: MODEL ----------
        if progress_callback:
            progress_callback(3, 6)

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=self.random_state,
        )

        print(f"\nEntrenando Random Forest...")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")

        model.fit(X_train, y_train)

        # ---------- STEP 4: EVAL ----------
        if progress_callback:
            progress_callback(4, 6)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Métricas completas
        print(f"\n{'='*60}")
        print(f"RESULTADOS")
        print(f"{'='*60}")
        print(f"\nAccuracy: {acc:.4f}")

        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        # Feature importance (top 10)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            print(f"\nTop 10 Features Más Importantes:")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                print(f"  {i+1}. {feat_name}: {importances[idx]:.4f}")

        # ---------- STEP 5: SAVE ----------
        if progress_callback:
            progress_callback(5, 6)

        # Guardar modelo tradicional (para compatibilidad)
        joblib.dump(model, self.model_output)
        print(f"\n✓ Modelo guardado (legacy): {self.model_output}")

        # Guardar con ModelManager si está habilitado
        model_dir = None
        if self.use_model_manager and self.model_manager:
            metadata = {
                "accuracy": float(acc),
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "test_size": self.test_size,
                "dataset": str(self.csv_file),
                "features": feature_names,
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
                "confusion_matrix": cm.tolist(),
                "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            }

            model_dir = self.model_manager.save_model(
                model=model,
                name=self.model_name,
                version=self.model_version,
                metadata=metadata
            )

        # ---------- STEP 6: COMPLETE ----------
        if progress_callback:
            progress_callback(6, 6)

        print(f"\n{'='*60}")
        print(f"✓ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}\n")

        return {
            "accuracy": acc,
            "model_path": self.model_output,
            "model_dir": model_dir,
            "confusion_matrix": cm.tolist(),
            "feature_importances": importances.tolist() if hasattr(model, 'feature_importances_') else None,
            "feature_names": feature_names
        }
