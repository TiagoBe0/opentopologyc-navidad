import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .model_manager import ModelManager


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
        model_version="1.0",
        target_column=None,  # Columna target configurable
        task_type="regression",  # "regression" o "classification"
        output_dir=None  # Directorio para gráficos y métricas
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
        self.target_column = target_column
        self.task_type = task_type
        self.output_dir = output_dir or str(Path(model_output).parent)
        self.model_manager = ModelManager() if use_model_manager else None

    # ======================================================
    # DATA
    # ======================================================
    def load_data(self):
        import pandas as pd

        df = pd.read_csv(self.csv_file)

        # Detectar columna target
        target_col = self._detect_target_column(df)

        # Separar features y target
        # Columnas a excluir de features (target y metadata)
        exclude_cols = [target_col, "file", "num_points", "num_atoms_real"]

        # Agregar aliases comunes de target a excluir
        target_aliases = ["label", "target", "n_vacancies", "vacancies", "y"]
        exclude_cols.extend([col for col in target_aliases if col in df.columns and col != target_col])

        X = df.drop(columns=exclude_cols, errors="ignore")
        y = df[target_col]

        # Obtener nombres de features
        feature_names = X.columns.tolist()

        print(f"✓ Columna target detectada: '{target_col}'")
        print(f"✓ Features a usar: {len(feature_names)}")

        return X.values, y.values, feature_names

    def _detect_target_column(self, df):
        """
        Detecta automáticamente la columna target

        Prioridad:
        1. self.target_column (si fue especificado)
        2. Columnas candidatas comunes
        3. Error si no encuentra ninguna

        Returns:
            Nombre de la columna target
        """
        # Si se especificó explícitamente
        if self.target_column:
            if self.target_column in df.columns:
                return self.target_column
            else:
                raise ValueError(
                    f"Columna target especificada '{self.target_column}' no existe en el CSV.\n"
                    f"Columnas disponibles: {list(df.columns)}"
                )

        # Buscar columnas candidatas (orden de prioridad)
        candidates = ["n_vacancies", "label", "target", "vacancies", "y", "class"]

        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        # Si no encuentra, mostrar error con columnas disponibles
        raise ValueError(
            f"No se encontró columna target en el CSV.\n"
            f"Columnas disponibles: {list(df.columns)}\n\n"
            f"Soluciones:\n"
            f"1. Especificar columna target explícitamente:\n"
            f"   pipeline = TrainingPipeline(..., target_column='tu_columna')\n\n"
            f"2. Renombrar una columna a 'n_vacancies' o 'label':\n"
            f"   import pandas as pd\n"
            f"   df = pd.read_csv('{self.csv_file}')\n"
            f"   df['label'] = df['tu_columna']  # Renombrar\n"
            f"   df.to_csv('{self.csv_file}', index=False)\n"
        )

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

    # ======================================================
    # EXPORTACIÓN DE MÉTRICAS Y GRÁFICOS
    # ======================================================
    def create_regression_plots(self, y_train, y_pred_train, y_test, y_pred_test,
                               feature_names, feature_importances):
        """
        Crea gráficos de evaluación para regresión

        Genera 4 subplots:
        1. Predicciones vs Reales
        2. Análisis de Residuos
        3. Feature Importance
        4. Distribución de Errores
        """
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Calcular métricas para el título
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Evaluación del Modelo - Regresión', fontsize=16, fontweight='bold')

        # 1. Predicciones vs Reales
        axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, s=50, color='steelblue')
        axes[0, 0].plot([y_test.min(), y_test.max()],
                        [y_test.min(), y_test.max()],
                        'r--', lw=2, label='Ideal')
        axes[0, 0].set_xlabel('Vacancias Reales', fontsize=12)
        axes[0, 0].set_ylabel('Vacancias Predichas', fontsize=12)
        axes[0, 0].set_title(f'Predicciones vs Reales\nR² = {r2_test:.4f}', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuos
        residuos = y_test - y_pred_test
        axes[0, 1].scatter(y_pred_test, residuos, alpha=0.6, s=50, color='steelblue')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicciones', fontsize=12)
        axes[0, 1].set_ylabel('Residuos (Real - Predicción)', fontsize=12)
        axes[0, 1].set_title('Análisis de Residuos', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Feature Importance (Top 15)
        indices = np.argsort(feature_importances)[::-1][:15]
        top_features = [feature_names[i] for i in indices]
        top_importances = feature_importances[indices]

        axes[1, 0].barh(range(len(indices)), top_importances, color='steelblue')
        axes[1, 0].set_yticks(range(len(indices)))
        axes[1, 0].set_yticklabels(top_features, fontsize=9)
        axes[1, 0].set_xlabel('Importancia', fontsize=12)
        axes[1, 0].set_title('Top 15 Features', fontsize=12)
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # 4. Distribución de Errores
        axes[1, 1].hist(residuos, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Error (Real - Predicción)', fontsize=12)
        axes[1, 1].set_ylabel('Frecuencia', fontsize=12)
        axes[1, 1].set_title(f'Distribución de Errores\nRMSE={rmse_test:.2f}, MAE={mae_test:.2f}',
                            fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        plot_path = Path(self.output_dir) / 'evaluacion_modelo.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráficos guardados: {plot_path}")

        plt.close()

        return str(plot_path)

    def export_metrics(self, metrics_dict):
        """Exporta métricas a CSV"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        metrics_path = Path(self.output_dir) / 'metricas.csv'

        # Convertir dict a DataFrame
        df = pd.DataFrame([metrics_dict])
        df.to_csv(metrics_path, index=False)

        print(f"📋 Métricas guardadas: {metrics_path}")
        return str(metrics_path)

    def export_feature_importance(self, feature_names, feature_importances):
        """Exporta importancia de features a CSV"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Crear DataFrame con importancias
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        importance_path = Path(self.output_dir) / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)

        print(f"⭐ Importancia de features guardada: {importance_path}")
        return str(importance_path)

    # ======================================================
    # ENTRENAMIENTO CON REGRESIÓN
    # ======================================================
    def train_regression(self, progress_callback=None):
        """
        Entrena modelo de REGRESIÓN con métricas completas y exportación

        Args:
            progress_callback: Función callback(step, total)

        Returns:
            Dict con métricas, rutas de archivos generados
        """

        # ---------- STEP 1: LOAD ----------
        if progress_callback:
            progress_callback(1, 7)

        X, y, feature_names = self.load_data()

        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO DE MODELO - REGRESIÓN")
        print(f"{'='*60}")
        print(f"Dataset: {self.csv_file}")
        print(f"Muestras: {len(X)}")
        print(f"Features: {len(feature_names)}")
        print(f"Target: Min={y.min():.1f}, Max={y.max():.1f}, Media={y.mean():.1f}")

        # ---------- STEP 2: SPLIT ----------
        if progress_callback:
            progress_callback(2, 7)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        print(f"\nSplit:")
        print(f"  Train: {len(X_train)} muestras")
        print(f"  Test:  {len(X_test)} muestras")

        # ---------- STEP 3: MODEL ----------
        if progress_callback:
            progress_callback(3, 7)

        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=self.random_state,
        )

        print(f"\nEntrenando Random Forest Regressor...")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")

        model.fit(X_train, y_train)

        # ---------- STEP 4: EVAL ----------
        if progress_callback:
            progress_callback(4, 7)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Métricas train
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        # Métricas test
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        print(f"\n{'='*60}")
        print(f"RESULTADOS")
        print(f"{'='*60}")

        print(f"\n🔵 ENTRENAMIENTO:")
        print(f"   RMSE: {rmse_train:.4f}")
        print(f"   MAE:  {mae_train:.4f}")
        print(f"   R²:   {r2_train:.4f}")

        print(f"\n🟢 PRUEBA:")
        print(f"   RMSE: {rmse_test:.4f}")
        print(f"   MAE:  {mae_test:.4f}")
        print(f"   R²:   {r2_test:.4f}")

        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"\n⭐ TOP 15 FEATURES MÁS IMPORTANTES:")
        for i in range(min(15, len(indices))):
            idx = indices[i]
            feat_name = feature_names[idx]
            print(f"   {i+1}. {feat_name:30s}: {importances[idx]:.4f}")

        # ---------- STEP 5: CREAR GRÁFICOS ----------
        if progress_callback:
            progress_callback(5, 7)

        plot_path = self.create_regression_plots(
            y_train, y_pred_train,
            y_test, y_pred_test,
            feature_names, importances
        )

        # ---------- STEP 6: EXPORTAR MÉTRICAS ----------
        if progress_callback:
            progress_callback(6, 7)

        metrics_dict = {
            'rmse_train': rmse_train,
            'mae_train': mae_train,
            'r2_train': r2_train,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'r2_test': r2_test,
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_features': len(feature_names),
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth
        }

        metrics_path = self.export_metrics(metrics_dict)
        importance_path = self.export_feature_importance(feature_names, importances)

        # ---------- STEP 7: GUARDAR MODELO ----------
        if progress_callback:
            progress_callback(7, 7)

        # Guardar modelo
        joblib.dump(model, self.model_output)
        print(f"\n💾 Modelo guardado: {self.model_output}")

        # Guardar nombres de features
        features_path = Path(self.output_dir) / 'feature_names.txt'
        with open(features_path, 'w') as f:
            for fname in feature_names:
                f.write(f"{fname}\n")
        print(f"📝 Features guardadas: {features_path}")

        print(f"\n{'='*60}")
        print(f"✓ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        print(f"\n📁 Archivos generados en: {self.output_dir}/")
        print(f"   • {Path(self.model_output).name}")
        print(f"   • feature_names.txt")
        print(f"   • feature_importance.csv")
        print(f"   • metricas.csv")
        print(f"   • evaluacion_modelo.png")
        print(f"{'='*60}\n")

        return {
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'r2_test': r2_test,
            'model_path': self.model_output,
            'plot_path': plot_path,
            'metrics_path': metrics_path,
            'importance_path': importance_path,
            'features_path': str(features_path)
        }
