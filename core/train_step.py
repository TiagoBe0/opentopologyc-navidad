#en este scrip hacemos la clase para entrenar el modelo y exportar un pkl a un directorio de modelos
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class RandomForestTrainer:
    """
    Clase para entrenar y evaluar modelos Random Forest con pipeline completo.
    
    Características:
    - Carga y preprocesamiento de datos
    - Pipeline con imputación y escalado
    - Entrenamiento de Random Forest
    - Evaluación completa con métricas
    - Análisis de importancia de features
    - Guardado de modelos y resultados
    """
    
    # Features que NO deben usarse para entrenamiento
    FORBIDDEN_FEATURES = ['n_vacancies', 'n_atoms_surface', 'vacancies', 'file']
    
    def __init__(self, random_state=42, logger=None):
        """
        Inicializa el entrenador.
        
        Args:
            random_state: Semilla para reproducibilidad
            logger: Logger personalizado (opcional)
        """
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)

        # Atributos que se llenarán durante el entrenamiento
        self.name_file = None
        self.pipeline = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.feature_names = None
        self.importance_df = None
        self.metrics = None
        
    def load_data(self, csv_path):
        """
        Carga el dataset y separa features del target.

        Args:
            csv_path: Ruta al archivo CSV

        Returns:
            tuple: (X, y) donde X son features e y es el target

        Raises:
            ValueError: Si no se encuentra 'n_vacancies' en el dataset
        """
        self.name_file = csv_path
        self.logger.info(f"📂 Cargando dataset: {csv_path}")
        df = pd.read_csv(csv_path, index_col='file')
        
        self.logger.info(f"   Muestras: {df.shape[0]} | Columnas: {df.shape[1]}")
        
        # Verificar target
        if 'n_vacancies' not in df.columns:
            raise ValueError("❌ No se encontró 'n_vacancies' en el dataset")
        
        # Separar target (y)
        y = df['n_vacancies'].astype(float)
        
        # Eliminar features prohibidas
        X = df.copy()
        for forbidden in self.FORBIDDEN_FEATURES:
            if forbidden in X.columns:
                X = X.drop(columns=[forbidden])
        
        # Solo columnas numéricas
        X = X.select_dtypes(include=[np.number])
        
        # Eliminar columnas completamente vacías (todas NaN)
        columns_before = len(X.columns)
        X = X.dropna(axis=1, how='all')
        columns_after = len(X.columns)
        
        if columns_before != columns_after:
            dropped = columns_before - columns_after
            self.logger.info(f"   ℹ️  Columnas completamente vacías eliminadas: {dropped}")
        
        self.logger.info(f"   Features de entrada: {len(X.columns)}")
        self.logger.info(f"   Features eliminadas: {[f for f in self.FORBIDDEN_FEATURES if f in df.columns]}")
        
        # Verificar valores faltantes
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            self.logger.info(f"   ⚠️  Valores faltantes detectados: {missing_count}")
            self.logger.info(f"   Serán imputados con la mediana")
        
        self.logger.info(f"\n📊 Estadísticas de vacancias:")
        self.logger.info(f"   Min: {y.min():.0f} | Max: {y.max():.0f} | Media: {y.mean():.1f} ± {y.std():.1f}")
        
        return X, y
    
    def create_pipeline(self, X):
        """
        Crea pipeline de preprocesamiento + modelo.
        
        Pipeline:
        1. Imputación de valores faltantes (mediana)
        2. Escalado (StandardScaler)
        3. Random Forest Regressor
        
        Args:
            X: DataFrame de features para obtener nombres de columnas
            
        Returns:
            Pipeline: Pipeline de sklearn completo
        """
        # Obtener nombres de columnas
        numeric_features = list(X.columns)
        
        # Preprocesador
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features)
            ],
            remainder='drop'
        )
        
        # Pipeline completo
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
                bootstrap=True,
                oob_score=True
            ))
        ])
        
        return pipeline
    
    def train(self, X, y, test_size=0.2):
        """
        Entrena el modelo con pipeline completo.
        
        Args:
            X: Features
            y: Target
            test_size: Proporción para test
            
        Returns:
            dict: Resultados del entrenamiento
        """
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        self.logger.info(f"\n🔀 División de datos:")
        self.logger.info(f"   Entrenamiento: {len(X_train)} muestras")
        self.logger.info(f"   Prueba: {len(X_test)} muestras")
        
        # Crear y entrenar pipeline
        self.logger.info(f"\n🌲 Entrenando Random Forest con pipeline...")
        self.logger.info(f"   Configuración:")
        self.logger.info(f"     • n_estimators: 200")
        self.logger.info(f"     • max_features: sqrt ({int(np.sqrt(len(X.columns)))} features por split)")
        self.logger.info(f"     • Preprocesamiento: Imputación + Escalado")
        
        pipeline = self.create_pipeline(X)
        pipeline.fit(X_train, y_train)
        
        self.logger.info("   ✅ Entrenamiento completo")
        
        # OOB Score
        oob_score = pipeline.named_steps['regressor'].oob_score_
        self.logger.info(f"   📊 OOB Score: {oob_score:.4f}")
        
        # Predicciones
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Guardar atributos
        self.pipeline = pipeline
        self.model = pipeline.named_steps['regressor']
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test
        self.feature_names = list(X.columns)
        
        results = {
            'pipeline': pipeline,
            'model': pipeline.named_steps['regressor'],
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'feature_names': list(X.columns)
        }
        
        return results
    
    def evaluate(self):
        """
        Evalúa el modelo y muestra métricas.
        
        Returns:
            dict: Métricas de evaluación
        """
        if self.model is None:
            raise ValueError("❌ El modelo no ha sido entrenado. Llama a train() primero.")
        
        # Métricas train
        rmse_train = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
        mae_train = mean_absolute_error(self.y_train, self.y_pred_train)
        r2_train = r2_score(self.y_train, self.y_pred_train)
        
        # Métricas test
        rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))
        mae_test = mean_absolute_error(self.y_test, self.y_pred_test)
        r2_test = r2_score(self.y_test, self.y_pred_test)
        
        # Error porcentual medio
        mape_train = np.mean(np.abs((self.y_train - self.y_pred_train) / (self.y_train + 1e-10))) * 100
        mape_test = np.mean(np.abs((self.y_test - self.y_pred_test) / (self.y_test + 1e-10))) * 100
        
        self.logger.info("\n" + "="*70)
        self.logger.info("📈 RESULTADOS DEL MODELO")
        self.logger.info("="*70)
        
        self.logger.info("\n🔵 ENTRENAMIENTO:")
        self.logger.info(f"   RMSE:  {rmse_train:.4f}")
        self.logger.info(f"   MAE:   {mae_train:.4f}")
        self.logger.info(f"   R²:    {r2_train:.4f}")
        self.logger.info(f"   MAPE:  {mape_train:.2f}%")
        
        self.logger.info("\n🟢 PRUEBA:")
        self.logger.info(f"   RMSE:  {rmse_test:.4f}")
        self.logger.info(f"   MAE:   {mae_test:.4f}")
        self.logger.info(f"   R²:    {r2_test:.4f}")
        self.logger.info(f"   MAPE:  {mape_test:.2f}%")
        
        # Indicador de overfitting
        overfit_indicator = (r2_train - r2_test) / r2_train * 100 if r2_train > 0 else 0
        self.logger.info(f"\n📉 Diferencia Train-Test R²: {overfit_indicator:.2f}%")
        if overfit_indicator > 10:
            self.logger.warning("   ⚠️  Posible overfitting detectado")
        else:
            self.logger.info("   ✅ Buen balance entre train y test")
        
        self.metrics = {
            'train': {'rmse': rmse_train, 'mae': mae_train, 'r2': r2_train, 'mape': mape_train},
            'test': {'rmse': rmse_test, 'mae': mae_test, 'r2': r2_test, 'mape': mape_test}
        }
        
        return self.metrics
    
    def analyze_feature_importance(self, top_n=20):
        """
        Analiza la importancia de cada feature con categorización.
        
        Args:
            top_n: Número de top features a mostrar
            
        Returns:
            DataFrame: DataFrame con importancias de features
        """
        if self.model is None:
            raise ValueError("❌ El modelo no ha sido entrenado. Llama a train() primero.")
        
        # CRÍTICO: Obtener los nombres de features que realmente usó el modelo
        try:
            if hasattr(self.pipeline, 'named_steps') and 'preprocessor' in self.pipeline.named_steps:
                preprocessor = self.pipeline.named_steps['preprocessor']
                if hasattr(preprocessor, 'get_feature_names_out'):
                    actual_feature_names = list(preprocessor.get_feature_names_out())
                else:
                    actual_feature_names = self.feature_names
            else:
                actual_feature_names = self.feature_names
        except:
            actual_feature_names = self.feature_names
        
        # Verificar que las longitudes coincidan
        n_importances = len(self.model.feature_importances_)
        if len(actual_feature_names) != n_importances:
            self.logger.warning(f"   ⚠️  Ajustando feature names: {len(actual_feature_names)} -> {n_importances}")
            actual_feature_names = actual_feature_names[:n_importances]
        
        # Crear DataFrame con importancias
        importance_df = pd.DataFrame({
            'feature': actual_feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Categorizar features
        def categorize_feature(fname):
            if 'occupancy' in fname or 'grid' in fname or 'fragment' in fname:
                return '📊 Grid3D'
            elif 'hull' in fname:
                return '🔷 ConvexHull'
            elif 'moi' in fname:
                return '⚖️ Momentos'
            elif 'rdf' in fname:
                return '📍 Radial'
            elif 'entropy' in fname:
                return '🌀 Entropía'
            elif 'bandwidth' in fname:
                return '🔍 Bandwidth'
            elif 'energy' in fname:
                return '⚡ Energía'
            else:
                return '❓ Otro'
        
        importance_df['category'] = importance_df['feature'].apply(categorize_feature)
        
        self.logger.info("\n" + "="*70)
        self.logger.info(f"⭐ TOP {min(top_n, len(importance_df))} FEATURES MÁS IMPORTANTES")
        self.logger.info("="*70)
        
        cumsum = 0
        for idx, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
            cumsum += row['importance']
            self.logger.info(f"{idx:2d}. {row['category']:15s} {row['feature']:35s}: "
                           f"{row['importance']:.4f} (acum: {cumsum:.3f})")
        
        # Resumen por categoría
        self.logger.info("\n" + "="*70)
        self.logger.info("📊 IMPORTANCIA POR CATEGORÍA")
        self.logger.info("="*70)
        
        category_importance = importance_df.groupby('category')['importance'].agg(['sum', 'count'])
        category_importance = category_importance.sort_values('sum', ascending=False)
        
        for cat, row in category_importance.iterrows():
            self.logger.info(f"{cat:15s}: {row['sum']:.4f} ({row['count']:2.0f} features)")
        
        self.importance_df = importance_df
        return importance_df
    
    def create_plots(self, output_dir):
        """
        Crea gráficos de evaluación mejorados.
        
        Args:
            output_dir: Directorio donde guardar los gráficos
        """
        if self.model is None:
            raise ValueError("❌ El modelo no ha sido entrenado. Llama a train() primero.")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Predicciones vs Reales (Test)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(self.y_test, self.y_pred_test, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
        ax1.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 
                 'r--', lw=2, label='Ideal')
        ax1.set_xlabel('Vacancias Reales', fontsize=11)
        ax1.set_ylabel('Vacancias Predichas', fontsize=11)
        ax1.set_title(f'Test: Predicciones vs Reales\nR² = {self.metrics["test"]["r2"]:.4f}, MAE = {self.metrics["test"]["mae"]:.2f}', 
                      fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Predicciones vs Reales (Train)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(self.y_train, self.y_pred_train, alpha=0.4, s=30, color='lightcoral', edgecolors='black', linewidth=0.3)
        ax2.plot([self.y_train.min(), self.y_train.max()], 
                 [self.y_train.min(), self.y_train.max()], 
                 'r--', lw=2, label='Ideal')
        ax2.set_xlabel('Vacancias Reales', fontsize=11)
        ax2.set_ylabel('Vacancias Predichas', fontsize=11)
        ax2.set_title(f'Train: Predicciones vs Reales\nR² = {self.metrics["train"]["r2"]:.4f}', 
                      fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuos (Test)
        ax3 = fig.add_subplot(gs[0, 2])
        residuos_test = self.y_test - self.y_pred_test
        ax3.scatter(self.y_pred_test, residuos_test, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
        ax3.axhline(y=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Predicciones', fontsize=11)
        ax3.set_ylabel('Residuos (Real - Pred)', fontsize=11)
        ax3.set_title('Análisis de Residuos (Test)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature Importance (Top 20)
        ax4 = fig.add_subplot(gs[1, :])
        importances = self.model.feature_importances_
        
        # Obtener feature names del análisis de importancia
        if self.importance_df is not None:
            feature_names = self.importance_df['feature'].tolist()
        else:
            feature_names = self.feature_names
        
        # Ajustar feature_names si es necesario
        if len(feature_names) != len(importances):
            feature_names = feature_names[:len(importances)]
        
        indices = np.argsort(importances)[::-1][:20]
        
        colors = []
        for i in indices:
            fname = feature_names[i]
            if 'occupancy' in fname or 'grid' in fname:
                colors.append('steelblue')
            elif 'hull' in fname:
                colors.append('coral')
            elif 'moi' in fname:
                colors.append('gold')
            elif 'energy' in fname:
                colors.append('lightgreen')
            else:
                colors.append('lightgray')
        
        ax4.barh(range(len(indices)), importances[indices], color=colors, edgecolor='black')
        ax4.set_yticks(range(len(indices)))
        ax4.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
        ax4.set_xlabel('Importancia', fontsize=11)
        ax4.set_title('Top 20 Features Más Importantes', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Distribución de Errores (Test)
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(residuos_test, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax5.axvline(x=0, color='r', linestyle='--', lw=2)
        ax5.set_xlabel('Error (Real - Pred)', fontsize=11)
        ax5.set_ylabel('Frecuencia', fontsize=11)
        ax5.set_title(f'Distribución de Errores\nMedia: {residuos_test.mean():.3f}', 
                      fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Q-Q Plot (normalidad de residuos)
        ax6 = fig.add_subplot(gs[2, 1])
        from scipy import stats
        stats.probplot(residuos_test, dist="norm", plot=ax6)
        ax6.set_title('Q-Q Plot (Normalidad)', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Error Absoluto vs Predicción
        ax7 = fig.add_subplot(gs[2, 2])
        abs_errors = np.abs(residuos_test)
        ax7.scatter(self.y_pred_test, abs_errors, alpha=0.6, s=50, color='coral', edgecolors='black', linewidth=0.5)
        ax7.set_xlabel('Predicciones', fontsize=11)
        ax7.set_ylabel('Error Absoluto', fontsize=11)
        ax7.set_title('Error Absoluto vs Predicción', fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        plot_path = Path(output_dir) / 'evaluacion_completa.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"\n📊 Gráficos guardados: {plot_path}")
        
        plt.close()
    
    def save_model(self, output_dir):
        """
        Guarda el modelo completo con metadata.
        
        Args:
            output_dir: Directorio donde guardar el modelo
        """
        if self.pipeline is None:
            raise ValueError("❌ No hay modelo para guardar. Entrena primero.")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Bundle con todo
        bundle = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'model_type': 'RandomForest_Complete',
            'n_features': len(self.feature_names),
            'forbidden_features': self.FORBIDDEN_FEATURES,
            'random_state': self.random_state
        }
        
        # Guardar bundle completo
        model_path = Path(output_dir) / 'modelo_completo.joblib'
        joblib.dump(bundle, model_path)
        self.logger.info(f"💾 Modelo completo guardado: {model_path}")
        
        # Guardar solo pipeline (para compatibilidad)
        pipeline_path = Path(output_dir) / 'pipeline.joblib'
        joblib.dump(self.pipeline, pipeline_path)
        self.logger.info(f"💾 Pipeline guardado: {pipeline_path}")
        
        # Guardar nombres de features
        features_path = Path(output_dir) / 'feature_names.txt'
        with open(features_path, 'w') as f:
            for fname in self.feature_names:
                f.write(f"{fname}\n")
        
        self.logger.info(f"📝 Features guardadas: {features_path}")
        
        # Guardar importancias si están disponibles
        if self.importance_df is not None:
            importance_path = Path(output_dir) / 'feature_importance.csv'
            self.importance_df.to_csv(importance_path, index=False)
            self.logger.info(f"📋 Importancias guardadas: {importance_path}")
    
    def predict(self, X_new):
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X_new: Nuevos datos para predecir
            
        Returns:
            array: Predicciones
        """
        if self.pipeline is None:
            raise ValueError("❌ El modelo no ha sido entrenado. Llama a train() primero.")
        
        return self.pipeline.predict(X_new)
    
    def get_model_info(self):
        """
        Obtiene información del modelo.
        
        Returns:
            dict: Información del modelo
        """
        if self.model is None:
            raise ValueError("❌ El modelo no ha sido entrenado.")
        
        info = {
            'model_type': 'RandomForestRegressor',
            'n_estimators': self.model.n_estimators,
            'n_features': self.model.n_features_in_,
            'feature_names': self.feature_names,
            'oob_score': getattr(self.model, 'oob_score_', None),
            'metrics': self.metrics
        }
        
        return info


def main():
    """Función principal para uso desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Entrena Random Forest con features completos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python train_random_forest.py \\
      -i features_complete/dataset_complete_features.csv \\
      -o modelo_completo \\
      --test-size 0.2

Features incluidas (~90):
  • Grid 3D Mejorado (26 features)
  • ConvexHull (2 features)
  • Momentos de Inercia (3 features)
  • Distribución Radial (2 features)
  • Entropía Espacial (1 feature)
  • Bandwidth (1 feature)
  • Energía (50 features)
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='CSV con dataset completo')
    parser.add_argument('-o', '--output', default='modelo_completo',
                       help='Directorio de salida')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proporción para test (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Semilla aleatoria')
    parser.add_argument('--top-features', type=int, default=20,
                       help='Número de top features a mostrar (default: 20)')
    
    args = parser.parse_args()
    
    # Validar archivo
    if not Path(args.input).exists():
        print(f"❌ Archivo no encontrado: {args.input}")
        return
    
    try:
        print("="*70)
        print("🚀 ENTRENAMIENTO CON FEATURES COMPLETOS")
        print("="*70)
        
        # Crear entrenador
        trainer = RandomForestTrainer(random_state=args.random_state)
        
        # 1. Cargar datos
        X, y = trainer.load_data(args.input)
        
        # 2. Entrenar
        trainer.train(X, y, test_size=args.test_size)
        
        # 3. Evaluar
        trainer.evaluate()
        
        # 4. Feature importance
        trainer.analyze_feature_importance(args.top_features)
        
        # 5. Crear gráficos
        trainer.create_plots(args.output)
        
        # 6. Guardar modelo
        trainer.save_model(args.output)
        
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*70)
        print(f"\n📁 Archivos generados en: {args.output}/")
        print(f"   • modelo_completo.joblib (bundle con metadata)")
        print(f"   • pipeline.joblib (solo pipeline)")
        print(f"   • feature_names.txt")
        print(f"   • feature_importance.csv")
        print(f"   • evaluacion_completa.png")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()