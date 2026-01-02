# opentopologyc/core/pipeline.py

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from core.loader import DumpLoader
from core.surface_extractor import SurfaceExtractor
from config.extractor_config import ExtractorConfig
from core.feature_extractor import FeatureExtractor
from core.normalizer import PositionNormalizer
from core.train_step import RandomForestTrainer

class ExtractorPipeline:
    """
    Pipeline principal de extracción de features.
    """

    def __init__(self, config: ExtractorConfig):
        self.cfg = config
        self.cfg.validate()

        # --- instanciamos módulos base ---
        self.loader = DumpLoader()
        self.surface = SurfaceExtractor(self.cfg)
        self.normalizer = PositionNormalizer(scale_factor=self.cfg.a0)
        self.features = FeatureExtractor(self.cfg)



    # --------------------------------------------------------------
    # PROCESAR UN SOLO ARCHIVO
    # --------------------------------------------------------------
    def process_single(self, file_path):
        try:
            surface_dump = self.surface.extract(file_path)
            raw = self.loader.load(surface_dump)

            pos = raw["positions"]

            # NORMALIZACIÓN (ya la tenés)
            pos_norm, box_size, _ = self.normalizer.normalize(pos)

            feats = {}

            if self.cfg.compute_grid_features:
                feats.update(self.features.grid_features(pos_norm, box_size))

            if self.cfg.compute_inertia_features:
                feats.update(self.features.inertia_feature(pos))

            if self.cfg.compute_radial_features:
                feats.update(self.features.radial_features(pos))

            if self.cfg.compute_entropy_features:
                feats.update(self.features.entropy_spatial(pos_norm))

            if self.cfg.compute_clustering_features:
                feats.update(self.features.bandwidth(pos_norm))

            feats["file"] = Path(file_path).name
            feats["num_points"] = pos.shape[0]

            return feats

        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")
            return None





    # --------------------------------------------------------------
    # PROCESAR DIRECTORIO ENTERO
    # --------------------------------------------------------------
    def run(self):
        """
        Ejecuta el pipeline completo sobre el directorio configurado.
        """
        input_dir = Path(self.cfg.input_dir)

        files = sorted(
            str(f) for f in input_dir.glob("*")
            if not f.name.endswith("_surface_normalized.dump")
        )


        if not files:
            print(f"No se encontraron archivos .dump en: {input_dir}")
            return None

        print(f"Archivos encontrados: {len(files)}")
        rows = []

        for f in tqdm(files, desc="Procesando dumps"):
            result = self.process_single(f)
            if result is not None:
                rows.append(result)

         

        if not rows:
            print("No se extrajo ningún resultado.")
            return None

        df = pd.DataFrame(rows).set_index("file")

        # guardar CSV
        output_csv = input_dir / "dataset_features.csv"
        df.to_csv(output_csv)

        print(f"Dataset guardado en:\n{output_csv}")
        #etapa de entrenamiento
        # Uso programático
        trainer = RandomForestTrainer(random_state=42)

        # Cargar datos
        X, y = trainer.load_data("dataset.csv")

        # Entrenar
        trainer.train(X, y, test_size=0.2)

        # Evaluar
        metrics = trainer.evaluate()

        # Analizar importancia
        importance_df = trainer.analyze_feature_importance(top_n=20)

        # Guardar
        trainer.save_model("model_rf")


        return df
