# opentopologyc/core/pipeline.py

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from .loader import DumpLoader
from .surface_extractor import SurfaceExtractor
from ..config.extractor_config import ExtractorConfig
from .feature_extractor import FeatureExtractor
from .normalizer import PositionNormalizer

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
            # Primero cargar el archivo ORIGINAL para obtener el número real de átomos
            original_data = self.loader.load(file_path)
            num_atoms_real = original_data["num_atoms"]

            # Ahora aplicar el filtro de superficie
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
            feats["num_points"] = pos.shape[0]  # Átomos de superficie
            feats["num_atoms_real"] = num_atoms_real  # Átomos totales en simulación

            # Calcular número de vacancias CORRECTAMENTE
            feats["n_vacancies"] = self.cfg.total_atoms - num_atoms_real

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

        # Extensiones a IGNORAR (no son dumps LAMMPS)
        ignore_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp',  # Imágenes
            '.csv', '.xlsx', '.xls',  # Hojas de cálculo
            '.py', '.pyc', '.pyo', '.pyx',  # Python
            '.txt', '.log', '.out',  # Logs/texto
            '.json', '.xml', '.yaml', '.yml',  # Config
            '.pdf', '.doc', '.docx',  # Documentos
            '.zip', '.tar', '.gz', '.bz2',  # Comprimidos
        }

        files = sorted(
            str(f) for f in input_dir.glob("*")
            if f.is_file()  # Solo archivos (no directorios, ignora normalized_dumps/)
            and not f.name.endswith("_surface_normalized.dump")  # No archivos procesados (legacy)
            and f.suffix.lower() not in ignore_extensions  # No extensiones ignoradas
        )

        if not files:
            print(f"No se encontraron archivos dump válidos en: {input_dir}")
            return None

        print(f"Archivos dump encontrados: {len(files)}")
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
        print(f"Total de muestras extraídas: {len(df)}")
        print(f"Columnas generadas: {list(df.columns)}")

        return df
