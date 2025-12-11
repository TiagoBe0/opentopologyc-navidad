# opentopologyc/core/pipeline.py

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from core.loader import DumpLoader
from core.surface_extractor import SurfaceExtractor
from core.normalizer import PositionNormalizer
from config.extractor_config import ExtractorConfig


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

        self.normalizer = PositionNormalizer()


    # --------------------------------------------------------------
    # PROCESAR UN SOLO ARCHIVO
    # --------------------------------------------------------------
    def process_single(self, file_path):
        try:
            # 1. Carga del dump filtrado
            raw = self.loader.load(file_path)
            positions = raw["positions"]

            # 2. Extrae superficie -> devuelve nuevo dump con superficie filtrada
            filtered_dump = self.surface.extract(file_path)

            # 3. Cargar el dump filtrado
            raw_filtered = self.loader.load(filtered_dump)
            pos_surface = raw_filtered["positions"]


            # aligned es Nx3 listo para features




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

        files = sorted(str(f) for f in input_dir.glob("*"))

        if not files:
            print(f"No se encontraron archivos .dump en: {input_dir}")
            return None

        print(f"Archivos encontrados: {len(files)}")
        rows = []

        for f in tqdm(files, desc="Procesando dumps"):
            result = self.process_single(f)
            if result:
                rows.append(result)

        if not rows:
            print("No se extrajo ningún resultado.")
            return None

        df = pd.DataFrame(rows).set_index("file")

        # guardar CSV
        output_csv = input_dir / "dataset_features.csv"
        df.to_csv(output_csv)

        print(f"Dataset guardado en:\n{output_csv}")

        return df
