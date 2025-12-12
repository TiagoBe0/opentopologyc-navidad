# opentopologyc/core/pipeline.py

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from core.loader import DumpLoader
from core.surface_extractor import SurfaceExtractor
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



    # --------------------------------------------------------------
    # PROCESAR UN SOLO ARCHIVO
    # --------------------------------------------------------------
    def process_single(self, file_path):
        try:
            filtered = self.surface.extract(file_path)
            raw = self.loader.load(filtered)

            pos = raw["positions"]

            # Ejemplo choronga: número de puntos
            n = pos.shape[0]

            return {
                "file": Path(file_path).name,
                "num_points": n
            }

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
         

        if not rows:
            print("No se extrajo ningún resultado.")
            return None

        df = pd.DataFrame(rows).set_index("file")

        # guardar CSV
        output_csv = input_dir / "dataset_features.csv"
        df.to_csv(output_csv)

        print(f"Dataset guardado en:\n{output_csv}")

        return df
