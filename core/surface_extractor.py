from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

from ovito.io import import_file, export_file
from ovito.modifiers import (
    ConstructSurfaceModifier,
    DeleteSelectedModifier,
    InvertSelectionModifier,
    ExpressionSelectionModifier,
    AffineTransformationModifier
)

from config.extractor_config import ExtractorConfig
from core.dump_validator import DumpValidator


class SurfaceExtractor:
    """
    Filtra la superficie de la partícula y normaliza coordenadas:
      1. Centrado en el origen
      2. Rotación PCA (alinear ejes principales)
    Exporta un archivo .dump ya normalizado.
    """

    def __init__(self, config: ExtractorConfig):
        self.config = config
        self.probe_radius = config.probe_radius
        self.use_surface_distance = config.surface_distance
        self.surface_distance_value = config.surface_distance_value

    # ----------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ----------------------------------------------------------
    def extract(self, dump_file_path: str):
        dump_file_path = Path(dump_file_path)

        # Archivo de salida
        output_path = dump_file_path.with_name(
            dump_file_path.stem + "_surface_normalized.dump"
        )

        # 1) Validar y pre-procesar dump si es necesario
        try:
            validated_path = DumpValidator.validate_and_fix(str(dump_file_path))
        except ValueError as e:
            raise ValueError(f"Dump inválido: {e}")

        # 2) Importar archivo (validado o corregido)
        try:
            pipeline = import_file(validated_path)
        except Exception as e:
            raise ValueError(f"OVITO no pudo importar el archivo: {e}")

        # ------------------------------------------------------
        # 2) FILTRADO DE SUPERFICIE
        # ------------------------------------------------------
        if not self.use_surface_distance:
            # Método: seleccionar partículas de superficie según el radio
            pipeline.modifiers.append(
                ConstructSurfaceModifier(
                    radius=self.probe_radius,
                    smoothing_level=8,
                    select_surface_particles=True
                )
            )
            pipeline.modifiers.append(InvertSelectionModifier())
            pipeline.modifiers.append(DeleteSelectedModifier())

        else:
            # Método: seleccionar por distancia a la superficie
            pipeline.modifiers.append(
                ConstructSurfaceModifier(
                    radius=self.probe_radius,
                    smoothing_level=8,
                    surface_distance=True
                )
            )
            pipeline.modifiers.append(
                ExpressionSelectionModifier(
                    expression=f"SurfaceDistance < {self.surface_distance_value}"
                )
            )
            pipeline.modifiers.append(InvertSelectionModifier())
            pipeline.modifiers.append(DeleteSelectedModifier())

        data = pipeline.compute()

        positions = data.particles.positions
        if positions.shape[0] == 0:
            raise ValueError("No quedaron partículas de superficie")

        else:
            com = np.mean(positions, axis=0)
            export_file(
                pipeline,
                str(output_path),
                "lammps/dump",
                columns=["Position.X", "Position.Y", "Position.Z"]
            )
            #pipeline.clear()
            return str(output_path)
