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


class SurfaceExtractor:
    """
    Filtra la superficie y normaliza coordenadas (centrado + PCA).
    Exporta un .dump final completamente normalizado.
    """

    def __init__(self, config: ExtractorConfig):
        self.config = config
        self.probe_radius = config.probe_radius
        self.use_surface_distance = config.surface_distance
        self.surface_distance_value = config.surface_distance_value

    # ----------------------------------------------------------
    # METODO PRINCIPAL
    # ----------------------------------------------------------
    def extract(self, dump_file_path: str):
        dump_file_path = Path(dump_file_path)

        # Archivo final normalizado
        output_path = dump_file_path.with_name(
            dump_file_path.stem + "_surface_normalized.dump"
        )

        # 1) Importar archivo
        pipeline = import_file(str(dump_file_path))

        # ------------------------------------------------------
        # 2) MODIFICADORES DE FILTRADO DE SUPERFICIE
        # ------------------------------------------------------
        if not self.use_surface_distance:
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
            pipeline.modifiers.append(
                ConstructSurfaceModifier(
                    radius=self.probe_radius,
                    smoothing_level=8,
                    surface_distance=True
                )
            )
            pipeline.modifiers.append(
                ExpressionSelectionModifier(
                    expression=f"SurfaceDistance < { self.surface_distance_value}"
                )
            )
            pipeline.modifiers.append(InvertSelectionModifier())
            pipeline.modifiers.append(DeleteSelectedModifier())

        # Ejecutar filtrado
        data = pipeline.compute()

        # ------------------------------------------------------
        # 3) EXTRAER POSICIONES PARA NORMALIZAR
        # ------------------------------------------------------
        pos = data.particles.positions.numpy().copy()

        # 3.1 Centrado
        centroid = pos.mean(axis=0)
        pos_centered = pos - centroid

        # 3.2 PCA (alinear ejes)
        pca = PCA(n_components=3)
        pos_pca = pca.fit_transform(pos_centered)

        # matriz de rotación PCA (para aplicar a OVITO)
        R = pca.components_

        # ------------------------------------------------------
        # 4) APLICAR TRANSFORMACION A OVITO
        # ------------------------------------------------------
        # Centrar primero (traslación)
        T = np.eye(4)
        T[:3, 3] = -centroid
        pipeline.modifiers.append(AffineTransformationModifier(transformation=T))

        # Luego rotación PCA
        Trot = np.eye(4)
        Trot[:3, :3] = R
        pipeline.modifiers.append(AffineTransformationModifier(transformation=Trot))

        # Recalcular datos con las transformaciones aplicadas
        pipeline.compute()

        # ------------------------------------------------------
        # 5) EXPORTAR RESULTADO FINAL NORMALIZADO
        # ------------------------------------------------------
        export_file(
            pipeline,
            str(output_path),
            "lammps/dump",
            columns=["Position.X", "Position.Y", "Position.Z"]
        )

        return str(output_path)
