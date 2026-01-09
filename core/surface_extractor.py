from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

# IMPORTANTE: Importación lazy de OVITO
# OVITO requiere PySide6/shiboken6, que puede no estar instalado
# o puede conflictuar con PyQt5. Importamos solo cuando se use.
_ovito_imported = False
_ovito_import_error = None

def _import_ovito():
    """Importa OVITO de forma lazy (solo cuando se necesite)"""
    global _ovito_imported, _ovito_import_error
    global import_file, export_file
    global ConstructSurfaceModifier, DeleteSelectedModifier
    global InvertSelectionModifier, ExpressionSelectionModifier
    global AffineTransformationModifier

    if _ovito_imported:
        return

    if _ovito_import_error is not None:
        raise _ovito_import_error

    try:
        from ovito.io import import_file as _import_file
        from ovito.io import export_file as _export_file
        from ovito.modifiers import (
            ConstructSurfaceModifier as _ConstructSurfaceModifier,
            DeleteSelectedModifier as _DeleteSelectedModifier,
            InvertSelectionModifier as _InvertSelectionModifier,
            ExpressionSelectionModifier as _ExpressionSelectionModifier,
            AffineTransformationModifier as _AffineTransformationModifier
        )

        # Asignar a variables globales
        import_file = _import_file
        export_file = _export_file
        ConstructSurfaceModifier = _ConstructSurfaceModifier
        DeleteSelectedModifier = _DeleteSelectedModifier
        InvertSelectionModifier = _InvertSelectionModifier
        ExpressionSelectionModifier = _ExpressionSelectionModifier
        AffineTransformationModifier = _AffineTransformationModifier

        _ovito_imported = True

    except ImportError as e:
        error_msg = (
            f"No se pudo importar OVITO: {e}\n\n"
            "OVITO requiere PySide6/shiboken6. Para instalar:\n"
            "  python scripts/fix_ovito_pyside6.py\n\n"
            "O manualmente:\n"
            "  pip install PySide6 shiboken6"
        )
        _ovito_import_error = ImportError(error_msg)
        raise _ovito_import_error

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
        # IMPORTANTE: Importar OVITO solo cuando se necesite
        _import_ovito()

        dump_file_path = Path(dump_file_path)

        # Crear subdirectorio para archivos normalizados
        normalized_dir = dump_file_path.parent / "normalized_dumps"
        normalized_dir.mkdir(exist_ok=True)

        # Archivo de salida en subdirectorio
        output_path = normalized_dir / (dump_file_path.stem + "_surface_normalized.dump")

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
