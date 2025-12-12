# opentopologyc/config/extractor_config.py

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class ExtractorConfig:
    """
    Configuración inicial del extractor OpenTopologyC.
    
    Esta clase define:
      - Directorio que contiene los dumps
      - Radio de sonda (ej: para alpha shape o ConstructSurface)
      - Flags booleanos para activar/desactivar features
    """

    input_dir: str = ""                 # directorio con dumps
    probe_radius: float = 2.0           # radio de sonda para alpha shape
    surface_distance:bool = False                                    # para entrenamiento con mas particulas del clustering
    surface_distance_value:float = 2.0                                # valor de distancia de superficie para filtrar particulas(solo si surface_distance=True)
    total_atoms:int =16384
    a0: float = 3.532
    lattice_type: str = "fcc"            # tipo de celda unidad (fcc, bcc, hcp, diamond, sc)
    compute_grid_features: bool = True
    compute_hull_features: bool = True
    compute_inertia_features: bool = True
    compute_radial_features: bool = True
    compute_entropy_features: bool = True
    compute_clustering_features: bool = True
    

    def validate(self):
        """Validaciones simples para evitar errores de usuario."""
        if not self.input_dir:
            raise ValueError("input_dir no puede estar vacío.")

        if not Path(self.input_dir).exists():
            raise FileNotFoundError(f"El directorio no existe: {self.input_dir}")

        if self.probe_radius <= 0:
            raise ValueError("probe_radius debe ser mayor a cero.")

        return True

    # Serialización opcional
    def to_dict(self):
        return asdict(self)

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        return path

    @classmethod
    def load_json(cls, path):
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        cfg = cls(**data)
        cfg.validate()
        return cfg


"""

from opentopologyc.config.extractor_config import ExtractorConfig

cfg = ExtractorConfig(
    input_dir="databases/db_integrate",
    probe_radius=2.0,
    compute_grid_features=True,
    compute_hull_features=False,   # por ejemplo, desactivados
)

"""