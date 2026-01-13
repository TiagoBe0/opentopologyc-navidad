# opentopologyc/core/loader.py

from pathlib import Path
import numpy as np

class DumpLoader:
    """
    Loader ultragenérico para dumps LAMMPS.
    
    - Busca la línea que empieza con 'ITEM: ATOMS'
    - Extrae las columnas declaradas (dinámicas)
    - Construye:
        positions = Nx3 (x,y,z)
        extra = dict(col -> Nx1) (hasta 7 columnas)
    """

    def __init__(self, max_extra_cols=7):
        self.max_extra_cols = max_extra_cols

    def load(self, file_path):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()

        atoms_header_idx = None
        atom_columns = None

        # -----------------------------------------------------
        # 1) BUSCAR LA LÍNEA 'ITEM: ATOMS'
        # -----------------------------------------------------
        for i, line in enumerate(lines):
            if line.startswith("ITEM: ATOMS"):
                atoms_header_idx = i
                atom_columns = line.strip().split()[2:]  # después de 'ITEM: ATOMS'
                break

        if atoms_header_idx is None:
            raise ValueError("No se encontró 'ITEM: ATOMS' en el dump.")

        # -----------------------------------------------------
        # 2) DETERMINAR COLUMNAS X, Y, Z
        # -----------------------------------------------------
        if not all(col in atom_columns for col in ("x", "y", "z")):
            raise ValueError(f"El dump no tiene x,y,z. Header encontrado: {atom_columns}")

        col_idx = {c: k for k, c in enumerate(atom_columns)}

        ix, iy, iz = col_idx["x"], col_idx["y"], col_idx["z"]

        # -----------------------------------------------------
        # 3) LEER LAS LÍNEAS DE ÁTOMOS (todas las siguientes
        #    hasta encontrar otra línea que empiece con ITEM:)
        # -----------------------------------------------------
        atom_lines = []
        for j in range(atoms_header_idx + 1, len(lines)):
            if lines[j].startswith("ITEM:"):
                break
            if lines[j].strip():  # evitar líneas vacías
                atom_lines.append(lines[j].strip())

        num_atoms = len(atom_lines)

        # -----------------------------------------------------
        # 4) POSICIONES Nx3
        # -----------------------------------------------------
        positions = np.zeros((num_atoms, 3), dtype=float)

        # -----------------------------------------------------
        # 5) OTRAS COLUMNAS (hasta 7 matrices Nx1)
        # -----------------------------------------------------
        extra_columns = [c for c in atom_columns if c not in ("x", "y", "z")]
        extra_columns = extra_columns[: self.max_extra_cols]

        extra = {c: np.zeros((num_atoms, 1), dtype=float) for c in extra_columns}

        # -----------------------------------------------------
        # 6) PARSEO DE LOS DATOS
        # -----------------------------------------------------
        for n, line in enumerate(atom_lines):
            parts = line.split()

            # posiciones
            positions[n, 0] = float(parts[ix])
            positions[n, 1] = float(parts[iy])
            positions[n, 2] = float(parts[iz])

            # extra
            for c in extra_columns:
                extra[c][n, 0] = float(parts[col_idx[c]])

        # -----------------------------------------------------
        # 7) RESULTADO FINAL
        # -----------------------------------------------------
        return {
            "positions": positions,              # Nx3
            "extra": extra,                      # dict(col -> Nx1)
            "columns": atom_columns,             # columnas originales
            "num_atoms": num_atoms,              # cantidad real
        }
