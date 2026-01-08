#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo para Analisis Wigner-Seitz de defectos puntuales.

Este modulo implementa el metodo tradicional de Wigner-Seitz para
detectar vacancias e intersticiales comparando una estructura de
referencia con una estructura defectuosa.
"""

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Tuple, Dict, Optional


class SimulationBox:
    """
    Representa una caja de simulacion con condiciones periodicas.
    """

    def __init__(self, xlo: float, xhi: float, ylo: float, yhi: float,
                 zlo: float, zhi: float):
        """
        Inicializa la caja de simulacion.

        Args:
            xlo, xhi: Limites en X
            ylo, yhi: Limites en Y
            zlo, zhi: Limites en Z
        """
        self.xlo, self.xhi = xlo, xhi
        self.ylo, self.yhi = ylo, yhi
        self.zlo, self.zhi = zlo, zhi

        self.lx = xhi - xlo
        self.ly = yhi - ylo
        self.lz = zhi - zlo

    def get_volume(self) -> float:
        """Retorna el volumen de la caja"""
        return self.lx * self.ly * self.lz

    def get_strain(self, reference_box: 'SimulationBox') -> float:
        """
        Calcula el strain volumetrico respecto a una referencia.

        Args:
            reference_box: Caja de referencia

        Returns:
            Strain volumetrico (V - V0) / V0
        """
        v0 = reference_box.get_volume()
        v = self.get_volume()
        return (v - v0) / v0

    def apply_pbc(self, positions: np.ndarray) -> np.ndarray:
        """
        Aplica condiciones periodicas de contorno.

        Args:
            positions: Array (N, 3) de posiciones

        Returns:
            Posiciones con PBC aplicadas
        """
        pos = positions.copy()

        # Centrar en origen
        center = np.array([
            (self.xhi + self.xlo) / 2,
            (self.yhi + self.ylo) / 2,
            (self.zhi + self.zlo) / 2
        ])
        pos -= center

        # Aplicar PBC
        L = np.array([self.lx, self.ly, self.lz])
        pos = pos - L * np.round(pos / L)

        return pos

    def minimum_image(self, dr: np.ndarray) -> np.ndarray:
        """
        Aplica convencion de minima imagen a vectores de distancia.

        Args:
            dr: Vector(es) de distancia

        Returns:
            Vectores con minima imagen aplicada
        """
        L = np.array([self.lx, self.ly, self.lz])
        return dr - L * np.round(dr / L)


class WignerSeitzAnalyzer:
    """
    Analizador de defectos usando el metodo de Wigner-Seitz.

    El metodo asigna cada atomo de la estructura defectuosa al sitio
    de red mas cercano en la estructura de referencia. Los sitios
    con ocupacion 0 son vacancias, y los atomos no asignados son
    intersticiales.
    """

    def __init__(self, reference: np.ndarray, defective: np.ndarray,
                 ref_box: SimulationBox, def_box: SimulationBox,
                 use_pbc: bool = True, use_affine_mapping: bool = False):
        """
        Inicializa el analizador.

        Args:
            reference: Posiciones de referencia (N_ref, 3)
            defective: Posiciones defectuosas (N_def, 3)
            ref_box: Caja de la estructura de referencia
            def_box: Caja de la estructura defectuosa
            use_pbc: Usar condiciones periodicas de contorno
            use_affine_mapping: Aplicar mapeo afin para strain
        """
        self.reference = reference.copy()
        self.defective = defective.copy()
        self.ref_box = ref_box
        self.def_box = def_box
        self.use_pbc = use_pbc
        self.use_affine_mapping = use_affine_mapping

        # Resultados
        self.occupancy = None
        self.site_assignment = None

        # Validar compatibilidad de las estructuras
        self._validate_structures()

        # IMPORTANTE: Aplicar mapeo afin ANTES de PBC
        # El mapeo afin escala las coordenadas, y debe hacerse antes de centrar
        if use_affine_mapping:
            self._apply_affine_mapping()

        # Aplicar PBC despues del mapeo afin
        if use_pbc:
            self.reference = ref_box.apply_pbc(self.reference)
            self.defective = def_box.apply_pbc(self.defective)

    def _validate_structures(self):
        """
        Valida que las estructuras sean compatibles para el analisis.

        Raises:
            ValueError: Si las estructuras no son validas o compatibles
        """
        # Validar que las estructuras no esten vacias
        if len(self.reference) == 0:
            raise ValueError("La estructura de referencia esta vacia")

        if len(self.defective) == 0:
            raise ValueError("La estructura defectuosa esta vacia")

        # Validar dimensiones de las cajas
        if self.ref_box.lx <= 0 or self.ref_box.ly <= 0 or self.ref_box.lz <= 0:
            raise ValueError(
                f"Dimensiones invalidas de caja de referencia: "
                f"lx={self.ref_box.lx}, ly={self.ref_box.ly}, lz={self.ref_box.lz}"
            )

        if self.def_box.lx <= 0 or self.def_box.ly <= 0 or self.def_box.lz <= 0:
            raise ValueError(
                f"Dimensiones invalidas de caja defectuosa: "
                f"lx={self.def_box.lx}, ly={self.def_box.ly}, lz={self.def_box.lz}"
            )

        # Advertir si el numero de atomos es muy diferente
        n_ref = len(self.reference)
        n_def = len(self.defective)
        atom_diff_ratio = abs(n_ref - n_def) / n_ref if n_ref > 0 else 0

        if atom_diff_ratio > 0.5:
            import warnings
            warnings.warn(
                f"Diferencia significativa en numero de atomos: "
                f"referencia={n_ref}, defectuoso={n_def} "
                f"({atom_diff_ratio*100:.1f}% diferencia). "
                f"Verifique que las estructuras sean compatibles.",
                UserWarning
            )

        # Advertir si hay strain significativo sin mapeo afin
        strain = self.def_box.get_strain(self.ref_box)
        if abs(strain) > 0.05 and not self.use_affine_mapping:
            import warnings
            warnings.warn(
                f"Strain volumetrico significativo detectado ({strain*100:.2f}%) "
                f"sin mapeo afin activado. Considere activar use_affine_mapping=True "
                f"para mejores resultados.",
                UserWarning
            )

        # Validar que las dimensiones de las cajas sean razonablemente similares
        dim_ratios = [
            self.def_box.lx / self.ref_box.lx,
            self.def_box.ly / self.ref_box.ly,
            self.def_box.lz / self.ref_box.lz
        ]

        max_ratio = max(dim_ratios)
        min_ratio = min(dim_ratios)

        if max_ratio > 2.0 or min_ratio < 0.5:
            import warnings
            warnings.warn(
                f"Las dimensiones de las cajas son muy diferentes: "
                f"ratios = {dim_ratios}. "
                f"Verifique que las estructuras sean del mismo sistema.",
                UserWarning
            )

    def _apply_affine_mapping(self):
        """Aplica mapeo afin para compensar strain uniforme"""
        # Factores de escala
        sx = self.def_box.lx / self.ref_box.lx
        sy = self.def_box.ly / self.ref_box.ly
        sz = self.def_box.lz / self.ref_box.lz

        # Escalar posiciones de referencia
        self.reference[:, 0] *= sx
        self.reference[:, 1] *= sy
        self.reference[:, 2] *= sz

    def analyze(self) -> Dict:
        """
        Ejecuta el analisis Wigner-Seitz.

        Returns:
            Diccionario con resultados:
            - n_vacancies: Numero de vacancias
            - n_interstitials: Numero de intersticiales
            - vacancies: Indices de sitios vacios
            - interstitial_atoms: Indices de atomos intersticiales
            - occupancy: Ocupacion de cada sitio
            - vacancy_concentration: Concentracion de vacancias
            - interstitial_concentration: Concentracion de intersticiales
            - volumetric_strain: Strain volumetrico
        """
        n_ref = len(self.reference)
        n_def = len(self.defective)

        # Inicializar ocupacion
        self.occupancy = np.zeros(n_ref, dtype=int)
        self.site_assignment = np.full(n_def, -1, dtype=int)

        # Construir KD-Tree de sitios de referencia
        if self.use_pbc:
            # Para PBC, necesitamos considerar imagenes periodicas
            tree = self._build_pbc_tree()
        else:
            tree = cKDTree(self.reference)

        # Asignar cada atomo defectuoso al sitio mas cercano
        for i, pos in enumerate(self.defective):
            if self.use_pbc:
                # Buscar considerando PBC
                dist, idx = self._query_pbc(tree, pos)
            else:
                dist, idx = tree.query(pos)

            self.site_assignment[i] = idx
            self.occupancy[idx] += 1

        # Identificar defectos
        vacancies = np.where(self.occupancy == 0)[0]

        # Calcular umbral de distancia para intersticiales
        # Usamos la mitad de la distancia promedio entre sitios de red
        volume = self.ref_box.get_volume()
        avg_site_distance = (volume / n_ref) ** (1.0 / 3.0) if n_ref > 0 else 1.0
        distance_threshold = 0.5 * avg_site_distance

        # Intersticiales: dos criterios
        # 1. Atomos en sitios con ocupacion > 1
        # 2. Atomos muy lejos de su sitio mas cercano
        interstitial_atoms = []
        interstitial_distances = []

        # Criterio 1: Multiples atomos en un sitio
        for site_idx in np.where(self.occupancy > 1)[0]:
            # Encontrar atomos asignados a este sitio
            atoms_at_site = np.where(self.site_assignment == site_idx)[0]
            # El primero es el "legitimo", los demas son intersticiales
            if len(atoms_at_site) > 1:
                interstitial_atoms.extend(atoms_at_site[1:].tolist())
                # Registrar distancias para los intersticiales
                for atom_idx in atoms_at_site[1:]:
                    diff = self.defective[atom_idx] - self.reference[site_idx]
                    if self.use_pbc:
                        diff = self.def_box.minimum_image(diff)
                    dist = np.linalg.norm(diff)
                    interstitial_distances.append(dist)

        # Criterio 2: Atomos muy lejos de su sitio asignado
        for i in range(n_def):
            site_idx = self.site_assignment[i]
            if site_idx >= 0 and i not in interstitial_atoms:
                diff = self.defective[i] - self.reference[site_idx]
                if self.use_pbc:
                    diff = self.def_box.minimum_image(diff)
                dist = np.linalg.norm(diff)
                if dist > distance_threshold:
                    interstitial_atoms.append(i)
                    interstitial_distances.append(dist)

        interstitial_atoms = np.array(interstitial_atoms, dtype=int)
        interstitial_distances = np.array(interstitial_distances) if interstitial_distances else np.array([])

        # Calcular concentraciones
        vacancy_concentration = len(vacancies) / n_ref if n_ref > 0 else 0
        interstitial_concentration = len(interstitial_atoms) / n_ref if n_ref > 0 else 0

        # Strain volumetrico
        volumetric_strain = self.def_box.get_strain(self.ref_box)

        results = {
            'n_vacancies': len(vacancies),
            'n_interstitials': len(interstitial_atoms),
            'vacancies': vacancies,
            'interstitial_atoms': interstitial_atoms,
            'interstitial_distances': interstitial_distances,
            'distance_threshold': distance_threshold,
            'occupancy': self.occupancy,
            'site_assignment': self.site_assignment,
            'vacancy_concentration': vacancy_concentration,
            'interstitial_concentration': interstitial_concentration,
            'volumetric_strain': volumetric_strain,
            'n_reference_sites': n_ref,
            'n_defective_atoms': n_def
        }

        return results

    def _build_pbc_tree(self) -> cKDTree:
        """Construye KD-Tree para busqueda con PBC"""
        return cKDTree(self.reference)

    def _query_pbc(self, tree: cKDTree, pos: np.ndarray) -> Tuple[float, int]:
        """
        Busca el sitio mas cercano considerando PBC.

        Args:
            tree: KD-Tree de sitios de referencia
            pos: Posicion a buscar

        Returns:
            (distancia, indice) del sitio mas cercano
        """
        # Vector de diferencia
        diff = self.reference - pos

        # Aplicar minima imagen
        diff = self.def_box.minimum_image(diff)

        # Calcular distancias
        distances = np.linalg.norm(diff, axis=1)

        # Encontrar minimo
        idx = np.argmin(distances)

        return distances[idx], idx

    def get_vacancy_positions(self) -> np.ndarray:
        """Retorna las posiciones de las vacancias"""
        if self.occupancy is None:
            raise ValueError("Ejecuta analyze() primero")

        vacancies = np.where(self.occupancy == 0)[0]
        return self.reference[vacancies]

    def get_interstitial_positions(self, results: Dict) -> np.ndarray:
        """Retorna las posiciones de los intersticiales"""
        if len(results['interstitial_atoms']) == 0:
            return np.array([]).reshape(0, 3)
        return self.defective[results['interstitial_atoms']]


def read_lammps_dump(filepath: str) -> Tuple[np.ndarray, SimulationBox]:
    """
    Lee un archivo LAMMPS dump.

    Args:
        filepath: Ruta al archivo dump

    Returns:
        (positions, box): Posiciones y caja de simulacion

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo esta malformado o no contiene datos validos
    """
    # Validar que el archivo existe
    filepath_obj = Path(filepath)
    if not filepath_obj.exists():
        raise FileNotFoundError(f"Archivo LAMMPS no encontrado: {filepath}")

    if not filepath_obj.is_file():
        raise ValueError(f"La ruta no es un archivo: {filepath}")

    positions = []
    box_bounds = {}
    n_atoms = 0

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        raise ValueError(f"Error al leer el archivo {filepath}: {e}")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if 'ITEM: NUMBER OF ATOMS' in line:
            i += 1
            n_atoms = int(lines[i].strip())

        elif 'ITEM: BOX BOUNDS' in line:
            i += 1
            xlo, xhi = map(float, lines[i].split()[:2])
            i += 1
            ylo, yhi = map(float, lines[i].split()[:2])
            i += 1
            zlo, zhi = map(float, lines[i].split()[:2])
            box_bounds = {
                'xlo': xlo, 'xhi': xhi,
                'ylo': ylo, 'yhi': yhi,
                'zlo': zlo, 'zhi': zhi
            }

        elif 'ITEM: ATOMS' in line:
            # Determinar columnas
            header_parts = line.split()[2:]  # Skip "ITEM: ATOMS"

            # Buscar indices de x, y, z y guardar el tipo de coordenada
            x_idx = y_idx = z_idx = None
            x_type = y_type = z_type = None

            for j, col in enumerate(header_parts):
                if col in ['x', 'xu', 'xs']:
                    x_idx = j
                    x_type = col
                elif col in ['y', 'yu', 'ys']:
                    y_idx = j
                    y_type = col
                elif col in ['z', 'zu', 'zs']:
                    z_idx = j
                    z_type = col

            # Validar que se encontraron todas las coordenadas
            if x_idx is None or y_idx is None or z_idx is None:
                raise ValueError(
                    f"No se encontraron columnas de posicion en el header LAMMPS.\n"
                    f"Columnas encontradas: {header_parts}\n"
                    f"Se esperaban columnas: x/xu/xs, y/yu/ys, z/zu/zs"
                )

            # Validar que tenemos los limites de la caja (necesarios para xs, ys, zs)
            if not box_bounds:
                raise ValueError("No se encontraron los limites de la caja (BOX BOUNDS)")

            # Leer posiciones
            for _ in range(n_atoms):
                i += 1
                if i >= len(lines):
                    break
                parts = lines[i].split()
                if len(parts) > max(x_idx, y_idx, z_idx):
                    x = float(parts[x_idx])
                    y = float(parts[y_idx])
                    z = float(parts[z_idx])

                    # Convertir coordenadas escaladas a coordenadas reales
                    if x_type == 'xs':
                        x = box_bounds['xlo'] + x * (box_bounds['xhi'] - box_bounds['xlo'])
                    if y_type == 'ys':
                        y = box_bounds['ylo'] + y * (box_bounds['yhi'] - box_bounds['ylo'])
                    if z_type == 'zs':
                        z = box_bounds['zlo'] + z * (box_bounds['zhi'] - box_bounds['zlo'])

                    positions.append([x, y, z])

        i += 1

    # Validar que se leyeron datos validos
    if not box_bounds:
        raise ValueError(
            f"No se encontraron limites de caja (BOX BOUNDS) en {filepath}. "
            f"Verifique que sea un archivo LAMMPS dump valido."
        )

    if n_atoms == 0:
        raise ValueError(
            f"No se encontro el numero de atomos en {filepath}. "
            f"Verifique que sea un archivo LAMMPS dump valido."
        )

    if len(positions) == 0:
        raise ValueError(
            f"No se leyeron posiciones atomicas de {filepath}. "
            f"Verifique el formato del archivo."
        )

    if len(positions) != n_atoms:
        import warnings
        warnings.warn(
            f"Discrepancia en numero de atomos: header indica {n_atoms}, "
            f"pero se leyeron {len(positions)} posiciones.",
            UserWarning
        )

    positions = np.array(positions)
    box = SimulationBox(**box_bounds)

    return positions, box


def count_vacancies_wigner_seitz(reference_file: str, defective_file: str,
                                  use_pbc: bool = True,
                                  use_affine: bool = False) -> Dict:
    """
    Funcion de conveniencia para contar vacancias usando Wigner-Seitz.

    Args:
        reference_file: Ruta al archivo de referencia
        defective_file: Ruta al archivo defectuoso
        use_pbc: Usar condiciones periodicas
        use_affine: Usar mapeo afin

    Returns:
        Diccionario con resultados del analisis

    Raises:
        FileNotFoundError: Si alguno de los archivos no existe
        ValueError: Si los archivos estan malformados o las estructuras son incompatibles
    """
    try:
        # Leer archivos
        ref_pos, ref_box = read_lammps_dump(reference_file)
    except Exception as e:
        raise ValueError(f"Error al leer archivo de referencia: {e}")

    try:
        def_pos, def_box = read_lammps_dump(defective_file)
    except Exception as e:
        raise ValueError(f"Error al leer archivo defectuoso: {e}")

    try:
        # Crear analizador
        analyzer = WignerSeitzAnalyzer(
            ref_pos, def_pos, ref_box, def_box,
            use_pbc=use_pbc,
            use_affine_mapping=use_affine
        )

        # Ejecutar analisis
        results = analyzer.analyze()
    except Exception as e:
        raise ValueError(f"Error en el analisis Wigner-Seitz: {e}")

    return results


if __name__ == "__main__":
    # Ejemplo de uso
    import sys

    if len(sys.argv) < 3:
        print("Uso: python wigner_seitz.py <referencia.dump> <defectuoso.dump>")
        sys.exit(1)

    ref_file = sys.argv[1]
    def_file = sys.argv[2]

    results = count_vacancies_wigner_seitz(ref_file, def_file)

    print("\n" + "="*50)
    print("RESULTADOS WIGNER-SEITZ")
    print("="*50)
    print(f"Sitios de referencia: {results['n_reference_sites']}")
    print(f"Atomos defectuosos:   {results['n_defective_atoms']}")
    print(f"Vacancias:            {results['n_vacancies']}")
    print(f"Intersticiales:       {results['n_interstitials']}")
    print(f"Concentracion vac:    {results['vacancy_concentration']*100:.4f}%")
    print(f"Strain volumetrico:   {results['volumetric_strain']*100:.2f}%")
    print("="*50)
