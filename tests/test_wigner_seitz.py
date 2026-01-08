#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitarios para el modulo Wigner-Seitz.

Valida las correcciones realizadas:
1. Lectura correcta de coordenadas escaladas
2. Validacion de estructuras
3. Orden correcto de operaciones (mapeo afin antes de PBC)
4. Deteccion mejorada de intersticiales
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

# Import del modulo a testear
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.wigner_seitz import (
    SimulationBox,
    WignerSeitzAnalyzer,
    read_lammps_dump,
    count_vacancies_wigner_seitz
)


class TestSimulationBox(unittest.TestCase):
    """Tests para la clase SimulationBox"""

    def test_volume_calculation(self):
        """Verifica que el calculo de volumen es correcto"""
        box = SimulationBox(0, 10, 0, 10, 0, 10)
        self.assertEqual(box.get_volume(), 1000.0)

    def test_strain_calculation(self):
        """Verifica el calculo de strain volumetrico"""
        ref_box = SimulationBox(0, 10, 0, 10, 0, 10)
        def_box = SimulationBox(0, 11, 0, 11, 0, 11)

        strain = def_box.get_strain(ref_box)
        expected_strain = (11**3 - 10**3) / 10**3

        self.assertAlmostEqual(strain, expected_strain, places=5)

    def test_pbc_wrapping(self):
        """Verifica que las PBC envuelven correctamente las coordenadas"""
        box = SimulationBox(0, 10, 0, 10, 0, 10)

        # Posicion fuera de la caja
        pos = np.array([[15.0, 5.0, 5.0]])
        wrapped = box.apply_pbc(pos)

        # Debe estar dentro de [-5, 5] centrado en origen
        self.assertTrue(np.all(wrapped >= -5.0))
        self.assertTrue(np.all(wrapped <= 5.0))

    def test_minimum_image(self):
        """Verifica la convencion de minima imagen"""
        box = SimulationBox(0, 10, 0, 10, 0, 10)

        # Vector de distancia mayor que media caja
        dr = np.array([8.0, 0.0, 0.0])
        dr_min = box.minimum_image(dr)

        # Debe reducirse a -2.0
        self.assertAlmostEqual(dr_min[0], -2.0, places=5)


class TestLAMMPSReader(unittest.TestCase):
    """Tests para la lectura de archivos LAMMPS dump"""

    def setUp(self):
        """Crea archivos LAMMPS temporales para testing"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Limpia archivos temporales"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_dump_file(self, filename, use_scaled=False):
        """Helper para crear un archivo DUMP de prueba"""
        filepath = os.path.join(self.temp_dir, filename)

        coord_type = "xs ys zs" if use_scaled else "x y z"

        with open(filepath, 'w') as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write("4\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write("0.0 10.0\n")
            f.write(f"ITEM: ATOMS id type {coord_type}\n")

            if use_scaled:
                # Coordenadas escaladas [0, 1]
                f.write("1 1 0.0 0.0 0.0\n")
                f.write("2 1 0.5 0.0 0.0\n")
                f.write("3 1 0.0 0.5 0.0\n")
                f.write("4 1 0.0 0.0 0.5\n")
            else:
                # Coordenadas reales
                f.write("1 1 0.0 0.0 0.0\n")
                f.write("2 1 5.0 0.0 0.0\n")
                f.write("3 1 0.0 5.0 0.0\n")
                f.write("4 1 0.0 0.0 5.0\n")

        return filepath

    def test_read_regular_coordinates(self):
        """Verifica lectura de coordenadas regulares (x, y, z)"""
        filepath = self.create_dump_file("test_regular.dump", use_scaled=False)
        positions, box = read_lammps_dump(filepath)

        self.assertEqual(len(positions), 4)
        self.assertAlmostEqual(positions[1, 0], 5.0, places=5)
        self.assertEqual(box.lx, 10.0)

    def test_read_scaled_coordinates(self):
        """Verifica lectura y conversion de coordenadas escaladas (xs, ys, zs)"""
        filepath = self.create_dump_file("test_scaled.dump", use_scaled=True)
        positions, box = read_lammps_dump(filepath)

        self.assertEqual(len(positions), 4)
        # xs=0.5 debe convertirse a x=5.0 para caja [0, 10]
        self.assertAlmostEqual(positions[1, 0], 5.0, places=5)
        self.assertEqual(box.lx, 10.0)

    def test_file_not_found(self):
        """Verifica que se lanza error si el archivo no existe"""
        with self.assertRaises(FileNotFoundError):
            read_lammps_dump("/nonexistent/file.dump")

    def test_malformed_file(self):
        """Verifica que se lanza error para archivos malformados"""
        filepath = os.path.join(self.temp_dir, "malformed.dump")
        with open(filepath, 'w') as f:
            f.write("This is not a valid LAMMPS dump file\n")

        with self.assertRaises(ValueError):
            read_lammps_dump(filepath)


class TestWignerSeitzAnalyzer(unittest.TestCase):
    """Tests para el analizador Wigner-Seitz"""

    def setUp(self):
        """Configura estructuras de prueba"""
        # Estructura de referencia: red cubica simple 2x2x2
        self.reference = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])

        self.ref_box = SimulationBox(-0.5, 1.5, -0.5, 1.5, -0.5, 1.5)

    def test_no_defects(self):
        """Verifica analisis sin defectos"""
        # Estructura defectuosa identica a la referencia
        defective = self.reference.copy()
        def_box = self.ref_box

        analyzer = WignerSeitzAnalyzer(
            self.reference, defective, self.ref_box, def_box,
            use_pbc=False, use_affine_mapping=False
        )

        results = analyzer.analyze()

        self.assertEqual(results['n_vacancies'], 0)
        self.assertEqual(results['n_interstitials'], 0)
        self.assertAlmostEqual(results['volumetric_strain'], 0.0, places=10)

    def test_single_vacancy(self):
        """Verifica deteccion de una vacancia"""
        # Remover un atomo
        defective = self.reference[:-1].copy()  # Elimina el ultimo
        def_box = self.ref_box

        analyzer = WignerSeitzAnalyzer(
            self.reference, defective, self.ref_box, def_box,
            use_pbc=False, use_affine_mapping=False
        )

        results = analyzer.analyze()

        self.assertEqual(results['n_vacancies'], 1)
        self.assertEqual(results['n_defective_atoms'], 7)

    def test_single_interstitial(self):
        """Verifica deteccion de un intersticial"""
        # Agregar un atomo en posicion intersticial
        interstitial_pos = np.array([[0.5, 0.5, 0.5]])
        defective = np.vstack([self.reference, interstitial_pos])
        def_box = self.ref_box

        analyzer = WignerSeitzAnalyzer(
            self.reference, defective, self.ref_box, def_box,
            use_pbc=False, use_affine_mapping=False
        )

        results = analyzer.analyze()

        # Debe detectar al menos un intersticial
        self.assertGreaterEqual(results['n_interstitials'], 1)
        self.assertEqual(results['n_vacancies'], 0)

    def test_validation_empty_reference(self):
        """Verifica que se valida estructura de referencia vacia"""
        empty_ref = np.array([]).reshape(0, 3)

        with self.assertRaises(ValueError):
            WignerSeitzAnalyzer(
                empty_ref, self.reference, self.ref_box, self.ref_box,
                use_pbc=False
            )

    def test_validation_invalid_box(self):
        """Verifica que se validan dimensiones de caja invalidas"""
        invalid_box = SimulationBox(0, -10, 0, 10, 0, 10)  # lx negativo

        with self.assertRaises(ValueError):
            WignerSeitzAnalyzer(
                self.reference, self.reference, invalid_box, self.ref_box,
                use_pbc=False
            )

    def test_affine_mapping_order(self):
        """Verifica que el mapeo afin se aplica antes de PBC"""
        # Caja defectuosa con strain
        def_box = SimulationBox(-0.6, 1.8, -0.6, 1.8, -0.6, 1.8)
        defective = self.reference.copy() * 1.2  # Escalar atomos

        analyzer = WignerSeitzAnalyzer(
            self.reference, defective, self.ref_box, def_box,
            use_pbc=True, use_affine_mapping=True
        )

        results = analyzer.analyze()

        # Con mapeo afin correcto, deberia encontrar pocas o ninguna vacancia
        # Sin el orden correcto, habria muchos defectos falsos
        self.assertLessEqual(results['n_vacancies'], 2)


class TestIntegration(unittest.TestCase):
    """Tests de integracion end-to-end"""

    def setUp(self):
        """Crea archivos de prueba completos"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Limpia archivos temporales"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_perfect_lattice_dump(self, filename):
        """Crea archivo DUMP con red perfecta"""
        filepath = os.path.join(self.temp_dir, filename)

        with open(filepath, 'w') as f:
            f.write("ITEM: TIMESTEP\n0\n")
            f.write("ITEM: NUMBER OF ATOMS\n8\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("-0.5 1.5\n-0.5 1.5\n-0.5 1.5\n")
            f.write("ITEM: ATOMS id type x y z\n")

            idx = 1
            for i in [0.0, 1.0]:
                for j in [0.0, 1.0]:
                    for k in [0.0, 1.0]:
                        f.write(f"{idx} 1 {i} {j} {k}\n")
                        idx += 1

        return filepath

    def create_defective_lattice_dump(self, filename):
        """Crea archivo DUMP con vacancias"""
        filepath = os.path.join(self.temp_dir, filename)

        with open(filepath, 'w') as f:
            f.write("ITEM: TIMESTEP\n0\n")
            f.write("ITEM: NUMBER OF ATOMS\n6\n")  # 2 vacancias
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("-0.5 1.5\n-0.5 1.5\n-0.5 1.5\n")
            f.write("ITEM: ATOMS id type x y z\n")

            # Solo 6 de los 8 atomos
            positions = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                # Falta [0.0, 1.0, 1.0] y [1.0, 1.0, 1.0]
            ]

            for idx, pos in enumerate(positions, 1):
                f.write(f"{idx} 1 {pos[0]} {pos[1]} {pos[2]}\n")

        return filepath

    def test_end_to_end_vacancy_detection(self):
        """Test completo de deteccion de vacancias"""
        ref_file = self.create_perfect_lattice_dump("reference.dump")
        def_file = self.create_defective_lattice_dump("defective.dump")

        results = count_vacancies_wigner_seitz(
            ref_file, def_file,
            use_pbc=False, use_affine=False
        )

        self.assertEqual(results['n_vacancies'], 2)
        self.assertEqual(results['n_reference_sites'], 8)
        self.assertEqual(results['n_defective_atoms'], 6)
        self.assertAlmostEqual(results['vacancy_concentration'], 0.25, places=2)


if __name__ == '__main__':
    # Ejecutar todos los tests
    unittest.main(verbosity=2)
