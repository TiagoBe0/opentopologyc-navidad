#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validador y pre-procesador de dumps LAMMPS

Algunos dumps LAMMPS pueden tener formatos que OVITO no parsea bien:
- Box boundaries con notación científica muy pequeña
- Formatos no estándar

Este módulo valida y convierte dumps problemáticos a un formato
que OVITO puede leer correctamente.
"""

from pathlib import Path
import re


class DumpValidator:
    """Valida y corrige dumps LAMMPS con formato problemático"""

    @staticmethod
    def is_valid_dump(file_path: str) -> bool:
        """
        Verifica si un archivo es un dump LAMMPS válido

        Returns:
            True si el archivo tiene el formato correcto
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(500)  # Leer primeros 500 chars

                # Verificar headers obligatorios
                if 'ITEM: TIMESTEP' not in content:
                    return False
                if 'ITEM: NUMBER OF ATOMS' not in content:
                    return False
                if 'ITEM: BOX BOUNDS' not in content:
                    return False

                return True

        except (UnicodeDecodeError, IOError):
            return False

    @staticmethod
    def normalize_box_bounds(line: str) -> str:
        """
        Normaliza líneas de box bounds con notación científica

        Convierte: -3.17e-02 5.65e+01
        A formato: -0.0317 56.5
        """
        # Buscar dos números en notación científica o decimal
        pattern = r'([+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d+\.?\d+)\s+([+-]?\d+\.?\d*[eE][+-]?\d+|[+-]?\d+\.?\d+)'
        match = re.search(pattern, line)

        if match:
            try:
                val1 = float(match.group(1))
                val2 = float(match.group(2))

                # Convertir a formato decimal normal
                return f"{val1:.10f} {val2:.10f}\n"
            except ValueError:
                return line

        return line

    @staticmethod
    def preprocess_dump(input_path: str, output_path: str = None, add_id_if_missing: bool = True) -> str:
        """
        Pre-procesa un dump LAMMPS para hacerlo compatible con OVITO

        Args:
            input_path: Ruta al dump original
            output_path: Ruta de salida (opcional, usa _fixed.dump si no se especifica)
            add_id_if_missing: Si True, agrega columna 'id' si no existe

        Returns:
            Ruta al archivo procesado
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.with_name(input_path.stem + "_fixed.dump")

        try:
            with open(input_path, 'r', encoding='utf-8') as f_in:
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    in_box_bounds = False
                    box_line_count = 0
                    in_atoms = False
                    atom_id_counter = 1
                    atom_columns = None

                    for line in f_in:
                        # Detectar sección de box bounds
                        if 'ITEM: BOX BOUNDS' in line:
                            in_box_bounds = True
                            box_line_count = 0
                            in_atoms = False
                            f_out.write(line)
                            continue

                        # Normalizar las 3 líneas de box bounds
                        if in_box_bounds and box_line_count < 3:
                            normalized = DumpValidator.normalize_box_bounds(line)
                            f_out.write(normalized)
                            box_line_count += 1

                            if box_line_count == 3:
                                in_box_bounds = False
                            continue

                        # Detectar sección ATOMS
                        if line.startswith('ITEM: ATOMS'):
                            in_atoms = True
                            in_box_bounds = False
                            parts = line.strip().split()[2:]  # Columnas después de "ITEM: ATOMS"
                            atom_columns = parts

                            # Si no tiene 'id' y queremos agregarlo
                            if add_id_if_missing and 'id' not in atom_columns:
                                f_out.write('ITEM: ATOMS id ' + ' '.join(atom_columns) + '\n')
                            else:
                                f_out.write(line)
                            continue

                        # Si estamos en sección de átomos
                        if in_atoms and line.strip() and not line.startswith('ITEM:'):
                            # Si agregamos ID, insertarlo al inicio
                            if add_id_if_missing and atom_columns and 'id' not in atom_columns:
                                f_out.write(f"{atom_id_counter} {line}")
                                atom_id_counter += 1
                            else:
                                f_out.write(line)
                            continue

                        # Detectar nueva sección ITEM: (fin de atoms)
                        if in_atoms and line.startswith('ITEM:'):
                            in_atoms = False

                        f_out.write(line)

            return str(output_path)

        except Exception as e:
            raise ValueError(f"Error al pre-procesar dump: {e}")

    @staticmethod
    def validate_and_fix(file_path: str) -> str:
        """
        Valida un dump y lo corrige si es necesario

        Args:
            file_path: Ruta al dump

        Returns:
            Ruta al dump válido (original o corregido)
        """
        # Primero verificar si es un dump válido
        if not DumpValidator.is_valid_dump(file_path):
            raise ValueError("El archivo no es un dump LAMMPS válido")

        needs_preprocessing = False
        has_scientific = False
        missing_id = False

        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    # Verificar box bounds con notación científica
                    if 'ITEM: BOX BOUNDS' in line:
                        # Leer las siguientes 3 líneas
                        for _ in range(3):
                            next_line = next(f, '')
                            if 'e' in next_line.lower() or 'E' in next_line:
                                has_scientific = True
                                needs_preprocessing = True
                                break

                    # Verificar si falta columna 'id'
                    if line.startswith('ITEM: ATOMS'):
                        columns = line.strip().split()[2:]
                        if 'id' not in columns:
                            missing_id = True
                            needs_preprocessing = True
                        break

                    if i > 30:  # Solo buscar en primeras líneas
                        break
        except:
            pass

        # Si necesita pre-procesamiento, aplicarlo
        if needs_preprocessing:
            return DumpValidator.preprocess_dump(file_path, add_id_if_missing=missing_id)

        return file_path
