#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extractor GUI - OpenTopologyC

IMPORTANTE: Este módulo NO usa QThread porque OVITO no es thread-safe con PyQt5.
Usar QThread causa Segmentation Fault cuando OVITO intenta acceder a recursos Qt.

Solución implementada:
- QTimer para procesar archivos uno por uno
- QApplication.processEvents() para mantener UI responsive
- Todo se ejecuta en el main thread de Qt

Ver: https://www.ovito.org/docs/current/python/introduction/running.html#thread-safety
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTextEdit,
    QProgressBar, QCheckBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer

from gui_qt.base_window import BaseWindow
from core.pipeline import ExtractorPipeline
from config.extractor_config import ExtractorConfig
from pathlib import Path
import pandas as pd


class ExtractorGUIQt(BaseWindow):
    def __init__(self):
        super().__init__("OpenTopologyC - Extractor de Features", (700, 700))
        self.input_dir = ""

        # Pipeline y procesamiento
        self.pipeline = None
        self.files_to_process = []
        self.current_file_index = 0
        self.results = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_next_file)

        self._build_ui()

    # ======================================================
    # UI
    # ======================================================
    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignTop)
        layout.setSpacing(15)

        # -------- DATASET --------
        btn_input = QPushButton("📂 Seleccionar carpeta de dumps")
        btn_input.clicked.connect(self.select_input)
        self.lbl_input = QLabel("No seleccionado")

        layout.addWidget(btn_input)
        layout.addWidget(self.lbl_input)

        # -------- SURFACE PARAMS --------
        surface_group = QGroupBox("Parámetros de Superficie (OVITO)")
        surface_layout = QVBoxLayout()

        self.spin_probe = QDoubleSpinBox()
        self.spin_probe.setValue(2.0)
        self.spin_probe.setSingleStep(0.2)
        self.spin_probe.setRange(0.1, 10.0)

        surface_layout.addWidget(QLabel("Probe radius (Å):"))
        surface_layout.addWidget(self.spin_probe)

        # Checkbox para método de distancia a superficie
        self.chk_surface_distance = QCheckBox("Usar Surface Distance (método alternativo)")
        self.chk_surface_distance.setChecked(False)
        self.chk_surface_distance.toggled.connect(self.toggle_surface_distance)
        surface_layout.addWidget(self.chk_surface_distance)

        # SpinBox para surface distance value (deshabilitado por defecto)
        self.spin_surface_distance = QDoubleSpinBox()
        self.spin_surface_distance.setValue(2.0)
        self.spin_surface_distance.setSingleStep(0.1)
        self.spin_surface_distance.setRange(0.1, 10.0)
        self.spin_surface_distance.setDecimals(2)
        self.spin_surface_distance.setEnabled(False)

        surface_layout.addWidget(QLabel("  Surface Distance Value (Å):"))
        surface_layout.addWidget(self.spin_surface_distance)

        surface_layout.addWidget(QLabel("(Usa ConstructSurfaceModifier de OVITO)"))

        surface_group.setLayout(surface_layout)
        layout.addWidget(surface_group)

        # -------- MATERIAL PARAMS --------
        material_group = QGroupBox("Parámetros del Material")
        material_layout = QVBoxLayout()

        self.spin_total_atoms = QSpinBox()
        self.spin_total_atoms.setRange(100, 100000)
        self.spin_total_atoms.setValue(16384)

        self.spin_a0 = QDoubleSpinBox()
        self.spin_a0.setValue(3.532)
        self.spin_a0.setSingleStep(0.01)
        self.spin_a0.setRange(1.0, 10.0)
        self.spin_a0.setDecimals(4)

        material_layout.addWidget(QLabel("Átomos totales (perfectos):"))
        material_layout.addWidget(self.spin_total_atoms)
        material_layout.addWidget(QLabel("Parámetro de red a0 (Å):"))
        material_layout.addWidget(self.spin_a0)

        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        # -------- FEATURES --------
        features_group = QGroupBox("Features a Computar")
        features_layout = QVBoxLayout()

        self.chk_grid = QCheckBox("Grid features")
        self.chk_grid.setChecked(True)
        self.chk_hull = QCheckBox("Hull features")
        self.chk_hull.setChecked(True)
        self.chk_inertia = QCheckBox("Inertia features")
        self.chk_inertia.setChecked(True)
        self.chk_radial = QCheckBox("Radial features")
        self.chk_radial.setChecked(True)
        self.chk_entropy = QCheckBox("Entropy features")
        self.chk_entropy.setChecked(True)
        self.chk_clustering = QCheckBox("Clustering features")
        self.chk_clustering.setChecked(True)

        features_layout.addWidget(self.chk_grid)
        features_layout.addWidget(self.chk_hull)
        features_layout.addWidget(self.chk_inertia)
        features_layout.addWidget(self.chk_radial)
        features_layout.addWidget(self.chk_entropy)
        features_layout.addWidget(self.chk_clustering)

        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        # -------- RUN --------
        self.btn_run = QPushButton("🚀 Extraer Features")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self.run_extraction)
        layout.addWidget(self.btn_run)

        # -------- PROGRESS BAR --------
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.lbl_progress = QLabel("")
        layout.addWidget(self.lbl_progress)

        # -------- LOG --------
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Log de ejecución...")
        layout.addWidget(self.log)

        self.setCentralWidget(central)

    # ======================================================
    # FILES
    # ======================================================
    def select_input(self):
        path = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de dumps"
        )
        if path:
            self.input_dir = path
            self.lbl_input.setText(path)

    def toggle_surface_distance(self, checked):
        """Habilita/deshabilita el spinbox de surface distance"""
        self.spin_surface_distance.setEnabled(checked)

    # ======================================================
    # CORE - PROCESAMIENTO SIN THREADING (OVITO no es thread-safe)
    # ======================================================
    def run_extraction(self):
        """Inicia el proceso de extracción"""
        if not self.input_dir:
            QMessageBox.warning(self, "Error", "Debe seleccionar una carpeta de dumps")
            return

        # Crear configuración
        config = ExtractorConfig(
            input_dir=self.input_dir,
            probe_radius=self.spin_probe.value(),
            surface_distance=self.chk_surface_distance.isChecked(),
            surface_distance_value=self.spin_surface_distance.value(),
            total_atoms=self.spin_total_atoms.value(),
            a0=self.spin_a0.value(),
            lattice_type="fcc",
            compute_grid_features=self.chk_grid.isChecked(),
            compute_hull_features=self.chk_hull.isChecked(),
            compute_inertia_features=self.chk_inertia.isChecked(),
            compute_radial_features=self.chk_radial.isChecked(),
            compute_entropy_features=self.chk_entropy.isChecked(),
            compute_clustering_features=self.chk_clustering.isChecked()
        )

        # Limpiar log y preparar UI
        self.log.clear()
        self.log.append("🔬 Iniciando pipeline de extracción...\n")
        self.log.append(f"📂 Directorio: {config.input_dir}\n")
        self.log.append(f"⚙️ Probe radius: {config.probe_radius} Å\n")

        if config.surface_distance:
            self.log.append(f"⚙️ Método: Surface Distance ({config.surface_distance_value} Å)\n")
        else:
            self.log.append(f"⚙️ Método: Surface Selection (default)\n")

        self.log.append(f"⚙️ Total atoms: {config.total_atoms}\n")
        self.log.append(f"⚙️ a0: {config.a0} Å\n")
        self.log.append(f"⚙️ Lattice: {config.lattice_type}\n")

        # Mostrar features activas
        features_active = []
        if config.compute_grid_features:
            features_active.append("Grid")
        if config.compute_hull_features:
            features_active.append("Hull")
        if config.compute_inertia_features:
            features_active.append("Inertia")
        if config.compute_radial_features:
            features_active.append("Radial")
        if config.compute_entropy_features:
            features_active.append("Entropy")
        if config.compute_clustering_features:
            features_active.append("Clustering")

        self.log.append(f"✓ Features: {', '.join(features_active)}\n\n")

        # Deshabilitar botón
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ Procesando...")

        try:
            # Crear pipeline
            self.pipeline = ExtractorPipeline(config)

            # Obtener lista de archivos
            input_dir = Path(config.input_dir)

            # Extensiones a IGNORAR (no son dumps LAMMPS)
            ignore_extensions = {
                '.png', '.jpg', '.jpeg', '.gif', '.bmp',
                '.csv', '.xlsx', '.xls',
                '.py', '.pyc', '.pyo', '.pyx',
                '.txt', '.log', '.out',
                '.json', '.xml', '.yaml', '.yml',
                '.pdf', '.doc', '.docx',
                '.zip', '.tar', '.gz', '.bz2',
            }

            self.files_to_process = sorted(
                str(f) for f in input_dir.glob("*")
                if f.is_file()  # Solo archivos (no directorios, ignora normalized_dumps/)
                and not f.name.endswith("_surface_normalized.dump")  # No archivos procesados (legacy)
                and f.suffix.lower() not in ignore_extensions
            )

            if not self.files_to_process:
                self.log.append(f"❌ No se encontraron archivos en: {input_dir}\n")
                self.btn_run.setEnabled(True)
                self.btn_run.setText("🚀 Extraer Features")
                return

            self.log.append(f"📋 Archivos encontrados: {len(self.files_to_process)}\n\n")

            # Inicializar procesamiento
            self.current_file_index = 0
            self.results = []
            self.progress_bar.setMaximum(len(self.files_to_process))
            self.progress_bar.setValue(0)

            # Iniciar timer para procesar archivos
            self.timer.start(10)  # Procesa cada 10ms

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.log.append(f"\n❌ ERROR:\n{error_msg}\n")
            self.btn_run.setEnabled(True)
            self.btn_run.setText("🚀 Extraer Features")
            QMessageBox.critical(self, "Error", f"Error al iniciar extracción:\n\n{str(e)}")

    def process_next_file(self):
        """Procesa el siguiente archivo (llamado por QTimer)"""
        if self.current_file_index >= len(self.files_to_process):
            # Terminamos todos los archivos
            self.timer.stop()
            self.on_extraction_complete()
            return

        file_path = self.files_to_process[self.current_file_index]
        file_name = Path(file_path).name

        try:
            # Actualizar UI
            self.lbl_progress.setText(f"Procesando: {file_name}")
            self.progress_bar.setValue(self.current_file_index)
            QApplication.processEvents()  # Mantener UI responsive

            # Procesar archivo
            self.log.append(f"[{self.current_file_index + 1}/{len(self.files_to_process)}] {file_name}...")
            QApplication.processEvents()

            result = self.pipeline.process_single(file_path)

            if result is not None:
                self.results.append(result)
                self.log.append(" ✓\n")
            else:
                self.log.append(" ✗ (error)\n")

            QApplication.processEvents()

        except Exception as e:
            self.log.append(f" ✗ ERROR: {str(e)}\n")
            QApplication.processEvents()

        # Avanzar al siguiente archivo
        self.current_file_index += 1

    def on_extraction_complete(self):
        """Llamado cuando se completa la extracción de todos los archivos"""
        self.progress_bar.setValue(len(self.files_to_process))
        self.lbl_progress.setText("Completado")

        if self.results:
            # Guardar CSV
            df = pd.DataFrame(self.results)
            output_csv = f"{self.input_dir}/dataset_features.csv"
            df.to_csv(output_csv, index=False)

            self.log.append(f"\n✅ Extracción completada!\n")
            self.log.append(f"📊 Muestras procesadas: {len(df)}\n")
            self.log.append(f"💾 CSV guardado: {output_csv}\n")

            QMessageBox.information(
                self,
                "Éxito",
                f"Extracción completada!\n\nMuestras procesadas: {len(df)}\nCSV generado:\n{output_csv}"
            )
        else:
            self.log.append(f"\n❌ No se pudieron procesar archivos\n")
            QMessageBox.warning(self, "Advertencia", "No se pudieron procesar archivos")

        # Habilitar botón
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 Extraer Features")

