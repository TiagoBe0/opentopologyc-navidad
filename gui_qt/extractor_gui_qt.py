#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTextEdit,
    QProgressDialog, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from gui_qt.base_window import BaseWindow
from core.pipeline import ExtractorPipeline
from config.extractor_config import ExtractorConfig


class ExtractorWorker(QThread):
    """Worker thread para extraer features sin bloquear la UI"""

    # Señales
    log_message = pyqtSignal(str)  # Mensajes de log
    finished = pyqtSignal(str)  # Ruta del CSV generado
    error = pyqtSignal(str)  # Error message

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        """Ejecuta la extracción en thread separado"""
        try:
            self.log_message.emit("🔬 Iniciando pipeline de extracción...\n")

            pipeline = ExtractorPipeline(self.config)

            self.log_message.emit(f"📂 Directorio: {self.config.input_dir}\n")
            self.log_message.emit(f"⚙️ Probe radius: {self.config.probe_radius} Å\n")
            self.log_message.emit(f"⚙️ Total atoms: {self.config.total_atoms}\n")
            self.log_message.emit(f"⚙️ a0: {self.config.a0} Å\n")
            self.log_message.emit(f"⚙️ Lattice: {self.config.lattice_type}\n")
            self.log_message.emit(f"⚙️ Método: ConstructSurfaceModifier (OVITO)\n\n")

            # Ejecutar pipeline
            df = pipeline.run()

            if df is not None:
                output_csv = f"{self.config.input_dir}/dataset_features.csv"
                self.log_message.emit(f"\n✅ Extracción completada!\n")
                self.log_message.emit(f"📊 Muestras procesadas: {len(df)}\n")
                self.log_message.emit(f"💾 CSV guardado: {output_csv}\n")
                self.finished.emit(output_csv)
            else:
                self.error.emit("No se pudieron extraer features")

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class ExtractorGUIQt(BaseWindow):
    def __init__(self):
        super().__init__("OpenTopologyC - Extractor de Features", (700, 650))
        self.input_dir = ""
        self.worker = None
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

    # ======================================================
    # CORE
    # ======================================================
    def run_extraction(self):
        if not self.input_dir:
            QMessageBox.warning(self, "Error", "Debe seleccionar una carpeta de dumps")
            return

        # Crear configuración
        config = ExtractorConfig(
            input_dir=self.input_dir,
            probe_radius=self.spin_probe.value(),
            total_atoms=self.spin_total_atoms.value(),
            a0=self.spin_a0.value(),
            lattice_type="fcc",  # Por ahora fijo
            compute_grid_features=self.chk_grid.isChecked(),
            compute_hull_features=self.chk_hull.isChecked(),
            compute_inertia_features=self.chk_inertia.isChecked(),
            compute_radial_features=self.chk_radial.isChecked(),
            compute_entropy_features=self.chk_entropy.isChecked(),
            compute_clustering_features=self.chk_clustering.isChecked()
        )

        # NOTA: num_ghost_layers no va en ExtractorConfig
        # Se usa directamente en SurfaceExtractor, que lo toma del mismo config
        # Por ahora, el valor de ghost layers se ignora en esta GUI
        # TODO: Agregar num_ghost_layers a ExtractorConfig si es necesario

        # Limpiar log
        self.log.clear()

        # Deshabilitar botón
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ Procesando...")

        # Crear y configurar worker
        self.worker = ExtractorWorker(config)
        self.worker.log_message.connect(self.on_log_message)
        self.worker.finished.connect(self.on_extraction_finished)
        self.worker.error.connect(self.on_extraction_error)

        # Iniciar worker
        self.worker.start()

    def on_log_message(self, message):
        """Agrega mensaje al log"""
        self.log.append(message)

    def on_extraction_finished(self, csv_path):
        """Llamado cuando la extracción termina exitosamente"""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 Extraer Features")

        QMessageBox.information(
            self,
            "Éxito",
            f"Extracción completada!\n\nCSV generado:\n{csv_path}"
        )

        # Limpiar worker
        self.worker = None

    def on_extraction_error(self, error_msg):
        """Llamado cuando ocurre un error"""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 Extraer Features")

        self.log.append(f"\n❌ ERROR:\n{error_msg}\n")

        QMessageBox.critical(
            self,
            "Error",
            f"Error durante la extracción:\n\n{error_msg}"
        )

        # Limpiar worker
        self.worker = None

