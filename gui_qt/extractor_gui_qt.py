#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTextEdit
)
from PyQt5.QtCore import Qt

from gui_qt.base_window import BaseWindow
from core.feature_extractor import FeatureExtractor


class ExtractorGUIQt(BaseWindow):
    def __init__(self):
        super().__init__("OpenTopologyC - Extractor de Features", (700, 600))
        self.input_dir = ""
        self.output_csv = ""
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

        # -------- OUTPUT --------
        btn_output = QPushButton("💾 Seleccionar CSV de salida")
        btn_output.clicked.connect(self.select_output)
        self.lbl_output = QLabel("No seleccionado")

        layout.addWidget(btn_output)
        layout.addWidget(self.lbl_output)

        # -------- PARAMS --------
        params = QGroupBox("Parámetros Alpha Shape")
        p = QVBoxLayout()

        self.spin_probe = QDoubleSpinBox()
        self.spin_probe.setValue(2.0)
        self.spin_probe.setSingleStep(0.2)

        self.spin_ghost = QSpinBox()
        self.spin_ghost.setRange(1, 5)
        self.spin_ghost.setValue(2)

        p.addWidget(QLabel("Probe radius"))
        p.addWidget(self.spin_probe)
        p.addWidget(QLabel("Ghost layers"))
        p.addWidget(self.spin_ghost)

        params.setLayout(p)
        layout.addWidget(params)

        # -------- RUN --------
        btn_run = QPushButton("🚀 Extraer Features")
        btn_run.setMinimumHeight(40)
        btn_run.clicked.connect(self.run_extraction)
        layout.addWidget(btn_run)

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

    def select_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Seleccionar CSV de salida", "", "CSV (*.csv)"
        )
        if path:
            self.output_csv = path
            self.lbl_output.setText(path)

    # ======================================================
    # CORE
    # ======================================================
    def run_extraction(self):
        if not self.input_dir or not self.output_csv:
            QMessageBox.warning(self, "Error", "Faltan rutas")
            return

        try:
            self.log.append("🔬 Iniciando extracción...")
            extractor = FeatureExtractor(
                dump_folder=self.input_dir,
                output_csv=self.output_csv,
                probe_radius=self.spin_probe.value(),
                num_ghost_layers=self.spin_ghost.value(),
            )
            extractor.run()
            self.log.append("✅ Extracción finalizada")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
