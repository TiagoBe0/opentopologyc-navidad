#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt

from gui_qt.base_window import BaseWindow
from gui_qt.visualizer_3d_qt import AtomVisualizer3DQt

from core.prediction_pipeline import PredictionPipeline


class PredictionGUIQt(BaseWindow):
    def __init__(self):
        super().__init__("OpenTopologyC - Predicción", (1200, 700))
        self.pipeline = None
        self._build_ui()

    # ======================================================
    # UI
    # ======================================================
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        # ---------- PANEL IZQUIERDO ----------
        controls = QVBoxLayout()
        controls.setAlignment(Qt.AlignTop)

        # Dump
        self.dump_path = ""
        btn_dump = QPushButton("📂 Cargar Dump")
        btn_dump.clicked.connect(self.load_dump)
        controls.addWidget(btn_dump)

        # Modelo
        self.model_path = ""
        btn_model = QPushButton("📂 Cargar Modelo")
        btn_model.clicked.connect(self.load_model)
        controls.addWidget(btn_model)

        # Alpha Shape
        alpha_box = QGroupBox("Alpha Shape")
        alpha_layout = QVBoxLayout()
        self.chk_alpha = QCheckBox("Aplicar Alpha Shape")
        self.chk_alpha.setChecked(True)

        self.spin_probe = QDoubleSpinBox()
        self.spin_probe.setValue(2.0)
        self.spin_probe.setSingleStep(0.2)

        self.spin_ghost = QSpinBox()
        self.spin_ghost.setRange(1, 5)
        self.spin_ghost.setValue(2)

        alpha_layout.addWidget(self.chk_alpha)
        alpha_layout.addWidget(QLabel("Probe radius"))
        alpha_layout.addWidget(self.spin_probe)
        alpha_layout.addWidget(QLabel("Ghost layers"))
        alpha_layout.addWidget(self.spin_ghost)
        alpha_box.setLayout(alpha_layout)

        controls.addWidget(alpha_box)

        # Clustering
        cluster_box = QGroupBox("Clustering")
        cl = QVBoxLayout()

        self.chk_cluster = QCheckBox("Aplicar clustering")
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["KMeans", "MeanShift", "Aglomerativo", "HDBSCAN"])

        self.spin_clusters = QSpinBox()
        self.spin_clusters.setRange(2, 20)
        self.spin_clusters.setValue(5)

        cl.addWidget(self.chk_cluster)
        cl.addWidget(QLabel("Método"))
        cl.addWidget(self.cmb_method)
        cl.addWidget(QLabel("N clusters"))
        cl.addWidget(self.spin_clusters)

        cluster_box.setLayout(cl)
        controls.addWidget(cluster_box)

        # Botón Predecir
        btn_predict = QPushButton("🎯 Predecir y Visualizar")
        btn_predict.setMinimumHeight(40)
        btn_predict.clicked.connect(self.run_prediction)
        controls.addWidget(btn_predict)

        root.addLayout(controls, 1)

        # ---------- VISUALIZADOR ----------
        self.visualizer = AtomVisualizer3DQt()
        root.addWidget(self.visualizer, 3)

        self.setCentralWidget(central)

    # ======================================================
    # FILE LOADERS
    # ======================================================
    def load_dump(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar dump", "", "Dump (*.dump)"
        )
        if path:
            self.dump_path = path
            self.visualizer.load_dump_from_path(path)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar modelo", "", "Model (*.joblib *.pkl)"
        )
        if path:
            self.model_path = path

    # ======================================================
    # PIPELINE
    # ======================================================
    def run_prediction(self):
        if not self.dump_path or not self.model_path:
            QMessageBox.warning(self, "Error", "Debe cargar dump y modelo")
            return

        try:
            self.pipeline = PredictionPipeline(
                dump_file=self.dump_path,
                model_file=self.model_path,
                apply_alpha=self.chk_alpha.isChecked(),
                probe_radius=self.spin_probe.value(),
                num_ghost_layers=self.spin_ghost.value(),
            )

            result = self.pipeline.run()

            # Enviar datos al visualizador
            self.visualizer.positions = result["positions"]
            self.visualizer.colors = result.get("vacancy_prob", None)
            self.visualizer.plot()

            # Clustering visual
            if self.chk_cluster.isChecked():
                self.visualizer.apply_clustering(
                    method=self.cmb_method.currentText(),
                    n_clusters=self.spin_clusters.value(),
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
