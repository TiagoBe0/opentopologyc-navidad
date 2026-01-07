#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QProgressDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from gui_qt.base_window import BaseWindow
# IMPORTANTE: No importar AtomVisualizer3DQt aquí (lazy import)
# Esto previene que matplotlib se importe antes de configurarse

from core.prediction_pipeline import PredictionPipeline
from config.extractor_config import ExtractorConfig


class PredictionWorker(QThread):
    """Worker thread para ejecutar predicción sin bloquear la UI"""

    # Señales
    progress = pyqtSignal(int, int, str)  # step, total, message
    finished = pyqtSignal(dict)  # result
    error = pyqtSignal(str)  # error message

    def __init__(self, pipeline, dump_path, params):
        super().__init__()
        self.pipeline = pipeline
        self.dump_path = dump_path
        self.params = params

    def run(self):
        """Ejecuta la predicción en thread separado"""
        try:
            result = self.pipeline.predict_single(
                dump_file=self.dump_path,
                apply_alpha_shape=self.params['apply_alpha_shape'],
                probe_radius=self.params['probe_radius'],
                num_ghost_layers=self.params['num_ghost_layers'],
                apply_clustering=self.params['apply_clustering'],
                clustering_method=self.params['clustering_method'],
                clustering_params=self.params['clustering_params'],
                target_cluster=self.params['target_cluster'],
                save_intermediate_stages=True,
                progress_callback=self.on_progress
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def on_progress(self, step, total, message):
        """Callback de progreso"""
        self.progress.emit(step, total, message)


class PredictionGUIQt(BaseWindow):
    def __init__(self):
        super().__init__("OpenTopologyC - Predicción", (1200, 700))
        self.pipeline = None
        self.worker = None
        self.progress_dialog = None
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

        # Parámetros del Material
        material_box = QGroupBox("Parámetros del Material")
        material_layout = QVBoxLayout()

        self.spin_total_atoms = QSpinBox()
        self.spin_total_atoms.setRange(100, 100000)
        self.spin_total_atoms.setValue(16384)

        self.spin_a0 = QDoubleSpinBox()
        self.spin_a0.setValue(3.532)
        self.spin_a0.setSingleStep(0.001)
        self.spin_a0.setRange(1.0, 10.0)
        self.spin_a0.setDecimals(4)

        self.combo_lattice = QComboBox()
        self.combo_lattice.addItems(["fcc", "bcc", "hcp", "diamond", "sc"])
        self.combo_lattice.setCurrentText("fcc")

        material_layout.addWidget(QLabel("Átomos totales (perfectos):"))
        material_layout.addWidget(self.spin_total_atoms)
        material_layout.addWidget(QLabel("Parámetro de red a0 (Å):"))
        material_layout.addWidget(self.spin_a0)
        material_layout.addWidget(QLabel("Tipo de red:"))
        material_layout.addWidget(self.combo_lattice)

        material_box.setLayout(material_layout)
        controls.addWidget(material_box)

        # Botón Predecir
        btn_predict = QPushButton("🎯 Predecir y Visualizar")
        btn_predict.setMinimumHeight(40)
        btn_predict.clicked.connect(self.run_prediction)
        controls.addWidget(btn_predict)

        root.addLayout(controls, 1)

        # ---------- VISUALIZADOR ----------
        # Lazy import: importar AQUÍ para que matplotlib se configure primero
        from gui_qt.visualizer_3d_qt import AtomVisualizer3DQt
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
            # Crear configuración
            config = ExtractorConfig(
                input_dir=".",
                probe_radius=self.spin_probe.value(),
                total_atoms=self.spin_total_atoms.value(),
                a0=self.spin_a0.value(),
                lattice_type=self.combo_lattice.currentText(),
                compute_grid_features=True,
                compute_hull_features=True,
                compute_inertia_features=True,
                compute_radial_features=True,
                compute_entropy_features=True,
                compute_clustering_features=True
            )

            # Crear pipeline
            self.pipeline = PredictionPipeline(
                model_path=self.model_path,
                config=config,
                logger=lambda msg: print(msg)
            )

            # Preparar parámetros de clustering
            clustering_params = None
            if self.chk_cluster.isChecked():
                clustering_params = {'n_clusters': self.spin_clusters.value()}

            # Preparar parámetros para el worker
            params = {
                'apply_alpha_shape': self.chk_alpha.isChecked(),
                'probe_radius': self.spin_probe.value(),
                'num_ghost_layers': self.spin_ghost.value(),
                'apply_clustering': self.chk_cluster.isChecked(),
                'clustering_method': self.cmb_method.currentText(),
                'clustering_params': clustering_params,
                'target_cluster': 'largest'
            }

            # Crear progress dialog
            self.progress_dialog = QProgressDialog(
                "Iniciando predicción...",
                "Cancelar",
                0,
                5,
                self
            )
            self.progress_dialog.setWindowTitle("Procesando")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.canceled.connect(self.cancel_prediction)

            # Crear y configurar worker thread
            self.worker = PredictionWorker(self.pipeline, self.dump_path, params)
            self.worker.progress.connect(self.on_progress)
            self.worker.finished.connect(self.on_prediction_finished)
            self.worker.error.connect(self.on_prediction_error)

            # Iniciar worker
            self.worker.start()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))

    def on_progress(self, step, total, message):
        """Actualiza la barra de progreso"""
        if self.progress_dialog:
            self.progress_dialog.setValue(step)
            self.progress_dialog.setLabelText(f"[{step}/{total}] {message}")

    def on_prediction_finished(self, result):
        """Llamado cuando la predicción termina exitosamente"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Cargar etapas intermedias en visualizador
        if result.get("intermediate_stages"):
            self.visualizer.load_stages(result["intermediate_stages"])

        # Mostrar resultados
        msg = f"Predicción completada!\n\n"
        msg += f"Vacancias predichas: {result['predicted_vacancies']:.2f}\n"
        msg += f"Vacancias reales: {result['real_vacancies']}\n"
        msg += f"Error absoluto: {result['error']:.2f}\n\n"

        if result.get('clustering_applied'):
            cinfo = result.get('clustering_info', {})
            msg += f"Clusters encontrados: {cinfo.get('n_clusters', 'N/A')}\n"

        QMessageBox.information(self, "Éxito", msg)

        # Limpiar worker
        self.worker = None

    def on_prediction_error(self, error_msg):
        """Llamado cuando ocurre un error en la predicción"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        QMessageBox.critical(self, "Error", f"Error en predicción:\n{error_msg}")

        # Limpiar worker
        self.worker = None

    def cancel_prediction(self):
        """Cancela la predicción en curso"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.worker = None

        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
