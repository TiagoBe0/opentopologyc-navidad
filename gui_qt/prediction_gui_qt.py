#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI de Predicción con Pasos Separados
======================================

Permite ejecutar Alpha Shape, Clustering y Predicción por separado,
actualizando el visualizador 3D después de cada paso.
"""

import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QProgressDialog
)
from PySide6.QtCore import Qt, QThread, Signal

from gui_qt.base_window import BaseWindow
from core.alpha_shape_filter import filter_surface_atoms, LAMMPSDumpParser
from core.clustering_engine import cluster_surface_atoms
from core.loader import DumpLoader
from core.normalizer import PositionNormalizer
from core.feature_extractor import FeatureExtractor
from config.extractor_config import ExtractorConfig
import joblib


class AlphaShapeWorker(QThread):
    """Worker thread para ejecutar Alpha Shape"""
    finished = Signal(dict)  # {positions, n_atoms, dump_data}
    error = Signal(str)

    def __init__(self, dump_path, probe_radius, num_ghost_layers, lattice_param):
        super().__init__()
        self.dump_path = dump_path
        self.probe_radius = probe_radius
        self.num_ghost_layers = num_ghost_layers
        self.lattice_param = lattice_param

    def run(self):
        try:
            # Generar nombre de archivo de salida
            input_path = Path(self.dump_path)
            output_file = input_path.parent / f"{input_path.stem}_surface_filtered.dump"

            # Aplicar filtro Alpha Shape
            filter_surface_atoms(
                input_dump=self.dump_path,
                output_dump=str(output_file),
                probe_radius=self.probe_radius,
                lattice_param=self.lattice_param,
                num_ghost_layers=self.num_ghost_layers,
                smoothing=0
            )

            # Leer dump filtrado
            dump_data = LAMMPSDumpParser.read(str(output_file))
            positions = dump_data['positions']

            self.finished.emit({
                'positions': positions,
                'n_atoms': len(positions),
                'dump_data': dump_data,
                'output_file': str(output_file)
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class ClusteringWorker(QThread):
    """Worker thread para ejecutar Clustering"""
    finished = Signal(dict)  # {labels, cluster_info, positions}
    error = Signal(str)

    def __init__(self, positions, method, params):
        super().__init__()
        self.positions = positions
        self.method = method
        self.params = params

    def run(self):
        try:
            # Determinar si usar clustering jerárquico
            use_hierarchical = self.method == "Hierarchical"

            # Aplicar clustering
            labels, cluster_info = cluster_surface_atoms(
                positions=self.positions,
                method=self.method,
                hierarchical=use_hierarchical,
                **self.params
            )

            self.finished.emit({
                'labels': labels,
                'cluster_info': cluster_info,
                'positions': self.positions,
                'method': self.method
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class PredictionGUIQt(BaseWindow):
    def __init__(self):
        super().__init__("OpenTopologyC - Predicción (Pasos Separados)", (1200, 800))

        # Estado de la aplicación
        self.dump_path = ""
        self.model_path = ""
        self.original_dump_data = None
        self.filtered_positions = None
        self.filtered_dump_data = None
        self.clustering_labels = None
        self.clustering_info = None

        # Workers
        self.alpha_worker = None
        self.cluster_worker = None

        self._build_ui()

    # ======================================================
    # UI
    # ======================================================
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        # ---------- PANEL IZQUIERDO: CONTROLES ----------
        controls = QVBoxLayout()
        controls.setAlignment(Qt.AlignTop)

        # === SECCIÓN 1: ARCHIVOS ===
        files_box = QGroupBox("📂 Archivos")
        files_layout = QVBoxLayout()

        btn_dump = QPushButton("Cargar Dump Original")
        btn_dump.clicked.connect(self.load_dump)
        files_layout.addWidget(btn_dump)

        self.lbl_dump = QLabel("No cargado")
        self.lbl_dump.setStyleSheet("color: gray; font-size: 10px;")
        files_layout.addWidget(self.lbl_dump)

        btn_model = QPushButton("Cargar Modelo (.joblib/.pkl)")
        btn_model.clicked.connect(self.load_model)
        files_layout.addWidget(btn_model)

        self.lbl_model = QLabel("No cargado")
        self.lbl_model.setStyleSheet("color: gray; font-size: 10px;")
        files_layout.addWidget(self.lbl_model)

        files_box.setLayout(files_layout)
        controls.addWidget(files_box)

        # === SECCIÓN 2: PASO 1 - ALPHA SHAPE ===
        step1_box = QGroupBox("🔵 PASO 1: Alpha Shape")
        step1_layout = QVBoxLayout()

        self.chk_alpha = QCheckBox("Activar filtro Alpha Shape")
        self.chk_alpha.setChecked(True)
        step1_layout.addWidget(self.chk_alpha)

        step1_layout.addWidget(QLabel("Probe radius (Å):"))
        self.spin_probe = QDoubleSpinBox()
        self.spin_probe.setValue(2.0)
        self.spin_probe.setSingleStep(0.2)
        self.spin_probe.setRange(0.1, 10.0)
        step1_layout.addWidget(self.spin_probe)

        step1_layout.addWidget(QLabel("Ghost layers:"))
        self.spin_ghost = QSpinBox()
        self.spin_ghost.setRange(1, 5)
        self.spin_ghost.setValue(2)
        step1_layout.addWidget(self.spin_ghost)

        self.btn_step1 = QPushButton("▶ Ejecutar Paso 1")
        self.btn_step1.setMinimumHeight(40)
        self.btn_step1.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_step1.clicked.connect(self.run_step1_alpha_shape)
        step1_layout.addWidget(self.btn_step1)

        self.lbl_step1_result = QLabel("")
        self.lbl_step1_result.setStyleSheet("color: green; font-size: 10px;")
        step1_layout.addWidget(self.lbl_step1_result)

        step1_box.setLayout(step1_layout)
        controls.addWidget(step1_box)

        # === SECCIÓN 3: PASO 2 - CLUSTERING ===
        step2_box = QGroupBox("🟢 PASO 2: Clustering")
        step2_layout = QVBoxLayout()

        self.chk_cluster = QCheckBox("Activar clustering")
        self.chk_cluster.setChecked(False)
        step2_layout.addWidget(self.chk_cluster)

        step2_layout.addWidget(QLabel("Método:"))
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["KMeans", "MeanShift", "Aglomerativo", "HDBSCAN", "Hierarchical"])
        step2_layout.addWidget(self.cmb_method)

        step2_layout.addWidget(QLabel("N clusters (ignorado en Hierarchical):"))
        self.spin_clusters = QSpinBox()
        self.spin_clusters.setRange(2, 20)
        self.spin_clusters.setValue(5)
        step2_layout.addWidget(self.spin_clusters)

        self.btn_step2 = QPushButton("▶ Ejecutar Paso 2")
        self.btn_step2.setMinimumHeight(40)
        self.btn_step2.setEnabled(False)
        self.btn_step2.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_step2.clicked.connect(self.run_step2_clustering)
        step2_layout.addWidget(self.btn_step2)

        self.lbl_step2_result = QLabel("")
        self.lbl_step2_result.setStyleSheet("color: blue; font-size: 10px;")
        step2_layout.addWidget(self.lbl_step2_result)

        step2_box.setLayout(step2_layout)
        controls.addWidget(step2_box)

        # === SECCIÓN 4: PASO 3 - PREDICCIÓN ===
        step3_box = QGroupBox("🟡 PASO 3: Predicción")
        step3_layout = QVBoxLayout()

        # Parámetros del material
        step3_layout.addWidget(QLabel("Átomos totales (cristal perfecto):"))
        self.spin_total_atoms = QSpinBox()
        self.spin_total_atoms.setRange(100, 100000)
        self.spin_total_atoms.setValue(16384)
        step3_layout.addWidget(self.spin_total_atoms)

        step3_layout.addWidget(QLabel("Parámetro de red a0 (Å):"))
        self.spin_a0 = QDoubleSpinBox()
        self.spin_a0.setValue(3.532)
        self.spin_a0.setSingleStep(0.001)
        self.spin_a0.setRange(1.0, 10.0)
        self.spin_a0.setDecimals(4)
        step3_layout.addWidget(self.spin_a0)

        step3_layout.addWidget(QLabel("Tipo de red:"))
        self.combo_lattice = QComboBox()
        self.combo_lattice.addItems(["fcc", "bcc", "hcp", "diamond", "sc"])
        step3_layout.addWidget(self.combo_lattice)

        self.btn_step3 = QPushButton("▶ Ejecutar Paso 3 (Predicción)")
        self.btn_step3.setMinimumHeight(40)
        self.btn_step3.setEnabled(False)
        self.btn_step3.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_step3.clicked.connect(self.run_step3_prediction)
        step3_layout.addWidget(self.btn_step3)

        self.lbl_step3_result = QLabel("")
        self.lbl_step3_result.setStyleSheet("color: orange; font-size: 10px;")
        step3_layout.addWidget(self.lbl_step3_result)

        step3_box.setLayout(step3_layout)
        controls.addWidget(step3_box)

        controls.addStretch()

        root.addLayout(controls, 1)

        # ---------- PANEL DERECHO: VISUALIZADOR ----------
        from gui_qt.visualizer_3d_qt import AtomVisualizer3DQt
        self.visualizer = AtomVisualizer3DQt()
        root.addWidget(self.visualizer, 3)

        self.setCentralWidget(central)

    # ======================================================
    # FILE LOADERS
    # ======================================================
    def load_dump(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar dump", "", "Dump files (*.dump);;All files (*)"
        )
        if path:
            self.dump_path = path
            self.lbl_dump.setText(f"✓ {Path(path).name}")
            self.lbl_dump.setStyleSheet("color: green; font-size: 10px;")

            # Cargar dump en visualizador
            self.visualizer.load_dump_from_path(path)

            # Leer dump original para referencia
            self.original_dump_data = LAMMPSDumpParser.read(path)

            # Resetear estado
            self.filtered_positions = None
            self.filtered_dump_data = None
            self.clustering_labels = None
            self.clustering_info = None

            self.lbl_step1_result.setText("")
            self.lbl_step2_result.setText("")
            self.lbl_step3_result.setText("")

            self.btn_step2.setEnabled(False)
            self.btn_step3.setEnabled(False)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar modelo", "", "Model (*.joblib *.pkl)"
        )
        if path:
            self.model_path = path
            self.lbl_model.setText(f"✓ {Path(path).name}")
            self.lbl_model.setStyleSheet("color: green; font-size: 10px;")

    # ======================================================
    # PASO 1: ALPHA SHAPE
    # ======================================================
    def run_step1_alpha_shape(self):
        if not self.dump_path:
            QMessageBox.warning(self, "Error", "Debe cargar un dump primero")
            return

        if not self.chk_alpha.isChecked():
            # Saltear Alpha Shape, usar dump original
            self.filtered_positions = self.original_dump_data['positions']
            self.filtered_dump_data = self.original_dump_data
            self.lbl_step1_result.setText(f"✓ Alpha Shape omitido. Átomos: {len(self.filtered_positions)}")
            self.btn_step2.setEnabled(True)
            self.btn_step3.setEnabled(True)
            return

        # Ejecutar Alpha Shape
        self.btn_step1.setEnabled(False)
        self.btn_step1.setText("⏳ Procesando...")

        self.alpha_worker = AlphaShapeWorker(
            self.dump_path,
            self.spin_probe.value(),
            self.spin_ghost.value(),
            self.spin_a0.value()  # lattice_param
        )
        self.alpha_worker.finished.connect(self.on_alpha_shape_finished)
        self.alpha_worker.error.connect(self.on_alpha_shape_error)
        self.alpha_worker.start()

    def on_alpha_shape_finished(self, result):
        self.filtered_positions = result['positions']
        self.filtered_dump_data = result['dump_data']

        # Actualizar visualizador con átomos filtrados
        self.visualizer.load_dump_from_path(result['output_file'])

        self.lbl_step1_result.setText(f"✓ Completado. Átomos superficiales: {result['n_atoms']}")

        # Habilitar paso 2 y 3
        self.btn_step2.setEnabled(True)
        self.btn_step3.setEnabled(True)

        # Restaurar botón
        self.btn_step1.setEnabled(True)
        self.btn_step1.setText("▶ Ejecutar Paso 1")

        QMessageBox.information(self, "Paso 1 Completado",
                               f"Alpha Shape aplicado\nÁtomos superficiales: {result['n_atoms']}")

    def on_alpha_shape_error(self, error_msg):
        self.lbl_step1_result.setText(f"✗ Error: {error_msg}")
        self.lbl_step1_result.setStyleSheet("color: red; font-size: 10px;")

        self.btn_step1.setEnabled(True)
        self.btn_step1.setText("▶ Ejecutar Paso 1")

        QMessageBox.critical(self, "Error en Paso 1", f"Error al aplicar Alpha Shape:\n{error_msg}")

    # ======================================================
    # PASO 2: CLUSTERING
    # ======================================================
    def run_step2_clustering(self):
        if self.filtered_positions is None:
            QMessageBox.warning(self, "Error", "Debe ejecutar Paso 1 primero")
            return

        if not self.chk_cluster.isChecked():
            # Saltear clustering
            self.clustering_labels = None
            self.clustering_info = None
            self.lbl_step2_result.setText("✓ Clustering omitido")

            # Actualizar visualizador sin colores de clustering
            self.visualizer.positions = self.filtered_positions
            self.visualizer.colors = None
            self.visualizer.cluster_labels = None
            self.visualizer.plot()

            return

        # Preparar parámetros
        method = self.cmb_method.currentText()
        params = {}

        if method == "Hierarchical":
            params = {
                'min_atoms': 30,
                'max_iterations': 4,
                'silhouette_threshold': 0.3,
                'davies_bouldin_threshold': 1.5,
                'dispersion_threshold': 5.0,
                'quantile': 0.2
            }
        else:
            params = {'n_clusters': self.spin_clusters.value()}

        # Ejecutar clustering
        self.btn_step2.setEnabled(False)
        self.btn_step2.setText("⏳ Procesando...")

        self.cluster_worker = ClusteringWorker(
            self.filtered_positions,
            method,
            params
        )
        self.cluster_worker.finished.connect(self.on_clustering_finished)
        self.cluster_worker.error.connect(self.on_clustering_error)
        self.cluster_worker.start()

    def on_clustering_finished(self, result):
        self.clustering_labels = result['labels']
        self.clustering_info = result['cluster_info']

        n_clusters = self.clustering_info['n_clusters']

        # Actualizar visualizador con colores de clustering
        self.visualizer.positions = self.filtered_positions
        self.visualizer.apply_clustering(self.clustering_labels)

        self.lbl_step2_result.setText(f"✓ Completado. Clusters: {n_clusters}")

        # Restaurar botón
        self.btn_step2.setEnabled(True)
        self.btn_step2.setText("▶ Ejecutar Paso 2")

        QMessageBox.information(self, "Paso 2 Completado",
                               f"Clustering aplicado\nMétodo: {result['method']}\nClusters encontrados: {n_clusters}")

    def on_clustering_error(self, error_msg):
        self.lbl_step2_result.setText(f"✗ Error: {error_msg}")
        self.lbl_step2_result.setStyleSheet("color: red; font-size: 10px;")

        self.btn_step2.setEnabled(True)
        self.btn_step2.setText("▶ Ejecutar Paso 2")

        QMessageBox.critical(self, "Error en Paso 2", f"Error al aplicar clustering:\n{error_msg}")

    # ======================================================
    # PASO 3: PREDICCIÓN
    # ======================================================
    def run_step3_prediction(self):
        if self.filtered_positions is None:
            QMessageBox.warning(self, "Error", "Debe ejecutar Paso 1 primero")
            return

        if not self.model_path:
            QMessageBox.warning(self, "Error", "Debe cargar un modelo primero")
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

            # Cargar modelo
            model = joblib.load(self.model_path)

            # SI HAY CLUSTERING: Predecir para cada cluster por separado
            if self.clustering_labels is not None:
                self._predict_with_clusters(model, config)
            else:
                # Sin clustering: predicción normal
                self._predict_without_clusters(model, config)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.lbl_step3_result.setText(f"✗ Error")
            self.lbl_step3_result.setStyleSheet("color: red; font-size: 10px;")
            QMessageBox.critical(self, "Error en Predicción", f"Error:\n{str(e)}")

    def _predict_without_clusters(self, model, config):
        """Predicción sin clustering (método original)"""
        normalizer = PositionNormalizer(scale_factor=config.a0)
        extractor = FeatureExtractor(config)

        pos_norm, box_size, _ = normalizer.normalize(self.filtered_positions)

        features = {}
        if config.compute_grid_features:
            features.update(extractor.grid_features(pos_norm, box_size))
        if config.compute_inertia_features:
            features.update(extractor.inertia_feature(self.filtered_positions))
        if config.compute_radial_features:
            features.update(extractor.radial_features(self.filtered_positions))
        if config.compute_entropy_features:
            features.update(extractor.entropy_spatial(pos_norm))
        if config.compute_clustering_features:
            features.update(extractor.bandwidth(pos_norm))

        features["num_points"] = len(self.filtered_positions)

        # Predecir
        import pandas as pd
        df = pd.DataFrame([features])

        forbidden = [
            "n_vacancies", "n_atoms_surface",
            "vacancies", "file", "num_atoms_real", "num_points"
        ]
        X = df.drop(columns=[c for c in forbidden if c in df.columns], errors="ignore")

        prediction = model.predict(X)[0]

        # Calcular vacancias reales
        n_atoms_real = len(self.original_dump_data['positions'])
        n_vacancies_real = config.total_atoms - n_atoms_real
        error = abs(prediction - n_vacancies_real)

        # Mostrar resultado
        msg = f"🎯 Predicción Completada (Sin Clustering)\n\n"
        msg += f"Vacancias predichas: {prediction:.2f}\n"
        msg += f"Vacancias reales: {n_vacancies_real}\n"
        msg += f"Error absoluto: {error:.2f}\n\n"
        msg += f"Átomos en simulación: {n_atoms_real}\n"
        msg += f"Átomos superficiales: {len(self.filtered_positions)}"

        self.lbl_step3_result.setText(f"✓ Predicción: {prediction:.2f} vacancias (Error: {error:.2f})")

        QMessageBox.information(self, "Predicción Completada", msg)

    def _predict_with_clusters(self, model, config):
        """Predicción con clustering: predice para cada cluster y suma"""
        import pandas as pd

        normalizer = PositionNormalizer(scale_factor=config.a0)
        extractor = FeatureExtractor(config)

        # Obtener clusters únicos (excluyendo ruido si existe)
        unique_labels = np.unique(self.clustering_labels)
        clusters_to_process = [lbl for lbl in unique_labels if lbl != -1]

        cluster_predictions = []
        total_prediction = 0.0

        # Predecir para cada cluster
        for cluster_id in clusters_to_process:
            # Extraer posiciones del cluster
            mask = self.clustering_labels == cluster_id
            cluster_positions = self.filtered_positions[mask]
            n_atoms_cluster = len(cluster_positions)

            # Extraer features para este cluster
            pos_norm, box_size, _ = normalizer.normalize(cluster_positions)

            features = {}
            if config.compute_grid_features:
                features.update(extractor.grid_features(pos_norm, box_size))
            if config.compute_inertia_features:
                features.update(extractor.inertia_feature(cluster_positions))
            if config.compute_radial_features:
                features.update(extractor.radial_features(cluster_positions))
            if config.compute_entropy_features:
                features.update(extractor.entropy_spatial(pos_norm))
            if config.compute_clustering_features:
                features.update(extractor.bandwidth(pos_norm))

            features["num_points"] = n_atoms_cluster

            # Predecir vacancias para este cluster
            df = pd.DataFrame([features])
            forbidden = [
                "n_vacancies", "n_atoms_surface",
                "vacancies", "file", "num_atoms_real", "num_points"
            ]
            X = df.drop(columns=[c for c in forbidden if c in df.columns], errors="ignore")

            cluster_pred = model.predict(X)[0]
            total_prediction += cluster_pred

            cluster_predictions.append({
                'cluster_id': cluster_id,
                'n_atoms': n_atoms_cluster,
                'prediction': cluster_pred
            })

        # Calcular vacancias reales
        n_atoms_real = len(self.original_dump_data['positions'])
        n_vacancies_real = config.total_atoms - n_atoms_real
        error = abs(total_prediction - n_vacancies_real)

        # Construir mensaje detallado
        msg = f"🎯 Predicción Completada (Con Clustering)\n\n"
        msg += f"📊 Total de Clusters: {len(clusters_to_process)}\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        msg += "Predicción por Cluster:\n"
        for cp in cluster_predictions:
            msg += f"  • Cluster {cp['cluster_id']}: {cp['prediction']:.2f} vacancias ({cp['n_atoms']} átomos)\n"

        msg += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"Vacancias predichas (TOTAL): {total_prediction:.2f}\n"
        msg += f"Vacancias reales: {n_vacancies_real}\n"
        msg += f"Error absoluto: {error:.2f}\n\n"
        msg += f"Átomos en simulación: {n_atoms_real}\n"
        msg += f"Átomos superficiales: {len(self.filtered_positions)}"

        self.lbl_step3_result.setText(f"✓ {len(clusters_to_process)} clusters: {total_prediction:.2f} vacancias (Error: {error:.2f})")

        QMessageBox.information(self, "Predicción por Clusters", msg)
