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
    QProgressDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal

from .base_window import BaseWindow
from ..core.alpha_shape_filter import filter_surface_atoms, LAMMPSDumpParser
from ..core.clustering_engine import cluster_surface_atoms
from ..core.loader import DumpLoader
from ..core.normalizer import PositionNormalizer
from ..core.feature_extractor import FeatureExtractor
from ..config.extractor_config import ExtractorConfig
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

        # Estado de predicción por clusters
        self.cluster_predictions = []  # Lista de {cluster_id, n_atoms, prediction, model_name, features}
        self.current_prediction_positions = None
        self.current_prediction_source = ""
        self.current_config = None  # ExtractorConfig usado en la predicción

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

        # ---------- PANEL IZQUIERDO: CONTROLES CON SCROLL ----------
        # Crear scroll area para los controles
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(400)  # Ancho mínimo para los controles

        # Widget contenedor para los controles
        controls_widget = QWidget()
        controls = QVBoxLayout(controls_widget)
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

        # Selector de modelos disponibles
        files_layout.addWidget(QLabel("Modelos disponibles:"))
        self.combo_models = QComboBox()
        self.combo_models.currentIndexChanged.connect(self.on_model_selected)
        files_layout.addWidget(self.combo_models)

        # Botón para refrescar lista de modelos
        btn_refresh = QPushButton("🔄 Actualizar lista")
        btn_refresh.clicked.connect(self.refresh_model_list)
        files_layout.addWidget(btn_refresh)

        # Botón para cargar modelo manualmente
        btn_model = QPushButton("Cargar Modelo Manual (.joblib/.pkl)")
        btn_model.clicked.connect(self.load_model_manual)
        files_layout.addWidget(btn_model)

        self.lbl_model = QLabel("No cargado")
        self.lbl_model.setStyleSheet("color: gray; font-size: 10px;")
        files_layout.addWidget(self.lbl_model)

        files_box.setLayout(files_layout)
        controls.addWidget(files_box)

        # Cargar lista inicial de modelos
        self.refresh_model_list()

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
        self.spin_clusters.setRange(1, 1000)
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

        # --- FILTRO DE CLUSTERS ---
        step2_layout.addWidget(QLabel("─" * 30))
        step2_layout.addWidget(QLabel("🔍 Visualizar Cluster Individual:"))

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Cluster:"))
        self.spin_cluster_filter = QSpinBox()
        self.spin_cluster_filter.setRange(0, 0)
        self.spin_cluster_filter.setValue(0)
        self.spin_cluster_filter.setEnabled(False)
        filter_layout.addWidget(self.spin_cluster_filter)
        step2_layout.addLayout(filter_layout)

        btn_filter_layout = QHBoxLayout()
        self.btn_view_cluster = QPushButton("👁️ Ver cluster")
        self.btn_view_cluster.setEnabled(False)
        self.btn_view_cluster.clicked.connect(self.view_selected_cluster)
        btn_filter_layout.addWidget(self.btn_view_cluster)

        self.btn_view_all = QPushButton("👁️ Ver todos")
        self.btn_view_all.setEnabled(False)
        self.btn_view_all.clicked.connect(self.view_all_clusters)
        btn_filter_layout.addWidget(self.btn_view_all)
        step2_layout.addLayout(btn_filter_layout)

        step2_box.setLayout(step2_layout)
        controls.addWidget(step2_box)

        # === SECCIÓN 4: PASO 3 - PREDICCIÓN ===
        step3_box = QGroupBox("🟡 PASO 3: Predicción")
        step3_layout = QVBoxLayout()

        # Parámetros del material
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

        # === SECCIÓN 5: PASO 4 - AJUSTE FINO (OPCIONAL) ===
        self.step4_box = QGroupBox("⚙️ PASO 4: Ajuste Fino por Cluster (Opcional)")
        step4_layout = QVBoxLayout()

        step4_layout.addWidget(QLabel("Predicción Total Actual:"))
        self.lbl_total_prediction = QLabel("-- vacancias")
        self.lbl_total_prediction.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3;")
        step4_layout.addWidget(self.lbl_total_prediction)

        step4_layout.addWidget(QLabel("\nPrediciones por Cluster:"))

        # Tabla de clusters
        self.table_clusters = QTableWidget()
        self.table_clusters.setColumnCount(5)
        self.table_clusters.setHorizontalHeaderLabels(["Cluster", "Átomos", "Predicción", "Modelo", "Acción"])
        self.table_clusters.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_clusters.setMinimumHeight(150)
        self.table_clusters.setMaximumHeight(300)  # Aumentado para mejor visibilidad
        self.table_clusters.setSelectionBehavior(QTableWidget.SelectRows)
        step4_layout.addWidget(self.table_clusters)

        # Selector de modelo alternativo
        step4_layout.addWidget(QLabel("\nModelo alternativo para re-predicción:"))
        self.combo_alt_model = QComboBox()
        step4_layout.addWidget(self.combo_alt_model)

        # Botón para re-predecir
        self.btn_repredict = QPushButton("🔄 Re-predecir Cluster Seleccionado")
        self.btn_repredict.setStyleSheet("background-color: #9C27B0; color: white;")
        self.btn_repredict.clicked.connect(self.repredict_selected_cluster)
        step4_layout.addWidget(self.btn_repredict)

        self.lbl_repredict_status = QLabel("")
        self.lbl_repredict_status.setStyleSheet("font-size: 10px;")
        step4_layout.addWidget(self.lbl_repredict_status)

        self.step4_box.setLayout(step4_layout)
        self.step4_box.setVisible(False)  # Oculto inicialmente
        controls.addWidget(self.step4_box)

        controls.addStretch()

        # Asignar el widget de controles al scroll area
        scroll_area.setWidget(controls_widget)

        # Agregar scroll area al layout raíz (panel izquierdo)
        root.addWidget(scroll_area, 1)

        # ---------- PANEL DERECHO: VISUALIZADOR ----------
        from .visualizer_3d_qt import AtomVisualizer3DQt
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

            # HABILITAR todos los pasos al cargar dump (pasos independientes)
            self.btn_step2.setEnabled(True)
            self.btn_step3.setEnabled(True)

            # Resetear controles de filtro de clusters
            self.spin_cluster_filter.setEnabled(False)
            self.spin_cluster_filter.setRange(0, 0)
            self.btn_view_cluster.setEnabled(False)
            self.btn_view_all.setEnabled(False)

    def refresh_model_list(self):
        """Actualiza el combo box con los modelos disponibles en models/"""
        self.combo_models.clear()

        # Obtener carpeta models/
        models_dir = Path(__file__).parent.parent.parent / "models"

        if not models_dir.exists():
            models_dir.mkdir(exist_ok=True)
            self.combo_models.addItem("(No hay modelos disponibles)")
            return

        # Buscar archivos .joblib y .pkl
        model_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pkl"))

        if not model_files:
            self.combo_models.addItem("(No hay modelos disponibles)")
        else:
            # Ordenar por fecha de modificación (más reciente primero)
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for model_file in model_files:
                self.combo_models.addItem(model_file.name, userData=str(model_file))

    def on_model_selected(self, index):
        """Cuando se selecciona un modelo del dropdown"""
        if index < 0:
            return

        model_path = self.combo_models.itemData(index)
        if model_path and model_path != "(No hay modelos disponibles)":
            self.model_path = model_path
            self.lbl_model.setText(f"✓ {Path(model_path).name}")
            self.lbl_model.setStyleSheet("color: green; font-size: 10px;")

    def load_model_manual(self):
        """Carga un modelo manualmente desde cualquier ubicación"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar modelo", "", "Model (*.joblib *.pkl)"
        )
        if path:
            self.model_path = path
            self.lbl_model.setText(f"✓ {Path(path).name}")
            self.lbl_model.setStyleSheet("color: green; font-size: 10px;")

    # ======================================================
    # MÉTODOS AUXILIARES
    # ======================================================
    def _get_current_positions(self):
        """
        Devuelve las posiciones actuales a usar:
        - Si hay Alpha Shape aplicado, devuelve filtered_positions
        - Si no, devuelve posiciones originales del dump

        Returns:
            tuple: (positions, source_description)
        """
        if self.filtered_positions is not None:
            return self.filtered_positions, "átomos filtrados (Alpha Shape)"
        elif self.original_dump_data is not None:
            return self.original_dump_data['positions'], "átomos originales (sin Alpha Shape)"
        else:
            return None, "sin datos"

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
        # Obtener posiciones actuales (filtradas o originales)
        positions, source_desc = self._get_current_positions()

        if positions is None:
            QMessageBox.warning(self, "Error", "Debe cargar un dump primero")
            return

        if not self.chk_cluster.isChecked():
            # Saltear clustering
            self.clustering_labels = None
            self.clustering_info = None
            self.lbl_step2_result.setText("✓ Clustering omitido")

            # Deshabilitar controles de filtro de clusters
            self.spin_cluster_filter.setEnabled(False)
            self.spin_cluster_filter.setRange(0, 0)
            self.btn_view_cluster.setEnabled(False)
            self.btn_view_all.setEnabled(False)

            # Actualizar visualizador sin colores de clustering
            self.visualizer.positions = positions
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

        # Guardar las posiciones que se están usando (para el visualizador después)
        self.current_clustering_positions = positions
        self.current_clustering_source = source_desc

        self.cluster_worker = ClusteringWorker(
            positions,
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
        self.visualizer.positions = self.current_clustering_positions
        self.visualizer.apply_clustering(self.clustering_labels)

        # Mensaje con información de origen de datos
        source_info = f" ({self.current_clustering_source})"
        self.lbl_step2_result.setText(f"✓ Completado. Clusters: {n_clusters}{source_info}")

        # Habilitar controles de filtro de clusters
        unique_labels = np.unique(self.clustering_labels)
        cluster_ids = [lbl for lbl in unique_labels if lbl != -1]

        if len(cluster_ids) > 0:
            self.spin_cluster_filter.setEnabled(True)
            self.spin_cluster_filter.setRange(min(cluster_ids), max(cluster_ids))
            self.spin_cluster_filter.setValue(min(cluster_ids))
            self.btn_view_cluster.setEnabled(True)
            self.btn_view_all.setEnabled(True)

        # Restaurar botón
        self.btn_step2.setEnabled(True)
        self.btn_step2.setText("▶ Ejecutar Paso 2")

        QMessageBox.information(self, "Paso 2 Completado",
                               f"Clustering aplicado\nMétodo: {result['method']}\nClusters encontrados: {n_clusters}\n\nOrigen: {self.current_clustering_source}")

    def on_clustering_error(self, error_msg):
        self.lbl_step2_result.setText(f"✗ Error: {error_msg}")
        self.lbl_step2_result.setStyleSheet("color: red; font-size: 10px;")

        self.btn_step2.setEnabled(True)
        self.btn_step2.setText("▶ Ejecutar Paso 2")

        QMessageBox.critical(self, "Error en Paso 2", f"Error al aplicar clustering:\n{error_msg}")

    # ======================================================
    # CÁLCULO DE ÁTOMOS TOTALES
    # ======================================================
    def _calculate_total_atoms(self, box_bounds, a0, lattice_type):
        """
        Calcula el número de átomos en un cristal perfecto

        Args:
            box_bounds: Lista de tuplas [(xlo, xhi), (ylo, yhi), (zlo, zhi)]
            a0: Parámetro de red en Angstroms
            lattice_type: Tipo de red ('fcc', 'bcc', 'hcp', 'diamond', 'sc')

        Returns:
            Número de átomos totales en cristal perfecto
        """
        # Calcular volumen de la caja
        lx = box_bounds[0][1] - box_bounds[0][0]
        ly = box_bounds[1][1] - box_bounds[1][0]
        lz = box_bounds[2][1] - box_bounds[2][0]
        volume_box = lx * ly * lz

        # Volumen de celda unitaria (cúbica)
        volume_cell = a0 ** 3

        # Número de átomos por celda según tipo de red
        atoms_per_cell = {
            'fcc': 4,
            'bcc': 2,
            'sc': 1,
            'hcp': 2,  # Aproximación
            'diamond': 8
        }

        n_atoms_cell = atoms_per_cell.get(lattice_type, 4)  # Default: fcc

        # Número de celdas unitarias
        n_cells = volume_box / volume_cell

        # Átomos totales
        total_atoms = int(n_cells * n_atoms_cell)

        return total_atoms

    # ======================================================
    # PASO 3: PREDICCIÓN
    # ======================================================
    def run_step3_prediction(self):
        # Obtener posiciones actuales (filtradas o originales)
        positions, source_desc = self._get_current_positions()

        if positions is None:
            QMessageBox.warning(self, "Error", "Debe cargar un dump primero")
            return

        if not self.model_path:
            QMessageBox.warning(self, "Error", "Debe cargar un modelo primero")
            return

        try:
            # Guardar posiciones y origen para usar en predicción
            self.current_prediction_positions = positions
            self.current_prediction_source = source_desc

            # Calcular total de átomos del cristal perfecto
            box_bounds = self.original_dump_data['box_bounds']
            total_atoms = self._calculate_total_atoms(
                box_bounds,
                self.spin_a0.value(),
                self.combo_lattice.currentText()
            )

            # Crear configuración
            config = ExtractorConfig(
                input_dir=".",
                probe_radius=self.spin_probe.value(),
                total_atoms=total_atoms,
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

        pos_norm, box_size, _ = normalizer.normalize(self.current_prediction_positions)

        features = {}
        if config.compute_grid_features:
            features.update(extractor.grid_features(pos_norm, box_size))
        if config.compute_hull_features:
            features.update(extractor.hull_features(self.current_prediction_positions))
        if config.compute_inertia_features:
            features.update(extractor.inertia_feature(self.current_prediction_positions))
        if config.compute_radial_features:
            features.update(extractor.radial_features(self.current_prediction_positions))
        if config.compute_entropy_features:
            features.update(extractor.entropy_spatial(pos_norm))
        if config.compute_clustering_features:
            features.update(extractor.bandwidth(pos_norm))

        features["num_points"] = len(self.current_prediction_positions)

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
        msg += f"Átomos usados: {len(self.current_prediction_positions)} ({self.current_prediction_source})"

        self.lbl_step3_result.setText(f"✓ Predicción: {prediction:.2f} vacancias (Error: {error:.2f})")

        # Ocultar paso 4 (no aplica sin clustering)
        self.step4_box.setVisible(False)

        QMessageBox.information(self, "Predicción Completada", msg)

    def _predict_with_clusters(self, model, config):
        """Predicción con clustering: predice para cada cluster y suma"""
        import pandas as pd

        normalizer = PositionNormalizer(scale_factor=config.a0)
        extractor = FeatureExtractor(config)

        # Normalizar todas las posiciones una vez para obtener el box_size de referencia
        all_pos_norm, reference_box_size, _ = normalizer.normalize(self.current_prediction_positions)

        # Obtener clusters únicos (excluyendo ruido si existe)
        unique_labels = np.unique(self.clustering_labels)
        clusters_to_process = [lbl for lbl in unique_labels if lbl != -1]

        cluster_predictions = []
        total_prediction = 0.0

        # Predecir para cada cluster
        for cluster_id in clusters_to_process:
            # Extraer posiciones del cluster
            mask = self.clustering_labels == cluster_id
            cluster_positions = self.current_prediction_positions[mask]
            n_atoms_cluster = len(cluster_positions)

            # Saltar clusters muy pequeños (menos de 3 átomos)
            if n_atoms_cluster < 3:
                print(f"⚠️ Cluster {cluster_id} tiene solo {n_atoms_cluster} átomos, se omite.")
                continue

            # Normalizar el cluster con el mismo scale_factor
            pos_norm, _, _ = normalizer.normalize(cluster_positions)

            # IMPORTANTE: Usar el reference_box_size de toda la muestra, no el del cluster individual
            # Esto asegura que las features sean comparables con el entrenamiento
            features = {}
            if config.compute_grid_features:
                features.update(extractor.grid_features(pos_norm, reference_box_size))
            if config.compute_hull_features:
                features.update(extractor.hull_features(cluster_positions))
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

            try:
                cluster_pred = model.predict(X)[0]
                total_prediction += cluster_pred

                cluster_predictions.append({
                    'cluster_id': cluster_id,
                    'n_atoms': n_atoms_cluster,
                    'prediction': cluster_pred,
                    'model_name': Path(self.model_path).name,
                    'features': features.copy(),
                    'positions': cluster_positions.copy()
                })
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"⚠️ Error al predecir cluster {cluster_id} ({n_atoms_cluster} átomos): {str(e)}")
                print(error_details)
                # Continuar con el siguiente cluster
                continue

        # Calcular vacancias reales
        n_atoms_real = len(self.original_dump_data['positions'])
        n_vacancies_real = config.total_atoms - n_atoms_real
        error = abs(total_prediction - n_vacancies_real)

        # Construir mensaje detallado
        msg = f"🎯 Predicción Completada (Con Clustering)\n\n"
        msg += f"📊 Clusters totales: {len(clusters_to_process)}\n"
        msg += f"✓ Clusters procesados: {len(cluster_predictions)}\n"
        if len(cluster_predictions) < len(clusters_to_process):
            omitidos = len(clusters_to_process) - len(cluster_predictions)
            msg += f"⚠️ Clusters omitidos: {omitidos} (muy pequeños o con errores)\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        msg += "Predicción por Cluster:\n"
        for cp in cluster_predictions:
            msg += f"  • Cluster {cp['cluster_id']}: {cp['prediction']:.2f} vacancias ({cp['n_atoms']} átomos)\n"

        msg += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"Vacancias predichas (TOTAL): {total_prediction:.2f}\n"
        msg += f"Vacancias reales: {n_vacancies_real}\n"
        msg += f"Error absoluto: {error:.2f}\n\n"
        msg += f"Átomos en simulación: {n_atoms_real}\n"
        msg += f"Átomos usados: {len(self.current_prediction_positions)} ({self.current_prediction_source})"

        self.lbl_step3_result.setText(f"✓ {len(clusters_to_process)} clusters: {total_prediction:.2f} vacancias (Error: {error:.2f})")

        # Guardar estado para ajuste fino
        self.cluster_predictions = cluster_predictions
        self.current_config = config

        # Mostrar mensaje
        QMessageBox.information(self, "Predicción por Clusters", msg)

        # Activar y poblar paso 4 (ajuste fino)
        if len(cluster_predictions) > 0:
            self._populate_fine_tuning_ui()

    # ======================================================
    # VISUALIZACIÓN DE CLUSTERS INDIVIDUALES
    # ======================================================
    def view_selected_cluster(self):
        """Muestra solo el cluster seleccionado en el visualizador"""
        if self.clustering_labels is None:
            QMessageBox.warning(self, "Error", "No hay clustering aplicado")
            return

        target_cluster = self.spin_cluster_filter.value()

        # Contar átomos en el cluster seleccionado
        mask = self.clustering_labels == target_cluster
        n_atoms = np.sum(mask)

        # Actualizar visualizador con cluster resaltado
        self.visualizer.apply_clustering(self.clustering_labels, target_cluster=target_cluster)

        self.lbl_step2_result.setText(f"✓ Visualizando cluster {target_cluster} ({n_atoms} átomos)")

    def view_all_clusters(self):
        """Muestra todos los clusters en el visualizador"""
        if self.clustering_labels is None:
            QMessageBox.warning(self, "Error", "No hay clustering aplicado")
            return

        # Actualizar visualizador sin filtro
        self.visualizer.apply_clustering(self.clustering_labels, target_cluster=None)

        n_clusters = self.clustering_info['n_clusters']
        self.lbl_step2_result.setText(f"✓ Visualizando todos los clusters ({n_clusters} clusters)")

    # ======================================================
    # AJUSTE FINO POR CLUSTER
    # ======================================================
    def _populate_fine_tuning_ui(self):
        """Pobla la tabla y controles del paso 4 (ajuste fino)"""
        # Mostrar sección
        self.step4_box.setVisible(True)

        # Actualizar lista de modelos disponibles
        self.refresh_model_list()  # Esto ya actualiza combo_models
        
        # Copiar modelos al combo alternativo
        self.combo_alt_model.clear()
        for i in range(self.combo_models.count()):
            model_name = self.combo_models.itemText(i)
            model_path = self.combo_models.itemData(i)
            self.combo_alt_model.addItem(model_name, userData=model_path)

        # Llenar tabla
        self.table_clusters.setRowCount(len(self.cluster_predictions))
        
        for row, cp in enumerate(self.cluster_predictions):
            # Cluster ID
            self.table_clusters.setItem(row, 0, QTableWidgetItem(str(cp['cluster_id'])))
            
            # N átomos
            self.table_clusters.setItem(row, 1, QTableWidgetItem(str(cp['n_atoms'])))
            
            # Predicción
            pred_item = QTableWidgetItem(f"{cp['prediction']:.2f}")
            self.table_clusters.setItem(row, 2, pred_item)
            
            # Modelo usado
            self.table_clusters.setItem(row, 3, QTableWidgetItem(cp['model_name']))
            
            # Botón de acción (solo indicador visual)
            action_item = QTableWidgetItem("🔄")
            self.table_clusters.setItem(row, 4, action_item)

        # Actualizar total
        self._update_total_prediction()

    def _update_total_prediction(self):
        """Recalcula y actualiza el total de vacancias predichas"""
        if not self.cluster_predictions:
            return
        
        total = sum(cp['prediction'] for cp in self.cluster_predictions)
        self.lbl_total_prediction.setText(f"{total:.2f} vacancias")
        
        # También actualizar el label del paso 3
        self.lbl_step3_result.setText(
            f"✓ {len(self.cluster_predictions)} clusters: {total:.2f} vacancias (ajustado)"
        )

    def repredict_selected_cluster(self):
        """Re-predice el cluster seleccionado con el modelo alternativo"""
        # Verificar que hay un cluster seleccionado
        selected_rows = self.table_clusters.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "Error", "Seleccione un cluster de la tabla")
            return
        
        row = self.table_clusters.currentRow()
        if row < 0 or row >= len(self.cluster_predictions):
            QMessageBox.warning(self, "Error", "Cluster no válido")
            return
        
        # Obtener modelo alternativo
        alt_model_path = self.combo_alt_model.currentData()
        if not alt_model_path or alt_model_path == "(No hay modelos disponibles)":
            QMessageBox.warning(self, "Error", "Seleccione un modelo alternativo")
            return
        
        try:
            # Cargar modelo alternativo
            alt_model = joblib.load(alt_model_path)
            alt_model_name = Path(alt_model_path).name
            
            # Obtener cluster data
            cluster_data = self.cluster_predictions[row]
            features = cluster_data['features']
            
            # Re-predecir
            import pandas as pd
            df = pd.DataFrame([features])
            forbidden = [
                "n_vacancies", "n_atoms_surface",
                "vacancies", "file", "num_atoms_real", "num_points"
            ]
            X = df.drop(columns=[c for c in forbidden if c in df.columns], errors="ignore")
            
            new_prediction = alt_model.predict(X)[0]
            
            # Actualizar datos
            old_prediction = cluster_data['prediction']
            cluster_data['prediction'] = new_prediction
            cluster_data['model_name'] = alt_model_name
            
            # Actualizar tabla
            self.table_clusters.item(row, 2).setText(f"{new_prediction:.2f}")
            self.table_clusters.item(row, 3).setText(alt_model_name)
            
            # Actualizar total
            self._update_total_prediction()
            
            # Mensaje de éxito
            diff = new_prediction - old_prediction
            sign = "+" if diff > 0 else ""
            self.lbl_repredict_status.setText(
                f"✓ Cluster {cluster_data['cluster_id']}: {old_prediction:.2f} → {new_prediction:.2f} ({sign}{diff:.2f})"
            )
            self.lbl_repredict_status.setStyleSheet("color: green; font-size: 10px;")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al re-predecir:\n{str(e)}")
            self.lbl_repredict_status.setText(f"✗ Error: {str(e)}")
            self.lbl_repredict_status.setStyleSheet("color: red; font-size: 10px;")
