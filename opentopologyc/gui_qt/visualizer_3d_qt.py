#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizador 3D de átomos en Qt + Matplotlib

✔ Qt puro (sin Tkinter)
✔ Backend QtAgg
✔ Coloreado por:
    - Clustering (labels)
    - Probabilidad de vacancia (fallback)
✔ Highlight de cluster objetivo
✔ Control de tamaño, alpha, ejes y grilla
"""

# CRÍTICO: Configurar QT_API ANTES de importar matplotlib
import os
os.environ['QT_API'] = 'pyside6'

import numpy as np
import matplotlib
# IMPORTANTE: Usar backend QtAgg para PySide6
matplotlib.use("QtAgg", force=True)

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox,
    QFileDialog, QGroupBox, QComboBox
)
from PySide6.QtCore import Qt

# Importar específicamente desde backend QtAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm

from ..core.alpha_shape_filter import LAMMPSDumpParser


class AtomVisualizer3DQt(QWidget):
    # ======================================================
    # INIT
    # ======================================================
    def __init__(self, parent=None):
        super().__init__(parent)

        # Datos
        self.positions = None
        self.colors = None
        self.cluster_labels = None
        self.highlight_cluster = None

        # Etapas múltiples (para visualizar pipeline)
        self.stages = {}  # {'key': {'name':str, 'positions':ndarray, 'labels':ndarray, 'info':dict}}
        self.current_stage = None

        # Parámetros visuales
        self.atom_size = 20
        self.alpha = 0.85
        self.show_axes = True
        self.show_grid = True

        # Control de zoom
        self.original_limits = None
        self.zoom_factor = 1.0
        self.original_ticks = None  # Guardar ticks originales para zoom

        self._build_ui()

    # ======================================================
    # UI
    # ======================================================
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Sin márgenes externos
        layout.setSpacing(2)  # Espacio mínimo entre widgets

        # ---------- CONTROLES COMPACTOS ----------
        controls = QGroupBox("Visualización")
        controls.setMaximumHeight(80)  # Altura máxima reducida
        cl = QVBoxLayout()
        cl.setContentsMargins(5, 5, 5, 5)  # Padding reducido
        cl.setSpacing(3)  # Espacio mínimo

        # Primera fila: Cargar dump, checkboxes y zoom
        first_row = QHBoxLayout()
        btn_load = QPushButton("📂")
        btn_load.setMaximumWidth(40)
        btn_load.setToolTip("Cargar dump")
        btn_load.clicked.connect(self.load_dump_dialog)
        first_row.addWidget(btn_load)

        chk_axes = QCheckBox("Ejes")
        chk_axes.setChecked(True)
        chk_axes.stateChanged.connect(self.toggle_axes)
        first_row.addWidget(chk_axes)

        chk_grid = QCheckBox("Grid")
        chk_grid.setChecked(True)
        chk_grid.stateChanged.connect(self.toggle_grid)
        first_row.addWidget(chk_grid)

        # Botones de zoom
        btn_zoom_in = QPushButton("🔍+")
        btn_zoom_in.setMaximumWidth(40)
        btn_zoom_in.clicked.connect(self.zoom_in)
        first_row.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("🔍-")
        btn_zoom_out.setMaximumWidth(40)
        btn_zoom_out.clicked.connect(self.zoom_out)
        first_row.addWidget(btn_zoom_out)

        btn_reset_view = QPushButton("🔄")
        btn_reset_view.setMaximumWidth(40)
        btn_reset_view.setToolTip("Resetear vista")
        btn_reset_view.clicked.connect(self.reset_view)
        first_row.addWidget(btn_reset_view)

        # Sliders en la misma fila
        first_row.addWidget(QLabel("Tam"))
        size_slider = QSlider(Qt.Horizontal)
        size_slider.setRange(5, 60)
        size_slider.setValue(self.atom_size)
        size_slider.setMaximumWidth(80)
        size_slider.valueChanged.connect(self.set_atom_size)
        first_row.addWidget(size_slider)

        first_row.addWidget(QLabel("α"))
        alpha_slider = QSlider(Qt.Horizontal)
        alpha_slider.setRange(10, 100)
        alpha_slider.setValue(int(self.alpha * 100))
        alpha_slider.setMaximumWidth(80)
        alpha_slider.valueChanged.connect(self.set_alpha)
        first_row.addWidget(alpha_slider)

        first_row.addStretch()
        cl.addLayout(first_row)

        # Segunda fila: Selector de etapas (inicialmente oculto)
        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Etapa:"))
        self.stage_combo = QComboBox()
        self.stage_combo.currentIndexChanged.connect(self.change_stage)
        stage_row.addWidget(self.stage_combo)
        self.stage_label = QLabel("")
        stage_row.addWidget(self.stage_label)
        stage_row.addStretch()
        cl.addLayout(stage_row)

        controls.setLayout(cl)
        layout.addWidget(controls)

        # ---------- FIGURA (MÁS ESPACIO) ----------
        self.fig = Figure(figsize=(8, 7))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")

        layout.addWidget(self.canvas)

    # ======================================================
    # LOADERS
    # ======================================================
    def load_dump_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar dump", "", "Dump files (*.dump);;All files (*)"
        )
        if path:
            self.load_dump_from_path(path)

    def load_dump_from_path(self, path):
        data = LAMMPSDumpParser.read(path)
        self.positions = data["positions"]
        self.colors = None
        self.cluster_labels = None
        self.highlight_cluster = None
        self.original_limits = None  # Resetear límites para el nuevo dump
        self.original_ticks = None  # Resetear ticks para el nuevo dump
        self.plot()

    # ======================================================
    # STAGES (ETAPAS MÚLTIPLES)
    # ======================================================
    def load_stages(self, stages_dict):
        """
        Carga múltiples etapas del pipeline

        Args:
            stages_dict: Dict con etapas {'key': {'name', 'positions', 'labels', 'info'}}
        """
        self.stages = stages_dict
        self.stage_combo.clear()

        if not stages_dict:
            return

        # Agregar etapas al combo (ordenadas por key)
        sorted_keys = sorted(stages_dict.keys())
        for key in sorted_keys:
            stage = stages_dict[key]
            self.stage_combo.addItem(stage['name'], key)

        # Seleccionar última etapa por defecto
        self.stage_combo.setCurrentIndex(len(sorted_keys) - 1)

    def change_stage(self, index):
        """Cambia la etapa visualizada"""
        if index < 0 or not self.stages:
            return

        key = self.stage_combo.currentData()
        if key not in self.stages:
            return

        self.current_stage = key
        stage = self.stages[key]

        # Actualizar datos
        self.positions = stage['positions']
        self.cluster_labels = stage.get('labels')

        # Actualizar colores según labels
        if self.cluster_labels is not None:
            self.apply_clustering(self.cluster_labels)
        else:
            self.colors = None
            self.plot()

        # Actualizar label de información
        info_str = " | ".join([f"{k}: {v}" for k, v in stage.get('info', {}).items()])
        self.stage_label.setText(info_str)

    # ======================================================
    # CLUSTERING VISUAL
    # ======================================================
    def apply_clustering(self, labels, target_cluster=None):
        """
        Colorea los puntos según cluster y resalta el cluster objetivo
        """
        self.cluster_labels = labels
        self.highlight_cluster = target_cluster

        labels = np.asarray(labels)
        unique = np.unique(labels)

        colors = np.zeros((len(labels), 4))
        cmap = cm.get_cmap("tab10", len(unique))

        for i, lbl in enumerate(unique):
            if lbl == -1:
                colors[labels == lbl] = [0.6, 0.6, 0.6, 0.2]  # ruido
            else:
                colors[labels == lbl] = cmap(i)

        # Highlight cluster objetivo
        if target_cluster is not None:
            mask = labels == target_cluster
            colors[~mask, 3] = 0.15
            colors[mask, 3] = 1.0

        self.colors = colors
        self.plot()

    # ======================================================
    # PLOT
    # ======================================================
    def plot(self):
        if self.positions is None:
            return

        self.ax.clear()

        # Base kwargs para scatter
        scatter_kwargs = {
            "s": self.atom_size,
            "edgecolors": "black",
            "linewidth": 0.25,
        }

        # IMPORTANTE: Solo usar alpha global si NO hay un cluster resaltado
        # Cuando hay highlight_cluster, los colores ya tienen alphas individuales
        if self.highlight_cluster is None:
            scatter_kwargs["alpha"] = self.alpha

        if self.colors is not None:
            self.ax.scatter(
                self.positions[:, 0],
                self.positions[:, 1],
                self.positions[:, 2],
                c=self.colors,
                **scatter_kwargs
            )
        else:
            # Sin colores custom, usar color steelblue
            # Alpha ya está en scatter_kwargs si no hay highlight_cluster
            self.ax.scatter(
                self.positions[:, 0],
                self.positions[:, 1],
                self.positions[:, 2],
                color="steelblue",
                **scatter_kwargs
            )

        self.ax.set_xlabel("X (Å)")
        self.ax.set_ylabel("Y (Å)")
        self.ax.set_zlabel("Z (Å)")
        self.ax.grid(self.show_grid)

        if not self.show_axes:
            self.ax.set_axis_off()

        # Guardar límites y ticks originales la primera vez
        if self.original_limits is None:
            self.original_limits = {
                'xlim': self.ax.get_xlim(),
                'ylim': self.ax.get_ylim(),
                'zlim': self.ax.get_zlim()
            }
            # Guardar ticks originales para mantenerlos fijos durante zoom
            self.original_ticks = {
                'xticks': self.ax.get_xticks().copy(),
                'yticks': self.ax.get_yticks().copy(),
                'zticks': self.ax.get_zticks().copy()
            }

        self.canvas.draw_idle()

    # ======================================================
    # CALLBACKS
    # ======================================================
    def set_atom_size(self, v):
        self.atom_size = v
        self.plot()

    def set_alpha(self, v):
        self.alpha = v / 100
        self.plot()

    def toggle_axes(self, state):
        self.show_axes = state == Qt.Checked
        self.plot()

    def toggle_grid(self, state):
        self.show_grid = state == Qt.Checked
        self.plot()

    # ======================================================
    # ZOOM
    # ======================================================
    def zoom_in(self):
        """Acercar la vista (zoom in) en un 40%"""
        if self.positions is None or self.original_limits is None:
            return

        # Obtener límites actuales
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()

        # Calcular centros
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2

        # Reducir rangos en 40% (más agresivo que el 20% anterior)
        x_range = (xlim[1] - xlim[0]) * 0.6
        y_range = (ylim[1] - ylim[0]) * 0.6
        z_range = (zlim[1] - zlim[0]) * 0.6

        # Aplicar nuevos límites
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        self.ax.set_zlim(z_center - z_range/2, z_center + z_range/2)

        # Mantener ticks originales fijos
        if self.original_ticks is not None:
            self.ax.set_xticks(self.original_ticks['xticks'])
            self.ax.set_yticks(self.original_ticks['yticks'])
            self.ax.set_zticks(self.original_ticks['zticks'])

        self.canvas.draw_idle()

    def zoom_out(self):
        """Alejar la vista (zoom out) en un 40%"""
        if self.positions is None or self.original_limits is None:
            return

        # Obtener límites actuales
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()

        # Calcular centros
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2

        # Aumentar rangos en 67% (inverso del 40% de zoom in)
        x_range = (xlim[1] - xlim[0]) * 1.67
        y_range = (ylim[1] - ylim[0]) * 1.67
        z_range = (zlim[1] - zlim[0]) * 1.67

        # Aplicar nuevos límites
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        self.ax.set_zlim(z_center - z_range/2, z_center + z_range/2)

        # Mantener ticks originales fijos
        if self.original_ticks is not None:
            self.ax.set_xticks(self.original_ticks['xticks'])
            self.ax.set_yticks(self.original_ticks['yticks'])
            self.ax.set_zticks(self.original_ticks['zticks'])

        self.canvas.draw_idle()

    def reset_view(self):
        """Resetear la vista a los límites originales"""
        if self.positions is None or self.original_limits is None:
            return

        self.ax.set_xlim(self.original_limits['xlim'])
        self.ax.set_ylim(self.original_limits['ylim'])
        self.ax.set_zlim(self.original_limits['zlim'])

        # Restaurar ticks originales
        if self.original_ticks is not None:
            self.ax.set_xticks(self.original_ticks['xticks'])
            self.ax.set_yticks(self.original_ticks['yticks'])
            self.ax.set_zticks(self.original_ticks['zticks'])

        self.canvas.draw_idle()
