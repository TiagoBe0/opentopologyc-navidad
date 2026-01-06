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

import numpy as np
import matplotlib
matplotlib.use("QtAgg")

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox,
    QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm

from core.alpha_shape_filter import LAMMPSDumpParser


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

        # Parámetros visuales
        self.atom_size = 20
        self.alpha = 0.85
        self.show_axes = True
        self.show_grid = True

        self._build_ui()

    # ======================================================
    # UI
    # ======================================================
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---------- CONTROLES ----------
        controls = QGroupBox("Visualización")
        cl = QHBoxLayout()

        btn_load = QPushButton("📂 Cargar dump")
        btn_load.clicked.connect(self.load_dump_dialog)
        cl.addWidget(btn_load)

        chk_axes = QCheckBox("Ejes")
        chk_axes.setChecked(True)
        chk_axes.stateChanged.connect(self.toggle_axes)
        cl.addWidget(chk_axes)

        chk_grid = QCheckBox("Grid")
        chk_grid.setChecked(True)
        chk_grid.stateChanged.connect(self.toggle_grid)
        cl.addWidget(chk_grid)

        controls.setLayout(cl)
        layout.addWidget(controls)

        # ---------- SLIDERS ----------
        sliders = QHBoxLayout()

        sliders.addWidget(QLabel("Tamaño"))
        size_slider = QSlider(Qt.Horizontal)
        size_slider.setRange(5, 60)
        size_slider.setValue(self.atom_size)
        size_slider.valueChanged.connect(self.set_atom_size)
        sliders.addWidget(size_slider)

        sliders.addWidget(QLabel("Alpha"))
        alpha_slider = QSlider(Qt.Horizontal)
        alpha_slider.setRange(10, 100)
        alpha_slider.setValue(int(self.alpha * 100))
        alpha_slider.valueChanged.connect(self.set_alpha)
        sliders.addWidget(alpha_slider)

        layout.addLayout(sliders)

        # ---------- FIGURA ----------
        self.fig = Figure(figsize=(7, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")

        layout.addWidget(self.canvas)

    # ======================================================
    # LOADERS
    # ======================================================
    def load_dump_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar dump", "", "Dump (*.dump)"
        )
        if path:
            self.load_dump_from_path(path)

    def load_dump_from_path(self, path):
        data = LAMMPSDumpParser.read(path)
        self.positions = data["positions"]
        self.colors = None
        self.cluster_labels = None
        self.highlight_cluster = None
        self.plot()

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

        scatter_kwargs = {
            "s": self.atom_size,
            "alpha": self.alpha,
            "edgecolors": "black",
            "linewidth": 0.25,
        }

        if self.colors is not None:
            self.ax.scatter(
                self.positions[:, 0],
                self.positions[:, 1],
                self.positions[:, 2],
                c=self.colors,
                **scatter_kwargs
            )
        else:
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
