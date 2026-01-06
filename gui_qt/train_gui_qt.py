#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox,
    QSpinBox, QTextEdit, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from gui_qt.base_window import BaseWindow
from core.training_pipeline import TrainingPipeline


# ======================================================
# THREAD DE ENTRENAMIENTO
# ======================================================
class TrainingWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, csv_path, model_path, n_estimators, max_depth):
        super().__init__()
        self.csv_path = csv_path
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def run(self):
        try:
            self.log.emit("📊 Cargando dataset...")
            pipeline = TrainingPipeline(
                csv_file=self.csv_path,
                model_output=self.model_path,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
            )

            def callback(step, total):
                pct = int((step / total) * 100)
                self.progress.emit(pct)

            pipeline.train(progress_callback=callback)

            self.progress.emit(100)
            self.finished.emit("✅ Entrenamiento finalizado")

        except Exception as e:
            self.error.emit(str(e))


# ======================================================
# GUI PRINCIPAL
# ======================================================
class TrainingGUIQt(BaseWindow):
    def __init__(self):
        super().__init__("OpenTopologyC - Entrenamiento", (700, 600))
        self.csv_path = ""
        self.model_path = ""
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
        btn_csv = QPushButton("📂 Seleccionar CSV de entrenamiento")
        btn_csv.clicked.connect(self.select_csv)
        self.lbl_csv = QLabel("No seleccionado")

        layout.addWidget(btn_csv)
        layout.addWidget(self.lbl_csv)

        # -------- MODELO --------
        btn_model = QPushButton("💾 Seleccionar salida del modelo")
        btn_model.clicked.connect(self.select_model)
        self.lbl_model = QLabel("No seleccionado")

        layout.addWidget(btn_model)
        layout.addWidget(self.lbl_model)

        # -------- PARAMS --------
        params = QGroupBox("Parámetros del modelo")
        p = QVBoxLayout()

        self.spin_estimators = QSpinBox()
        self.spin_estimators.setRange(10, 500)
        self.spin_estimators.setValue(100)

        self.spin_depth = QSpinBox()
        self.spin_depth.setRange(1, 50)
        self.spin_depth.setValue(10)

        p.addWidget(QLabel("n_estimators"))
        p.addWidget(self.spin_estimators)
        p.addWidget(QLabel("max_depth"))
        p.addWidget(self.spin_depth)

        params.setLayout(p)
        layout.addWidget(params)

        # -------- RUN --------
        btn_train = QPushButton("🤖 Entrenar modelo")
        btn_train.setMinimumHeight(40)
        btn_train.clicked.connect(self.start_training)
        layout.addWidget(btn_train)

        # -------- PROGRESS --------
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # -------- LOG --------
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Log de entrenamiento...")
        layout.addWidget(self.log)

        self.setCentralWidget(central)

    # ======================================================
    # FILES
    # ======================================================
    def select_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar CSV", "", "CSV (*.csv)"
        )
        if path:
            self.csv_path = path
            self.lbl_csv.setText(path)

    def select_model(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar modelo", "", "Model (*.joblib *.pkl)"
        )
        if path:
            self.model_path = path
            self.lbl_model.setText(path)

    # ======================================================
    # TRAIN
    # ======================================================
    def start_training(self):
        if not self.csv_path or not self.model_path:
            QMessageBox.warning(self, "Error", "Faltan rutas")
            return

        self.log.clear()
        self.progress.setValue(0)

        self.worker = TrainingWorker(
            self.csv_path,
            self.model_path,
            self.spin_estimators.value(),
            self.spin_depth.value(),
        )

        self.worker.log.connect(self.log.append)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.worker.start()

    def on_finished(self, msg):
        self.log.append(msg)
        QMessageBox.information(self, "Listo", msg)

    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.log.append("❌ " + msg)
