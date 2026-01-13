#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox,
    QSpinBox, QTextEdit, QProgressBar, QLineEdit,
    QComboBox, QDialog, QScrollArea, QHBoxLayout
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap

from .base_window import BaseWindow
from ..core.training_pipeline import TrainingPipeline


# ======================================================
# THREAD DE ENTRENAMIENTO
# ======================================================
class TrainingWorker(QThread):
    log = Signal(str)
    progress = Signal(int)
    finished = Signal(dict)  # Cambiar a dict para pasar resultados
    error = Signal(str)

    def __init__(self, csv_path, model_path, n_estimators, max_depth, target_column=None, task_type="regression"):
        super().__init__()
        self.csv_path = csv_path
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.target_column = target_column
        self.task_type = task_type

    def run(self):
        try:
            self.log.emit("📊 Cargando dataset...")
            pipeline = TrainingPipeline(
                csv_file=self.csv_path,
                model_output=self.model_path,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                target_column=self.target_column,
                task_type=self.task_type
            )

            def callback(step, total):
                pct = int((step / total) * 100)
                self.progress.emit(pct)

            # Ejecutar según tipo de tarea
            if self.task_type == "regression":
                results = pipeline.train_regression(progress_callback=callback)
            else:
                results = pipeline.train(progress_callback=callback)

            self.progress.emit(100)
            self.finished.emit(results)

        except Exception as e:
            import traceback
            traceback.print_exc()
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

        # Tipo de tarea
        p.addWidget(QLabel("Tipo de tarea:"))
        self.combo_task = QComboBox()
        self.combo_task.addItems(["Regresión", "Clasificación"])
        self.combo_task.setCurrentIndex(0)  # Default: Regresión
        p.addWidget(self.combo_task)

        p.addWidget(QLabel("n_estimators"))
        self.spin_estimators = QSpinBox()
        self.spin_estimators.setRange(10, 500)
        self.spin_estimators.setValue(100)
        p.addWidget(self.spin_estimators)

        p.addWidget(QLabel("max_depth"))
        self.spin_depth = QSpinBox()
        self.spin_depth.setRange(1, 50)
        self.spin_depth.setValue(10)
        p.addWidget(self.spin_depth)

        # Target column (opcional)
        p.addWidget(QLabel("Columna target (dejar vacío para auto-detectar)"))
        self.txt_target = QLineEdit()
        self.txt_target.setPlaceholderText("(auto-detectar)")
        p.addWidget(self.txt_target)

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

        # Obtener columna target (None si está vacío)
        target_col = self.txt_target.text().strip() or None

        # Obtener tipo de tarea
        task_type = "regression" if self.combo_task.currentText() == "Regresión" else "classification"

        self.worker = TrainingWorker(
            self.csv_path,
            self.model_path,
            self.spin_estimators.value(),
            self.spin_depth.value(),
            target_col,
            task_type
        )

        self.worker.log.connect(self.log.append)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.worker.start()

    def on_finished(self, results):
        """Maneja los resultados del entrenamiento"""
        self.log.append("\n" + "="*60)
        self.log.append("✅ ENTRENAMIENTO COMPLETADO")
        self.log.append("="*60)

        # Mostrar métricas según tipo de tarea
        if 'r2_test' in results:
            # Regresión
            msg = f"🎯 Resultados de Regresión:\n\n"
            msg += f"R² Score: {results['r2_test']:.4f}\n"
            msg += f"RMSE: {results['rmse_test']:.4f}\n"
            msg += f"MAE: {results['mae_test']:.4f}\n\n"
            msg += f"📁 Archivos generados:\n"
            msg += f"• Modelo: {results['model_path']}\n"
            msg += f"• Gráficos: {results['plot_path']}\n"
            msg += f"• Métricas: {results['metrics_path']}\n"
            msg += f"• Feature Importance: {results['importance_path']}"

            self.log.append(f"\n📊 R² Score: {results['r2_test']:.4f}")
            self.log.append(f"📊 RMSE: {results['rmse_test']:.4f}")
            self.log.append(f"📊 MAE: {results['mae_test']:.4f}")
        else:
            # Clasificación
            msg = f"🎯 Resultados de Clasificación:\n\n"
            msg += f"Accuracy: {results['accuracy']:.4f}\n\n"
            msg += f"📁 Modelo guardado:\n{results['model_path']}"

            self.log.append(f"\n📊 Accuracy: {results['accuracy']:.4f}")

        self.log.append("\n" + "="*60)

        QMessageBox.information(self, "Entrenamiento Completado", msg)

        # Si es regresión, mostrar automáticamente el gráfico de resultados
        if 'plot_path' in results:
            self.show_results_plot(results['plot_path'])

    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.log.append("❌ " + msg)

    def show_results_plot(self, plot_path):
        """
        Muestra el gráfico de resultados en un diálogo

        Args:
            plot_path: Ruta al archivo PNG de resultados
        """
        from pathlib import Path

        # Verificar que el archivo existe
        if not Path(plot_path).exists():
            QMessageBox.warning(self, "Error", f"No se encontró el gráfico: {plot_path}")
            return

        # Crear diálogo
        dialog = QDialog(self)
        dialog.setWindowTitle("📊 Resultados del Entrenamiento")
        dialog.resize(1400, 1000)

        layout = QVBoxLayout(dialog)

        # Crear área de scroll por si la imagen es muy grande
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Label para mostrar la imagen
        image_label = QLabel()
        pixmap = QPixmap(plot_path)

        # Escalar la imagen si es necesario (mantener aspect ratio)
        if pixmap.width() > 1350 or pixmap.height() > 900:
            pixmap = pixmap.scaled(1350, 900, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        scroll_area.setWidget(image_label)
        layout.addWidget(scroll_area)

        # Botones
        btn_layout = QHBoxLayout()

        btn_save_as = QPushButton("💾 Guardar como...")
        btn_save_as.clicked.connect(lambda: self.save_plot_as(plot_path))
        btn_layout.addWidget(btn_save_as)

        btn_close = QPushButton("✓ Cerrar")
        btn_close.clicked.connect(dialog.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

        # Mostrar diálogo
        dialog.exec()

    def save_plot_as(self, original_path):
        """Permite guardar el gráfico en otra ubicación"""
        from pathlib import Path

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar gráfico como",
            str(Path.home() / "evaluacion_modelo.png"),
            "PNG Image (*.png);;All Files (*)"
        )

        if save_path:
            import shutil
            shutil.copy(original_path, save_path)
            QMessageBox.information(self, "Guardado", f"Gráfico guardado en:\n{save_path}")
