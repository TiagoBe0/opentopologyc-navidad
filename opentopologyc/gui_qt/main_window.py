import sys
from pathlib import Path

# Agregar directorio raíz al path si se ejecuta directamente
if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(root_dir))

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QPushButton, QLabel
)
from PySide6.QtCore import Qt

from .base_window import BaseWindow
from .extractor_gui_qt import ExtractorGUIQt
from .train_gui_qt import TrainingGUIQt
from .prediction_gui_qt import PredictionGUIQt


class MainWindow(BaseWindow):
    def __init__(self):
        super().__init__("OpenTopologyC - Suite Completa", (500, 500))
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("OpenTopologyC")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:26px; font-weight:bold;")

        subtitle = QLabel("Suite de Análisis Topológico")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color:gray;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        btn_extractor = QPushButton("🔬 Extractor de Features")
        btn_extractor.clicked.connect(self.open_extractor)

        btn_train = QPushButton("🤖 Entrenamiento")
        btn_train.clicked.connect(self.open_train)

        btn_predict = QPushButton("🎯 Predicción + Visualizador")
        btn_predict.clicked.connect(self.open_prediction)

        for b in (btn_extractor, btn_train, btn_predict):
            b.setMinimumHeight(40)
            layout.addWidget(b)

        self.setCentralWidget(central)

    def open_extractor(self):
        self.child = ExtractorGUIQt()
        self.child.show()

    def open_train(self):
        self.child = TrainingGUIQt()
        self.child.show()

    def open_prediction(self):
        self.child = PredictionGUIQt()
        self.child.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
