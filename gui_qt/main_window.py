import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QPushButton, QLabel
)
from PyQt5.QtCore import Qt

from gui_qt.base_window import BaseWindow
from gui_qt.extractor_gui_qt import ExtractorGUIQt
from gui_qt.train_gui_qt import TrainingGUIQt
from gui_qt.prediction_gui_qt import PredictionGUIQt


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
