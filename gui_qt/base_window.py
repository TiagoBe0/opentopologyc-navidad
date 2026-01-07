from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Qt


class BaseWindow(QMainWindow):
    def __init__(self, title="OpenTopologyC", size=(900, 700)):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(*size)
        self.setMinimumSize(800, 600)
        self._center()

    def _center(self):
        frame = self.frameGeometry()
        screen = self.screen().availableGeometry().center()
        frame.moveCenter(screen)
        self.move(frame.topLeft())
