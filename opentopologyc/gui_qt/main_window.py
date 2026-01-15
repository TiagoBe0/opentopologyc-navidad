#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenTopologyC Kit-Tools - Main Window
======================================

Aplicación simplificada que solo incluye herramientas de predicción:
- Alpha Shape filtering
- Clustering (KMeans, MeanShift, HDBSCAN, Hierarchical)
- Predicción con modelos pre-entrenados

Nota: Esta versión NO incluye entrenamiento ni extracción de features.
Los modelos deben ser entrenados previamente.
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path si se ejecuta directamente
if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(root_dir))

from PySide6.QtWidgets import QApplication
from .prediction_gui_qt import PredictionGUIQt


def main():
    """Launch the prediction-only GUI"""
    app = QApplication(sys.argv)

    # Launch prediction GUI directly
    window = PredictionGUIQt()
    window.setWindowTitle("OpenTopologyC Kit-Tools - Predicción")
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
