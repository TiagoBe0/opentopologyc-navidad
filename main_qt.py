#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# opentopologyc/main_qt.py

"""
Punto de entrada principal para OpenTopologyC (versión Qt)
Lanza la interfaz gráfica Qt que permite acceder a:
- Extractor de Features
- Entrenamiento de Modelos
- Predicción de Vacancias con Visualizador 3D
"""

import sys
from pathlib import Path
import warnings

# Suprimir warnings de incompatibilidad Qt5/Qt6 de OVITO
# OVITO usa Qt6 pero esta aplicación usa PyQt5
# Ambos pueden coexistir aunque haya warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
warnings.filterwarnings('ignore', message='.*Incompatible version of the Qt.*')

# Agregar el directorio raíz al PYTHONPATH
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from PyQt5.QtWidgets import QApplication
from gui_qt.main_window import MainWindow


def main():
    """Inicia la aplicación OpenTopologyC Qt"""
    app = QApplication(sys.argv)

    # Configurar estilo de aplicación
    app.setApplicationName("OpenTopologyC")
    app.setOrganizationName("OpenTopologyC")

    # Crear y mostrar ventana principal
    window = MainWindow()
    window.show()

    # Ejecutar aplicación
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
