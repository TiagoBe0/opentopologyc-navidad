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

# CRÍTICO: Configurar matplotlib ANTES DE CUALQUIER OTRO IMPORT
# Esto debe ser lo PRIMERO para que matplotlib use PyQt5 en lugar de PySide6
import os
os.environ['QT_API'] = 'pyqt5'
os.environ['MPLBACKEND'] = 'Qt5Agg'

# Forzar matplotlib a usar PyQt5 inmediatamente
import matplotlib
matplotlib.use('Qt5Agg', force=True)

import sys
from pathlib import Path
import warnings

# Suprimir warnings de incompatibilidad Qt5/Qt6 de OVITO
# OVITO usa PySide6 pero esta aplicación usa PyQt5
# Matplotlib ahora usa PyQt5 gracias a la configuración arriba
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
warnings.filterwarnings('ignore', message='.*Incompatible version of the Qt.*')
warnings.filterwarnings('ignore', message='.*Matplotlib is using.*')

# Agregar el directorio raíz al PYTHONPATH
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from PyQt5.QtWidgets import QApplication
from gui_qt.main_window import MainWindow
from core.logger import setup_logger, log_session_start, log_session_end


def main():
    """Inicia la aplicación OpenTopologyC Qt"""
    # Configurar logging
    logger = setup_logger(
        name="opentopologyc",
        log_file="opentopologyc.log",
        console=False  # No duplicar salida en consola
    )

    log_session_start(logger, "GUI Qt")
    logger.info("Iniciando aplicación OpenTopologyC Qt")

    try:
        app = QApplication(sys.argv)

        # Configurar estilo de aplicación
        app.setApplicationName("OpenTopologyC")
        app.setOrganizationName("OpenTopologyC")

        logger.info("Aplicación Qt configurada")

        # Crear y mostrar ventana principal
        window = MainWindow()
        window.show()

        logger.info("Ventana principal mostrada")

        # Ejecutar aplicación
        exit_code = app.exec_()

        logger.info(f"Aplicación cerrada con código: {exit_code}")
        log_session_end(logger)

        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Error en aplicación: {str(e)}", exc_info=True)
        log_session_end(logger)
        raise


if __name__ == "__main__":
    main()
