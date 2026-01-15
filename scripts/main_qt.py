#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# opentopologyc/main_qt.py

"""
Punto de entrada principal para OpenTopologyC Kit-Tools
========================================================

Versión simplificada enfocada en predicción.

Herramientas incluidas:
- Alpha Shape filtering (filtrado de átomos superficiales)
- Clustering (KMeans, MeanShift, HDBSCAN, Hierarchical)
- Predicción de vacancias con modelos pre-entrenados
- Visualizador 3D interactivo

NOTA: Esta versión NO incluye entrenamiento ni extracción de features.
      Los modelos ML deben ser entrenados previamente en otra herramienta.
"""

# CRÍTICO: Configurar matplotlib ANTES DE CUALQUIER OTRO IMPORT
# Esto debe ser lo PRIMERO para que matplotlib use PySide6
import os
os.environ['QT_API'] = 'pyside6'
os.environ['MPLBACKEND'] = 'QtAgg'

# Forzar matplotlib a usar PySide6 inmediatamente
import matplotlib
matplotlib.use('QtAgg', force=True)

import sys
from pathlib import Path
import warnings

# Suprimir warnings de incompatibilidad Qt
# OVITO y matplotlib ahora usan PySide6
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
warnings.filterwarnings('ignore', message='.*Incompatible version of the Qt.*')
warnings.filterwarnings('ignore', message='.*Matplotlib is using.*')

# Agregar el directorio raíz al PYTHONPATH
root_dir = Path(__file__).parent.parent  # Subir un nivel desde scripts/ a la raíz
sys.path.insert(0, str(root_dir))

from PySide6.QtWidgets import QApplication
from opentopologyc.gui_qt.prediction_gui_qt import PredictionGUIQt
from opentopologyc.core.logger import setup_logger, log_session_start, log_session_end


def main():
    """Inicia la aplicación OpenTopologyC Kit-Tools (Predicción)"""
    # Configurar logging
    logger = setup_logger(
        name="opentopologyc_kittools",
        log_file="opentopologyc_kittools.log",
        console=False  # No duplicar salida en consola
    )

    log_session_start(logger, "Kit-Tools (Predicción)")
    logger.info("Iniciando OpenTopologyC Kit-Tools - Prediction Suite")

    try:
        app = QApplication(sys.argv)

        # Configurar estilo de aplicación
        app.setApplicationName("OpenTopologyC Kit-Tools")
        app.setOrganizationName("OpenTopologyC")

        logger.info("Aplicación Qt configurada")

        # Crear y mostrar GUI de predicción directamente
        window = PredictionGUIQt()
        window.setWindowTitle("OpenTopologyC Kit-Tools - Predicción")
        window.show()

        logger.info("GUI de predicción mostrada")

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
