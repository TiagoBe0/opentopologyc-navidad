#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenTopologyC Kit-Tools - Launcher
===================================

Punto de entrada principal para OpenTopologyC Kit-Tools.

Esta versión simplificada incluye solo herramientas de predicción:
- Alpha Shape filtering
- Clustering (múltiples algoritmos)
- Predicción con modelos ML pre-entrenados
- Visualizador 3D interactivo

Para ejecutar:
    python main.py

O directamente:
    python scripts/main_qt.py
"""

import sys
from pathlib import Path

# Agregar raíz al path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

# Importar y ejecutar el launcher principal
from scripts.main_qt import main

if __name__ == "__main__":
    main()
