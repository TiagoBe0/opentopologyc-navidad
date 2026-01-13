#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# opentopologyc/main.py

"""
Punto de entrada principal para OpenTopologyC.
Lanza la interfaz gráfica principal que permite acceder a:
- Extractor de Features
- Entrenamiento de Modelos
"""

from opentopologyc.gui.main_gui import MainGUI


def main():
    """Inicia la aplicación OpenTopologyC"""
    app = MainGUI()
    app.run()


if __name__ == "__main__":
    main()



    