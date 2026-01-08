#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenTopologyC - Punto de entrada principal

Ejecuta la GUI principal que permite elegir entre:
- Extractor de Features
- Entrenamiento de Modelos
"""

from gui.main_gui import MainGUI


def main():
    """Inicia la aplicación GUI principal."""
    gui = MainGUI()
    gui.run()


if __name__ == "__main__":
    main()
