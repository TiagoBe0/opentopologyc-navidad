#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnóstico y reparación para el error de FigureCanvasQTAgg

Este script diagnostica y corrige el problema:
TypeError: addWidget(...): argument 1 has unexpected type 'FigureCanvasQTAgg'
"""

import sys
import os

print("=" * 80)
print("DIAGNÓSTICO: Error FigureCanvasQTAgg + PyQt5")
print("=" * 80)

# Paso 1: Verificar instalaciones
print("\n1. Verificando instalaciones...")
print("-" * 80)

try:
    import PyQt5.QtCore
    print(f"✓ PyQt5 versión: {PyQt5.QtCore.PYQT_VERSION_STR}")
    pyqt5_version = PyQt5.QtCore.PYQT_VERSION_STR
    pyqt5_ok = True
except ImportError as e:
    print(f"✗ PyQt5 no disponible: {e}")
    pyqt5_ok = False

try:
    import matplotlib
    print(f"✓ Matplotlib versión: {matplotlib.__version__}")
    mpl_ok = True
except ImportError as e:
    print(f"✗ Matplotlib no disponible: {e}")
    mpl_ok = False

# Paso 2: Verificar backend de matplotlib
print("\n2. Verificando backend de matplotlib...")
print("-" * 80)

if mpl_ok:
    import matplotlib
    backend = matplotlib.get_backend()
    print(f"Backend actual: {backend}")

    # Intentar forzar Qt5Agg
    try:
        matplotlib.use("Qt5Agg", force=True)
        print(f"✓ Backend forzado a: {matplotlib.get_backend()}")
    except Exception as e:
        print(f"✗ Error al forzar backend: {e}")

# Paso 3: Verificar FigureCanvasQTAgg
print("\n3. Verificando FigureCanvasQTAgg...")
print("-" * 80)

canvas_ok = False
canvas_is_qwidget = False

if pyqt5_ok and mpl_ok:
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PyQt5.QtWidgets import QWidget

        print("✓ FigureCanvasQTAgg importado correctamente")

        # Crear una instancia de prueba
        fig = Figure()
        canvas = FigureCanvasQTAgg(fig)

        print(f"Tipo de canvas: {type(canvas)}")
        print(f"MRO de canvas: {type(canvas).__mro__}")

        # Verificar si es QWidget
        if isinstance(canvas, QWidget):
            print("✓ FigureCanvasQTAgg ES un QWidget")
            canvas_is_qwidget = True
            canvas_ok = True
        else:
            print("✗ FigureCanvasQTAgg NO es un QWidget")
            print(f"  Hereda de: {[c.__name__ for c in type(canvas).__mro__]}")
            canvas_ok = False

    except Exception as e:
        print(f"✗ Error al importar FigureCanvasQTAgg: {e}")
        import traceback
        traceback.print_exc()

# Paso 4: Diagnóstico
print("\n4. Diagnóstico...")
print("-" * 80)

if canvas_ok and canvas_is_qwidget:
    print("\n✓ TODO ESTÁ CORRECTO")
    print("El problema puede ser específico de tu entorno.")
    print("\nPosibles causas:")
    print("1. Variables de entorno conflictivas")
    print("2. Importaciones en orden incorrecto")
    print("3. Caché de Python corrupto")
    print("\nSoluciones sugeridas:")
    print("  python -m pip cache purge")
    print("  python -m pip install --force-reinstall --no-cache-dir matplotlib")
    sys.exit(0)

if not pyqt5_ok:
    print("\n✗ PROBLEMA: PyQt5 no está instalado")
    print("\nSolución:")
    print("  pip install PyQt5")
    sys.exit(1)

if not mpl_ok:
    print("\n✗ PROBLEMA: Matplotlib no está instalado")
    print("\nSolución:")
    print("  pip install matplotlib")
    sys.exit(1)

if not canvas_is_qwidget:
    print("\n✗ PROBLEMA DETECTADO: FigureCanvasQTAgg no es un QWidget")
    print("\nCausas posibles:")
    print("  1. Matplotlib compilado contra PySide6 en lugar de PyQt5")
    print("  2. Conflicto de backends Qt")
    print("  3. Instalación corrupta de matplotlib")

    print("\n" + "=" * 80)
    print("APLICANDO SOLUCIÓN AUTOMÁTICA")
    print("=" * 80)

    print("\n1. Desinstalando paquetes conflictivos...")
    import subprocess

    # Desinstalar PySide6
    for pkg in ["PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6"]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],
                capture_output=True,
                text=True
            )
            if "Successfully uninstalled" in result.stdout:
                print(f"  ✓ Desinstalado {pkg}")
        except Exception as e:
            print(f"  ⚠ Error desinstalando {pkg}: {e}")

    print("\n2. Limpiando caché de pip...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "cache", "purge"],
            capture_output=True
        )
        print("  ✓ Caché limpiado")
    except:
        print("  ⚠ No se pudo limpiar caché")

    print("\n3. Reinstalando matplotlib con PyQt5...")
    try:
        # Desinstalar matplotlib
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "matplotlib", "-y"],
            capture_output=True
        )
        print("  ✓ Matplotlib desinstalado")

        # Reinstalar matplotlib
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "matplotlib"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✓ Matplotlib reinstalado")
        else:
            print(f"  ✗ Error reinstalando: {result.stderr}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 80)
    print("REPARACIÓN COMPLETADA")
    print("=" * 80)
    print("\nPor favor, REINICIA Python y vuelve a ejecutar:")
    print("  python main_qt.py")
    print("\nSi el problema persiste, ejecuta:")
    print("  python scripts/fix_qt_backend.py")

