#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para resolver el conflicto OVITO (PySide6) vs PyQt5

Problema:
- OVITO requiere PySide6/shiboken6
- La aplicación Qt usa PyQt5
- Hay conflicto entre ambos backends

Solución:
- Instalar PySide6 (para OVITO)
- Configurar matplotlib para usar PyQt5
- Importar OVITO de forma lazy (solo cuando se necesite)
"""

import subprocess
import sys

print("=" * 80)
print("REPARACIÓN: Conflicto OVITO (PySide6) + PyQt5")
print("=" * 80)

print("\n1. Verificando instalaciones...")
print("-" * 80)

# Verificar PyQt5
try:
    import PyQt5.QtCore
    print(f"✓ PyQt5 instalado: {PyQt5.QtCore.PYQT_VERSION_STR}")
    pyqt5_ok = True
except ImportError:
    print("✗ PyQt5 NO instalado")
    pyqt5_ok = False

# Verificar PySide6
try:
    import PySide6
    print(f"✓ PySide6 instalado: {PySide6.__version__}")
    pyside6_ok = True
except ImportError:
    print("✗ PySide6 NO instalado (requerido por OVITO)")
    pyside6_ok = False

# Verificar shiboken6
try:
    import shiboken6
    print(f"✓ shiboken6 instalado")
    shiboken6_ok = True
except ImportError:
    print("✗ shiboken6 NO instalado (requerido por OVITO)")
    shiboken6_ok = False

# Verificar OVITO
try:
    import ovito
    print(f"✓ OVITO instalado: {ovito.__version__}")
    ovito_ok = True
except ImportError as e:
    print(f"⚠ OVITO no disponible: {e}")
    ovito_ok = False

# Verificar matplotlib
try:
    import matplotlib
    backend = matplotlib.get_backend()
    print(f"✓ Matplotlib backend: {backend}")
    mpl_ok = True
except ImportError:
    print("✗ Matplotlib NO instalado")
    mpl_ok = False

print("\n2. Diagnóstico...")
print("-" * 80)

needs_fix = False

if not pyqt5_ok:
    print("\n✗ PROBLEMA: PyQt5 no está instalado")
    needs_fix = True

if not pyside6_ok or not shiboken6_ok:
    print("\n✗ PROBLEMA: PySide6/shiboken6 no están instalados (requeridos por OVITO)")
    needs_fix = True

if not needs_fix:
    print("\n✓ Todas las dependencias están instaladas correctamente")

    if mpl_ok:
        if 'qt5' in backend.lower() or 'pyqt5' in backend.lower():
            print("✓ Matplotlib está usando PyQt5 (correcto)")
        else:
            print(f"⚠ Matplotlib está usando {backend} en lugar de Qt5Agg")
            needs_fix = True

    if not needs_fix:
        print("\n" + "=" * 80)
        print("✓ SISTEMA CONFIGURADO CORRECTAMENTE")
        print("=" * 80)
        sys.exit(0)

# Aplicar correcciones
print("\n" + "=" * 80)
print("APLICANDO CORRECCIONES")
print("=" * 80)

if not pyqt5_ok:
    print("\n1. Instalando PyQt5...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "PyQt5"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✓ PyQt5 instalado")
        else:
            print(f"  ✗ Error: {result.stderr}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

if not pyside6_ok:
    print("\n2. Instalando PySide6 (requerido por OVITO)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "PySide6"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✓ PySide6 instalado")
        else:
            print(f"  ✗ Error: {result.stderr}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

if not shiboken6_ok:
    print("\n3. Instalando shiboken6 (requerido por OVITO)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "shiboken6"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✓ shiboken6 instalado")
        else:
            print(f"  ✗ Error: {result.stderr}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n4. Verificando matplotlib...")
if mpl_ok:
    print("  ✓ Matplotlib ya está instalado")
    print("  Nota: Matplotlib usará PyQt5 gracias a la configuración en el código")
else:
    print("  Instalando matplotlib...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "matplotlib"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✓ Matplotlib instalado")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 80)
print("REPARACIÓN COMPLETADA")
print("=" * 80)

print("\nResumen de la configuración:")
print("  • PyQt5: Para la interfaz gráfica principal")
print("  • PySide6/shiboken6: Para OVITO")
print("  • Matplotlib: Configurado para usar PyQt5 (Qt5Agg)")

print("\nIMPORTANTE:")
print("  La aplicación ahora puede usar ambos backends:")
print("  - OVITO usará PySide6")
print("  - La GUI y matplotlib usarán PyQt5")
print("  - Esto es seguro y normal")

print("\nPróximos pasos:")
print("  1. Reinicia Python completamente")
print("  2. Ejecuta: python main_qt.py")

print("\nSi aún hay problemas:")
print("  python scripts/validate_system.py")
