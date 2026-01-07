#!/usr/bin/env python3
"""
Script para reparar el problema de backends Qt en matplotlib

Problema:
- PySide6 está instalado pero corrupto
- Matplotlib intenta usarlo en lugar de PyQt5
- La aplicación falla con ImportError

Solución:
- Desinstalar PySide6 completamente
- Verificar que PyQt5 funcione correctamente
"""

import subprocess
import sys

print("=" * 80)
print("REPARACIÓN DE BACKEND Qt PARA MATPLOTLIB")
print("=" * 80)

print("\n1. Verificando estado actual...")
print("-" * 80)

# Check current state
try:
    import PySide6
    print("⚠ PySide6 está instalado (y probablemente corrupto)")
    pyside6_installed = True
except ImportError as e:
    print("✓ PySide6 no está instalado")
    pyside6_installed = False

try:
    import PyQt5.QtCore
    print(f"✓ PyQt5 está instalado: {PyQt5.QtCore.PYQT_VERSION_STR}")
    pyqt5_works = True
except ImportError:
    print("✗ PyQt5 NO está instalado o no funciona")
    pyqt5_works = False

if not pyside6_installed and pyqt5_works:
    print("\n✓ No hay problemas detectados. El sistema está correcto.")
    sys.exit(0)

if pyside6_installed:
    print("\n2. Desinstalando PySide6...")
    print("-" * 80)

    # Try pip uninstall
    print("\nIntentando: pip uninstall PySide6 -y")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "PySide6", "-y"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode == 0:
            print("✓ PySide6 desinstalado con pip")
        else:
            print("⚠ pip uninstall falló o PySide6 no estaba instalado con pip")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Try conda remove
    print("\nIntentando: conda remove pyside6 -y")
    try:
        result = subprocess.run(
            ["conda", "remove", "pyside6", "-y"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ PySide6 desinstalado con conda")
        else:
            print("⚠ conda remove falló (puede ser normal si no se instaló con conda)")
    except FileNotFoundError:
        print("⚠ conda no está disponible")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Also try PySide6-Essentials and PySide6-Addons
    print("\nDesinstalando paquetes relacionados...")
    for pkg in ["PySide6-Essentials", "PySide6-Addons", "shiboken6"]:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],
                capture_output=True,
                text=True
            )
            print(f"  Intentado desinstalar {pkg}")
        except:
            pass

print("\n3. Limpiando cachés de Python...")
print("-" * 80)
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "cache", "purge"],
        capture_output=True
    )
    print("✓ Caché de pip limpiado")
except:
    print("⚠ No se pudo limpiar caché de pip")

print("\n4. Verificando estado después de desinstalación...")
print("-" * 80)

# Force reimport to check
import importlib
import sys

# Remove from cache
for mod in list(sys.modules.keys()):
    if 'PySide6' in mod:
        del sys.modules[mod]

try:
    import PySide6
    print("⚠ PySide6 AÚN está presente (puede requerir desinstalación manual)")
    print("\nPara desinstalar manualmente:")
    print("  pip uninstall PySide6 PySide6-Essentials PySide6-Addons shiboken6 -y")
except ImportError:
    print("✓ PySide6 ha sido desinstalado correctamente")

try:
    import PyQt5.QtCore
    print(f"✓ PyQt5 funciona correctamente: {PyQt5.QtCore.PYQT_VERSION_STR}")
except ImportError:
    print("✗ PyQt5 no funciona. Instalando...")
    subprocess.run([sys.executable, "-m", "pip", "install", "PyQt5"])

print("\n5. Verificando matplotlib...")
print("-" * 80)

# Set environment before importing matplotlib
import os
os.environ['QT_API'] = 'pyqt5'
os.environ['MPLBACKEND'] = 'Qt5Agg'

# Remove matplotlib from cache
for mod in list(sys.modules.keys()):
    if 'matplotlib' in mod:
        del sys.modules[mod]

try:
    import matplotlib
    matplotlib.use('Qt5Agg', force=True)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    print(f"✓ Matplotlib configurado correctamente")
    print(f"  Backend: {matplotlib.get_backend()}")
except Exception as e:
    print(f"✗ Error al configurar matplotlib: {e}")

print("\n" + "=" * 80)
print("RESULTADO:")
print("=" * 80)

# Final verification
all_good = True
try:
    import PySide6
    print("✗ PySide6 aún está instalado")
    all_good = False
except ImportError:
    print("✓ PySide6 desinstalado")

try:
    import PyQt5.QtCore
    print("✓ PyQt5 funciona")
except ImportError:
    print("✗ PyQt5 no funciona")
    all_good = False

try:
    import matplotlib
    matplotlib.use('Qt5Agg', force=True)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    print("✓ Matplotlib con PyQt5 funciona")
except Exception as e:
    print(f"✗ Matplotlib no funciona: {e}")
    all_good = False

print("\n" + "=" * 80)
if all_good:
    print("✅ TODO CORRECTO - La aplicación debería funcionar ahora")
    print("\nPuedes ejecutar:")
    print("  python main_qt.py")
else:
    print("⚠ AÚN HAY PROBLEMAS")
    print("\nIntenta desinstalar manualmente:")
    print("  pip uninstall PySide6 PySide6-Essentials PySide6-Addons shiboken6 -y")
    print("  conda remove pyside6 -y")

print("=" * 80)
