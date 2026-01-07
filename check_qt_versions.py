#!/usr/bin/env python3
"""Script para verificar versiones de Qt y matplotlib instaladas"""

import sys

print("=" * 80)
print("VERIFICACIÓN DE VERSIONES Qt y Matplotlib")
print("=" * 80)

# Check Python version
print(f"\n1. Python: {sys.version}")

# Check PyQt5
print("\n2. PyQt5:")
try:
    import PyQt5.QtCore
    print(f"   ✓ Instalado: {PyQt5.QtCore.PYQT_VERSION_STR}")
    print(f"   ✓ Qt version: {PyQt5.QtCore.QT_VERSION_STR}")
except ImportError as e:
    print(f"   ✗ No instalado: {e}")

# Check PySide6
print("\n3. PySide6:")
try:
    import PySide6.QtCore
    print(f"   ✓ Instalado: {PySide6.__version__}")
    print(f"   ⚠ PROBLEMA: PySide6 está instalado y causa conflicto")
    print(f"   → Solución: pip uninstall PySide6")
except ImportError:
    print(f"   ✓ No instalado (correcto)")

# Check PySide2
print("\n3b. PySide2:")
try:
    import PySide2.QtCore
    print(f"   ✓ Instalado: {PySide2.__version__}")
except ImportError:
    print(f"   ✓ No instalado")

# Check matplotlib
print("\n4. Matplotlib:")
try:
    import matplotlib
    print(f"   ✓ Versión: {matplotlib.__version__}")
    print(f"   ✓ Backend por defecto: {matplotlib.get_backend()}")

    # Check available Qt backends
    print("\n5. Backends Qt disponibles en matplotlib:")
    try:
        import matplotlib.backends.backend_qt5agg
        print(f"   ✓ backend_qt5agg (PyQt5)")
    except ImportError as e:
        print(f"   ✗ backend_qt5agg no disponible: {e}")

    try:
        import matplotlib.backends.backend_qtagg
        print(f"   ✓ backend_qtagg (genérico - puede usar PyQt5 o PySide6)")
    except ImportError:
        print(f"   ✗ backend_qtagg no disponible")

except ImportError as e:
    print(f"   ✗ No instalado: {e}")

# Check QT_API environment variable
print("\n6. Variables de entorno:")
import os
print(f"   QT_API: {os.environ.get('QT_API', 'No configurada')}")
print(f"   MPLBACKEND: {os.environ.get('MPLBACKEND', 'No configurada')}")

# Try importing with our configuration
print("\n7. Test de configuración:")
os.environ['QT_API'] = 'pyqt5'
os.environ['MPLBACKEND'] = 'Qt5Agg'

try:
    # Reimport matplotlib with new config
    import importlib
    if 'matplotlib' in sys.modules:
        importlib.reload(matplotlib)

    import matplotlib
    matplotlib.use('Qt5Agg', force=True)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    print(f"   ✓ Configuración funcional")
    print(f"   ✓ Backend activo: {matplotlib.get_backend()}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Check matplotlib config file
print("\n8. Configuración de matplotlib:")
try:
    import matplotlib
    config_dir = matplotlib.get_configdir()
    print(f"   Config dir: {config_dir}")

    import os
    matplotlibrc = os.path.join(config_dir, 'matplotlibrc')
    if os.path.exists(matplotlibrc):
        print(f"   ⚠ Archivo matplotlibrc encontrado: {matplotlibrc}")
        with open(matplotlibrc, 'r') as f:
            for line in f:
                if 'backend' in line.lower() and not line.strip().startswith('#'):
                    print(f"     → {line.strip()}")
    else:
        print(f"   ✓ No hay matplotlibrc personalizado")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 80)
print("DIAGNÓSTICO:")
print("=" * 80)

has_problem = False

try:
    import PySide6
    print("\n⚠ PROBLEMA ENCONTRADO:")
    print("   PySide6 está instalado en tu sistema.")
    print("   Matplotlib prefiere PySide6 sobre PyQt5 cuando ambos están presentes.")
    print("\n✅ SOLUCIÓN RECOMENDADA:")
    print("   Desinstalar PySide6:")
    print("   $ pip uninstall PySide6")
    print("   $ conda remove pyside6  # si instalado con conda")
    has_problem = True
except ImportError:
    pass

if not has_problem:
    print("\n✓ No se encontraron problemas obvios con PySide6.")
    print("  Verifica si matplotlib puede importar PyQt5 correctamente.")

print("\n" + "=" * 80)
print("\nPara ejecutar este script con tu Python:")
print("/home/santi/miniconda3/bin/python check_qt_versions.py")
print("=" * 80)
