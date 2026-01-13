# Guía Rápida: Usar Wigner-Seitz en la GUI

## 🚀 Inicio Rápido

### 1. Abrir la GUI de Predicciones

```bash
cd /home/user/opentopologyc-navidad
python gui/predict_gui.py
```

### 2. Interfaz Principal

La ventana muestra tres secciones de archivos de entrada:

```
┌─────────────────────────────────────────────────────────────┐
│ Archivos de Entrada                                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ► Archivo defectuoso (DUMP):                                │
│   [________________________________] [Buscar]                │
│   → Para ambos métodos (ML y W-S)                           │
│                                                              │
│ ► Archivo de referencia (DUMP) - Solo Wigner-Seitz:        │
│   [________________________________] [Buscar]                │
│   → CONFIGURACIÓN DE REFERENCIA (estructura perfecta)       │
│                                                              │
│ ► Modelo entrenado (.joblib) - Solo ML:                    │
│   [________________________________] [Buscar]                │
│   → Solo para predicción con Random Forest                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3. Opciones Wigner-Seitz

```
┌─────────────────────────────────────────────────────────────┐
│ Opciones Wigner-Seitz                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ☑ Condiciones Periodicas (PBC)                              │
│   → Activar si la simulación usa PBC (recomendado)          │
│                                                              │
│ ☐ Mapeo Afin (para strain > 5%)                            │
│   → Activar si la celda se deformó uniformemente            │
│                                                              │
│ Nota: El mapeo afin compensa deformaciones uniformes        │
│       de la celda                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4. Botones de Acción

```
┌─────────────────────────────────────────────────────────────┐
│ Acciones                                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ [Predecir con ML] [Analizar Wigner-Seitz] [Comparar Ambos] │
│                                                              │
│ Predecir con ML:                                             │
│   - Requiere: archivo defectuoso + modelo ML                │
│   - NO requiere: archivo de referencia                      │
│                                                              │
│ Analizar Wigner-Seitz:                                      │
│   - Requiere: archivo defectuoso + archivo de referencia    │
│   - NO requiere: modelo ML                                  │
│                                                              │
│ Comparar Ambos:                                             │
│   - Requiere: TODOS los archivos                            │
│   - Ejecuta ambos métodos y compara resultados              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5. Resultados

```
┌──────────────────────────┬──────────────────────────────────┐
│ Modelo ML                │ Wigner-Seitz                     │
├──────────────────────────┼──────────────────────────────────┤
│                          │                                  │
│ Vacancias predichas: 160 │ Vacancias: 156                   │
│ Features usadas: 45      │ Intersticiales: 12               │
│                          │ Sitios ref: 32000                │
│                          │ Atomos def: 31856                │
│                          │ Conc. vac: 0.488%                │
│                          │ Strain: 1.23%                    │
│                          │                                  │
└──────────────────────────┴──────────────────────────────────┘
```

---

## 📋 Casos de Uso

### Caso 1: Solo Análisis Wigner-Seitz

**Objetivo**: Detectar vacancias comparando con configuración de referencia

**Pasos**:
1. Clic en "Buscar" → Archivo defectuoso
   - Seleccionar: `simulation_after_irradiation.dump`

2. Clic en "Buscar" → Archivo de referencia
   - Seleccionar: `perfect_lattice.dump`

3. Configurar opciones:
   - ☑ Condiciones Periódicas (si usa PBC)
   - ☐ Mapeo Afín (solo si hay strain grande)

4. Clic en "Analizar Wigner-Seitz"

5. Ver resultados en panel derecho

**Archivos necesarios**: 2
- ✅ Archivo defectuoso
- ✅ Archivo de referencia
- ❌ Modelo ML (no necesario)

---

### Caso 2: Comparación ML vs Wigner-Seitz

**Objetivo**: Validar predicción ML contra método tradicional

**Pasos**:
1. Cargar archivo defectuoso
2. Cargar archivo de referencia
3. Cargar modelo ML entrenado
4. Clic en "Comparar Ambos"
5. Ver comparación detallada abajo

**Archivos necesarios**: 3
- ✅ Archivo defectuoso
- ✅ Archivo de referencia
- ✅ Modelo ML

**Resultado**: Tabla comparativa con diferencias y conclusiones

---

## 🔍 Interpretación de Resultados

### Métricas Wigner-Seitz

| Métrica | Significado | Ejemplo |
|---------|-------------|---------|
| **Vacancias** | Número de sitios de red vacíos | 156 |
| **Intersticiales** | Átomos en posiciones no de red | 12 |
| **Sitios ref** | Total de sitios en referencia | 32000 |
| **Atomos def** | Total de átomos en defectuoso | 31856 |
| **Conc. vac** | Porcentaje de sitios vacíos | 0.488% |
| **Strain** | Deformación volumétrica | 1.23% |

### Análisis de Comparación

Cuando usas "Comparar Ambos", el sistema muestra:

```
===========================================================
DIFERENCIA:
===========================================================
  ML - WS = 4.5 vacancias
  Diferencia relativa: 2.9%

  CONCLUSION: Excelente concordancia entre metodos
```

**Niveles de concordancia**:
- < 5% → **Excelente concordancia**
- 5-15% → **Buena concordancia**
- > 15% → **Diferencia significativa** (revisar parámetros)

---

## ❓ Preguntas Frecuentes

### ¿Qué es una "configuración de referencia"?

Es un archivo LAMMPS dump que contiene la estructura **sin defectos** o en estado **inicial conocido**. Por ejemplo:
- Red cristalina perfecta antes de la irradiación
- Estructura optimizada sin vacancias
- Configuración inicial del sistema

### ¿Cuándo activar "Mapeo Afín"?

Actívalo cuando:
- La celda de simulación se deformó uniformemente
- El strain volumétrico es > 5%
- Hay cambio de presión o temperatura
- El sistema se expandió/contrajo homogéneamente

**No activar** si solo hay defectos locales sin deformación global.

### ¿Cuándo usar PBC?

**Siempre** que tu simulación LAMMPS use condiciones periódicas (boundary p p p).

**No usar** solo si tu simulación tiene fronteras fijas (boundary f f f).

### ¿Puedo usar coordenadas escaladas (xs ys zs)?

**Sí**. El lector soporta automáticamente:
- Coordenadas regulares: `x y z`
- Coordenadas unwrapped: `xu yu zu`
- Coordenadas escaladas: `xs ys zs` ← Se convierten automáticamente

### ¿Qué formato deben tener los archivos?

Formato estándar LAMMPS dump:

```
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
32000
ITEM: BOX BOUNDS pp pp pp
0.0 100.0
0.0 100.0
0.0 100.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 3.532 0.0 0.0
...
```

---

## ⚠️ Errores Comunes

### Error: "Seleccione un archivo de referencia"

**Causa**: No se cargó el archivo de referencia

**Solución**:
1. Buscar el campo "Archivo de referencia (DUMP) - Solo Wigner-Seitz"
2. Clic en "Buscar" a la derecha
3. Seleccionar archivo de referencia

### Error: "El archivo de referencia no existe"

**Causa**: Ruta inválida o archivo movido

**Solución**:
1. Verificar que el archivo existe en el sistema
2. Volver a seleccionar con el botón "Buscar"

### Error: "Diferencia significativa en número de átomos"

**Causa**: Los archivos tienen muy diferente cantidad de átomos

**Solución**:
- Esto es una advertencia, no un error
- Verifica que los archivos corresponden al mismo sistema
- Si es correcto (muchas vacancias), ignora la advertencia

### Error: "Strain volumétrico significativo sin mapeo afín"

**Causa**: Hay deformación > 5% sin mapeo afín activado

**Solución**:
1. Ir a "Opciones Wigner-Seitz"
2. Activar ☑ "Mapeo Afín (para strain > 5%)"
3. Re-ejecutar el análisis

---

## 📊 Ejemplo Completo

### Escenario: Análisis de Irradiación

**Archivos**:
- `fcc_Cu_perfect.dump` - Estructura FCC de Cu perfecta (referencia)
- `fcc_Cu_after_100keV.dump` - Después de irradiación con 100 keV (defectuoso)
- `rf_vacancy_model.joblib` - Modelo Random Forest entrenado

**Procedimiento**:

1. **Cargar archivos**
   ```
   Defectuoso:  fcc_Cu_after_100keV.dump
   Referencia:  fcc_Cu_perfect.dump
   Modelo:      rf_vacancy_model.joblib
   ```

2. **Configurar opciones**
   ```
   ☑ Condiciones Periódicas (PBC)
   ☐ Mapeo Afín (no hay strain global)
   ```

3. **Ejecutar**
   ```
   Clic en "Comparar Ambos"
   ```

4. **Resultados esperados**
   ```
   ML:    ~155 vacancias (predicción estadística)
   W-S:   156 vacancias (conteo exacto)
   Diff:  <5% → Excelente concordancia
   ```

**Interpretación**:
- El modelo ML está bien calibrado
- La irradiación produjo ~156 vacancias
- Concentración: 0.48% (aceptable para 100 keV)
- Sin strain significativo (solo defectos locales)

---

## 🎯 Tips y Mejores Prácticas

### 1. Preparación de Archivos

✅ **Hacer**:
- Usar el mismo timestep o equilibrar antes de dump
- Asegurar que las cajas sean comparables
- Verificar que el formato LAMMPS sea consistente

❌ **Evitar**:
- Comparar sistemas de diferente tamaño
- Mezclar diferentes materiales
- Usar dumps de diferentes códigos de simulación

### 2. Configuración de Opciones

✅ **PBC activada**: Para sistemas periódicos (mayoría de casos)
✅ **Mapeo afín activado**: Si hay presión/temperatura diferente
❌ **Mapeo afín innecesario**: Si solo hay defectos puntuales

### 3. Interpretación

✅ **Validar**: Comparar ML con W-S para verificar modelo
✅ **Contextualizar**: Considerar la energía de irradiación
✅ **Revisar strain**: Si > 5%, considerar efectos de volumen

### 4. Solución de Problemas

Si los resultados no tienen sentido:
1. Verificar que los archivos corresponden al mismo sistema
2. Revisar si hay deformación global (activar mapeo afín)
3. Comprobar que PBC está correctamente configurada
4. Validar que las coordenadas se leen correctamente

---

## 📞 Soporte

Si necesitas ayuda adicional:

1. **Documentación completa**: `ANALISIS_WIGNER_SEITZ.md`
2. **Tests**: `tests/test_wigner_seitz.py`
3. **Código fuente**:
   - Core: `core/wigner_seitz.py`
   - GUI: `gui/predict_gui.py`

---

**Versión**: 1.0
**Última actualización**: 2026-01-13
**Estado**: ✅ Documentación completa
