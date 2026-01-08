#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI para prediccion de vacancias con comparacion de metodos.

Permite comparar:
- Prediccion con modelo Random Forest (ML)
- Metodo tradicional Wigner-Seitz

Autor: OpenTopologyC Team
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import threading
import os
import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Imports del proyecto
try:
    from core.wigner_seitz import (
        WignerSeitzAnalyzer,
        SimulationBox,
        read_lammps_dump,
        count_vacancies_wigner_seitz
    )
except ImportError as e:
    print(f"Warning: Could not import wigner_seitz: {e}")
    WignerSeitzAnalyzer = None

try:
    from core.feature_extractor import FeatureExtractor
    from core.loader import DumpLoader
    from core.surface_extractor import SurfaceExtractor
    from config.extractor_config import ExtractorConfig
    import joblib
except ImportError as e:
    print(f"Warning: Could not import ML modules: {e}")


class PredictGUI:
    """
    GUI para prediccion de vacancias comparando metodo ML vs Wigner-Seitz.
    """

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("OpenTopologyC - Prediccion de Vacancias")
        self.window.geometry("800x900")
        self.window.resizable(True, True)

        # Variables GUI
        self.var_defective_file = tk.StringVar()
        self.var_reference_file = tk.StringVar()
        self.var_model_file = tk.StringVar()
        self.var_use_pbc = tk.BooleanVar(value=True)
        self.var_use_affine = tk.BooleanVar(value=False)

        # Variables de resultados
        self.ml_result = None
        self.ws_result = None

        # Estado
        self.running = False

        self.build_layout()

    def build_layout(self):
        """Construye la interfaz grafica"""
        # Frame principal con scroll
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content_frame = ttk.Frame(self.scrollable_frame, padding=10)
        content_frame.pack(fill="both", expand=True)

        # Titulo
        title_label = ttk.Label(
            content_frame,
            text="Prediccion de Vacancias - Comparacion de Metodos",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 20))

        # --------------------------------------------------
        # SECCION 1: ARCHIVOS DE ENTRADA
        # --------------------------------------------------
        section1 = ttk.LabelFrame(content_frame, text="Archivos de Entrada", padding=10)
        section1.pack(fill="x", pady=(0, 15))

        # Archivo defectuoso (para ambos metodos)
        def_frame = ttk.Frame(section1)
        def_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(def_frame, text="Archivo defectuoso (DUMP):",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))

        def_entry_frame = ttk.Frame(def_frame)
        def_entry_frame.pack(fill="x")

        ttk.Entry(def_entry_frame, textvariable=self.var_defective_file,
                  width=70).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(def_entry_frame, text="Buscar",
                   command=self.select_defective_file, width=10).pack(side="right")

        # Archivo de referencia (solo para Wigner-Seitz)
        ref_frame = ttk.Frame(section1)
        ref_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(ref_frame, text="Archivo de referencia (DUMP) - Solo Wigner-Seitz:",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))

        ref_entry_frame = ttk.Frame(ref_frame)
        ref_entry_frame.pack(fill="x")

        ttk.Entry(ref_entry_frame, textvariable=self.var_reference_file,
                  width=70).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(ref_entry_frame, text="Buscar",
                   command=self.select_reference_file, width=10).pack(side="right")

        # Modelo ML
        model_frame = ttk.Frame(section1)
        model_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(model_frame, text="Modelo entrenado (.joblib) - Solo ML:",
                  font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))

        model_entry_frame = ttk.Frame(model_frame)
        model_entry_frame.pack(fill="x")

        ttk.Entry(model_entry_frame, textvariable=self.var_model_file,
                  width=70).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(model_entry_frame, text="Buscar",
                   command=self.select_model_file, width=10).pack(side="right")

        # --------------------------------------------------
        # SECCION 2: OPCIONES WIGNER-SEITZ
        # --------------------------------------------------
        section2 = ttk.LabelFrame(content_frame, text="Opciones Wigner-Seitz", padding=10)
        section2.pack(fill="x", pady=(0, 15))

        options_frame = ttk.Frame(section2)
        options_frame.pack(fill="x")

        ttk.Checkbutton(
            options_frame,
            text="Condiciones Periodicas (PBC)",
            variable=self.var_use_pbc
        ).pack(side="left", padx=(0, 20))

        ttk.Checkbutton(
            options_frame,
            text="Mapeo Afin (para strain > 5%)",
            variable=self.var_use_affine
        ).pack(side="left")

        info_ws = ttk.Label(
            section2,
            text="Nota: El mapeo afin compensa deformaciones uniformes de la celda",
            font=("Arial", 9, "italic"),
            foreground="gray"
        )
        info_ws.pack(anchor="w", pady=(10, 0))

        # --------------------------------------------------
        # SECCION 3: ACCIONES
        # --------------------------------------------------
        section3 = ttk.LabelFrame(content_frame, text="Acciones", padding=10)
        section3.pack(fill="x", pady=(0, 15))

        buttons_frame = ttk.Frame(section3)
        buttons_frame.pack(fill="x", pady=5)

        # Boton ML
        self.ml_button = ttk.Button(
            buttons_frame,
            text="Predecir con ML",
            command=self.predict_ml,
            width=20
        )
        self.ml_button.pack(side="left", padx=5)

        # Boton Wigner-Seitz
        self.ws_button = ttk.Button(
            buttons_frame,
            text="Analizar Wigner-Seitz",
            command=self.analyze_wigner_seitz,
            width=20
        )
        self.ws_button.pack(side="left", padx=5)

        # Boton Comparar
        self.compare_button = ttk.Button(
            buttons_frame,
            text="Comparar Ambos",
            command=self.compare_methods,
            width=20
        )
        self.compare_button.pack(side="left", padx=5)

        # Boton Salir
        ttk.Button(
            buttons_frame,
            text="Salir",
            command=self.window.quit,
            width=15
        ).pack(side="right", padx=5)

        # --------------------------------------------------
        # SECCION 4: RESULTADOS
        # --------------------------------------------------
        section4 = ttk.LabelFrame(content_frame, text="Resultados", padding=10)
        section4.pack(fill="both", expand=True, pady=(0, 15))

        # Frame para mostrar resultados lado a lado
        results_container = ttk.Frame(section4)
        results_container.pack(fill="both", expand=True)

        # Columna ML
        ml_frame = ttk.LabelFrame(results_container, text="Modelo ML", padding=5)
        ml_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.ml_result_label = ttk.Label(
            ml_frame,
            text="Sin resultados",
            font=("Consolas", 11),
            justify="left"
        )
        self.ml_result_label.pack(anchor="w", pady=10, padx=10)

        # Columna Wigner-Seitz
        ws_frame = ttk.LabelFrame(results_container, text="Wigner-Seitz", padding=5)
        ws_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.ws_result_label = ttk.Label(
            ws_frame,
            text="Sin resultados",
            font=("Consolas", 11),
            justify="left"
        )
        self.ws_result_label.pack(anchor="w", pady=10, padx=10)

        # --------------------------------------------------
        # SECCION 5: COMPARACION
        # --------------------------------------------------
        section5 = ttk.LabelFrame(content_frame, text="Comparacion", padding=10)
        section5.pack(fill="x", pady=(0, 15))

        self.comparison_text = scrolledtext.ScrolledText(
            section5,
            height=8,
            width=80,
            font=("Consolas", 10),
            bg="#f5f5f5"
        )
        self.comparison_text.pack(fill="both", expand=True)

        # --------------------------------------------------
        # SECCION 6: ESTADO
        # --------------------------------------------------
        section6 = ttk.LabelFrame(content_frame, text="Estado", padding=10)
        section6.pack(fill="x")

        status_frame = ttk.Frame(section6)
        status_frame.pack(fill="x")

        self.status_label = ttk.Label(
            status_frame,
            text="Listo",
            font=("Arial", 10),
            foreground="green"
        )
        self.status_label.pack(side="left")

        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode='indeterminate',
            length=200
        )

    # --------------------------------------------------
    # METODOS DE SELECCION DE ARCHIVOS
    # --------------------------------------------------
    def select_defective_file(self):
        """Selecciona archivo defectuoso"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo defectuoso",
            filetypes=[("LAMMPS dump", "*.dump"), ("All files", "*.*")]
        )
        if filepath:
            self.var_defective_file.set(filepath)
            self.update_status(f"Archivo defectuoso: {Path(filepath).name}")

    def select_reference_file(self):
        """Selecciona archivo de referencia"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo de referencia",
            filetypes=[("LAMMPS dump", "*.dump"), ("All files", "*.*")]
        )
        if filepath:
            self.var_reference_file.set(filepath)
            self.update_status(f"Archivo referencia: {Path(filepath).name}")

    def select_model_file(self):
        """Selecciona modelo entrenado"""
        filepath = filedialog.askopenfilename(
            title="Seleccionar modelo entrenado",
            filetypes=[("Joblib files", "*.joblib"), ("PKL files", "*.pkl"), ("All files", "*.*")]
        )
        if filepath:
            self.var_model_file.set(filepath)
            self.update_status(f"Modelo: {Path(filepath).name}")

    # --------------------------------------------------
    # METODO ML
    # --------------------------------------------------
    def predict_ml(self):
        """Predice vacancias usando el modelo ML"""
        if self.running:
            return

        # Validaciones
        if not self.var_defective_file.get():
            messagebox.showerror("Error", "Seleccione un archivo defectuoso")
            return

        if not self.var_model_file.get():
            messagebox.showerror("Error", "Seleccione un modelo entrenado")
            return

        if not Path(self.var_defective_file.get()).exists():
            messagebox.showerror("Error", "El archivo defectuoso no existe")
            return

        if not Path(self.var_model_file.get()).exists():
            messagebox.showerror("Error", "El modelo no existe")
            return

        self.running = True
        self.ml_button.config(state="disabled")
        self.update_status("Ejecutando prediccion ML...")
        self.progress_bar.pack(side="right", padx=(10, 0))
        self.progress_bar.start()

        thread = threading.Thread(target=self._execute_ml_prediction, daemon=True)
        thread.start()

    def _execute_ml_prediction(self):
        """Ejecuta prediccion ML en hilo separado"""
        try:
            # Cargar modelo
            bundle = joblib.load(self.var_model_file.get())

            if isinstance(bundle, dict):
                pipeline = bundle.get('pipeline')
                feature_names = bundle.get('feature_names', [])
            else:
                pipeline = bundle
                feature_names = []

            # Crear configuracion temporal
            cfg = ExtractorConfig(
                input_dir=str(Path(self.var_defective_file.get()).parent),
                probe_radius=2.0,
                a0=3.532,
                compute_grid_features=True,
                compute_inertia_features=True,
                compute_radial_features=True,
                compute_entropy_features=True,
                compute_clustering_features=True
            )

            # Extraer features
            extractor = FeatureExtractor(cfg)
            loader = DumpLoader()

            # Cargar posiciones
            raw = loader.load(self.var_defective_file.get())
            positions = raw["positions"]

            # Extraer features
            features = extractor.extract_all_features(positions)

            # Preparar para prediccion
            import pandas as pd
            X = pd.DataFrame([features])

            # Eliminar columnas que no son features
            for col in ['n_vacancies', 'file', 'num_points']:
                if col in X.columns:
                    X = X.drop(columns=[col])

            # Predecir
            prediction = pipeline.predict(X)[0]

            self.ml_result = {
                'n_vacancies': prediction,
                'features_used': len(X.columns)
            }

            def update_success():
                self.progress_bar.stop()
                self.progress_bar.pack_forget()
                self.ml_button.config(state="normal")
                self.running = False

                result_text = (
                    f"Vacancias predichas: {prediction:.1f}\n"
                    f"Features usadas: {len(X.columns)}"
                )
                self.ml_result_label.config(text=result_text)
                self.update_status("Prediccion ML completada")

            self.window.after(0, update_success)

        except Exception as e:
            error_msg = str(e)

            def update_error():
                self.progress_bar.stop()
                self.progress_bar.pack_forget()
                self.ml_button.config(state="normal")
                self.running = False
                self.ml_result_label.config(text=f"Error: {error_msg[:50]}...")
                self.update_status(f"Error ML: {error_msg}", error=True)

            self.window.after(0, update_error)

    # --------------------------------------------------
    # METODO WIGNER-SEITZ
    # --------------------------------------------------
    def analyze_wigner_seitz(self):
        """Analiza vacancias usando metodo Wigner-Seitz"""
        if self.running:
            return

        if WignerSeitzAnalyzer is None:
            messagebox.showerror("Error", "Modulo Wigner-Seitz no disponible")
            return

        # Validaciones
        if not self.var_defective_file.get():
            messagebox.showerror("Error", "Seleccione un archivo defectuoso")
            return

        if not self.var_reference_file.get():
            messagebox.showerror("Error", "Seleccione un archivo de referencia")
            return

        if not Path(self.var_defective_file.get()).exists():
            messagebox.showerror("Error", "El archivo defectuoso no existe")
            return

        if not Path(self.var_reference_file.get()).exists():
            messagebox.showerror("Error", "El archivo de referencia no existe")
            return

        self.running = True
        self.ws_button.config(state="disabled")
        self.update_status("Ejecutando analisis Wigner-Seitz...")
        self.progress_bar.pack(side="right", padx=(10, 0))
        self.progress_bar.start()

        thread = threading.Thread(target=self._execute_ws_analysis, daemon=True)
        thread.start()

    def _execute_ws_analysis(self):
        """Ejecuta analisis Wigner-Seitz en hilo separado"""
        try:
            results = count_vacancies_wigner_seitz(
                self.var_reference_file.get(),
                self.var_defective_file.get(),
                use_pbc=self.var_use_pbc.get(),
                use_affine=self.var_use_affine.get()
            )

            self.ws_result = results

            def update_success():
                self.progress_bar.stop()
                self.progress_bar.pack_forget()
                self.ws_button.config(state="normal")
                self.running = False

                result_text = (
                    f"Vacancias: {results['n_vacancies']}\n"
                    f"Intersticiales: {results['n_interstitials']}\n"
                    f"Sitios ref: {results['n_reference_sites']}\n"
                    f"Atomos def: {results['n_defective_atoms']}\n"
                    f"Conc. vac: {results['vacancy_concentration']*100:.3f}%\n"
                    f"Strain: {results['volumetric_strain']*100:.2f}%"
                )
                self.ws_result_label.config(text=result_text)
                self.update_status("Analisis Wigner-Seitz completado")

            self.window.after(0, update_success)

        except Exception as e:
            error_msg = str(e)

            def update_error():
                self.progress_bar.stop()
                self.progress_bar.pack_forget()
                self.ws_button.config(state="normal")
                self.running = False
                self.ws_result_label.config(text=f"Error: {error_msg[:50]}...")
                self.update_status(f"Error WS: {error_msg}", error=True)

            self.window.after(0, update_error)

    # --------------------------------------------------
    # COMPARACION
    # --------------------------------------------------
    def compare_methods(self):
        """Compara los resultados de ambos metodos"""
        self.comparison_text.delete(1.0, tk.END)

        if self.ml_result is None and self.ws_result is None:
            self.comparison_text.insert(tk.END, "No hay resultados para comparar.\n")
            self.comparison_text.insert(tk.END, "Ejecute al menos uno de los metodos primero.")
            return

        self.comparison_text.insert(tk.END, "="*60 + "\n")
        self.comparison_text.insert(tk.END, "COMPARACION DE METODOS\n")
        self.comparison_text.insert(tk.END, "="*60 + "\n\n")

        # Resultados ML
        self.comparison_text.insert(tk.END, "MODELO ML (Random Forest):\n")
        self.comparison_text.insert(tk.END, "-"*30 + "\n")
        if self.ml_result:
            self.comparison_text.insert(
                tk.END,
                f"  Vacancias predichas: {self.ml_result['n_vacancies']:.1f}\n"
            )
            self.comparison_text.insert(
                tk.END,
                f"  Features utilizadas: {self.ml_result['features_used']}\n"
            )
        else:
            self.comparison_text.insert(tk.END, "  No ejecutado\n")

        self.comparison_text.insert(tk.END, "\n")

        # Resultados Wigner-Seitz
        self.comparison_text.insert(tk.END, "METODO WIGNER-SEITZ:\n")
        self.comparison_text.insert(tk.END, "-"*30 + "\n")
        if self.ws_result:
            self.comparison_text.insert(
                tk.END,
                f"  Vacancias detectadas: {self.ws_result['n_vacancies']}\n"
            )
            self.comparison_text.insert(
                tk.END,
                f"  Intersticiales: {self.ws_result['n_interstitials']}\n"
            )
            self.comparison_text.insert(
                tk.END,
                f"  Concentracion: {self.ws_result['vacancy_concentration']*100:.4f}%\n"
            )
            self.comparison_text.insert(
                tk.END,
                f"  Strain volumetrico: {self.ws_result['volumetric_strain']*100:.2f}%\n"
            )
        else:
            self.comparison_text.insert(tk.END, "  No ejecutado\n")

        # Comparacion directa si ambos estan disponibles
        if self.ml_result and self.ws_result:
            self.comparison_text.insert(tk.END, "\n")
            self.comparison_text.insert(tk.END, "="*60 + "\n")
            self.comparison_text.insert(tk.END, "DIFERENCIA:\n")
            self.comparison_text.insert(tk.END, "="*60 + "\n")

            ml_vac = self.ml_result['n_vacancies']
            ws_vac = self.ws_result['n_vacancies']
            diff = ml_vac - ws_vac
            rel_diff = (diff / ws_vac * 100) if ws_vac > 0 else 0

            self.comparison_text.insert(
                tk.END,
                f"  ML - WS = {diff:.1f} vacancias\n"
            )
            self.comparison_text.insert(
                tk.END,
                f"  Diferencia relativa: {rel_diff:.1f}%\n"
            )

            if abs(rel_diff) < 5:
                self.comparison_text.insert(
                    tk.END,
                    "\n  CONCLUSION: Excelente concordancia entre metodos\n"
                )
            elif abs(rel_diff) < 15:
                self.comparison_text.insert(
                    tk.END,
                    "\n  CONCLUSION: Buena concordancia entre metodos\n"
                )
            else:
                self.comparison_text.insert(
                    tk.END,
                    "\n  CONCLUSION: Diferencia significativa - revisar parametros\n"
                )

    # --------------------------------------------------
    # UTILIDADES
    # --------------------------------------------------
    def update_status(self, message, error=False):
        """Actualiza el mensaje de estado"""
        self.status_label.config(text=message)
        if error:
            self.status_label.config(foreground="red")
        elif "completad" in message.lower():
            self.status_label.config(foreground="green")
        else:
            self.status_label.config(foreground="blue")

    def run(self):
        """Ejecuta la GUI"""
        self.window.bind('<Escape>', lambda e: self.window.quit())

        # Centrar ventana
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

        self.window.mainloop()


if __name__ == "__main__":
    gui = PredictGUI()
    gui.run()
