#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# opentopologyc/gui/prediction_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..config.extractor_config import ExtractorConfig
from ..core.prediction_pipeline import PredictionPipeline


class PredictionGUI:
    """
    GUI para predicción de vacancias usando modelo entrenado
    """

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("OpenTopologyC - Predicción de Vacancias")
        self.window.geometry("650x900")
        self.window.resizable(True, True)

        # Variables GUI
        self.var_input_dump = tk.StringVar()
        self.var_model_file = tk.StringVar()
        self.var_apply_alpha = tk.BooleanVar(value=True)
        self.var_probe_radius = tk.DoubleVar(value=2.0)
        self.var_num_ghost_layers = tk.IntVar(value=2)

        # Parámetros de clustering
        self.var_apply_clustering = tk.BooleanVar(value=False)
        self.var_clustering_method = tk.StringVar(value="KMeans")
        self.var_n_clusters = tk.IntVar(value=5)
        self.var_quantile = tk.DoubleVar(value=0.2)
        self.var_linkage = tk.StringVar(value="ward")
        self.var_min_cluster_size = tk.IntVar(value=10)
        self.var_target_cluster = tk.StringVar(value="largest")
        self.var_specific_cluster = tk.IntVar(value=0)

        # Parámetros del material
        self.var_total_atoms = tk.IntVar(value=16384)
        self.var_a0 = tk.DoubleVar(value=3.532)
        self.var_lattice_type = tk.StringVar(value="fcc")

        # Features a computar
        self.var_grid = tk.BooleanVar(value=True)
        self.var_hull = tk.BooleanVar(value=True)
        self.var_inertia = tk.BooleanVar(value=True)
        self.var_radial = tk.BooleanVar(value=True)
        self.var_entropy = tk.BooleanVar(value=True)
        self.var_clustering = tk.BooleanVar(value=True)

        # Estado
        self.running = False
        self.pipeline = None

        self.build_layout()

    def build_layout(self):
        """Construye la interfaz"""
        # Frame principal con scroll
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content_frame = ttk.Frame(scrollable_frame, padding=10)
        content_frame.pack(fill="both", expand=True)

        # Título
        title = ttk.Label(
            content_frame,
            text="OpenTopologyC - Predicción de Vacancias",
            font=("Arial", 14, "bold")
        )
        title.pack(pady=(0, 20))

        # SECCIÓN 1: ARCHIVOS DE ENTRADA
        section1 = ttk.LabelFrame(content_frame, text="Archivos de Entrada", padding=10)
        section1.pack(fill="x", pady=(0, 15))

        # Archivo dump
        ttk.Label(section1, text="Archivo dump:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        dump_frame = ttk.Frame(section1)
        dump_frame.pack(fill="x", pady=(0, 10))
        ttk.Entry(dump_frame, textvariable=self.var_input_dump, width=60).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(dump_frame, text="Buscar", command=self.select_dump_file, width=10).pack(side="right")

        # Modelo entrenado
        ttk.Label(section1, text="Modelo entrenado:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))

        # Combobox para modelos disponibles
        ttk.Label(section1, text="Modelos disponibles en carpeta models/:", font=("Arial", 9)).pack(anchor="w", pady=(0, 2))
        models_combo_frame = ttk.Frame(section1)
        models_combo_frame.pack(fill="x", pady=(0, 5))
        self.models_combo = ttk.Combobox(models_combo_frame, state="readonly", width=57)
        self.models_combo.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.models_combo.bind("<<ComboboxSelected>>", self.on_model_selected_combo)
        ttk.Button(models_combo_frame, text="🔄", command=self.refresh_models_list, width=3).pack(side="right")

        # Entry manual
        ttk.Label(section1, text="O seleccione manualmente:", font=("Arial", 9)).pack(anchor="w", pady=(5, 2))
        model_frame = ttk.Frame(section1)
        model_frame.pack(fill="x", pady=(0, 5))
        ttk.Entry(model_frame, textvariable=self.var_model_file, width=60).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(model_frame, text="Buscar", command=self.select_model_file, width=10).pack(side="right")

        # Cargar lista inicial
        self.refresh_models_list()

        # SECCIÓN 2: PARÁMETROS ALPHA SHAPE
        section2 = ttk.LabelFrame(content_frame, text="Parámetros Alpha Shape", padding=10)
        section2.pack(fill="x", pady=(0, 15))

        # Checkbox para activar/desactivar
        ttk.Checkbutton(
            section2,
            text="Aplicar filtro Alpha Shape (recomendado)",
            variable=self.var_apply_alpha,
            command=self.toggle_alpha_params
        ).pack(anchor="w", pady=(0, 10))

        # Frame para parámetros
        self.alpha_params_frame = ttk.Frame(section2)
        self.alpha_params_frame.pack(fill="x")

        # Probe radius
        probe_frame = ttk.Frame(self.alpha_params_frame)
        probe_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(probe_frame, text="Probe radius:", width=20, anchor="w").pack(side="left")
        ttk.Entry(probe_frame, textvariable=self.var_probe_radius, width=15).pack(side="left")

        # Ghost layers
        ghost_frame = ttk.Frame(self.alpha_params_frame)
        ghost_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(ghost_frame, text="Ghost layers:", width=20, anchor="w").pack(side="left")
        ttk.Spinbox(ghost_frame, from_=1, to=5, textvariable=self.var_num_ghost_layers, width=15).pack(side="left")

        # SECCIÓN 2.5: PARÁMETROS DE CLUSTERING
        section2_5 = ttk.LabelFrame(content_frame, text="Parámetros de Clustering", padding=10)
        section2_5.pack(fill="x", pady=(0, 15))

        # Checkbox para activar/desactivar
        ttk.Checkbutton(
            section2_5,
            text="Aplicar clustering (para aislar nanoporos individuales)",
            variable=self.var_apply_clustering,
            command=self.toggle_clustering_params
        ).pack(anchor="w", pady=(0, 10))

        # Frame para parámetros de clustering
        self.clustering_params_frame = ttk.Frame(section2_5)
        self.clustering_params_frame.pack(fill="x")

        # Método de clustering
        method_frame = ttk.Frame(self.clustering_params_frame)
        method_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(method_frame, text="Método:", width=20, anchor="w").pack(side="left")
        self.method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.var_clustering_method,
            values=["KMeans", "MeanShift", "Aglomerativo", "HDBSCAN"],
            width=18,
            state="readonly"
        )
        self.method_combo.pack(side="left")
        self.method_combo.bind("<<ComboboxSelected>>", self.update_clustering_params)

        # Frame dinámico para parámetros específicos del método
        self.method_params_frame = ttk.Frame(self.clustering_params_frame)
        self.method_params_frame.pack(fill="x", pady=(0, 10))
        self.update_clustering_params()

        # Target cluster
        target_frame = ttk.Frame(self.clustering_params_frame)
        target_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(target_frame, text="Cluster objetivo:", width=20, anchor="w").pack(side="left")
        target_combo = ttk.Combobox(
            target_frame,
            textvariable=self.var_target_cluster,
            values=["largest", "all", "specific"],
            width=18,
            state="readonly"
        )
        target_combo.pack(side="left", padx=(0, 10))
        target_combo.bind("<<ComboboxSelected>>", self.toggle_specific_cluster)

        # Spinbox para cluster específico
        ttk.Label(target_frame, text="Cluster #:", anchor="w").pack(side="left", padx=(10, 5))
        self.specific_cluster_spinbox = ttk.Spinbox(
            target_frame,
            from_=0,
            to=100,
            textvariable=self.var_specific_cluster,
            width=10,
            state="disabled"
        )
        self.specific_cluster_spinbox.pack(side="left")

        # SECCIÓN 3: PARÁMETROS DEL MATERIAL
        section3 = ttk.LabelFrame(content_frame, text="Parámetros del Material", padding=10)
        section3.pack(fill="x", pady=(0, 15))

        # Total atoms
        atoms_frame = ttk.Frame(section3)
        atoms_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(atoms_frame, text="Átomos totales:", width=20, anchor="w").pack(side="left")
        ttk.Entry(atoms_frame, textvariable=self.var_total_atoms, width=15).pack(side="left")

        # a0
        a0_frame = ttk.Frame(section3)
        a0_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(a0_frame, text="Parámetro de red (a0):", width=20, anchor="w").pack(side="left")
        ttk.Entry(a0_frame, textvariable=self.var_a0, width=15).pack(side="left")

        # Lattice type
        lattice_frame = ttk.Frame(section3)
        lattice_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(lattice_frame, text="Tipo de red:", width=20, anchor="w").pack(side="left")
        lattice_combo = ttk.Combobox(
            lattice_frame,
            textvariable=self.var_lattice_type,
            values=["fcc", "bcc", "hcp", "sc", "diamond"],
            width=13,
            state="readonly"
        )
        lattice_combo.pack(side="left")

        # SECCIÓN 4: FEATURES A COMPUTAR
        section4 = ttk.LabelFrame(content_frame, text="Features a Computar (deben coincidir con el modelo)", padding=10)
        section4.pack(fill="x", pady=(0, 15))

        features_frame = ttk.Frame(section4)
        features_frame.pack(fill="x")

        col1 = ttk.Frame(features_frame)
        col1.pack(side="left", fill="both", expand=True, padx=(0, 10))
        ttk.Checkbutton(col1, text="Grid features", variable=self.var_grid).pack(anchor="w", pady=2)
        ttk.Checkbutton(col1, text="Hull features", variable=self.var_hull).pack(anchor="w", pady=2)
        ttk.Checkbutton(col1, text="Inertia moments", variable=self.var_inertia).pack(anchor="w", pady=2)

        col2 = ttk.Frame(features_frame)
        col2.pack(side="left", fill="both", expand=True)
        ttk.Checkbutton(col2, text="Radial features", variable=self.var_radial).pack(anchor="w", pady=2)
        ttk.Checkbutton(col2, text="Entropy", variable=self.var_entropy).pack(anchor="w", pady=2)
        ttk.Checkbutton(col2, text="Clustering", variable=self.var_clustering).pack(anchor="w", pady=2)

        # Botones select all/none
        select_frame = ttk.Frame(section4)
        select_frame.pack(fill="x", pady=(10, 0))
        ttk.Button(select_frame, text="Seleccionar Todos", command=self.select_all_features, width=15).pack(side="left", padx=(0, 5))
        ttk.Button(select_frame, text="Deseleccionar Todos", command=self.deselect_all_features, width=15).pack(side="left")

        # SECCIÓN 5: ACCIONES
        section5 = ttk.LabelFrame(content_frame, text="Acciones", padding=10)
        section5.pack(fill="x", pady=(0, 15))

        buttons_frame = ttk.Frame(section5)
        buttons_frame.pack(fill="x", pady=5)

        # Botón Predecir
        self.predict_button = ttk.Button(
            buttons_frame,
            text="🎯 Predecir Vacancias",
            command=self.predict,
            width=20
        )
        self.predict_button.pack(side="left", padx=5)

        # Botón Salir
        ttk.Button(
            buttons_frame,
            text="❌ Salir",
            command=self.window.destroy,
            width=20
        ).pack(side="right", padx=5)

        # SECCIÓN 6: RESULTADOS
        section6 = ttk.LabelFrame(content_frame, text="Resultados", padding=10)
        section6.pack(fill="both", expand=True, pady=(0, 10))

        self.results_text = scrolledtext.ScrolledText(
            section6,
            height=15,
            width=70,
            font=("Consolas", 9),
            bg="black",
            fg="white",
            insertbackground="white"
        )
        self.results_text.pack(fill="both", expand=True)

        # Configurar colores
        self.results_text.tag_config("HEADER", foreground="cyan", font=("Consolas", 9, "bold"))
        self.results_text.tag_config("SUCCESS", foreground="lightgreen")
        self.results_text.tag_config("ERROR", foreground="red")
        self.results_text.tag_config("INFO", foreground="white")
        self.results_text.tag_config("WARNING", foreground="yellow")

        # Botones para resultados
        results_buttons = ttk.Frame(section6)
        results_buttons.pack(fill="x", pady=(5, 0))
        ttk.Button(results_buttons, text="🧹 Limpiar", command=self.clear_results, width=15).pack(side="left")

        # SECCIÓN 7: ESTADO
        section7 = ttk.LabelFrame(content_frame, text="Estado", padding=10)
        section7.pack(fill="x", pady=(0, 10))

        status_frame = ttk.Frame(section7)
        status_frame.pack(fill="x")

        self.status_label = ttk.Label(
            status_frame,
            text="✅ Listo para predecir",
            font=("Arial", 10),
            foreground="green"
        )
        self.status_label.pack(side="left")

        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode='indeterminate',
            length=200
        )

        # Inicializar estado de controles
        self.toggle_clustering_params()

    def toggle_alpha_params(self):
        """Habilita/deshabilita parámetros de Alpha Shape"""
        if self.var_apply_alpha.get():
            for child in self.alpha_params_frame.winfo_children():
                for widget in child.winfo_children():
                    widget.config(state="normal")
        else:
            for child in self.alpha_params_frame.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Entry, ttk.Spinbox)):
                        widget.config(state="disabled")

    def toggle_clustering_params(self):
        """Habilita/deshabilita parámetros de clustering"""
        state = "normal" if self.var_apply_clustering.get() else "disabled"

        # Habilitar/deshabilitar todos los controles en clustering_params_frame
        for child in self.clustering_params_frame.winfo_children():
            for widget in child.winfo_children():
                if isinstance(widget, (ttk.Entry, ttk.Spinbox, ttk.Combobox)):
                    widget.config(state=state if state == "normal" else "readonly" if isinstance(widget, ttk.Combobox) else "disabled")

    def update_clustering_params(self, event=None):
        """Actualiza los parámetros según el método seleccionado"""
        # Limpiar frame de parámetros
        for widget in self.method_params_frame.winfo_children():
            widget.destroy()

        method = self.var_clustering_method.get()

        if method == "KMeans":
            # Parámetro: n_clusters
            frame = ttk.Frame(self.method_params_frame)
            frame.pack(fill="x")
            ttk.Label(frame, text="Número de clusters:", width=20, anchor="w").pack(side="left")
            ttk.Spinbox(frame, from_=2, to=20, textvariable=self.var_n_clusters, width=10).pack(side="left")

        elif method == "MeanShift":
            # Parámetro: quantile
            frame = ttk.Frame(self.method_params_frame)
            frame.pack(fill="x")
            ttk.Label(frame, text="Quantile (0-1):", width=20, anchor="w").pack(side="left")
            ttk.Entry(frame, textvariable=self.var_quantile, width=10).pack(side="left")

        elif method == "Aglomerativo":
            # Parámetros: n_clusters y linkage
            frame1 = ttk.Frame(self.method_params_frame)
            frame1.pack(fill="x", pady=(0, 5))
            ttk.Label(frame1, text="Número de clusters:", width=20, anchor="w").pack(side="left")
            ttk.Spinbox(frame1, from_=2, to=20, textvariable=self.var_n_clusters, width=10).pack(side="left")

            frame2 = ttk.Frame(self.method_params_frame)
            frame2.pack(fill="x")
            ttk.Label(frame2, text="Linkage:", width=20, anchor="w").pack(side="left")
            ttk.Combobox(
                frame2,
                textvariable=self.var_linkage,
                values=["ward", "complete", "average", "single"],
                width=8,
                state="readonly"
            ).pack(side="left")

        elif method == "HDBSCAN":
            # Parámetro: min_cluster_size
            frame = ttk.Frame(self.method_params_frame)
            frame.pack(fill="x")
            ttk.Label(frame, text="Tamaño mínimo cluster:", width=20, anchor="w").pack(side="left")
            ttk.Spinbox(frame, from_=5, to=100, textvariable=self.var_min_cluster_size, width=10).pack(side="left")

    def toggle_specific_cluster(self, event=None):
        """Habilita/deshabilita el spinbox de cluster específico"""
        if self.var_target_cluster.get() == "specific":
            self.specific_cluster_spinbox.config(state="normal")
        else:
            self.specific_cluster_spinbox.config(state="disabled")

    def select_all_features(self):
        """Selecciona todas las features"""
        self.var_grid.set(True)
        self.var_hull.set(True)
        self.var_inertia.set(True)
        self.var_radial.set(True)
        self.var_entropy.set(True)
        self.var_clustering.set(True)

    def deselect_all_features(self):
        """Deselecciona todas las features"""
        self.var_grid.set(False)
        self.var_hull.set(False)
        self.var_inertia.set(False)
        self.var_radial.set(False)
        self.var_entropy.set(False)
        self.var_clustering.set(False)

    def select_dump_file(self):
        """Selecciona archivo dump"""
        dump_file = filedialog.askopenfilename(
            title="Seleccionar archivo dump",
            filetypes=[("Dump files", "*.dump"), ("All files", "*.*")]
        )
        if dump_file:
            self.var_input_dump.set(dump_file)
            self.log_result(f"Dump seleccionado: {Path(dump_file).name}", "INFO")

    def select_model_file(self):
        """Selecciona modelo entrenado"""
        model_file = filedialog.askopenfilename(
            title="Seleccionar modelo entrenado",
            filetypes=[("Joblib files", "*.joblib"), ("PKL files", "*.pkl"), ("All files", "*.*")]
        )
        if model_file:
            self.var_model_file.set(model_file)
            self.log_result(f"Modelo seleccionado: {Path(model_file).name}", "INFO")

    def refresh_models_list(self):
        """Actualiza la lista de modelos disponibles en models/"""
        from pathlib import Path

        # Obtener carpeta models/
        models_dir = Path(__file__).parent.parent.parent / "models"

        if not models_dir.exists():
            models_dir.mkdir(exist_ok=True)
            self.models_combo["values"] = ["(No hay modelos disponibles)"]
            return

        # Buscar archivos .joblib y .pkl
        model_files = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pkl"))

        if not model_files:
            self.models_combo["values"] = ["(No hay modelos disponibles)"]
        else:
            # Ordenar por fecha de modificación (más reciente primero)
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Guardar paths completos para recuperar después
            self.model_paths = {f.name: str(f) for f in model_files}

            # Mostrar solo nombres en el combo
            self.models_combo["values"] = [f.name for f in model_files]

            # Seleccionar el primero por defecto
            if model_files:
                self.models_combo.current(0)
                self.var_model_file.set(str(model_files[0]))

    def on_model_selected_combo(self, event=None):
        """Cuando se selecciona un modelo del combobox"""
        selected = self.models_combo.get()
        if selected and selected != "(No hay modelos disponibles)":
            if hasattr(self, 'model_paths') and selected in self.model_paths:
                model_path = self.model_paths[selected]
                self.var_model_file.set(model_path)
                self.log_result(f"Modelo seleccionado: {selected}", "INFO")

    def log_result(self, message, level="INFO"):
        """Agrega mensaje al área de resultados"""
        if level == "HEADER":
            formatted = f"\n{'='*70}\n{message}\n{'='*70}\n"
        elif level == "SUCCESS":
            formatted = f"✅ {message}\n"
        elif level == "ERROR":
            formatted = f"❌ {message}\n"
        elif level == "WARNING":
            formatted = f"⚠️  {message}\n"
        else:
            formatted = f"📝 {message}\n"

        self.results_text.insert(tk.END, formatted, level)
        self.results_text.see(tk.END)

    def clear_results(self):
        """Limpia el área de resultados"""
        self.results_text.delete(1.0, tk.END)
        self.log_result("Resultados limpiados", "INFO")

    def update_status(self, message, color="green"):
        """Actualiza el estado"""
        self.status_label.config(text=message, foreground=color)

    def predict(self):
        """Ejecuta la predicción"""
        if self.running:
            return

        # Validaciones
        if not self.var_input_dump.get():
            messagebox.showerror("Error", "Debe seleccionar un archivo dump")
            return

        if not self.var_model_file.get():
            messagebox.showerror("Error", "Debe seleccionar un modelo entrenado")
            return

        if not Path(self.var_input_dump.get()).exists():
            messagebox.showerror("Error", f"El archivo dump no existe:\n{self.var_input_dump.get()}")
            return

        if not Path(self.var_model_file.get()).exists():
            messagebox.showerror("Error", f"El modelo no existe:\n{self.var_model_file.get()}")
            return

        # Confirmar
        clustering_info = ""
        if self.var_apply_clustering.get():
            clustering_info = f"Clustering: Sí ({self.var_clustering_method.get()})\n"
        else:
            clustering_info = "Clustering: No\n"

        confirm = messagebox.askyesno(
            "Confirmar predicción",
            f"¿Está seguro de realizar la predicción?\n\n"
            f"Dump: {Path(self.var_input_dump.get()).name}\n"
            f"Modelo: {Path(self.var_model_file.get()).name}\n"
            f"Alpha Shape: {'Sí' if self.var_apply_alpha.get() else 'No'}\n"
            f"{clustering_info}\n"
            f"Este proceso puede tomar algunos minutos."
        )

        if not confirm:
            return

        # Configurar estado
        self.running = True
        self.predict_button.config(state="disabled", text="Prediciendo...")
        self.update_status("⏳ Prediciendo... (la ventana puede no responder)", "blue")
        self.progress_bar.pack(side="right", padx=(10, 0))
        self.progress_bar.start()

        # Forzar actualización
        self.window.update()

        # Ejecutar después de un delay
        self.window.after(100, self._execute_prediction)

    def _execute_prediction(self):
        """Ejecuta la predicción (en hilo principal)"""
        try:
            # Crear configuración
            config = ExtractorConfig(
                input_dir=".",  # No se usa en predicción
                probe_radius=self.var_probe_radius.get(),
                total_atoms=self.var_total_atoms.get(),
                a0=self.var_a0.get(),
                lattice_type=self.var_lattice_type.get(),
                compute_grid_features=self.var_grid.get(),
                compute_hull_features=self.var_hull.get(),
                compute_inertia_features=self.var_inertia.get(),
                compute_radial_features=self.var_radial.get(),
                compute_entropy_features=self.var_entropy.get(),
                compute_clustering_features=self.var_clustering.get()
            )

            # Crear pipeline
            self.log_result("Inicializando pipeline de predicción...", "HEADER")
            pipeline = PredictionPipeline(
                model_path=self.var_model_file.get(),
                config=config
            )

            # Preparar parámetros de clustering
            clustering_params = None
            if self.var_apply_clustering.get():
                method = self.var_clustering_method.get()
                if method == "KMeans":
                    clustering_params = {'n_clusters': self.var_n_clusters.get()}
                elif method == "MeanShift":
                    clustering_params = {'quantile': self.var_quantile.get()}
                elif method == "Aglomerativo":
                    clustering_params = {
                        'n_clusters': self.var_n_clusters.get(),
                        'linkage': self.var_linkage.get()
                    }
                elif method == "HDBSCAN":
                    clustering_params = {
                        'min_cluster_size': self.var_min_cluster_size.get(),
                        'min_samples': None
                    }

            # Determinar target cluster
            target_cluster = self.var_target_cluster.get()
            if target_cluster == "specific":
                target_cluster = self.var_specific_cluster.get()

            # Predecir
            self.log_result("Ejecutando predicción...", "INFO")
            result = pipeline.predict_single(
                dump_file=self.var_input_dump.get(),
                apply_alpha_shape=self.var_apply_alpha.get(),
                probe_radius=self.var_probe_radius.get(),
                num_ghost_layers=self.var_num_ghost_layers.get(),
                apply_clustering=self.var_apply_clustering.get(),
                clustering_method=self.var_clustering_method.get(),
                clustering_params=clustering_params,
                target_cluster=target_cluster
            )

            # Mostrar resultados
            self.log_result("RESULTADOS DE LA PREDICCIÓN", "HEADER")
            self.log_result(f"Archivo: {result['file']}", "INFO")
            self.log_result(f"Átomos en simulación: {result['num_atoms_in_simulation']}", "INFO")
            self.log_result(f"Átomos superficiales: {result['num_surface_atoms']}", "INFO")

            # Mostrar info de clustering si aplica
            if result.get('clustering_applied') and result.get('clustering_info'):
                cinfo = result['clustering_info']
                self.log_result(f"Clustering aplicado: {cinfo['method']}", "INFO")
                self.log_result(f"Clusters encontrados: {cinfo['n_clusters']}", "INFO")
                if 'target_cluster' in cinfo:
                    self.log_result(f"Cluster procesado: {cinfo['target_cluster']} ({cinfo['cluster_size']} átomos)", "INFO")

            self.log_result(f"Vacancias predichas: {result['predicted_vacancies']:.2f}", "SUCCESS")
            self.log_result(f"Vacancias reales: {result['real_vacancies']}", "INFO")
            self.log_result(f"Error absoluto: {result['error']:.2f}", "WARNING" if result['error'] > 5 else "SUCCESS")

            # Finalizar
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.update_status("✅ Predicción completada", "green")
            self.predict_button.config(state="normal", text="🎯 Predecir Vacancias")
            self.running = False

            messagebox.showinfo(
                "Éxito",
                f"Predicción completada!\n\n"
                f"Vacancias predichas: {result['predicted_vacancies']:.2f}\n"
                f"Vacancias reales: {result['real_vacancies']}\n"
                f"Error: {result['error']:.2f}"
            )

        except Exception as error:
            error_message = str(error)
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.update_status(f"❌ Error en predicción", "red")
            self.predict_button.config(state="normal", text="🎯 Predecir Vacancias")
            self.running = False
            self.log_result(f"Error: {error_message}", "ERROR")
            messagebox.showerror("Error", f"Error en predicción:\n{error_message}")

    def run(self):
        """Ejecuta la GUI"""
        # Atajos de teclado
        self.window.bind('<Escape>', lambda e: self.window.destroy())
        self.window.bind('<F5>', lambda e: self.predict() if not self.running else None)

        # Centrar ventana
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

        # Mensaje inicial
        self.log_result("OpenTopologyC - Predicción de Vacancias", "HEADER")
        self.log_result("Seleccione un archivo dump y un modelo para comenzar.", "INFO")

        # Ejecutar
        self.window.mainloop()


if __name__ == "__main__":
    try:
        gui = PredictionGUI()
        gui.run()
    except Exception as e:
        print(f"Error al iniciar la GUI: {e}")
        import traceback
        traceback.print_exc()
