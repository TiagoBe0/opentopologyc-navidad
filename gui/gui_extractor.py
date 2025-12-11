# opentopologyc/gui/gui_extractor.py

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import os
import warnings

# Ignorar warning de OVITO temporalmente
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

from config.extractor_config import ExtractorConfig
try:
    from core.pipeline import ExtractorPipeline  # Importar el pipeline
except ImportError as e:
    print(f"Advertencia: No se pudo importar ExtractorPipeline: {e}")
    ExtractorPipeline = None


class ExtractorGUI:
    """
    GUI sencilla para configurar parámetros del extractor.
    """

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("OpenTopologyC - Configuración Inicial")
        self.window.geometry("420x600")
        self.window.resizable(False, False)

        # Variables GUI
        self.var_input_dir = tk.StringVar()
        self.var_probe_radius = tk.DoubleVar(value=2.0)
        self.var_surface_distance = tk.BooleanVar(value=False)
        self.var_surface_distance_value = tk.DoubleVar(value=4.0)
        # Boolean features
        self.var_grid = tk.BooleanVar(value=True)
        self.var_hull = tk.BooleanVar(value=True)
        self.var_inertia = tk.BooleanVar(value=True)
        self.var_radial = tk.BooleanVar(value=True)
        self.var_entropy = tk.BooleanVar(value=True)
        self.var_clustering = tk.BooleanVar(value=True)

        # Variable para estado del botón Run
        self.running = False

        self.build_layout()

    # ---------------------------------------------------
    # Construcción de la Interfaz
    # ---------------------------------------------------
    def build_layout(self):
        frame = ttk.Frame(self.window, padding=15)
        frame.pack(fill="both", expand=True)

        # --------------------------------------
        # DIRECTORIO
        # --------------------------------------
        ttk.Label(frame, text="Directorio de dumps:", font=("Arial", 11, "bold")).pack(anchor="w", pady=5)
        
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill="x")

        ttk.Entry(dir_frame, textvariable=self.var_input_dir, width=40).pack(side="left", padx=5)
        ttk.Button(dir_frame, text="Buscar", command=self.select_directory).pack(side="left")

        # --------------------------------------
        # RADIO DE SONDA
        # --------------------------------------
        ttk.Label(frame, text="Radio de sonda (alpha / surface):", font=("Arial", 11, "bold")).pack(anchor="w", pady=10)

        ttk.Entry(frame, textvariable=self.var_probe_radius, width=10).pack(anchor="w", padx=5)
        ttk.Checkbutton(frame, text="surface distance", variable=self.var_surface_distance).pack(anchor="w")
        ttk.Label(frame, text="surface distance value:", font=("Arial", 11, "bold")).pack(anchor="w", pady=10)
        ttk.Entry(frame, textvariable=self.var_surface_distance_value, width=10).pack(anchor="w", padx=5)
        
        # --------------------------------------
        # FEATURES
        # --------------------------------------
        ttk.Label(frame, text="Cálculo de features:", font=("Arial", 11, "bold")).pack(anchor="w", pady=10)

        ttk.Checkbutton(frame, text="Grid features", variable=self.var_grid).pack(anchor="w")
        ttk.Checkbutton(frame, text="Hull (Convex Hull)", variable=self.var_hull).pack(anchor="w")
        ttk.Checkbutton(frame, text="Inertia moments", variable=self.var_inertia).pack(anchor="w")
        ttk.Checkbutton(frame, text="Radial features", variable=self.var_radial).pack(anchor="w")
        ttk.Checkbutton(frame, text="Entropy", variable=self.var_entropy).pack(anchor="w")
        ttk.Checkbutton(frame, text="Clustering / Bandwidth", variable=self.var_clustering).pack(anchor="w")

        # --------------------------------------
        # BOTONES
        # --------------------------------------
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(pady=20)

        # Botón para crear configuración
        ttk.Button(buttons_frame, text="Crear Configuración", 
                  command=self.create_config).pack(side="left", padx=5)
        
        # Nuevo botón RUN - deshabilitado inicialmente
        self.run_button = ttk.Button(buttons_frame, text="Run Pipeline", 
                                    command=self.run_pipeline, state="disabled")
        self.run_button.pack(side="left", padx=5)

        # Botón para cargar configuración existente
        ttk.Button(buttons_frame, text="Cargar Configuración", 
                  command=self.load_config).pack(side="left", padx=5)

        ttk.Button(frame, text="Salir", command=self.window.quit).pack()

        # Frame para estado/progreso
        self.status_frame = ttk.LabelFrame(frame, text="Estado", padding=10)
        self.status_frame.pack(fill="x", pady=10)

        self.status_label = ttk.Label(self.status_frame, text="Listo")
        self.status_label.pack()

    # ---------------------------------------------------
    # Acciones
    # ---------------------------------------------------
    def select_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.var_input_dir.set(directory)

    def create_config(self):
        try:
            # Validar que hay un directorio
            if not self.var_input_dir.get():
                messagebox.showerror("Error", "Debe seleccionar un directorio")
                return

            # Crear configuración
            cfg = ExtractorConfig(
                input_dir=self.var_input_dir.get(),
                probe_radius=self.var_probe_radius.get(),
                surface_distance=self.var_surface_distance.get(),
                surface_distance_value=self.var_surface_distance_value.get(),
                compute_grid_features=self.var_grid.get(),
                compute_hull_features=self.var_hull.get(),
                compute_inertia_features=self.var_inertia.get(),
                compute_radial_features=self.var_radial.get(),
                compute_entropy_features=self.var_entropy.get(),
                compute_clustering_features=self.var_clustering.get(),
            )
            cfg.validate()

            # Guardar configuración
            cfg.save_json("config_extractor.json")
            
            messagebox.showinfo("Éxito", "Configuración creada y guardada correctamente")
            
            # Habilitar botón Run
            self.run_button.config(state="normal")
            self.update_status("Configuración guardada. Listo para ejecutar.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

    def load_config(self):
        """Cargar configuración desde archivo existente"""
        try:
            config_file = filedialog.askopenfilename(
                title="Seleccionar archivo de configuración",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if config_file:
                cfg = ExtractorConfig.load_json(config_file)
                
                # Actualizar variables GUI
                self.var_input_dir.set(cfg.input_dir)
                self.var_probe_radius.set(cfg.probe_radius)
                self.var_surface_distance.set(cfg.surface_distance)
                self.var_surface_distance_value.set(cfg.surface_distance_value)
                self.var_grid.set(cfg.compute_grid_features)
                self.var_hull.set(cfg.compute_hull_features)
                self.var_inertia.set(cfg.compute_inertia_features)
                self.var_radial.set(cfg.compute_radial_features)
                self.var_entropy.set(cfg.compute_entropy_features)
                self.var_clustering.set(cfg.compute_clustering_features)
                
                # Guardar como archivo por defecto
                cfg.save_json("config_extractor.json")
                
                messagebox.showinfo("Éxito", "Configuración cargada correctamente")
                
                # Habilitar botón Run
                self.run_button.config(state="normal")
                self.update_status("Configuración cargada. Listo para ejecutar.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar configuración: {str(e)}")

    def run_pipeline(self):
        """Ejecutar el pipeline en un hilo separado"""
        if self.running:
            return
        
        # Verificar que existe configuración
        if not os.path.exists("config_extractor.json"):
            messagebox.showerror("Error", "Primero debe crear o cargar una configuración")
            return
        
        # Verificar que el pipeline está disponible
        if ExtractorPipeline is None:
            messagebox.showerror("Error", "No se puede cargar el módulo ExtractorPipeline")
            return
        
        # Crear hilo para ejecutar pipeline
        self.running = True
        self.run_button.config(state="disabled", text="Ejecutando...")
        self.update_status("Ejecutando pipeline...")
        
        thread = threading.Thread(target=self._execute_pipeline, daemon=True)
        thread.start()

    def _execute_pipeline(self):
        """Función que ejecuta el pipeline (en hilo separado)"""
        try:
            # Cargar configuración
            cfg = ExtractorConfig.load_json("config_extractor.json")
            
            # Crear y ejecutar pipeline
            pipeline = ExtractorPipeline(cfg)
            
            # Actualizar GUI desde hilo principal
            def update_success():
                self.update_status("Pipeline ejecutado exitosamente")
                self.run_button.config(state="normal", text="Run Pipeline")
                self.running = False
                messagebox.showinfo("Éxito", "Pipeline ejecutado exitosamente")
            
            # Ejecutar pipeline
            result = pipeline.run()
            
            # Actualizar interfaz en hilo principal
            self.window.after(0, update_success)
            
        except Exception as error:
            # Manejar errores en hilo principal
            error_message = str(error)
            
            def update_error():
                self.update_status(f"Error: {error_message}")
                self.run_button.config(state="normal", text="Run Pipeline")
                self.running = False
                messagebox.showerror("Error", f"Error en pipeline: {error_message}")
            
            self.window.after(0, update_error)

    def update_status(self, message):
        """Actualizar mensaje de estado"""
        self.status_label.config(text=message)

    # ---------------------------------------------------
    # Ejecución
    # ---------------------------------------------------
    def run(self):
        self.window.mainloop()


# Entrada directa
if __name__ == "__main__":
    gui = ExtractorGUI()
    gui.run()