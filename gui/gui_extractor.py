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
    GUI para configurar parámetros del extractor.
    """

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("OpenTopologyC - Configuración Inicial")
        self.window.geometry("550x750")  # Aumentado el tamaño
        self.window.resizable(True, True)  # Permitir redimensionar

        # Variables GUI
        self.var_input_dir = tk.StringVar()
        self.var_probe_radius = tk.DoubleVar(value=2.0)
        self.var_surface_distance = tk.BooleanVar(value=False)
        self.var_surface_distance_value = tk.DoubleVar(value=4.0)
        # Training parameters
        self.var_total_atoms = tk.IntVar(value=16384)
        self.var_a0 = tk.DoubleVar(value=3.532)
        self.var_lattice_type = tk.StringVar(value="fcc")
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
        # Crear un canvas con scrollbar
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Canvas para scroll
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Empacar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame para todo el contenido
        content_frame = ttk.Frame(self.scrollable_frame, padding=10)
        content_frame.pack(fill="both", expand=True)
        
        # Título
        title_label = ttk.Label(
            content_frame, 
            text="OpenTopologyC - Extractor de Features",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # --------------------------------------
        # SECCIÓN 1: DIRECTORIO
        # --------------------------------------
        section1 = ttk.LabelFrame(content_frame, text="Directorio de Datos", padding=10)
        section1.pack(fill="x", pady=(0, 15))
        
        # Directorio de entrada
        dir_label = ttk.Label(section1, text="Directorio de dumps:", font=("Arial", 10, "bold"))
        dir_label.pack(anchor="w", pady=(0, 5))
        
        dir_frame = ttk.Frame(section1)
        dir_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Entry(dir_frame, textvariable=self.var_input_dir, width=60).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(dir_frame, text="Buscar", command=self.select_directory, width=10).pack(side="right")
        
        # --------------------------------------
        # SECCIÓN 2: PARÁMETROS DE EXTRACCIÓN
        # --------------------------------------
        section2 = ttk.LabelFrame(content_frame, text="Parámetros de Extracción", padding=10)
        section2.pack(fill="x", pady=(0, 15))
        
        # Radio de sonda
        probe_frame = ttk.Frame(section2)
        probe_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(probe_frame, text="Radio de sonda:", width=20, anchor="w").pack(side="left")
        ttk.Entry(probe_frame, textvariable=self.var_probe_radius, width=15).pack(side="left", padx=(0, 20))
        
        # Surface distance
        surface_frame = ttk.Frame(section2)
        surface_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Checkbutton(
            surface_frame, 
            text="Usar surface distance", 
            variable=self.var_surface_distance,
            command=self.toggle_surface_distance
        ).pack(side="left")
        
        # Surface distance value (inicialmente deshabilitado)
        self.surface_value_frame = ttk.Frame(section2)
        self.surface_value_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(self.surface_value_frame, text="Valor surface distance:", width=20, anchor="w").pack(side="left")
        self.surface_value_entry = ttk.Entry(self.surface_value_frame, textvariable=self.var_surface_distance_value, width=15, state="disabled")
        self.surface_value_entry.pack(side="left")
        
        # --------------------------------------
        # SECCIÓN 3: PARÁMETROS DEL MATERIAL
        # --------------------------------------
        section3 = ttk.LabelFrame(content_frame, text="Parámetros del Material", padding=10)
        section3.pack(fill="x", pady=(0, 15))
        
        # Total atoms
        atoms_frame = ttk.Frame(section3)
        atoms_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(atoms_frame, text="Átomos totales:", width=20, anchor="w").pack(side="left")
        ttk.Entry(atoms_frame, textvariable=self.var_total_atoms, width=15).pack(side="left", padx=(0, 20))
        
        # a0 parameter
        a0_frame = ttk.Frame(section3)
        a0_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(a0_frame, text="Parámetro de red (a0):", width=20, anchor="w").pack(side="left")
        ttk.Entry(a0_frame, textvariable=self.var_a0, width=15).pack(side="left", padx=(0, 20))
        
        # Lattice type
        lattice_frame = ttk.Frame(section3)
        lattice_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(lattice_frame, text="Tipo de red:", width=20, anchor="w").pack(side="left")
        
        # ComboBox para lattice type
        lattice_types = ["fcc", "bcc", "hcp", "sc", "diamond"]
        lattice_combo = ttk.Combobox(lattice_frame, textvariable=self.var_lattice_type, 
                                    values=lattice_types, width=13, state="readonly")
        lattice_combo.pack(side="left")
        
        # --------------------------------------
        # SECCIÓN 4: FEATURES A CALCULAR
        # --------------------------------------
        section4 = ttk.LabelFrame(content_frame, text="Features a Calcular", padding=10)
        section4.pack(fill="x", pady=(0, 15))
        
        # Crear grid de checkbuttons
        features_frame = ttk.Frame(section4)
        features_frame.pack(fill="x")
        
        # Columna 1
        col1 = ttk.Frame(features_frame)
        col1.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ttk.Checkbutton(col1, text="Grid features", variable=self.var_grid).pack(anchor="w", pady=2)
        ttk.Checkbutton(col1, text="Hull (Convex Hull)", variable=self.var_hull).pack(anchor="w", pady=2)
        ttk.Checkbutton(col1, text="Inertia moments", variable=self.var_inertia).pack(anchor="w", pady=2)
        
        # Columna 2
        col2 = ttk.Frame(features_frame)
        col2.pack(side="left", fill="both", expand=True)
        
        ttk.Checkbutton(col2, text="Radial features", variable=self.var_radial).pack(anchor="w", pady=2)
        ttk.Checkbutton(col2, text="Entropy", variable=self.var_entropy).pack(anchor="w", pady=2)
        ttk.Checkbutton(col2, text="Clustering / Bandwidth", variable=self.var_clustering).pack(anchor="w", pady=2)
        
        # Botón para seleccionar/deseleccionar todos
        select_frame = ttk.Frame(section4)
        select_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Button(select_frame, text="Seleccionar Todos", 
                  command=self.select_all_features, width=15).pack(side="left", padx=(0, 5))
        ttk.Button(select_frame, text="Deseleccionar Todos", 
                  command=self.deselect_all_features, width=15).pack(side="left")
        
        # --------------------------------------
        # SECCIÓN 5: BOTONES DE ACCIÓN
        # --------------------------------------
        section5 = ttk.LabelFrame(content_frame, text="Acciones", padding=10)
        section5.pack(fill="x", pady=(0, 15))
        
        # Frame para botones principales
        buttons_frame = ttk.Frame(section5)
        buttons_frame.pack(fill="x", pady=5)
        
        # Botón para crear configuración
        ttk.Button(buttons_frame, text="💾 Crear Configuración", 
                  command=self.create_config, width=20).pack(side="left", padx=5)
        
        # Botón RUN - deshabilitado inicialmente
        self.run_button = ttk.Button(buttons_frame, text="🚀 Run Pipeline", 
                                    command=self.run_pipeline, state="disabled", width=20)
        self.run_button.pack(side="left", padx=5)
        
        # Botón para cargar configuración existente
        ttk.Button(buttons_frame, text="📂 Cargar Configuración", 
                  command=self.load_config, width=20).pack(side="left", padx=5)
        
        # Frame para botones secundarios
        secondary_buttons = ttk.Frame(section5)
        secondary_buttons.pack(fill="x", pady=(10, 0))

        ttk.Button(secondary_buttons, text="❌ Salir",
                  command=self.window.destroy, width=20).pack(side="right")
        
        # --------------------------------------
        # SECCIÓN 6: ESTADO
        # --------------------------------------
        section6 = ttk.LabelFrame(content_frame, text="Estado del Sistema", padding=10)
        section6.pack(fill="x", pady=(0, 10))
        
        # Frame para estado
        status_display = ttk.Frame(section6)
        status_display.pack(fill="x")
        
        # Etiqueta de estado
        self.status_label = ttk.Label(status_display, text="✅ Listo", 
                                     font=("Arial", 10), foreground="green")
        self.status_label.pack(side="left")
        
        # Progress bar (inicialmente oculta)
        self.progress_bar = ttk.Progressbar(status_display, mode='indeterminate', length=200)
        
        # Información adicional
        info_label = ttk.Label(
            content_frame, 
            text="Nota: Los parámetros del material se usarán para normalizar las coordenadas",
            font=("Arial", 9, "italic"),
            foreground="gray"
        )
        info_label.pack(pady=(5, 0))

    # ---------------------------------------------------
    # Métodos auxiliares
    # ---------------------------------------------------
    def toggle_surface_distance(self):
        """Habilita/deshabilita el campo de surface distance value"""
        if self.var_surface_distance.get():
            self.surface_value_entry.config(state="normal")
        else:
            self.surface_value_entry.config(state="disabled")
    
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

    # ---------------------------------------------------
    # Acciones principales
    # ---------------------------------------------------
    def select_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.var_input_dir.set(directory)
            self.update_status(f"Directorio seleccionado: {directory}")

    def create_config(self):
        try:
            # Validar que hay un directorio
            if not self.var_input_dir.get():
                messagebox.showerror("Error", "Debe seleccionar un directorio")
                return
            
            # Validar parámetros del material
            if self.var_total_atoms.get() <= 0:
                messagebox.showerror("Error", "El número de átomos debe ser positivo")
                return
            
            if self.var_a0.get() <= 0:
                messagebox.showerror("Error", "El parámetro de red (a0) debe ser positivo")
                return
            
            if not self.var_lattice_type.get():
                messagebox.showerror("Error", "Debe seleccionar un tipo de red")
                return

            # Crear configuración extendida
            cfg = ExtractorConfig(
                input_dir=self.var_input_dir.get(),
                probe_radius=self.var_probe_radius.get(),
                surface_distance=self.var_surface_distance.get(),
                surface_distance_value=self.var_surface_distance_value.get(),
                total_atoms=self.var_total_atoms.get(),
                a0=self.var_a0.get(),
                lattice_type=self.var_lattice_type.get(),
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
                self.var_total_atoms.set(getattr(cfg, 'total_atoms', 16384))
                self.var_a0.set(getattr(cfg, 'a0', 3.532))
                self.var_lattice_type.set(getattr(cfg, 'lattice_type', 'fcc'))
                self.var_grid.set(cfg.compute_grid_features)
                self.var_hull.set(cfg.compute_hull_features)
                self.var_inertia.set(cfg.compute_inertia_features)
                self.var_radial.set(cfg.compute_radial_features)
                self.var_entropy.set(cfg.compute_entropy_features)
                self.var_clustering.set(cfg.compute_clustering_features)
                
                # Actualizar estado del campo surface distance
                self.toggle_surface_distance()
                
                # Guardar como archivo por defecto
                cfg.save_json("config_extractor.json")
                
                messagebox.showinfo("Éxito", "Configuración cargada correctamente")
                
                # Habilitar botón Run
                self.run_button.config(state="normal")
                self.update_status("Configuración cargada. Listo para ejecutar.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar configuración: {str(e)}")

    def run_pipeline(self):
        """Ejecutar el pipeline directamente en el hilo principal"""
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

        # Confirmar ejecución con advertencia
        confirm = messagebox.askyesno(
            "Confirmar ejecución",
            "¿Está seguro de ejecutar el pipeline?\n\n"
            "Esto puede tomar varios minutos.\n"
            "La ventana puede no responder durante el proceso.\n\n"
            "IMPORTANTE: No cierre la aplicación hasta que termine."
        )

        if not confirm:
            return

        # Configurar estado de ejecución
        self.running = True
        self.run_button.config(state="disabled", text="Ejecutando...")
        self.update_status("⏳ Ejecutando pipeline... (La ventana puede no responder)")
        self.progress_bar.pack(side="right", padx=(10, 0))
        self.progress_bar.start()

        # Forzar actualización de la GUI antes de empezar
        self.window.update()

        # Ejecutar pipeline después de un pequeño delay para que se actualice la GUI
        self.window.after(100, self._execute_pipeline)

    def _execute_pipeline(self):
        """Función que ejecuta el pipeline directamente"""
        try:
            # Cargar configuración
            cfg = ExtractorConfig.load_json("config_extractor.json")

            # Crear y ejecutar pipeline
            pipeline = ExtractorPipeline(cfg)

            # Ejecutar pipeline (directo, sin threading)
            result = pipeline.run()

            # Actualizar interfaz
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.update_status("✅ Pipeline ejecutado exitosamente")
            self.run_button.config(state="normal", text="🚀 Run Pipeline")
            self.running = False
            messagebox.showinfo("Éxito", "Pipeline ejecutado exitosamente!\n\nRevisa el archivo dataset_features.csv generado.")

        except Exception as error:
            # Manejar errores
            error_message = str(error)
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.update_status(f"❌ Error: {error_message}")
            self.run_button.config(state="normal", text="🚀 Run Pipeline")
            self.running = False
            messagebox.showerror("Error", f"Error en pipeline:\n{error_message}")

    def update_status(self, message):
        """Actualizar mensaje de estado"""
        self.status_label.config(text=message)
        
        # Cambiar color según el estado
        if "Error" in message or "❌" in message:
            self.status_label.config(foreground="red")
        elif "Listo" in message or "✅" in message:
            self.status_label.config(foreground="green")
        else:
            self.status_label.config(foreground="blue")

    # ---------------------------------------------------
    # Ejecución
    # ---------------------------------------------------
    def run(self):
        # Configurar manejo de teclas
        self.window.bind('<Escape>', lambda e: self.window.destroy())
        self.window.bind('<Return>', lambda e: self.run_pipeline() if self.run_button['state'] == 'normal' else None)
        
        # Centrar ventana
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')
        
        # Ejecutar
        self.window.mainloop()


# Entrada directa
if __name__ == "__main__":
    gui = ExtractorGUI()
    gui.run()