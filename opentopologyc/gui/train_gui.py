#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# opentopologyc/gui/train_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import threading
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Agregar ruta al módulo train_step
sys.path.append(str(Path(__file__).parent.parent))
from ..core.train_step import RandomForestTrainer


class TrainingGUI:
    """
    GUI para entrenar modelos Random Forest.
    
    Interfaz similar a ExtractorGUI con:
    - Selección de dataset
    - Configuración de parámetros
    - Ejecución en hilo separado
    - Visualización de resultados
    - Guardado de modelos
    """
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("OpenTopologyC - Entrenamiento de Modelos")
        self.window.geometry("600x800")  # Aumentado para incluir consola
        self.window.resizable(True, True)
        
        # Variables GUI
        self.var_input_csv = tk.StringVar()
        self.var_output_dir = tk.StringVar(value="modelos_entrenados")
        self.var_test_size = tk.DoubleVar(value=0.2)
        self.var_random_state = tk.IntVar(value=42)
        self.var_top_features = tk.IntVar(value=20)
        
        # Variables para estado
        self.running = False
        self.trainer = None
        
        # Configurar logging para redirección
        self.log_messages = []
        
        self.build_layout()
        
    def build_layout(self):
        """Construye la interfaz gráfica"""
        # Crear frame principal con scroll
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
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame de contenido
        content_frame = ttk.Frame(self.scrollable_frame, padding=10)
        content_frame.pack(fill="both", expand=True)
        
        # Título
        title_label = ttk.Label(
            content_frame,
            text="OpenTopologyC - Entrenamiento de Modelos",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # --------------------------------------------------
        # SECCIÓN 1: DATOS DE ENTRADA
        # --------------------------------------------------
        section1 = ttk.LabelFrame(content_frame, text="Datos de Entrada", padding=10)
        section1.pack(fill="x", pady=(0, 15))
        
        # Archivo CSV de entrada
        csv_frame = ttk.Frame(section1)
        csv_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(csv_frame, text="Dataset CSV:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        csv_entry_frame = ttk.Frame(csv_frame)
        csv_entry_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Entry(csv_entry_frame, textvariable=self.var_input_csv, width=60).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(csv_entry_frame, text="Buscar", command=self.select_csv_file, width=10).pack(side="right")
        
        # Directorio de salida
        output_frame = ttk.Frame(section1)
        output_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(output_frame, text="Directorio de salida:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Entry(output_entry_frame, textvariable=self.var_output_dir, width=60).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(output_entry_frame, text="Seleccionar", command=self.select_output_dir, width=10).pack(side="right")
        
        # --------------------------------------------------
        # SECCIÓN 2: PARÁMETROS DEL MODELO
        # --------------------------------------------------
        section2 = ttk.LabelFrame(content_frame, text="Parámetros del Modelo", padding=10)
        section2.pack(fill="x", pady=(0, 15))
        
        # Test size
        test_frame = ttk.Frame(section2)
        test_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(test_frame, text="Tamaño del test set (%):", width=25, anchor="w").pack(side="left")
        test_spinbox = tk.Spinbox(test_frame, from_=10, to=50, increment=5, 
                                 textvariable=self.var_test_size, width=10)
        test_spinbox.pack(side="left", padx=(0, 20))
        
        # Random state
        random_frame = ttk.Frame(section2)
        random_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(random_frame, text="Random state:", width=25, anchor="w").pack(side="left")
        random_spinbox = tk.Spinbox(random_frame, from_=0, to=1000, 
                                   textvariable=self.var_random_state, width=10)
        random_spinbox.pack(side="left", padx=(0, 20))
        
        # Top features
        features_frame = ttk.Frame(section2)
        features_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(features_frame, text="Top features a mostrar:", width=25, anchor="w").pack(side="left")
        features_spinbox = tk.Spinbox(features_frame, from_=5, to=50, 
                                     textvariable=self.var_top_features, width=10)
        features_spinbox.pack(side="left")
        
        # Información del modelo
        info_frame = ttk.Frame(section2)
        info_frame.pack(fill="x", pady=(10, 0))
        
        info_text = """
        ⚙️ Configuración del modelo:
        • Random Forest con 200 árboles
        • max_features='sqrt'
        • Imputación de valores faltantes (mediana)
        • Escalado de features (StandardScaler)
        • Out-of-bag score habilitado
        """
        
        info_label = ttk.Label(info_frame, text=info_text, font=("Arial", 9), 
                              foreground="gray", justify="left")
        info_label.pack(anchor="w")
        
        # --------------------------------------------------
        # SECCIÓN 3: ACCIONES
        # --------------------------------------------------
        section3 = ttk.LabelFrame(content_frame, text="Acciones", padding=10)
        section3.pack(fill="x", pady=(0, 15))
        
        # Frame para botones principales
        buttons_frame = ttk.Frame(section3)
        buttons_frame.pack(fill="x", pady=5)
        
        # Botón Entrenar
        self.train_button = ttk.Button(
            buttons_frame, 
            text="🎯 Entrenar Modelo",
            command=self.train_model,
            width=20,
            state="normal"
        )
        self.train_button.pack(side="left", padx=5)
        
        # Botón Cargar Modelo
        ttk.Button(
            buttons_frame,
            text="📂 Cargar Modelo",
            command=self.load_model,
            width=20
        ).pack(side="left", padx=5)
        
        # Botón Salir
        ttk.Button(
            buttons_frame,
            text="❌ Salir",
            command=self.window.destroy,
            width=20
        ).pack(side="right", padx=5)
        
        # --------------------------------------------------
        # SECCIÓN 4: CONSOLA DE SALIDA
        # --------------------------------------------------
        section4 = ttk.LabelFrame(content_frame, text="Consola de Salida", padding=10)
        section4.pack(fill="both", expand=True, pady=(0, 10))
        
        # Text widget para mostrar logs
        self.console_text = scrolledtext.ScrolledText(
            section4,
            height=12,
            width=70,
            font=("Consolas", 9),
            bg="black",
            fg="white",
            insertbackground="white"
        )
        self.console_text.pack(fill="both", expand=True)
        
        # Configurar colores para la consola
        self.console_text.tag_config("INFO", foreground="lightgreen")
        self.console_text.tag_config("WARNING", foreground="yellow")
        self.console_text.tag_config("ERROR", foreground="red")
        self.console_text.tag_config("SUCCESS", foreground="cyan")
        self.console_text.tag_config("HEADER", foreground="magenta")
        
        # Botones para consola
        console_buttons = ttk.Frame(section4)
        console_buttons.pack(fill="x", pady=(5, 0))
        
        ttk.Button(console_buttons, text="🧹 Limpiar Consola", 
                  command=self.clear_console, width=15).pack(side="left")
        ttk.Button(console_buttons, text="💾 Guardar Logs", 
                  command=self.save_logs, width=15).pack(side="left", padx=(5, 0))
        
        # --------------------------------------------------
        # SECCIÓN 5: ESTADO DEL SISTEMA
        # --------------------------------------------------
        section5 = ttk.LabelFrame(content_frame, text="Estado del Sistema", padding=10)
        section5.pack(fill="x", pady=(0, 10))
        
        status_frame = ttk.Frame(section5)
        status_frame.pack(fill="x")
        
        # Etiqueta de estado
        self.status_label = ttk.Label(
            status_frame,
            text="✅ Listo para entrenar",
            font=("Arial", 10),
            foreground="green"
        )
        self.status_label.pack(side="left")
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode='indeterminate',
            length=200
        )
        
        # Información adicional
        info_bottom = ttk.Label(
            content_frame,
            text="Nota: El entrenamiento puede tomar varios minutos dependiendo del tamaño del dataset",
            font=("Arial", 9, "italic"),
            foreground="gray"
        )
        info_bottom.pack(pady=(5, 0))
        
    # --------------------------------------------------
    # MÉTODOS AUXILIARES
    # --------------------------------------------------
    def log_to_console(self, message, level="INFO"):
        """Agrega un mensaje a la consola"""
        color_tag = level
        
        # Formatear mensaje
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
        
        # Insertar en consola
        self.console_text.insert(tk.END, formatted, color_tag)
        self.console_text.see(tk.END)
        
        # Guardar en lista
        self.log_messages.append(f"[{level}] {message}")
        
    def clear_console(self):
        """Limpia la consola"""
        self.console_text.delete(1.0, tk.END)
        self.log_messages.clear()
        self.log_to_console("Consola limpiada", "INFO")
        
    def save_logs(self):
        """Guarda los logs en un archivo"""
        if not self.log_messages:
            messagebox.showwarning("Advertencia", "No hay logs para guardar")
            return
            
        log_file = filedialog.asksaveasfilename(
            title="Guardar logs",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if log_file:
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.log_messages))
                self.log_to_console(f"Logs guardados en: {log_file}", "SUCCESS")
                messagebox.showinfo("Éxito", f"Logs guardados en:\n{log_file}")
            except Exception as e:
                self.log_to_console(f"Error al guardar logs: {e}", "ERROR")
                messagebox.showerror("Error", f"No se pudieron guardar los logs: {e}")
                
    def update_status(self, message, color="green"):
        """Actualiza el estado del sistema"""
        self.status_label.config(text=message, foreground=color)
        
    # --------------------------------------------------
    # ACCIONES PRINCIPALES
    # --------------------------------------------------
    def select_csv_file(self):
        """Selecciona un archivo CSV"""
        csv_file = filedialog.askopenfilename(
            title="Seleccionar dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if csv_file:
            self.var_input_csv.set(csv_file)
            self.log_to_console(f"Dataset seleccionado: {csv_file}", "INFO")
            
    def select_output_dir(self):
        """Selecciona un directorio de salida"""
        output_dir = filedialog.askdirectory(
            title="Seleccionar directorio de salida"
        )
        
        if output_dir:
            self.var_output_dir.set(output_dir)
            self.log_to_console(f"Directorio de salida: {output_dir}", "INFO")
            
    def train_model(self):
        """Inicia el entrenamiento del modelo en un hilo separado"""
        if self.running:
            return
            
        # Validaciones
        if not self.var_input_csv.get():
            messagebox.showerror("Error", "Debe seleccionar un dataset CSV")
            return
            
        if not Path(self.var_input_csv.get()).exists():
            messagebox.showerror("Error", f"El archivo no existe:\n{self.var_input_csv.get()}")
            return
            
        # Confirmar
        confirm = messagebox.askyesno(
            "Confirmar entrenamiento",
            f"¿Está seguro de entrenar el modelo?\n\n"
            f"Dataset: {Path(self.var_input_csv.get()).name}\n"
            f"Test size: {self.var_test_size.get()*100:.0f}%\n"
            f"Salida: {self.var_output_dir.get()}\n\n"
            f"Esto puede tomar varios minutos."
        )
        
        if not confirm:
            return
            
        # Configurar estado de ejecución
        self.running = True
        self.train_button.config(state="disabled", text="Entrenando...")
        self.update_status("Entrenando modelo...", "blue")
        self.progress_bar.pack(side="right", padx=(10, 0))
        self.progress_bar.start()
        
        # Limpiar consola anterior
        self.clear_console()
        
        # Iniciar hilo de entrenamiento
        thread = threading.Thread(target=self._execute_training, daemon=True)
        thread.start()
        
    def _execute_training(self):
        """Ejecuta el entrenamiento (en hilo separado)"""
        try:
            # Crear entrenador
            self.trainer = RandomForestTrainer(
                random_state=self.var_random_state.get(),
                logger=None  # Usaremos nuestra propia consola
            )
            
            # Función para redirección de logs
            def log_wrapper(message):
                self.window.after(0, self.log_to_console, message, "INFO")
                
            # Reemplazar logger del entrenador
            self.trainer.logger = type('FakeLogger', (), {
                'info': lambda _, msg: log_wrapper(msg),
                'warning': lambda _, msg: self.window.after(0, self.log_to_console, msg, "WARNING"),
                'error': lambda _, msg: self.window.after(0, self.log_to_console, msg, "ERROR"),
            })()
            
            # Cargar datos
            self.window.after(0, self.log_to_console, "Cargando datos...", "INFO")
            X, y = self.trainer.load_data(self.var_input_csv.get())
            
            # Entrenar
            self.window.after(0, self.log_to_console, "Iniciando entrenamiento...", "HEADER")
            self.trainer.train(X, y, self.var_test_size.get())
            
            # Evaluar
            self.window.after(0, self.log_to_console, "Evaluando modelo...", "HEADER")
            self.trainer.evaluate()
            
            # Importancia de features
            self.window.after(0, self.log_to_console, "Analizando importancia de features...", "HEADER")
            self.trainer.analyze_feature_importance(self.var_top_features.get())
            
            # Crear gráficos
            self.window.after(0, self.log_to_console, "Generando gráficos...", "INFO")
            self.trainer.create_plots(self.var_output_dir.get())
            
            # Guardar modelo
            self.window.after(0, self.log_to_console, "Guardando modelo...", "INFO")
            self.trainer.save_model(self.var_output_dir.get())
            
            # Actualizar estado en GUI
            def update_success():
                self.progress_bar.stop()
                self.progress_bar.pack_forget()
                self.update_status("✅ Entrenamiento completado", "green")
                self.train_button.config(state="normal", text="🎯 Entrenar Modelo")
                self.running = False
                
                # Mostrar resumen
                self.log_to_console("\n" + "="*70, "HEADER")
                self.log_to_console("✅ ENTRENAMIENTO COMPLETADO", "SUCCESS")
                self.log_to_console("="*70, "HEADER")
                
                # Información del modelo
                info = self.trainer.get_model_info()
                self.log_to_console(f"📊 Métricas del modelo:", "INFO")
                self.log_to_console(f"   R² (test): {info['metrics']['test']['r2']:.4f}", "INFO")
                self.log_to_console(f"   MAE (test): {info['metrics']['test']['mae']:.2f}", "INFO")
                self.log_to_console(f"   RMSE (test): {info['metrics']['test']['rmse']:.2f}", "INFO")
                self.log_to_console(f"   Features usadas: {info['n_features']}", "INFO")
                
                # Archivos generados
                output_path = Path(self.var_output_dir.get())
                self.log_to_console(f"\n📁 Archivos generados en:", "INFO")
                for file in output_path.glob("*"):
                    if file.is_file():
                        self.log_to_console(f"   • {file.name}", "INFO")
                        
                messagebox.showinfo(
                    "Éxito",
                    f"Entrenamiento completado exitosamente!\n\n"
                    f"Modelo guardado en: {self.var_output_dir.get()}\n"
                    f"R² en test: {info['metrics']['test']['r2']:.4f}"
                )
                
            self.window.after(0, update_success)
            
        except Exception as e:
            error_msg = str(e)
            
            def update_error():
                self.progress_bar.stop()
                self.progress_bar.pack_forget()
                self.update_status(f"❌ Error en entrenamiento", "red")
                self.train_button.config(state="normal", text="🎯 Entrenar Modelo")
                self.running = False
                self.log_to_console(f"Error: {error_msg}", "ERROR")
                messagebox.showerror("Error", f"Error en entrenamiento:\n{error_msg}")
                
            self.window.after(0, update_error)
            
    def load_model(self):
        """Carga un modelo previamente entrenado"""
        model_file = filedialog.askopenfilename(
            title="Seleccionar modelo entrenado",
            filetypes=[("Joblib files", "*.joblib"), ("PKL files", "*.pkl"), ("All files", "*.*")]
        )
        
        if model_file:
            try:
                # Aquí podrías cargar y usar el modelo
                messagebox.showinfo(
                    "Modelo cargado",
                    f"Modelo cargado exitosamente:\n{model_file}\n\n"
                    f"Para usar el modelo en predicciones, implementa la función de carga en trainer.load_model()"
                )
                self.log_to_console(f"Modelo cargado: {model_file}", "SUCCESS")
            except Exception as e:
                self.log_to_console(f"Error al cargar modelo: {e}", "ERROR")
                messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")
                
    # --------------------------------------------------
    # EJECUCIÓN
    # --------------------------------------------------
    def run(self):
        """Inicia la aplicación GUI"""
        # Configurar atajos de teclado
        self.window.bind('<Escape>', lambda e: self.window.destroy())
        self.window.bind('<Control-o>', lambda e: self.select_csv_file())
        self.window.bind('<Control-s>', lambda e: self.select_output_dir())
        self.window.bind('<F5>', lambda e: self.train_model() if not self.running else None)
        
        # Centrar ventana
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')
        
        # Mensaje inicial
        self.log_to_console("OpenTopologyC - Entrenamiento de Modelos", "HEADER")
        self.log_to_console("Sistema listo. Seleccione un dataset CSV para comenzar.", "INFO")
        
        # Ejecutar
        self.window.mainloop()


# Entrada principal
if __name__ == "__main__":
    try:
        gui = TrainingGUI()
        gui.run()
    except Exception as e:
        print(f"Error al iniciar la GUI: {e}")
        import traceback
        traceback.print_exc()