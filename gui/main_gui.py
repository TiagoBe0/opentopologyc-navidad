#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# opentopologyc/gui/main_gui.py

import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

# Agregar rutas
sys.path.append(str(Path(__file__).parent.parent))

try:
    from gui.gui_extractor import ExtractorGUI
    from gui.train_gui import TrainingGUI
    from gui.prediction_gui import PredictionGUI
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)


class MainGUI:
    """GUI principal que permite seleccionar entre extracción, entrenamiento y predicción"""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("OpenTopologyC - Suite Completa")
        self.window.geometry("550x550")
        self.window.resizable(False, False)
        
        self.build_layout()
        
    def build_layout(self):
        """Construye la interfaz principal"""
        # Frame principal
        main_frame = ttk.Frame(self.window, padding=30)
        main_frame.pack(fill="both", expand=True)
        
        # Logo/Título
        title_label = ttk.Label(
            main_frame,
            text="OpenTopologyC",
            font=("Arial", 24, "bold"),
            foreground="#2E86C1"
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(
            main_frame,
            text="Suite de Análisis de Topología de Superficies",
            font=("Arial", 12),
            foreground="#7D3C98"
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Separador
        separator = ttk.Separator(main_frame, orient="horizontal")
        separator.pack(fill="x", pady=20)
        
        # Botones de opciones
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill="both", expand=True)
        
        # Botón 1: Extractor de Features
        extractor_btn = ttk.Button(
            options_frame,
            text="🔬 Extractor de Features",
            command=self.open_extractor,
            width=30,
            style="Accent.TButton"
        )
        extractor_btn.pack(pady=10)
        
        # Descripción
        extractor_desc = ttk.Label(
            options_frame,
            text="Extrae características topológicas de archivos dump",
            font=("Arial", 9),
            foreground="gray"
        )
        extractor_desc.pack(pady=(0, 20))
        
        # Botón 2: Entrenamiento de Modelos
        trainer_btn = ttk.Button(
            options_frame,
            text="🤖 Entrenamiento de Modelos",
            command=self.open_trainer,
            width=30,
            style="Accent.TButton"
        )
        trainer_btn.pack(pady=10)
        
        # Descripción
        trainer_desc = ttk.Label(
            options_frame,
            text="Entrena modelos Random Forest para predecir vacancias",
            font=("Arial", 9),
            foreground="gray"
        )
        trainer_desc.pack(pady=(0, 20))

        # Botón 3: Predicción de Vacancias
        prediction_btn = ttk.Button(
            options_frame,
            text="🎯 Predicción de Vacancias",
            command=self.open_prediction,
            width=30,
            style="Accent.TButton"
        )
        prediction_btn.pack(pady=10)

        # Descripción
        prediction_desc = ttk.Label(
            options_frame,
            text="Predice vacancias en un dump usando modelo entrenado",
            font=("Arial", 9),
            foreground="gray"
        )
        prediction_desc.pack(pady=(0, 20))

        # Botón Salir
        exit_btn = ttk.Button(
            options_frame,
            text="🚪 Salir",
            command=self.window.quit,
            width=20
        )
        exit_btn.pack(pady=(20, 0))
        
        # Configurar estilo para botones principales
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 10, "bold"), padding=10)
        
        # Información de versión
        version_label = ttk.Label(
            main_frame,
            text="v1.0.0",
            font=("Arial", 8),
            foreground="darkgray"
        )
        version_label.pack(side="bottom", pady=(20, 0))
        
    def open_extractor(self):
        """Abre la GUI de extracción de features"""
        self.window.withdraw()  # Ocultar ventana principal
        extractor_gui = ExtractorGUI()
        extractor_gui.run()
        self.window.deiconify()  # Mostrar ventana principal al cerrar extractor
        
    def open_trainer(self):
        """Abre la GUI de entrenamiento"""
        self.window.withdraw()
        trainer_gui = TrainingGUI()
        trainer_gui.run()
        self.window.deiconify()

    def open_prediction(self):
        """Abre la GUI de predicción"""
        self.window.withdraw()
        prediction_gui = PredictionGUI()
        prediction_gui.run()
        self.window.deiconify()

    def run(self):
        """Ejecuta la GUI principal"""
        # Centrar ventana
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')
        
        # Configurar atajos
        self.window.bind('<Escape>', lambda e: self.window.quit())
        self.window.bind('<F1>', lambda e: self.open_extractor())
        self.window.bind('<F2>', lambda e: self.open_trainer())
        self.window.bind('<F3>', lambda e: self.open_prediction())
        
        # Ejecutar
        self.window.mainloop()


if __name__ == "__main__":
    app = MainGUI()
    app.run()