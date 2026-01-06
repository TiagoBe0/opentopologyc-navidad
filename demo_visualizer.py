#!/usr/bin/env python3
# opentopologyc/demo_visualizer.py

"""
Demo del visualizador 3D con datos de ejemplo - Versión simplificada
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sys

# Agregar ruta
sys.path.append(str(Path(__file__).parent.parent))

# Importar el visualizador corregido
from gui.visualizer_3d import SimpleVisualizerApp, run_visualizer


class MockData:
    """Genera datos de ejemplo para la demo"""
    @staticmethod
    def create_fcc_lattice(a=3.532, size=6):
        """Crea una red FCC de ejemplo"""
        positions = []
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    # Posiciones base FCC
                    base = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]])
                    for offset in base:
                        pos = (np.array([i,j,k]) + offset) * a
                        positions.append(pos)
        positions = np.array(positions)
        
        # Añadir algo de ruido para hacerlo más interesante
        positions += np.random.randn(*positions.shape) * 0.1 * a
        return positions
    
    @staticmethod
    def create_sphere(n_points=1000, radius=10):
        """Crea una esfera de puntos"""
        phi = np.random.uniform(0, 2*np.pi, n_points)
        costheta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(costheta)
        
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        
        return np.column_stack((x, y, z))
    
    @staticmethod
    def create_nanopore(lattice_size=8, pore_radius=3.0):
        """Crea una nanoporo en una red"""
        positions = MockData.create_fcc_lattice(size=lattice_size)
        
        # Filtrar átomos dentro del radio del poro
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        mask = distances > pore_radius
        
        return positions[mask]


class DemoVisualizer(tk.Tk):
    """Demo interactiva del visualizador"""
    def __init__(self):
        super().__init__()
        self.title("OpenTopologyC - Demo Visualizador 3D")
        self.geometry("1000x700")
        
        # Datos de ejemplo
        self.example_data = {
            "Red FCC pequeña": MockData.create_fcc_lattice(size=4),
            "Red FCC mediana": MockData.create_fcc_lattice(size=6),
            "Esfera de puntos": MockData.create_sphere(n_points=500),
            "Nanoporo simple": MockData.create_nanopore(lattice_size=6, pore_radius=2.5)
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz de demo"""
        # Frame principal
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Título
        title_label = ttk.Label(
            main_frame,
            text="OpenTopologyC - Demo Visualizador 3D",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Descripción
        desc_text = """
        Esta demo muestra las capacidades del visualizador 3D para:
        
        1. Visualizar diferentes tipos de estructuras atómicas
        2. Aplicar transformaciones como Alpha Shape y Clustering
        3. Interactuar con la visualización en 3D
        
        Selecciona un tipo de datos y presiona "Cargar Demo" para comenzar.
        """
        
        desc_label = ttk.Label(main_frame, text=desc_text, font=("Arial", 10),
                              justify="left")
        desc_label.pack(pady=(0, 20))
        
        # Selección de datos
        data_frame = ttk.LabelFrame(main_frame, text="Datos de Ejemplo", padding=15)
        data_frame.pack(fill="x", pady=(0, 20))
        
        self.data_var = tk.StringVar(value="Red FCC pequeña")
        
        # Radio buttons para seleccionar datos
        for data_name in self.example_data.keys():
            rb = ttk.Radiobutton(data_frame, text=data_name, value=data_name,
                                variable=self.data_var)
            rb.pack(anchor="w", pady=2)
        
        # Botón para cargar datos
        ttk.Button(data_frame, text="📥 Cargar Demo", 
                  command=self.load_demo_data, width=20).pack(pady=(10, 0))
        
        # Transformaciones rápidas
        trans_frame = ttk.LabelFrame(main_frame, text="Transformaciones Rápidas", padding=15)
        trans_frame.pack(fill="x", pady=(0, 20))
        
        trans_buttons = ttk.Frame(trans_frame)
        trans_buttons.pack()
        
        ttk.Button(trans_buttons, text="Aplicar Alpha Shape", 
                  command=self.demo_alpha_shape, width=20).pack(side="left", padx=(0, 10))
        ttk.Button(trans_buttons, text="Aplicar Clustering", 
                  command=self.demo_clustering, width=20).pack(side="left")
        
        # Separador
        separator = ttk.Separator(main_frame, orient="horizontal")
        separator.pack(fill="x", pady=20)
        
        # Botones de acción
        action_frame = ttk.Frame(main_frame)
        action_frame.pack()
        
        ttk.Button(action_frame, text="🚀 Iniciar Visualizador Completo", 
                  command=self.open_full_visualizer, width=25).pack(side="left", padx=(0, 10))
        ttk.Button(action_frame, text="❌ Salir", 
                  command=self.destroy, width=15).pack(side="left")
        
        # Barra de estado
        self.status_bar = ttk.Label(self, text="Selecciona datos y presiona 'Cargar Demo'", 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_demo_data(self):
        """Carga datos de ejemplo"""
        data_name = self.data_var.get()
        positions = self.example_data[data_name]
        
        # Abrir visualizador con estos datos
        self.withdraw()  # Ocultar ventana de demo
        self.open_visualizer_with_data(positions, data_name)
    
    def open_visualizer_with_data(self, positions, data_name):
        """Abre el visualizador con datos específicos"""
        # Crear una ventana de visualización
        vis_window = tk.Toplevel(self)
        vis_window.title(f"Visualizador 3D - {data_name}")
        vis_window.geometry("900x700")
        
        # Crear frame para el visualizador
        vis_frame = ttk.Frame(vis_window)
        vis_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Importar e instanciar AtomVisualizer3D
        from gui.visualizer_3d import AtomVisualizer3D
        visualizer = AtomVisualizer3D(vis_frame)
        
        # Cargar los datos
        visualizer.positions = positions
        visualizer.colors = np.ones(len(positions))
        visualizer.labels = np.zeros(len(positions), dtype=int)
        visualizer.plot_atoms()
        
        # Añadir botones de control específicos
        control_frame = ttk.Frame(vis_window)
        control_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ttk.Button(control_frame, text="Aplicar Alpha Shape", 
                  command=lambda: visualizer.apply_alpha_shape(2.0, 2)).pack(side="left", padx=(0, 5))
        ttk.Button(control_frame, text="Aplicar Clustering", 
                  command=lambda: visualizer.apply_clustering("KMeans", 5)).pack(side="left")
        ttk.Button(control_frame, text="Cerrar", 
                  command=vis_window.destroy).pack(side="right")
        
        # Configurar cierre
        def on_closing():
            vis_window.destroy()
            self.deiconify()  # Volver a mostrar la ventana de demo
        
        vis_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    def demo_alpha_shape(self):
        """Demo de Alpha Shape"""
        messagebox.showinfo("Demo Alpha Shape", 
                          "Alpha Shape identifica átomos superficiales.\n\n"
                          "En el visualizador completo podrás ver cómo se resaltan "
                          "los átomos en la superficie de la estructura.")
    
    def demo_clustering(self):
        """Demo de Clustering"""
        messagebox.showinfo("Demo Clustering", 
                          "Clustering agrupa átomos por proximidad.\n\n"
                          "En el visualizador, los átomos de cada cluster "
                          "se muestran con colores diferentes.")
    
    def open_full_visualizer(self):
        """Abre el visualizador completo"""
        self.withdraw()
        run_visualizer()
        self.deiconify()


def main():
    """Función principal"""
    print("="*60)
    print("OpenTopologyC - Demo Visualizador 3D")
    print("="*60)
    print("\nOpciones:")
    print("1. Ejecutar demo interactiva")
    print("2. Ejecutar visualizador completo")
    
    choice = input("\nSeleccione (1-2): ").strip()
    
    if choice == "1":
        app = DemoVisualizer()
        # Centrar ventana
        app.update_idletasks()
        width = app.winfo_width()
        height = app.winfo_height()
        x = (app.winfo_screenwidth() // 2) - (width // 2)
        y = (app.winfo_screenheight() // 2) - (height // 2)
        app.geometry(f'{width}x{height}+{x}+{y}')
        
        app.mainloop()
    elif choice == "2":
        run_visualizer()
    else:
        print("Opción no válida. Saliendo...")


if __name__ == "__main__":
    main()