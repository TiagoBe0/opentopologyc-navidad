#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# opentopologyc/core/prediction_pipeline.py

"""
Pipeline de predicción de vacancias
Integra: Alpha Shape → Clustering → Extracción de Features → Predicción
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from core.alpha_shape_filter import filter_surface_atoms, LAMMPSDumpParser
from core.clustering_engine import cluster_surface_atoms, ClusteringEngine
from core.loader import DumpLoader
from core.normalizer import PositionNormalizer
from core.feature_extractor import FeatureExtractor
from config.extractor_config import ExtractorConfig


class PredictionPipeline:
    """
    Pipeline completo para predecir vacancias en un dump
    Flujo: Alpha Shape → Clustering → Features → Predicción
    """

    def __init__(self, model_path, config: ExtractorConfig):
        """
        Inicializa el pipeline de predicción

        Args:
            model_path: Ruta al modelo entrenado (.joblib o .pkl)
            config: Configuración del extractor (parámetros de material, features, etc.)
        """
        self.model_path = Path(model_path)
        self.config = config
        self.config.validate()

        # Cargar modelo
        print(f"\n[CARGA DE MODELO]")
        print(f"  Cargando: {self.model_path}")
        self.model = joblib.load(self.model_path)
        print(f"  ✓ Modelo cargado correctamente")

        # Instanciar módulos
        self.loader = DumpLoader()
        self.normalizer = PositionNormalizer(scale_factor=self.config.a0)
        self.features_extractor = FeatureExtractor(self.config)

    def predict_single(self, dump_file,
                      apply_alpha_shape=True,
                      probe_radius=2.0,
                      num_ghost_layers=2,
                      apply_clustering=False,
                      clustering_method="KMeans",
                      clustering_params=None,
                      target_cluster="largest"):
        """
        Predice el número de vacancias para un archivo dump

        Args:
            dump_file: Ruta al archivo dump
            apply_alpha_shape: Si True, aplica filtro Alpha Shape primero
            probe_radius: Radio de prueba para Alpha Shape
            num_ghost_layers: Número de capas ghost
            apply_clustering: Si True, aplica clustering después de Alpha Shape
            clustering_method: Método de clustering ("KMeans", "MeanShift", "Aglomerativo", "HDBSCAN")
            clustering_params: Parámetros del clustering (dict)
            target_cluster: Qué cluster procesar ("largest", "all", o int para cluster específico)

        Returns:
            dict con resultados de la predicción
        """
        dump_file = Path(dump_file)

        print(f"\n{'='*80}")
        print(f"PREDICCIÓN DE VACANCIAS")
        print(f"{'='*80}")
        print(f"\nArchivo: {dump_file.name}")

        # PASO 1: Obtener número real de átomos en simulación
        print(f"\n[1/5] Leyendo archivo original...")
        original_data = self.loader.load(dump_file)
        num_atoms_real = original_data["num_atoms"]
        print(f"  ✓ Átomos en simulación: {num_atoms_real}")

        # PASO 2: Aplicar Alpha Shape (opcional)
        if apply_alpha_shape:
            print(f"\n[2/5] Aplicando filtro Alpha Shape...")
            surface_dump = dump_file.parent / f"{dump_file.stem}_surface_filtered.dump"

            filter_result = filter_surface_atoms(
                input_dump=str(dump_file),
                output_dump=str(surface_dump),
                probe_radius=probe_radius,
                lattice_param=self.config.a0,
                num_ghost_layers=num_ghost_layers,
                smoothing=0
            )

            print(f"  ✓ Átomos superficiales: {filter_result['output_atoms']}")
            file_to_process = surface_dump
            num_surface_atoms = filter_result['output_atoms']
        else:
            print(f"\n[2/5] Saltando filtro Alpha Shape...")
            file_to_process = dump_file
            num_surface_atoms = num_atoms_real

        # PASO 3: Aplicar Clustering (opcional)
        clustering_info = None
        if apply_clustering:
            print(f"\n[3/5] Aplicando Clustering ({clustering_method})...")

            # Leer archivo filtrado
            dump_data = LAMMPSDumpParser.read(str(file_to_process))
            positions = dump_data['positions']

            # Configurar parámetros por defecto si no se proporcionan
            if clustering_params is None:
                if clustering_method == "KMeans":
                    clustering_params = {'n_clusters': 5}
                elif clustering_method == "MeanShift":
                    clustering_params = {'quantile': 0.2}
                elif clustering_method == "Aglomerativo":
                    clustering_params = {'n_clusters': 5, 'linkage': 'ward'}
                elif clustering_method == "HDBSCAN":
                    clustering_params = {'min_cluster_size': 10, 'min_samples': None}

            # Aplicar clustering
            labels, cluster_info = cluster_surface_atoms(
                positions=positions,
                method=clustering_method,
                **clustering_params
            )

            print(f"  ✓ Clusters encontrados: {cluster_info['n_clusters']}")

            # Mostrar métricas si están disponibles
            if 'metrics' in cluster_info:
                metrics = cluster_info['metrics']
                if 'silhouette' in metrics:
                    print(f"    - Silhouette: {metrics['silhouette']:.3f}")
                if 'cluster_size_mean' in metrics:
                    print(f"    - Tamaño promedio: {metrics['cluster_size_mean']:.1f} átomos")

            clustering_info = {
                'method': clustering_method,
                'n_clusters': cluster_info['n_clusters'],
                'labels': labels,
                'info': cluster_info
            }

            # Determinar qué cluster(s) procesar
            if target_cluster == "largest":
                # Encontrar el cluster más grande
                cluster_sizes = {
                    label: (labels == label).sum()
                    for label in np.unique(labels)
                    if label != -1  # Excluir ruido en HDBSCAN
                }
                if cluster_sizes:
                    largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
                    print(f"  ✓ Procesando cluster más grande: Cluster {largest_cluster} ({cluster_sizes[largest_cluster]} átomos)")

                    # Filtrar átomos del cluster
                    cluster_mask = labels == largest_cluster
                    cluster_positions = positions[cluster_mask]

                    # Guardar cluster en archivo temporal
                    cluster_dump = file_to_process.parent / f"{file_to_process.stem}_cluster_{largest_cluster}.dump"
                    filtered_atom_ids = [dump_data['atom_ids_ordered'][i] for i, m in enumerate(cluster_mask) if m]
                    LAMMPSDumpParser.write(str(cluster_dump), dump_data, filtered_atom_ids)

                    file_to_process = cluster_dump
                    clustering_info['target_cluster'] = largest_cluster
                    clustering_info['cluster_size'] = cluster_sizes[largest_cluster]

            elif target_cluster == "all":
                print(f"  ⚠ Modo 'all' aún no implementado, usando cluster más grande")
                # TODO: Implementar procesamiento de todos los clusters
                target_cluster = "largest"

            elif isinstance(target_cluster, int):
                # Procesar cluster específico
                if target_cluster in np.unique(labels):
                    cluster_mask = labels == target_cluster
                    cluster_size = cluster_mask.sum()
                    print(f"  ✓ Procesando cluster específico: Cluster {target_cluster} ({cluster_size} átomos)")

                    # Guardar cluster
                    cluster_dump = file_to_process.parent / f"{file_to_process.stem}_cluster_{target_cluster}.dump"
                    filtered_atom_ids = [dump_data['atom_ids_ordered'][i] for i, m in enumerate(cluster_mask) if m]
                    LAMMPSDumpParser.write(str(cluster_dump), dump_data, filtered_atom_ids)

                    file_to_process = cluster_dump
                    clustering_info['target_cluster'] = target_cluster
                    clustering_info['cluster_size'] = cluster_size
                else:
                    print(f"  ⚠ Cluster {target_cluster} no existe, usando cluster más grande")
                    target_cluster = "largest"
        else:
            print(f"\n[3/5] Saltando Clustering...")

        # PASO 4: Extraer features
        print(f"\n[4/5] Extrayendo features...")
        features = self._extract_features(file_to_process)
        print(f"  ✓ Features extraídas: {len(features)}")

        # PASO 5: Predecir
        print(f"\n[5/5] Realizando predicción...")
        prediction = self._predict_features(features)

        # Calcular vacancias reales
        n_vacancies_real = self.config.total_atoms - num_atoms_real

        result = {
            'file': dump_file.name,
            'num_atoms_in_simulation': num_atoms_real,
            'num_surface_atoms': num_surface_atoms,
            'predicted_vacancies': prediction,
            'real_vacancies': n_vacancies_real,
            'error': abs(prediction - n_vacancies_real),
            'features': features,
            'alpha_shape_applied': apply_alpha_shape,
            'clustering_applied': apply_clustering,
            'clustering_info': clustering_info
        }

        print(f"\n{'='*80}")
        print(f"RESULTADO")
        print(f"{'='*80}")
        print(f"  Átomos en simulación: {num_atoms_real}")
        print(f"  Átomos superficiales:  {num_surface_atoms}")
        if clustering_info:
            print(f"  Clusters encontrados:  {clustering_info['n_clusters']}")
            if 'target_cluster' in clustering_info:
                print(f"  Cluster procesado:     {clustering_info['target_cluster']} ({clustering_info['cluster_size']} átomos)")
        print(f"  Vacancias predichas:   {prediction:.2f}")
        print(f"  Vacancias reales:      {n_vacancies_real}")
        print(f"  Error absoluto:        {result['error']:.2f}")
        print(f"{'='*80}\n")

        return result

    def _extract_features(self, dump_file):
        """
        Extrae features de un dump procesado
        """
        # Cargar dump
        raw = self.loader.load(dump_file)
        pos = raw["positions"]

        # Normalizar
        pos_norm, box_size, _ = self.normalizer.normalize(pos)

        # Extraer features
        feats = {}

        if self.config.compute_grid_features:
            feats.update(self.features_extractor.grid_features(pos_norm, box_size))

        if self.config.compute_inertia_features:
            feats.update(self.features_extractor.inertia_feature(pos))

        if self.config.compute_radial_features:
            feats.update(self.features_extractor.radial_features(pos))

        if self.config.compute_entropy_features:
            feats.update(self.features_extractor.entropy_spatial(pos_norm))

        if self.config.compute_clustering_features:
            feats.update(self.features_extractor.bandwidth(pos_norm))

        feats["num_points"] = pos.shape[0]

        return feats

    def _predict_features(self, features):
        """
        Hace predicción usando el modelo cargado
        """
        # Convertir features a DataFrame
        df = pd.DataFrame([features])

        # Remover columnas que no son features
        forbidden_features = ['n_vacancies', 'n_atoms_surface', 'vacancies', 'file', 'num_atoms_real']
        X = df.drop(columns=[col for col in forbidden_features if col in df.columns], errors='ignore')

        # Predecir
        prediction = self.model.predict(X)[0]

        return prediction

    def predict_batch(self, dump_dir, output_csv=None, apply_alpha_shape=True,
                     probe_radius=2.0, num_ghost_layers=2):
        """
        Predice vacancias para múltiples dumps en un directorio

        Args:
            dump_dir: Directorio con dumps
            output_csv: Ruta para guardar resultados (opcional)
            apply_alpha_shape: Aplicar filtro Alpha Shape
            probe_radius: Radio de prueba
            num_ghost_layers: Capas ghost

        Returns:
            DataFrame con resultados
        """
        dump_dir = Path(dump_dir)
        dump_files = sorted(dump_dir.glob("*.dump"))

        if not dump_files:
            raise ValueError(f"No se encontraron archivos .dump en {dump_dir}")

        print(f"\n{'='*80}")
        print(f"PREDICCIÓN BATCH")
        print(f"{'='*80}")
        print(f"  Directorio: {dump_dir}")
        print(f"  Archivos encontrados: {len(dump_files)}")
        print(f"{'='*80}\n")

        results = []
        for dump_file in dump_files:
            try:
                result = self.predict_single(
                    dump_file,
                    apply_alpha_shape=apply_alpha_shape,
                    probe_radius=probe_radius,
                    num_ghost_layers=num_ghost_layers
                )
                results.append(result)
            except Exception as e:
                print(f"[ERROR] {dump_file.name}: {e}")
                continue

        # Crear DataFrame
        df_results = pd.DataFrame([
            {
                'file': r['file'],
                'num_atoms_simulation': r['num_atoms_in_simulation'],
                'num_surface_atoms': r['num_surface_atoms'],
                'predicted_vacancies': r['predicted_vacancies'],
                'real_vacancies': r['real_vacancies'],
                'error': r['error']
            }
            for r in results
        ])

        # Guardar
        if output_csv:
            output_csv = Path(output_csv)
            df_results.to_csv(output_csv, index=False)
            print(f"\n✓ Resultados guardados en: {output_csv}")

        # Resumen
        print(f"\n{'='*80}")
        print(f"RESUMEN BATCH")
        print(f"{'='*80}")
        print(f"  Archivos procesados: {len(df_results)}")
        print(f"  Error medio absoluto: {df_results['error'].mean():.2f}")
        print(f"  Error std: {df_results['error'].std():.2f}")
        print(f"  Error máximo: {df_results['error'].max():.2f}")
        print(f"{'='*80}\n")

        return df_results
