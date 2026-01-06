#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# opentopologyc/core/clustering_engine.py

"""
Motor de clustering para agrupar átomos superficiales de nanoporos
Soporta múltiples algoritmos: KMeans, MeanShift, Aglomerativo, HDBSCAN
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Tuple, Optional, Any

# Intentar importar HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class ClusteringEngine:
    """
    Motor de clustering para análisis de nanoporos
    """

    def __init__(self, positions: np.ndarray, method: str = "KMeans", **kwargs):
        """
        Inicializa el motor de clustering

        Args:
            positions: Array Nx3 de posiciones (x, y, z)
            method: Método de clustering ("KMeans", "MeanShift", "Aglomerativo", "HDBSCAN")
            **kwargs: Parámetros específicos del método
        """
        self.positions = np.array(positions)
        self.method = method
        self.params = kwargs
        self.labels = None
        self.n_clusters = 0
        self.metrics = {}

    def apply_clustering(self) -> int:
        """
        Aplica el método de clustering seleccionado

        Returns:
            Número de clusters encontrados
        """
        if self.method == "KMeans":
            return self._apply_kmeans()
        elif self.method == "MeanShift":
            return self._apply_meanshift()
        elif self.method == "Aglomerativo":
            return self._apply_agglomerative()
        elif self.method == "HDBSCAN":
            if not HDBSCAN_AVAILABLE:
                raise ImportError("HDBSCAN no está instalado. Instala con: pip install hdbscan")
            return self._apply_hdbscan()
        else:
            raise ValueError(f"Método de clustering desconocido: {self.method}")

    def _apply_kmeans(self) -> int:
        """Aplica KMeans clustering"""
        n_clusters = self.params.get('n_clusters', 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(self.positions)
        self.n_clusters = n_clusters
        return n_clusters

    def _apply_meanshift(self) -> int:
        """Aplica MeanShift clustering"""
        quantile = self.params.get('quantile', 0.2)
        bandwidth = estimate_bandwidth(
            self.positions,
            quantile=quantile,
            n_samples=min(500, len(self.positions))
        )
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        self.labels = ms.fit_predict(self.positions)
        self.n_clusters = len(np.unique(self.labels))
        return self.n_clusters

    def _apply_agglomerative(self) -> int:
        """Aplica clustering aglomerativo"""
        n_clusters = self.params.get('n_clusters', 5)
        linkage = self.params.get('linkage', 'ward')
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels = agg.fit_predict(self.positions)
        self.n_clusters = n_clusters
        return n_clusters

    def _apply_hdbscan(self) -> int:
        """Aplica HDBSCAN clustering"""
        min_cluster_size = self.params.get('min_cluster_size', 10)
        min_samples = self.params.get('min_samples', None)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        self.labels = clusterer.fit_predict(self.positions)

        # Excluir ruido (-1) al contar clusters
        unique_labels = np.unique(self.labels)
        self.n_clusters = len(unique_labels[unique_labels != -1])

        return self.n_clusters

    def compute_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de calidad del clustering

        Returns:
            Diccionario con métricas
        """
        if self.labels is None:
            return {}

        self.metrics = {}

        # Para HDBSCAN, excluir ruido
        valid = self.labels != -1
        X = self.positions[valid]
        labels = self.labels[valid]

        if len(X) > 0 and len(np.unique(labels)) > 1:
            try:
                self.metrics['silhouette'] = silhouette_score(X, labels)
                self.metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
                self.metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
            except Exception as e:
                print(f"Warning: No se pudieron calcular algunas métricas: {e}")

        # Contar ruido si es HDBSCAN
        if self.method == "HDBSCAN":
            noise_count = (self.labels == -1).sum()
            self.metrics['noise_points'] = int(noise_count)
            self.metrics['noise_percentage'] = float(100 * noise_count / len(self.labels))

        # Estadísticas de clusters
        cluster_sizes = []
        for label in np.unique(self.labels):
            if label != -1:  # Excluir ruido
                size = (self.labels == label).sum()
                cluster_sizes.append(size)

        if cluster_sizes:
            self.metrics['cluster_size_mean'] = float(np.mean(cluster_sizes))
            self.metrics['cluster_size_std'] = float(np.std(cluster_sizes))
            self.metrics['cluster_size_min'] = int(np.min(cluster_sizes))
            self.metrics['cluster_size_max'] = int(np.max(cluster_sizes))

        return self.metrics

    def get_cluster_positions(self, cluster_id: int) -> np.ndarray:
        """
        Obtiene las posiciones de un cluster específico

        Args:
            cluster_id: ID del cluster

        Returns:
            Array Mx3 con posiciones del cluster
        """
        if self.labels is None:
            raise ValueError("Debe aplicar clustering primero")

        mask = self.labels == cluster_id
        return self.positions[mask]

    def get_cluster_indices(self, cluster_id: int) -> np.ndarray:
        """
        Obtiene los índices de átomos en un cluster específico

        Args:
            cluster_id: ID del cluster

        Returns:
            Array de índices
        """
        if self.labels is None:
            raise ValueError("Debe aplicar clustering primero")

        return np.where(self.labels == cluster_id)[0]

    def get_all_clusters(self) -> Dict[int, Dict[str, Any]]:
        """
        Obtiene información de todos los clusters

        Returns:
            Diccionario con info de cada cluster
        """
        if self.labels is None:
            raise ValueError("Debe aplicar clustering primero")

        clusters_info = {}

        for label in np.unique(self.labels):
            if label == -1:  # Ruido en HDBSCAN
                continue

            indices = self.get_cluster_indices(label)
            positions = self.positions[indices]

            # Calcular centroide
            centroid = positions.mean(axis=0)

            # Calcular dispersión
            distances = np.linalg.norm(positions - centroid, axis=1)
            dispersion = distances.std()

            clusters_info[int(label)] = {
                'cluster_id': int(label),
                'n_atoms': len(indices),
                'indices': indices,
                'positions': positions,
                'centroid': centroid,
                'dispersion': float(dispersion),
                'bounds': {
                    'x': (float(positions[:, 0].min()), float(positions[:, 0].max())),
                    'y': (float(positions[:, 1].min()), float(positions[:, 1].max())),
                    'z': (float(positions[:, 2].min()), float(positions[:, 2].max()))
                }
            }

        return clusters_info

    def get_largest_cluster(self) -> Tuple[int, Dict[str, Any]]:
        """
        Obtiene el cluster más grande

        Returns:
            Tuple (cluster_id, cluster_info)
        """
        clusters = self.get_all_clusters()
        if not clusters:
            raise ValueError("No hay clusters válidos")

        largest_id = max(clusters.keys(), key=lambda k: clusters[k]['n_atoms'])
        return largest_id, clusters[largest_id]

    def filter_small_clusters(self, min_size: int) -> List[int]:
        """
        Filtra clusters pequeños

        Args:
            min_size: Tamaño mínimo de cluster

        Returns:
            Lista de IDs de clusters válidos
        """
        clusters = self.get_all_clusters()
        valid_clusters = [
            cluster_id for cluster_id, info in clusters.items()
            if info['n_atoms'] >= min_size
        ]
        return valid_clusters


class RecursiveClusteringEngine:
    """
    Motor de clustering recursivo/jerárquico
    Subdivide clusters grandes automáticamente
    """

    def __init__(self, positions: np.ndarray, method: str = "KMeans", **kwargs):
        """
        Inicializa el motor recursivo

        Args:
            positions: Array Nx3 de posiciones
            method: Método base de clustering
            **kwargs: Parámetros del método
        """
        self.positions = np.array(positions)
        self.method = method
        self.params = kwargs
        self.final_labels = None
        self.cluster_hierarchy = []

    def apply_recursive_clustering(self, max_cluster_size: int = 5000,
                                  max_depth: int = 5) -> int:
        """
        Aplica clustering recursivo

        Args:
            max_cluster_size: Tamaño máximo antes de subdividir
            max_depth: Profundidad máxima de recursión

        Returns:
            Número total de clusters finales
        """
        self.final_labels = np.zeros(len(self.positions), dtype=int)
        self.cluster_counter = 0

        self._recursive_split(
            positions=self.positions,
            indices=np.arange(len(self.positions)),
            depth=0,
            max_size=max_cluster_size,
            max_depth=max_depth,
            parent_id=None
        )

        return len(np.unique(self.final_labels))

    def _recursive_split(self, positions: np.ndarray, indices: np.ndarray,
                        depth: int, max_size: int, max_depth: int,
                        parent_id: Optional[int]):
        """
        Función recursiva para subdividir clusters
        """
        # Condiciones de parada
        if depth >= max_depth or len(positions) <= max_size:
            # No subdividir más, asignar cluster final
            cluster_id = self.cluster_counter
            self.cluster_counter += 1
            self.final_labels[indices] = cluster_id

            self.cluster_hierarchy.append({
                'cluster_id': cluster_id,
                'parent_id': parent_id,
                'depth': depth,
                'n_atoms': len(positions),
                'is_leaf': True
            })
            return

        # Aplicar clustering en este nivel
        engine = ClusteringEngine(positions, self.method, **self.params)
        n_clusters = engine.apply_clustering()

        # Registrar nodo intermedio
        node_id = self.cluster_counter
        self.cluster_counter += 1

        self.cluster_hierarchy.append({
            'cluster_id': node_id,
            'parent_id': parent_id,
            'depth': depth,
            'n_atoms': len(positions),
            'n_subclusters': n_clusters,
            'is_leaf': False
        })

        # Subdividir cada subcluster
        for label in np.unique(engine.labels):
            if label == -1:  # Ruido en HDBSCAN
                continue

            mask = engine.labels == label
            sub_positions = positions[mask]
            sub_indices = indices[mask]

            self._recursive_split(
                positions=sub_positions,
                indices=sub_indices,
                depth=depth + 1,
                max_size=max_size,
                max_depth=max_depth,
                parent_id=node_id
            )


def cluster_surface_atoms(positions: np.ndarray,
                         method: str = "KMeans",
                         recursive: bool = False,
                         **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Función conveniente para clusterizar átomos superficiales

    Args:
        positions: Array Nx3 de posiciones
        method: Método de clustering
        recursive: Si True, usa clustering recursivo
        **kwargs: Parámetros adicionales

    Returns:
        Tuple (labels, info_dict)
    """
    if recursive:
        engine = RecursiveClusteringEngine(positions, method, **kwargs)
        max_cluster_size = kwargs.pop('max_cluster_size', 5000)
        max_depth = kwargs.pop('max_depth', 5)
        n_clusters = engine.apply_recursive_clustering(max_cluster_size, max_depth)

        info = {
            'method': f"{method}_recursive",
            'n_clusters': n_clusters,
            'hierarchy': engine.cluster_hierarchy
        }

        return engine.final_labels, info
    else:
        engine = ClusteringEngine(positions, method, **kwargs)
        n_clusters = engine.apply_clustering()
        engine.compute_metrics()

        info = {
            'method': method,
            'n_clusters': n_clusters,
            'metrics': engine.metrics,
            'clusters': engine.get_all_clusters()
        }

        return engine.labels, info
