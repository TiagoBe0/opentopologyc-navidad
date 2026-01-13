#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering Jerárquico Mejorado para OpenTopologyC

Basado en métricas de calidad (Silhouette, Davies-Bouldin, dispersión)
en lugar de solo tamaño de clusters.

Características:
- Evaluación automática de calidad del clustering
- Subdivisión recursiva basada en métricas
- Soporte para MeanShift y KMeans
- Exportación de clusters finales
- Integración con visualización 3D
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Tuple, Optional, Any
import shutil
import json


class HierarchicalMeanShiftClusterer:
    """
    Motor de clustering jerárquico con evaluación por métricas
    """

    def __init__(self):
        self.final_clusters = []
        self.cluster_counter = 0
        self.visualization_data = []

    def calcular_metricas_clustering(self, coords: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de calidad del clustering

        Args:
            coords: Posiciones de los átomos (Nx3)
            labels: Etiquetas de cluster

        Returns:
            Diccionario con métricas
        """
        metricas = {}
        n_clusters = len(np.unique(labels))

        if n_clusters > 1 and len(coords) > n_clusters:
            try:
                metricas['silhouette'] = silhouette_score(coords, labels)
                metricas['davies_bouldin'] = davies_bouldin_score(coords, labels)
                metricas['calinski_harabasz'] = calinski_harabasz_score(coords, labels)

                # Dispersión promedio por cluster
                dispersiones = []
                for label in np.unique(labels):
                    cluster_points = coords[labels == label]
                    if len(cluster_points) > 1:
                        centroid = cluster_points.mean(axis=0)
                        dispersiones.append(np.mean(np.linalg.norm(cluster_points - centroid, axis=1)))

                metricas['dispersion_promedio'] = np.mean(dispersiones) if dispersiones else 0

            except Exception as e:
                print(f"Warning: Error al calcular métricas: {e}")
                metricas = self._metricas_default()
        else:
            metricas = self._metricas_default()

        return metricas

    def _metricas_default(self) -> Dict[str, float]:
        """Métricas por defecto cuando no se pueden calcular"""
        return {
            'silhouette': -1,
            'davies_bouldin': float('inf'),
            'calinski_harabasz': 0,
            'dispersion_promedio': float('inf')
        }

    def evaluar_necesidad_subdivision(self,
                                     coords: np.ndarray,
                                     labels: np.ndarray,
                                     min_atoms: int,
                                     silhouette_threshold: float = 0.3,
                                     davies_bouldin_threshold: float = 1.5,
                                     dispersion_threshold: Optional[float] = None) -> Tuple[bool, str, Dict]:
        """
        Evalúa si un cluster necesita ser subdividido

        Args:
            coords: Posiciones de átomos
            labels: Etiquetas de cluster
            min_atoms: Mínimo de átomos para subdividir
            silhouette_threshold: Umbral de Silhouette (menor = necesita subdivisión)
            davies_bouldin_threshold: Umbral de Davies-Bouldin (mayor = necesita subdivisión)
            dispersion_threshold: Umbral de dispersión (mayor = necesita subdivisión)

        Returns:
            (necesita_subdivision, razon, metricas)
        """
        n_atoms = len(coords)
        n_clusters = len(np.unique(labels))

        # No subdividir si hay muy pocos átomos
        if n_atoms < min_atoms * 2:
            return False, f"Muy pocos átomos ({n_atoms} < {min_atoms * 2})", {}

        # Caso especial: cluster único
        if n_clusters == 1:
            dispersion = np.std(coords, axis=0).mean()

            if dispersion_threshold and dispersion > dispersion_threshold:
                return True, f"Alta dispersión en cluster único ({dispersion:.2f})", {'dispersion': dispersion}

            if n_atoms >= min_atoms * 3:
                return True, f"Cluster único con {n_atoms} átomos", {'dispersion': dispersion}

            return False, "Cluster único compacto", {'dispersion': dispersion}

        # Calcular métricas
        metricas = self.calcular_metricas_clustering(coords, labels)

        razones = []
        necesita_subdivision = False

        # Evaluar Silhouette (valores bajos indican mala separación)
        if metricas['silhouette'] < silhouette_threshold:
            razones.append(f"Silhouette bajo ({metricas['silhouette']:.3f} < {silhouette_threshold})")
            necesita_subdivision = True

        # Evaluar Davies-Bouldin (valores altos indican clusters mal separados)
        if metricas['davies_bouldin'] > davies_bouldin_threshold:
            razones.append(f"Davies-Bouldin alto ({metricas['davies_bouldin']:.3f} > {davies_bouldin_threshold})")
            necesita_subdivision = True

        # Evaluar dispersión
        if dispersion_threshold and metricas['dispersion_promedio'] > dispersion_threshold:
            razones.append(f"Alta dispersión ({metricas['dispersion_promedio']:.3f} > {dispersion_threshold})")
            necesita_subdivision = True

        razon_final = " | ".join(razones) if razones else "Métricas aceptables"

        return necesita_subdivision, razon_final, metricas

    def clustering_recursivo(self,
                            positions: np.ndarray,
                            nivel: int = 0,
                            min_atoms: int = 50,
                            max_iterations: int = 5,
                            n_clusters_target: Optional[int] = None,
                            silhouette_threshold: float = 0.3,
                            davies_bouldin_threshold: float = 1.5,
                            dispersion_threshold: Optional[float] = None,
                            quantile: float = 0.2,
                            parent_cluster_id: int = 0,
                            metadata: Optional[Dict] = None) -> Dict:
        """
        Aplica clustering recursivo basado en métricas

        Args:
            positions: Posiciones de átomos (Nx3)
            nivel: Nivel actual de recursión
            min_atoms: Mínimo de átomos por cluster final
            max_iterations: Máximo de niveles de recursión
            n_clusters_target: Número objetivo de clusters (None = automático con MeanShift)
            silhouette_threshold: Umbral de Silhouette
            davies_bouldin_threshold: Umbral de Davies-Bouldin
            dispersion_threshold: Umbral de dispersión
            quantile: Quantile para estimación de bandwidth en MeanShift
            parent_cluster_id: ID del cluster padre
            metadata: Metadata adicional (atom IDs, etc.)

        Returns:
            Diccionario con resultado
        """
        n_atoms = len(positions)

        # Condición de parada: nivel máximo
        if nivel >= max_iterations:
            self.cluster_counter += 1
            cluster_data = {
                'positions': positions.copy(),
                'cluster_id': self.cluster_counter,
                'nivel': nivel,
                'n_atoms': n_atoms,
                'razon_final': 'Nivel máximo alcanzado',
                'metadata': metadata or {}
            }

            self.final_clusters.append(cluster_data)
            self.visualization_data.append({
                'positions': positions.copy(),
                'cluster_id': self.cluster_counter,
                'nivel': nivel
            })

            return {'subdivided': False, 'razon': 'Nivel máximo alcanzado'}

        # Condición de parada: muy pocos átomos
        if n_atoms < min_atoms * 2:
            self.cluster_counter += 1
            cluster_data = {
                'positions': positions.copy(),
                'cluster_id': self.cluster_counter,
                'nivel': nivel,
                'n_atoms': n_atoms,
                'razon_final': f'Pocos átomos ({n_atoms})',
                'metadata': metadata or {}
            }

            self.final_clusters.append(cluster_data)
            self.visualization_data.append({
                'positions': positions.copy(),
                'cluster_id': self.cluster_counter,
                'nivel': nivel
            })

            return {'subdivided': False, 'razon': f'Pocos átomos ({n_atoms})'}

        # Aplicar clustering
        if n_clusters_target is not None:
            # KMeans con número fijo de clusters
            kmeans = KMeans(n_clusters=n_clusters_target, random_state=42, n_init=10)
            labels = kmeans.fit_predict(positions)
            n_clusters_found = n_clusters_target
        else:
            # MeanShift automático
            bandwidth = estimate_bandwidth(
                positions,
                quantile=quantile,
                n_samples=min(500, len(positions))
            )
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            labels = ms.fit_predict(positions)
            n_clusters_found = len(np.unique(labels))

        # Guardar datos de visualización de este nivel
        self.visualization_data.append({
            'positions': positions.copy(),
            'labels': labels.copy(),
            'nivel': nivel,
            'n_clusters': n_clusters_found
        })

        # Evaluar si necesita subdivisión
        necesita_subdivision, razon, metricas = self.evaluar_necesidad_subdivision(
            positions, labels, min_atoms,
            silhouette_threshold, davies_bouldin_threshold, dispersion_threshold
        )

        if not necesita_subdivision:
            # Cluster final, no subdividir
            self.cluster_counter += 1
            cluster_data = {
                'positions': positions.copy(),
                'cluster_id': self.cluster_counter,
                'nivel': nivel,
                'n_atoms': n_atoms,
                'razon_final': razon,
                'metricas': metricas,
                'metadata': metadata or {}
            }

            self.final_clusters.append(cluster_data)

            return {'subdivided': False, 'razon': razon, 'metricas': metricas}

        # Subdividir: procesar cada subcluster
        for label in np.unique(labels):
            mask = labels == label
            sub_positions = positions[mask]

            # Preparar metadata para el subcluster si existe
            sub_metadata = None
            if metadata and 'atom_indices' in metadata:
                sub_indices = np.array(metadata['atom_indices'])[mask]
                sub_metadata = {'atom_indices': sub_indices.tolist()}

            self.clustering_recursivo(
                sub_positions,
                nivel=nivel + 1,
                min_atoms=min_atoms,
                max_iterations=max_iterations,
                n_clusters_target=n_clusters_target,
                silhouette_threshold=silhouette_threshold,
                davies_bouldin_threshold=davies_bouldin_threshold,
                dispersion_threshold=dispersion_threshold,
                quantile=quantile,
                parent_cluster_id=self.cluster_counter,
                metadata=sub_metadata
            )

        return {'subdivided': True, 'razon': razon, 'metricas': metricas, 'n_subclusters': n_clusters_found}

    def get_final_clusters(self) -> List[Dict]:
        """
        Retorna los clusters finales ordenados por tamaño

        Returns:
            Lista de diccionarios con información de clusters
        """
        # Ordenar por número de átomos (descendente)
        sorted_clusters = sorted(self.final_clusters, key=lambda x: x['n_atoms'], reverse=True)
        return sorted_clusters

    def get_visualization_data(self) -> pd.DataFrame:
        """
        Retorna datos para visualización 3D

        Returns:
            DataFrame con posiciones, labels y niveles
        """
        if not self.visualization_data:
            return pd.DataFrame()

        rows = []
        for data in self.visualization_data:
            positions = data['positions']
            cluster_id = data.get('cluster_id', -1)
            nivel = data.get('nivel', 0)

            for pos in positions:
                rows.append({
                    'x': pos[0],
                    'y': pos[1],
                    'z': pos[2],
                    'Cluster': cluster_id,
                    'Cluster_Level': nivel
                })

        return pd.DataFrame(rows)


def apply_hierarchical_clustering(
    positions: np.ndarray,
    min_atoms: int = 50,
    max_iterations: int = 5,
    n_clusters_per_level: Optional[int] = None,
    silhouette_threshold: float = 0.3,
    davies_bouldin_threshold: float = 1.5,
    dispersion_threshold: Optional[float] = None,
    quantile: float = 0.2,
    atom_indices: Optional[List[int]] = None
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Función conveniente para aplicar clustering jerárquico

    Args:
        positions: Posiciones de átomos (Nx3)
        min_atoms: Mínimo de átomos por cluster final
        max_iterations: Máximo de niveles de recursión
        n_clusters_per_level: Número fijo de clusters por nivel (None = automático)
        silhouette_threshold: Umbral de Silhouette
        davies_bouldin_threshold: Umbral de Davies-Bouldin
        dispersion_threshold: Umbral de dispersión
        quantile: Quantile para bandwidth estimation
        atom_indices: Lista de índices de átomos originales

    Returns:
        (lista_clusters_finales, dataframe_visualizacion)
    """
    clusterer = HierarchicalMeanShiftClusterer()

    # Preparar metadata con índices de átomos
    metadata = None
    if atom_indices is not None:
        metadata = {'atom_indices': atom_indices}

    # Ejecutar clustering recursivo
    clusterer.clustering_recursivo(
        positions=positions,
        nivel=0,
        min_atoms=min_atoms,
        max_iterations=max_iterations,
        n_clusters_target=n_clusters_per_level,
        silhouette_threshold=silhouette_threshold,
        davies_bouldin_threshold=davies_bouldin_threshold,
        dispersion_threshold=dispersion_threshold,
        quantile=quantile,
        metadata=metadata
    )

    # Obtener resultados
    final_clusters = clusterer.get_final_clusters()
    viz_data = clusterer.get_visualization_data()

    return final_clusters, viz_data
