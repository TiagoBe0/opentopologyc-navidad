# core/feature_extractor.py

import numpy as np
from scipy.stats import kurtosis, entropy
from sklearn.cluster import estimate_bandwidth
from sklearn.decomposition import PCA


class FeatureExtractor:
    def __init__(self, config):
        self.cfg = config
        self.grid_size = (10, 10, 10)
        self.a0 = 3.532  # Parámetro de red

    # ----------------------------------------------------
    # NORMALIZACIÓN Y CENTRADO
    # ----------------------------------------------------
    def normalize_positions(self, positions):
        """
        Normaliza y alinea posiciones usando PCA
        
        Args:
            positions: array (N, 3) de coordenadas
            
        Returns:
            normalized: array (N, 3) centrado y normalizado
            box_size: tamaño de caja para grid
        """
        if len(positions) < 3:
            centered = positions - positions.mean(axis=0)
            return centered / self.a0, 2.0
        
        # Centrar en origin
        centered = positions - positions.mean(axis=0)
        
        # PCA
        try:
            pca = PCA(n_components=3, svd_solver='covariance_eigh')
            aligned = pca.fit_transform(centered)
        except:
            pca = PCA(n_components=3, svd_solver='auto')
            aligned = pca.fit_transform(centered)
        
        # Normalizar
        normalized = aligned / self.a0
        
        # Calcular tamaño de caja
        box_size_max = 10.0
        extent = normalized.max(axis=0) - normalized.min(axis=0)
        box_size = max(1.5, min(extent.max() * 1.5, box_size_max))
        
        return normalized, box_size

    # ----------------------------------------------------
    # GRID FEATURES COMPLETAS (14 features total)
    # ----------------------------------------------------
    def grid_features(self, positions, box_size):
        """
        Calcula features del grid 3D (14 features en total)

        Args:
            positions: array (N, 3) de coordenadas NORMALIZADAS
            box_size: tamaño de la caja para el grid

        Returns:
            dict con 14 features:
            - 2 occupancy básicas (total, fraction)
            - 3 media por eje (x, y, z)
            - 4 gradientes (x, y, z, total)
            - 1 superficie
            - 1 entropía
            - 3 momentos de inercia del grid (grid_moi_1, 2, 3)
        """
        N, M, L = self.grid_size
        grid = np.zeros((N, M, L), dtype=np.int8)

        half = box_size / 2.0
        cell = box_size / np.array(self.grid_size)

        # Llenar grid
        for p in positions:
            idx = np.floor((p + half) / cell).astype(int)
            if np.all(idx >= 0) and np.all(idx < self.grid_size):
                grid[tuple(idx)] = 1

        occ = grid.sum()
        
        # Inicializar diccionario de features
        features = {}

        # ========== 1. OCCUPANCY BÁSICAS (2 features) ==========
        features["occupancy_total"] = float(occ)
        features["occupancy_fraction"] = float(grid.mean())

        # ========== 2. MEDIA POR EJE (3 features) ==========
        for axis, name in enumerate("xyz"):
            features[f"occupancy_{name}_mean"] = float(
                grid.sum(axis=axis).mean()
            )

        # ========== 3. GRADIENTES (4 features) ==========
        gx = np.abs(np.diff(grid, axis=0)).sum() if occ > 0 else 0.0
        gy = np.abs(np.diff(grid, axis=1)).sum() if occ > 0 else 0.0
        gz = np.abs(np.diff(grid, axis=2)).sum() if occ > 0 else 0.0

        features.update({
            "occupancy_gradient_x": float(gx),
            "occupancy_gradient_y": float(gy),
            "occupancy_gradient_z": float(gz),
            "occupancy_gradient_total": float(gx + gy + gz)
        })

        # ========== 4. SUPERFICIE (1 feature) ==========
        features["occupancy_surface"] = float(gx + gy + gz)

        # ========== 5. ENTROPÍA DEL GRID (1 feature) ==========
        if occ > 0:
            p = grid.flatten()
            p = p[p > 0] / occ
            features["grid_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            features["grid_entropy"] = 0.0

        # ========== 6. MOMENTOS DE INERCIA DEL GRID (3 features) ==========
        if occ > 0:
            try:
                coords = np.argwhere(grid == 1)
                centered_grid = coords - coords.mean(axis=0)

                Ixx = np.sum(centered_grid[:, 1]**2 + centered_grid[:, 2]**2)
                Iyy = np.sum(centered_grid[:, 0]**2 + centered_grid[:, 2]**2)
                Izz = np.sum(centered_grid[:, 0]**2 + centered_grid[:, 1]**2)
                Ixy = -np.sum(centered_grid[:, 0] * centered_grid[:, 1])
                Ixz = -np.sum(centered_grid[:, 0] * centered_grid[:, 2])
                Iyz = -np.sum(centered_grid[:, 1] * centered_grid[:, 2])

                I_tensor = np.array([
                    [Ixx, Ixy, Ixz],
                    [Ixy, Iyy, Iyz],
                    [Ixz, Iyz, Izz]
                ])

                eigenvalues = np.sort(np.linalg.eigvalsh(I_tensor))[::-1]
                features['grid_moi_1'] = float(eigenvalues[0])
                features['grid_moi_2'] = float(eigenvalues[1])
                features['grid_moi_3'] = float(eigenvalues[2])
            except Exception as e:
                features['grid_moi_1'] = 0.0
                features['grid_moi_2'] = 0.0
                features['grid_moi_3'] = 0.0
        else:
            features['grid_moi_1'] = 0.0
            features['grid_moi_2'] = 0.0
            features['grid_moi_3'] = 0.0

        return features

    # ----------------------------------------------------
    # INERCIA PRINCIPAL (1 feature)
    # ----------------------------------------------------
    def inertia_feature(self, positions):
        """
        Calcula SOLO moi_principal_3

        Nota: Esto es DIFERENTE de grid_moi_3.
        grid_moi_3 es de las celdas ocupadas del grid.
        moi_principal_3 es de las posiciones atómicas reales.
        """
        if len(positions) < 3:
            return {"moi_principal_3": np.nan}

        try:
            c = positions - positions.mean(axis=0)

            Ixx = np.sum(c[:, 1]**2 + c[:, 2]**2)
            Iyy = np.sum(c[:, 0]**2 + c[:, 2]**2)
            Izz = np.sum(c[:, 0]**2 + c[:, 1]**2)

            Ixy = -np.sum(c[:, 0] * c[:, 1])
            Ixz = -np.sum(c[:, 0] * c[:, 2])
            Iyz = -np.sum(c[:, 1] * c[:, 2])

            I = np.array([
                [Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz]
            ])

            eig = np.sort(np.linalg.eigvalsh(I))[::-1]

            return {"moi_principal_3": float(eig[2])}
        except Exception as e:
            return {"moi_principal_3": np.nan}

    # ----------------------------------------------------
    # RADIAL FEATURES (2 features)
    # ----------------------------------------------------
    def radial_features(self, positions):
        """
        Calcula rdf_mean y rdf_kurtosis
        """
        if len(positions) < 2:
            return {
                "rdf_mean": np.nan,
                "rdf_kurtosis": np.nan
            }
        
        try:
            c = positions.mean(axis=0)
            d = np.linalg.norm(positions - c, axis=1)

            return {
                "rdf_mean": float(d.mean()),
                "rdf_kurtosis": 0.0 if np.std(d) < 1e-6 else float(kurtosis(d))
            }
        except Exception as e:
            return {
                "rdf_mean": np.nan,
                "rdf_kurtosis": np.nan
            }

    # ----------------------------------------------------
    # ENTROPÍA ESPACIAL (1 feature)
    # ----------------------------------------------------
    def entropy_spatial(self, positions):
        """
        Calcula entropy_spatial
        """
        if len(positions) < 2:
            return {"entropy_spatial": np.nan}
        
        try:
            H, _ = np.histogramdd(positions, bins=10)
            p = H.flatten()
            p = p[p > 0] / p.sum()
            return {"entropy_spatial": float(entropy(p))}
        except Exception as e:
            return {"entropy_spatial": np.nan}

    # ----------------------------------------------------
    # BANDWIDTH (1 feature)
    # ----------------------------------------------------
    def bandwidth(self, positions):
        """
        Calcula ms_bandwidth (Mean Shift bandwidth)
        """
        if len(positions) < 10:
            return {"ms_bandwidth": np.nan}
        
        try:
            bw = estimate_bandwidth(
                positions,
                quantile=0.2,
                n_samples=min(500, len(positions))
            )
            
            if bw <= 0:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=2)
                nn.fit(positions)
                distances, _ = nn.kneighbors(positions)
                bw = np.mean(distances[:, 1]) * 2.0
            
            return {"ms_bandwidth": float(bw)}
        except Exception as e:
            return {"ms_bandwidth": np.nan}

    # ----------------------------------------------------
    # CONVEX HULL FEATURES
    # ----------------------------------------------------
    def hull_features(self, positions):
        """
        Calcula características del convex hull (envoltura convexa)

        Args:
            positions: array (N, 3) de coordenadas

        Returns:
            dict con características del hull
        """
        from scipy.spatial import ConvexHull

        if len(positions) < 4:
            return {
                "hull_volume": np.nan,
                "hull_area": np.nan,
                "hull_n_simplices": np.nan,
                "hull_volume_area_ratio": np.nan
            }

        try:
            hull = ConvexHull(positions)

            volume = float(hull.volume)
            area = float(hull.area)
            n_simplices = float(len(hull.simplices))

            # Relación volumen/área (indicador de compacidad)
            vol_area_ratio = volume / area if area > 0 else 0.0

            return {
                "hull_volume": volume,
                "hull_area": area,
                "hull_n_simplices": n_simplices,
                "hull_volume_area_ratio": vol_area_ratio
            }
        except Exception as e:
            return {
                "hull_volume": np.nan,
                "hull_area": np.nan,
                "hull_n_simplices": np.nan,
                "hull_volume_area_ratio": np.nan
            }

    # ----------------------------------------------------
    # FEATURE EXTRACTION PIPELINE COMPLETO
    # ----------------------------------------------------
    def extract_all_features(self, positions, n_vacancies=None):
        """
        Pipeline completo para extraer todas las features

        Args:
            positions: array (N, 3) de coordenadas atómicas
            n_vacancies: número de vacancias (opcional)

        Returns:
            dict con todas las features (19 base):
            - 14 grid features (occupancy, gradients, entropy, MOI del grid)
            - 1 MOI principal (moi_principal_3 de posiciones atómicas)
            - 2 radial features (RDF)
            - 1 entropy espacial
            - 1 bandwidth
            Total: 19 features base (sin hull features, compatible con extractor de training)
        """
        # Normalizar posiciones
        normalized_pos, box_size = self.normalize_positions(positions)

        # Inicializar diccionario de features
        features = {}

        # 1. Grid features (14 features: occupancy, gradients, entropy, grid_moi)
        grid_feats = self.grid_features(normalized_pos, box_size)
        features.update(grid_feats)

        # 2. MOI principal de posiciones atómicas (1 feature: solo moi_principal_3)
        moi_feats = self.inertia_feature(positions)
        features.update(moi_feats)
        
        # 3. RDF features (2 features)
        radial_feats = self.radial_features(positions)
        features.update(radial_feats)
        
        # 4. Entropía espacial (1 feature)
        entropy_feats = self.entropy_spatial(positions)
        features.update(entropy_feats)
        
        # 5. Bandwidth (1 feature)
        bw_feats = self.bandwidth(positions)
        features.update(bw_feats)

        # 6. Hull features (4 features) - si está habilitado en config
        if hasattr(self.cfg, 'compute_hull_features') and self.cfg.compute_hull_features:
            hull_feats = self.hull_features(positions)
            features.update(hull_feats)

        # 7. Vacancias (si se proporciona)
        if n_vacancies is not None:
            features['n_vacancies'] = float(n_vacancies)

        # Lista final de todas las columnas esperadas (19 base features)
        # Compatible con el extractor de training del usuario
        final_features = [
            'occupancy_total',
            'occupancy_fraction',
            'occupancy_x_mean',
            'occupancy_y_mean',
            'occupancy_z_mean',
            'occupancy_gradient_x',
            'occupancy_gradient_y',
            'occupancy_gradient_z',
            'occupancy_gradient_total',
            'occupancy_surface',
            'grid_entropy',
            'grid_moi_1',
            'grid_moi_2',
            'grid_moi_3',
            'moi_principal_3',
            'rdf_mean',
            'rdf_kurtosis',
            'entropy_spatial',
            'ms_bandwidth'
        ]

        # Agregar hull features si están habilitadas
        if hasattr(self.cfg, 'compute_hull_features') and self.cfg.compute_hull_features:
            final_features.extend([
                'hull_volume',
                'hull_area',
                'hull_n_simplices',
                'hull_volume_area_ratio'
            ])
        
        # Asegurar que todas las features estén presentes
        for feat in final_features:
            if feat not in features:
                features[feat] = np.nan
        
        # Si se agregó n_vacancies, añadirlo a la lista
        if 'n_vacancies' in features:
            final_features.append('n_vacancies')
        
        # Reordenar features según el orden especificado
        ordered_features = {}
        for feat in final_features:
            if feat in features:
                ordered_features[feat] = features[feat]
        
        return ordered_features

    # ----------------------------------------------------
    # MÉTODOS DE CONVENIENCIA PARA ACCESO INDIVIDUAL
    # ----------------------------------------------------
    def get_all_feature_names(self, include_vacancies=True, include_hull=None):
        """
        Devuelve la lista de todas las features extraídas

        Args:
            include_vacancies: si incluir n_vacancies en la lista
            include_hull: si incluir hull features. Si es None, usa la config.

        Returns:
            list de nombres de features
        """
        # 19 features base - compatible con extractor de training
        features = [
            'occupancy_total',
            'occupancy_fraction',
            'occupancy_x_mean',
            'occupancy_y_mean',
            'occupancy_z_mean',
            'occupancy_gradient_x',
            'occupancy_gradient_y',
            'occupancy_gradient_z',
            'occupancy_gradient_total',
            'occupancy_surface',
            'grid_entropy',
            'grid_moi_1',
            'grid_moi_2',
            'grid_moi_3',
            'moi_principal_3',
            'rdf_mean',
            'rdf_kurtosis',
            'entropy_spatial',
            'ms_bandwidth'
        ]

        # Agregar hull features si corresponde
        if include_hull is None:
            include_hull = hasattr(self.cfg, 'compute_hull_features') and self.cfg.compute_hull_features

        if include_hull:
            features.extend([
                'hull_volume',
                'hull_area',
                'hull_n_simplices',
                'hull_volume_area_ratio'
            ])

        if include_vacancies:
            features.append('n_vacancies')

        return features
    
    def get_feature_categories(self):
        """
        Devuelve las features agrupadas por categoría
        """
        categories = {
            'occupancy': [
                'occupancy_total',
                'occupancy_fraction',
                'occupancy_x_mean',
                'occupancy_y_mean',
                'occupancy_z_mean'
            ],
            'gradients': [
                'occupancy_gradient_x',
                'occupancy_gradient_y',
                'occupancy_gradient_z',
                'occupancy_gradient_total',
                'occupancy_surface'
            ],
            'grid_analysis': [
                'grid_entropy',
                'grid_moi_1',
                'grid_moi_2',
                'grid_moi_3'
            ],
            'shape_analysis': [
                'moi_principal_3'
            ],
            'radial_distribution': [
                'rdf_mean',
                'rdf_kurtosis'
            ],
            'spatial_analysis': [
                'entropy_spatial',
                'ms_bandwidth'
            ],
            'target': [
                'n_vacancies'
            ]
        }

        # Agregar hull features si están habilitadas
        if hasattr(self.cfg, 'compute_hull_features') and self.cfg.compute_hull_features:
            categories['hull_analysis'] = [
                'hull_volume',
                'hull_area',
                'hull_n_simplices',
                'hull_volume_area_ratio'
            ]

        return categories