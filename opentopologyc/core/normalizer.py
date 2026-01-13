# opentopologyc/core/normalizer.py

import numpy as np
from sklearn.decomposition import PCA


class PositionNormalizer:
    """
    Normaliza coordenadas:
      1) Centro en el origen
      2) Alineación PCA (cambio de ejes)
      3) Cálculo de box_size para discretización
    """

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    # ---------------------------------------------------------
    # 1. CENTRADO
    # ---------------------------------------------------------
    def center(self, positions):
        """
        Centra los puntos en su centroide.
        """
        positions = np.asarray(positions)

        if len(positions) == 0:
            return positions

        centroid = positions.mean(axis=0)
        centered = positions - centroid

        return centered

    # ---------------------------------------------------------
    # 2. PCA (alineación de ejes)
    # ---------------------------------------------------------
    def align(self, positions):
        """
        Alinea los puntos con PCA.
        Devuelve:
          aligned -> Nx3
          box_size -> tamaño máximo proyectado
          R -> matriz de rotación PCA
        """
        positions = np.asarray(positions)

        if len(positions) < 3:
            # Sin PCA posible
            return positions, 1.0, np.eye(3)

        try:
            pca = PCA(n_components=3)
            aligned = pca.fit_transform(positions)
            R = pca.components_
        except Exception:
            # fallback
            pca = PCA(n_components=3)
            aligned = pca.fit_transform(positions)
            R = pca.components_

        # Calcular box_size como la extensión máxima
        extent = aligned.max(axis=0) - aligned.min(axis=0)
        box_size = float(extent.max())

        # Reescalar si se definió
        if self.scale_factor != 1.0:
            aligned /= self.scale_factor
            box_size /= self.scale_factor

        return aligned, box_size, R

    # ---------------------------------------------------------
    # PROCESO COMPLETO
    # ---------------------------------------------------------
    def normalize(self, positions):
        """
        Pipeline completo: centro + PCA.
        """
        centered = self.center(positions)
        aligned, box_size, R = self.align(centered)
        return aligned, box_size, R
