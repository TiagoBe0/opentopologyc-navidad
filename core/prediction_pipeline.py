#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline de predicción de vacancias
Integra: Alpha Shape → Clustering → Extracción de Features → Predicción

Versión integrada:
- Mantiene TODA la funcionalidad original
- Agrega progress_callback(step, total, message)
- Agrega logger opcional (Qt / CLI)
- No depende de Qt
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from core.alpha_shape_filter import filter_surface_atoms, LAMMPSDumpParser
from core.clustering_engine import cluster_surface_atoms
from core.loader import DumpLoader
from core.normalizer import PositionNormalizer
from core.feature_extractor import FeatureExtractor
from config.extractor_config import ExtractorConfig


class PredictionPipeline:
    """
    Pipeline completo para predecir vacancias en un dump
    Flujo: Alpha Shape → Clustering → Features → Predicción
    """

    # ======================================================
    # INIT
    # ======================================================
    def __init__(
        self,
        model_path,
        config: ExtractorConfig,
        logger=None
    ):
        self.model_path = Path(model_path)
        self.config = config
        self.config.validate()
        self.logger = logger

        self._log("\n[CARGA DE MODELO]")
        self._log(f"  Cargando: {self.model_path}")
        self.model = joblib.load(self.model_path)
        self._log("  ✓ Modelo cargado correctamente")

        self.loader = DumpLoader()
        self.normalizer = PositionNormalizer(scale_factor=self.config.a0)
        self.features_extractor = FeatureExtractor(self.config)

    # ======================================================
    # LOG
    # ======================================================
    def _log(self, msg):
        if self.logger:
            self.logger(msg)
        else:
            print(msg)

    # ======================================================
    # SINGLE PREDICTION
    # ======================================================
    def predict_single(
        self,
        dump_file,
        apply_alpha_shape=True,
        probe_radius=2.0,
        num_ghost_layers=2,
        apply_clustering=False,
        clustering_method="KMeans",
        clustering_params=None,
        target_cluster="largest",
        progress_callback=None
    ):
        dump_file = Path(dump_file)
        TOTAL = 5

        self._log("\n" + "=" * 80)
        self._log("PREDICCIÓN DE VACANCIAS")
        self._log("=" * 80)
        self._log(f"\nArchivo: {dump_file.name}")

        # --------------------------------------------------
        # STEP 1: LOAD ORIGINAL
        # --------------------------------------------------
        if progress_callback:
            progress_callback(1, TOTAL, "Leyendo dump original")

        self._log("\n[1/5] Leyendo archivo original...")
        original_data = self.loader.load(dump_file)
        num_atoms_real = original_data["num_atoms"]
        self._log(f"  ✓ Átomos en simulación: {num_atoms_real}")

        # --------------------------------------------------
        # STEP 2: ALPHA SHAPE
        # --------------------------------------------------
        if progress_callback:
            progress_callback(2, TOTAL, "Aplicando Alpha Shape")

        if apply_alpha_shape:
            self._log("\n[2/5] Aplicando filtro Alpha Shape...")
            surface_dump = dump_file.parent / f"{dump_file.stem}_surface_filtered.dump"

            filter_result = filter_surface_atoms(
                input_dump=str(dump_file),
                output_dump=str(surface_dump),
                probe_radius=probe_radius,
                lattice_param=self.config.a0,
                num_ghost_layers=num_ghost_layers,
                smoothing=0
            )

            num_surface_atoms = filter_result["output_atoms"]
            self._log(f"  ✓ Átomos superficiales: {num_surface_atoms}")
            file_to_process = surface_dump
        else:
            self._log("\n[2/5] Saltando filtro Alpha Shape...")
            num_surface_atoms = num_atoms_real
            file_to_process = dump_file

        # --------------------------------------------------
        # STEP 3: CLUSTERING
        # --------------------------------------------------
        if progress_callback:
            progress_callback(3, TOTAL, "Clustering")

        clustering_info = None

        if apply_clustering:
            self._log(f"\n[3/5] Aplicando Clustering ({clustering_method})...")

            dump_data = LAMMPSDumpParser.read(str(file_to_process))
            positions = dump_data["positions"]

            if clustering_params is None:
                if clustering_method == "KMeans":
                    clustering_params = {"n_clusters": 5}
                elif clustering_method == "MeanShift":
                    clustering_params = {"quantile": 0.2}
                elif clustering_method == "Aglomerativo":
                    clustering_params = {"n_clusters": 5, "linkage": "ward"}
                elif clustering_method == "HDBSCAN":
                    clustering_params = {"min_cluster_size": 10, "min_samples": None}

            labels, cluster_info = cluster_surface_atoms(
                positions=positions,
                method=clustering_method,
                **clustering_params
            )

            self._log(f"  ✓ Clusters encontrados: {cluster_info['n_clusters']}")

            clustering_info = {
                "method": clustering_method,
                "n_clusters": cluster_info["n_clusters"],
                "labels": labels,
                "info": cluster_info,
            }

            # Selección de cluster
            if target_cluster == "largest":
                cluster_sizes = {
                    lbl: (labels == lbl).sum()
                    for lbl in np.unique(labels)
                    if lbl != -1
                }
                if cluster_sizes:
                    largest = max(cluster_sizes, key=cluster_sizes.get)
                    size = cluster_sizes[largest]
                    self._log(f"  ✓ Procesando cluster más grande: {largest} ({size} átomos)")

                    mask = labels == largest
                    cluster_dump = file_to_process.parent / f"{file_to_process.stem}_cluster_{largest}.dump"
                    ids = [
                        dump_data["atom_ids_ordered"][i]
                        for i, m in enumerate(mask) if m
                    ]
                    LAMMPSDumpParser.write(str(cluster_dump), dump_data, ids)

                    file_to_process = cluster_dump
                    clustering_info["target_cluster"] = largest
                    clustering_info["cluster_size"] = size
        else:
            self._log("\n[3/5] Saltando Clustering...")

        # --------------------------------------------------
        # STEP 4: FEATURES
        # --------------------------------------------------
        if progress_callback:
            progress_callback(4, TOTAL, "Extracción de features")

        self._log("\n[4/5] Extrayendo features...")
        features = self._extract_features(file_to_process)
        self._log(f"  ✓ Features extraídas: {len(features)}")

        # --------------------------------------------------
        # STEP 5: PREDICT
        # --------------------------------------------------
        if progress_callback:
            progress_callback(5, TOTAL, "Predicción ML")

        self._log("\n[5/5] Realizando predicción...")
        prediction = self._predict_features(features)

        n_vacancies_real = self.config.total_atoms - num_atoms_real

        result = {
            "file": dump_file.name,
            "num_atoms_in_simulation": num_atoms_real,
            "num_surface_atoms": num_surface_atoms,
            "predicted_vacancies": prediction,
            "real_vacancies": n_vacancies_real,
            "error": abs(prediction - n_vacancies_real),
            "features": features,
            "alpha_shape_applied": apply_alpha_shape,
            "clustering_applied": apply_clustering,
            "clustering_info": clustering_info,
        }

        self._log("\n" + "=" * 80)
        self._log("RESULTADO")
        self._log("=" * 80)
        self._log(f"  Vacancias predichas: {prediction:.2f}")
        self._log(f"  Vacancias reales:    {n_vacancies_real}")
        self._log(f"  Error absoluto:      {result['error']:.2f}")
        self._log("=" * 80 + "\n")

        return result

    # ======================================================
    # FEATURE EXTRACTION
    # ======================================================
    def _extract_features(self, dump_file):
        raw = self.loader.load(dump_file)
        pos = raw["positions"]

        pos_norm, box_size, _ = self.normalizer.normalize(pos)

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

    # ======================================================
    # PREDICT FEATURES
    # ======================================================
    def _predict_features(self, features):
        df = pd.DataFrame([features])

        forbidden = [
            "n_vacancies", "n_atoms_surface",
            "vacancies", "file", "num_atoms_real"
        ]
        X = df.drop(columns=[c for c in forbidden if c in df.columns], errors="ignore")

        return self.model.predict(X)[0]

    # ======================================================
    # BATCH
    # ======================================================
    def predict_batch(
        self,
        dump_dir,
        output_csv=None,
        apply_alpha_shape=True,
        probe_radius=2.0,
        num_ghost_layers=2,
        progress_callback=None
    ):
        dump_dir = Path(dump_dir)
        dump_files = sorted(dump_dir.glob("*.dump"))

        if not dump_files:
            raise ValueError(f"No se encontraron dumps en {dump_dir}")

        results = []
        TOTAL = len(dump_files)

        for i, dump_file in enumerate(dump_files, start=1):
            if progress_callback:
                progress_callback(i, TOTAL, f"Procesando {dump_file.name}")

            try:
                res = self.predict_single(
                    dump_file,
                    apply_alpha_shape=apply_alpha_shape,
                    probe_radius=probe_radius,
                    num_ghost_layers=num_ghost_layers
                )
                results.append(res)
            except Exception as e:
                self._log(f"[ERROR] {dump_file.name}: {e}")

        df = pd.DataFrame([
            {
                "file": r["file"],
                "num_atoms_simulation": r["num_atoms_in_simulation"],
                "num_surface_atoms": r["num_surface_atoms"],
                "predicted_vacancies": r["predicted_vacancies"],
                "real_vacancies": r["real_vacancies"],
                "error": r["error"],
            }
            for r in results
        ])

        if output_csv:
            output_csv = Path(output_csv)
            df.to_csv(output_csv, index=False)
            self._log(f"\n✓ Resultados guardados en: {output_csv}")

        return df
