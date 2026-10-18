"""
InSAR Deformation Module: Accra Hydro-InSAR Engine
==================================================
Multi-temporal satellite radar interferometry processing:
- Line-of-Sight (LOS) to vertical velocity decomposition (single and dual-geometry)
- Temporal coherence filtering
- Localized 'subsidence bowl' depression identification and kinematic characterization
- Spatial deformation interpolation across Greater Accra
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import interpolate, spatial
from sklearn.cluster import DBSCAN

from .data_ingest import InSARPointDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SubsidenceBowl:
    """Quantitative characterization of an identified localized ground subsidence bowl."""
    bowl_id: str
    name: str
    centroid_lat: float
    centroid_lon: float
    radius_km: float
    area_km2: float
    max_subsidence_rate_mm_yr: float  # peak negative vertical velocity
    mean_subsidence_rate_mm_yr: float
    annual_volume_deficit_m3_yr: float
    point_count: int
    risk_level: str  # "Critical", "Severe", "Moderate"

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        return {
            "bowl_id": self.bowl_id,
            "name": self.name,
            "centroid_lat": round(self.centroid_lat, 5),
            "centroid_lon": round(self.centroid_lon, 5),
            "radius_km": round(self.radius_km, 3),
            "area_km2": round(self.area_km2, 3),
            "max_subsidence_rate_mm_yr": round(self.max_subsidence_rate_mm_yr, 2),
            "mean_subsidence_rate_mm_yr": round(self.mean_subsidence_rate_mm_yr, 2),
            "annual_volume_deficit_m3_yr": round(self.annual_volume_deficit_m3_yr, 1),
            "point_count": self.point_count,
            "risk_level": self.risk_level,
        }


# ---------------------------------------------------------------------------
# Core InSAR Processing Engine
# ---------------------------------------------------------------------------

class InSARDeformationEngine:
    """
    Production-grade SAR interferometric deformation modeling engine.
    Performs LOS decomposition, coherence thresholding, and kinematic subsidence bowl tracking.
    """

    KNOWN_LANDMARKS = [
        {"name": "Kwame Nkrumah Circle Interchange", "lat": 5.5592, "lon": -0.2085},
        {"name": "Glefe Coastal Lagoon & Wetland", "lat": 5.5170, "lon": -0.2790},
        {"name": "Weija Dam Downstream Floodplain", "lat": 5.5680, "lon": -0.3340},
        {"name": "Gbawe Valley Depression", "lat": 5.5780, "lon": -0.3100},
        {"name": "Odaw River Basin Central", "lat": 5.5650, "lon": -0.2100},
        {"name": "Korle Lagoon Outlet & Agbogbloshie", "lat": 5.5310, "lon": -0.2130},
        {"name": "Kaneshie First Light Commercial Corridor", "lat": 5.5640, "lon": -0.2340},
    ]

    def __init__(self, default_coherence_threshold: float = 0.35) -> None:
        self.coherence_threshold = default_coherence_threshold

    # -----------------------------------------------------------------------
    # 1. Line-of-Sight (LOS) to Vertical Decomposition
    # -----------------------------------------------------------------------
    def decompose_los_to_vertical(
        self,
        dataset: Union[InSARPointDataset, pd.DataFrame],
        incidence_col: str = "incidence_angle_deg",
        v_los_col: str = "v_los",
        v_los_std_col: str = "v_los_std",
    ) -> pd.DataFrame:
        """
        Decompose radar Line-of-Sight (LOS) velocity into vertical velocity:
        v_vert = v_los / cos(theta_inc)
        sigma_v_vert = sigma_v_los / cos(theta_inc)

        Under dominant vertical subsidence in urban coastal settings, this yields
        the true kinematic subsidence rate in mm/year (negative = subsidence).
        """
        df = dataset.df.copy() if isinstance(dataset, InSARPointDataset) else dataset.copy()

        theta_deg = df[incidence_col].to_numpy(dtype=np.float64)
        v_los = df[v_los_col].to_numpy(dtype=np.float64)
        cos_theta = np.cos(np.radians(theta_deg))

        # Avoid zero division
        cos_theta = np.where(np.abs(cos_theta) < 1e-4, 1.0, cos_theta)

        v_vert = v_los / cos_theta
        df["v_vert"] = np.round(v_vert, 2)

        if v_los_std_col in df.columns:
            v_std = df[v_los_std_col].to_numpy(dtype=np.float64)
            df["v_vert_std"] = np.round(v_std / np.abs(cos_theta), 2)
        else:
            df["v_vert_std"] = 1.0

        return df

    def decompose_dual_geometry(
        self,
        asc_dataset: InSARPointDataset,
        desc_dataset: InSARPointDataset,
        max_dist_km: float = 0.15,
    ) -> pd.DataFrame:
        """
        Resolve true 2D velocity vectors (v_vert, v_east) by combining ascending
        and descending Line-of-Sight geometries via nearest spatial cross-matching:
        [ v_los_asc  ] = [ cos(theta_asc)   -sin(theta_asc)*cos(alpha_asc)  ] [ v_vert ]
        [ v_los_desc ]   [ cos(theta_desc)  -sin(theta_desc)*cos(alpha_desc) ] [ v_east ]
        """
        df_asc = asc_dataset.df.copy()
        df_desc = desc_dataset.df.copy()

        # Build KD-tree on descending coordinates (in km approx)
        lat_ref = float(df_asc["latitude"].mean())
        cos_lat = math.cos(math.radians(lat_ref))

        def to_coords_km(df: pd.DataFrame) -> np.ndarray:
            y = df["latitude"].to_numpy() * 111.32
            x = df["longitude"].to_numpy() * 111.32 * cos_lat
            return np.column_stack([x, y])

        coords_desc = to_coords_km(df_desc)
        coords_asc = to_coords_km(df_asc)

        tree = spatial.cKDTree(coords_desc)
        dists, indices = tree.query(coords_asc, k=1, distance_upper_bound=max_dist_km)

        valid = dists < max_dist_km
        if not np.any(valid):
            raise ValueError(f"No matching points found within {max_dist_km} km between asc/desc tracks.")

        valid_asc = df_asc.iloc[valid].reset_index(drop=True)
        valid_desc = df_desc.iloc[indices[valid]].reset_index(drop=True)

        n_matches = len(valid_asc)
        v_vert_arr = np.zeros(n_matches, dtype=np.float64)
        v_east_arr = np.zeros(n_matches, dtype=np.float64)

        for i in range(n_matches):
            t_asc = math.radians(valid_asc.at[i, "incidence_angle_deg"])
            h_asc = math.radians(valid_asc.at[i, "heading_angle_deg"])
            v_a = valid_asc.at[i, "v_los"]

            t_desc = math.radians(valid_desc.at[i, "incidence_angle_deg"])
            h_desc = math.radians(valid_desc.at[i, "heading_angle_deg"])
            v_d = valid_desc.at[i, "v_los"]

            # Matrix A
            a11 = math.cos(t_asc)
            a12 = -math.sin(t_asc) * math.cos(h_asc)
            a21 = math.cos(t_desc)
            a22 = -math.sin(t_desc) * math.cos(h_desc)

            A = np.array([[a11, a12], [a21, a22]])
            b = np.array([v_a, v_d])

            try:
                sol = np.linalg.solve(A, b)
                v_vert_arr[i] = sol[0]
                v_east_arr[i] = sol[1]
            except np.linalg.LinAlgError:
                v_vert_arr[i] = v_d / math.cos(t_desc)
                v_east_arr[i] = 0.0

        res_df = valid_asc[["point_id", "latitude", "longitude"]].copy()
        res_df["v_vert"] = np.round(v_vert_arr, 2)
        res_df["v_east"] = np.round(v_east_arr, 2)
        res_df["v_los_asc"] = valid_asc["v_los"].to_numpy()
        res_df["v_los_desc"] = valid_desc["v_los"].to_numpy()
        res_df["temporal_coherence"] = np.minimum(
            valid_asc["temporal_coherence"].to_numpy(),
            valid_desc["temporal_coherence"].to_numpy()
        )
        return res_df

    # -----------------------------------------------------------------------
    # 2. Coherence Threshold Filtering
    # -----------------------------------------------------------------------
    def filter_by_coherence(
        self,
        df: pd.DataFrame,
        min_coherence: Optional[float] = None,
        coherence_col: str = "temporal_coherence",
    ) -> pd.DataFrame:
        """Filter out decorrelated radar points (e.g. water bodies, dense vegetation)."""
        threshold = min_coherence if min_coherence is not None else self.coherence_threshold
        filtered = df[df[coherence_col] >= threshold].copy()
        filtered.reset_index(drop=True, inplace=True)
        logger.info(
            "Retained %d of %d points (%.1f%%) above coherence threshold %.2f",
            len(filtered), len(df), (len(filtered) / max(1, len(df))) * 100, threshold
        )
        return filtered

    # -----------------------------------------------------------------------
    # 3. Localized Subsidence Bowl Detection & Characterization
    # -----------------------------------------------------------------------
    def identify_subsidence_bowls(
        self,
        df: pd.DataFrame,
        velocity_col: str = "v_vert",
        subsidence_threshold: float = -12.0,  # mm/year
        eps_km: float = 1.2,
        min_samples: int = 8,
    ) -> List[SubsidenceBowl]:
        """
        Identify distinct localized subsidence bowl depressions using spatial DBSCAN clustering
        applied to critical ground sinking points (v_vert <= subsidence_threshold).
        Calculates geometric centroid, effective radius, area, peak velocity, and volumetric deficit.
        """
        # Filter for active sinking points
        sinking_df = df[df[velocity_col] <= subsidence_threshold].copy()
        if len(sinking_df) < min_samples:
            logger.warning("Fewer than %d sinking points found; no bowls detected.", min_samples)
            return []

        # Convert lat/lon to approximate Cartesian coordinates (km)
        lat_ref = float(sinking_df["latitude"].mean())
        cos_lat = math.cos(math.radians(lat_ref))
        x_km = (sinking_df["longitude"].to_numpy() - float(sinking_df["longitude"].mean())) * 111.32 * cos_lat
        y_km = (sinking_df["latitude"].to_numpy() - lat_ref) * 111.32

        coords = np.column_stack([x_km, y_km])
        clustering = DBSCAN(eps=eps_km, min_samples=min_samples).fit(coords)
        sinking_df["cluster_id"] = clustering.labels_

        bowls: List[SubsidenceBowl] = []
        bowl_counter = 1

        for label in sorted(set(clustering.labels_)):
            if label == -1:
                # Noise points
                continue

            sub_cluster = sinking_df[sinking_df["cluster_id"] == label]
            k = len(sub_cluster)
            c_lat = float(sub_cluster["latitude"].mean())
            c_lon = float(sub_cluster["longitude"].mean())

            # Effective radius from centroid (km)
            dists_km = np.sqrt(
                ((sub_cluster["latitude"].to_numpy() - c_lat) * 111.32) ** 2 +
                ((sub_cluster["longitude"].to_numpy() - c_lon) * 111.32 * cos_lat) ** 2
            )
            # Use 90th percentile distance as boundary radius
            radius_km = float(max(0.3, np.percentile(dists_km, 90)))
            area_km2 = float(math.pi * (radius_km ** 2))

            max_sub = float(sub_cluster[velocity_col].min())
            mean_sub = float(sub_cluster[velocity_col].mean())

            # Annual volumetric ground loss (m3/yr):
            # Area (m2) * |mean subsidence| (m/yr)
            volume_deficit_m3 = (area_km2 * 1e6) * (abs(mean_sub) * 1e-3)

            # Match nearest known Accra landmark for clear naming
            nearest_name = self._match_nearest_landmark(c_lat, c_lon)
            bowl_name = f"{nearest_name} Subsidence Bowl"

            # Assign risk level based on sinking rate
            if max_sub <= -25.0:
                risk_lvl = "Critical"
            elif max_sub <= -18.0:
                risk_lvl = "Severe"
            else:
                risk_lvl = "Moderate"

            bowls.append(
                SubsidenceBowl(
                    bowl_id=f"BOWL_{bowl_counter:02d}",
                    name=bowl_name,
                    centroid_lat=c_lat,
                    centroid_lon=c_lon,
                    radius_km=radius_km,
                    area_km2=area_km2,
                    max_subsidence_rate_mm_yr=max_sub,
                    mean_subsidence_rate_mm_yr=mean_sub,
                    annual_volume_deficit_m3_yr=volume_deficit_m3,
                    point_count=k,
                    risk_level=risk_lvl,
                )
            )
            bowl_counter += 1

        # Sort bowls by maximum subsidence severity
        bowls.sort(key=lambda b: b.max_subsidence_rate_mm_yr)
        return bowls

    def _match_nearest_landmark(self, lat: float, lon: float) -> str:
        """Find the closest landmark name in Accra."""
        best_name = "Greater Accra Metropolitan"
        min_dist = float("inf")
        for lm in self.KNOWN_LANDMARKS:
            d = math.sqrt((lat - lm["lat"]) ** 2 + (lon - lm["lon"]) ** 2)
            if d < min_dist:
                min_dist = d
                best_name = lm["name"]
        return best_name

    # -----------------------------------------------------------------------
    # 4. Continuous Spatial Surface Interpolation
    # -----------------------------------------------------------------------
    def interpolate_velocity_grid(
        self,
        df: pd.DataFrame,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        velocity_col: str = "v_vert",
        method: str = "linear",
    ) -> np.ndarray:
        """
        Interpolate discrete InSAR velocity points onto a regular 2D raster grid.
        Uses linear Delaunay triangulation with nearest-neighbor fill for boundary regions.
        """
        points = df[["latitude", "longitude"]].to_numpy()
        values = df[velocity_col].to_numpy()

        grid_coords = (lat_grid, lon_grid)

        # Primary interpolation
        v_grid = interpolate.griddata(points, values, grid_coords, method=method)

        # Fill NaNs outside convex hull with nearest-neighbor
        if np.isnan(v_grid).any():
            v_grid_near = interpolate.griddata(points, values, grid_coords, method="nearest")
            v_grid = np.where(np.isnan(v_grid), v_grid_near, v_grid)

        return np.round(v_grid, 2)

    def calculate_cumulative_subsidence(
        self,
        v_vert_mm_yr: Union[float, np.ndarray],
        timespan_years: float = 10.0,
    ) -> Union[float, np.ndarray]:
        """Compute cumulative ground loss over an engineering design horizon (mm)."""
        return v_vert_mm_yr * timespan_years
