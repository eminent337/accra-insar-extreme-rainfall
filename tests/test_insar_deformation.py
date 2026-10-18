"""
Unit Tests: InSAR Radar Deformation Module
==========================================
"""

import math
import numpy as np
import pandas as pd
import pytest

from accra_engine.data_ingest import InSARPointDataset
from accra_engine.insar_deformation import (
    InSARDeformationEngine,
    SubsidenceBowl,
)


@pytest.fixture
def mock_insar_dataset():
    rng = np.random.default_rng(42)
    n = 300
    lats = rng.uniform(5.50, 5.65, n)
    lons = rng.uniform(-0.35, -0.15, n)

    # Inject localized subsidence bowl around Circle (5.559, -0.208)
    v_vert = rng.normal(-1.0, 1.0, n)
    dists = np.sqrt((lats - 5.5592) ** 2 + (lons - -0.2085) ** 2)
    bowl_mask = dists < 0.025
    v_vert[bowl_mask] -= 25.0

    inc = np.full(n, 37.5)
    theta_rad = np.radians(inc)
    v_los = v_vert * np.cos(theta_rad)

    df = pd.DataFrame({
        "point_id": [f"PS_{i:04d}" for i in range(n)],
        "latitude": lats,
        "longitude": lons,
        "v_los": v_los,
        "v_los_std": rng.uniform(0.5, 1.5, n),
        "incidence_angle_deg": inc,
        "heading_angle_deg": np.full(n, -167.0),
        "temporal_coherence": rng.uniform(0.20, 0.95, n),
    })

    return InSARPointDataset(
        df=df,
        sensor="Sentinel-1_C-band",
        track_type="Descending",
        wavelength_cm=5.55,
    )


def test_los_to_vertical_decomposition(mock_insar_dataset):
    engine = InSARDeformationEngine()
    decomp = engine.decompose_los_to_vertical(mock_insar_dataset)

    assert "v_vert" in decomp.columns
    assert "v_vert_std" in decomp.columns

    # Verify mathematical formula v_vert = v_los / cos(theta)
    theta_rad = np.radians(decomp["incidence_angle_deg"].iloc[0])
    expected_v = decomp["v_los"].iloc[0] / math.cos(theta_rad)
    assert abs(decomp["v_vert"].iloc[0] - expected_v) < 0.05


def test_dual_geometry_decomposition():
    engine = InSARDeformationEngine()

    # Exact known vertical and east velocities
    true_v_vert = -25.0
    true_v_east = 3.5

    t_asc, h_asc = 38.0, -13.0
    t_desc, h_desc = 37.0, -167.0

    # Project to LOS
    # v_los = v_vert*cos(theta) - v_east*sin(theta)*cos(heading)
    t_a_rad, h_a_rad = math.radians(t_asc), math.radians(h_asc)
    t_d_rad, h_d_rad = math.radians(t_desc), math.radians(h_desc)

    v_los_asc = true_v_vert * math.cos(t_a_rad) - true_v_east * math.sin(t_a_rad) * math.cos(h_a_rad)
    v_los_desc = true_v_vert * math.cos(t_d_rad) - true_v_east * math.sin(t_d_rad) * math.cos(h_d_rad)

    df_asc = pd.DataFrame([{
        "point_id": "P_01", "latitude": 5.56, "longitude": -0.21,
        "v_los": v_los_asc, "v_los_std": 0.5, "incidence_angle_deg": t_asc,
        "heading_angle_deg": h_asc, "temporal_coherence": 0.85
    }])
    df_desc = pd.DataFrame([{
        "point_id": "P_01_D", "latitude": 5.56, "longitude": -0.21,
        "v_los": v_los_desc, "v_los_std": 0.5, "incidence_angle_deg": t_desc,
        "heading_angle_deg": h_desc, "temporal_coherence": 0.80
    }])

    ds_asc = InSARPointDataset(df=df_asc, sensor="Sentinel-1", track_type="Ascending", wavelength_cm=5.55)
    ds_desc = InSARPointDataset(df=df_desc, sensor="Sentinel-1", track_type="Descending", wavelength_cm=5.55)

    dual_res = engine.decompose_dual_geometry(ds_asc, ds_desc, max_dist_km=0.5)

    assert "v_vert" in dual_res.columns
    assert "v_east" in dual_res.columns
    assert abs(dual_res["v_vert"].iloc[0] - true_v_vert) < 0.1
    assert abs(dual_res["v_east"].iloc[0] - true_v_east) < 0.1


def test_coherence_filtering(mock_insar_dataset):
    engine = InSARDeformationEngine(default_coherence_threshold=0.50)
    filtered = engine.filter_by_coherence(mock_insar_dataset.df)

    assert len(filtered) < len(mock_insar_dataset.df)
    assert (filtered["temporal_coherence"] >= 0.50).all()


def test_subsidence_bowl_identification(mock_insar_dataset):
    engine = InSARDeformationEngine()
    decomp = engine.decompose_los_to_vertical(mock_insar_dataset)

    bowls = engine.identify_subsidence_bowls(
        decomp,
        subsidence_threshold=-15.0,
        eps_km=1.2,
        min_samples=4,
    )

    assert len(bowls) >= 1
    b = bowls[0]
    assert isinstance(b, SubsidenceBowl)
    assert b.max_subsidence_rate_mm_yr <= -15.0
    assert b.area_km2 > 0.0
    assert b.radius_km > 0.0
    assert b.annual_volume_deficit_m3_yr > 0.0
    assert b.risk_level in ["Critical", "Severe", "Moderate"]


def test_velocity_grid_interpolation(mock_insar_dataset):
    engine = InSARDeformationEngine()
    decomp = engine.decompose_los_to_vertical(mock_insar_dataset)

    lats = np.linspace(5.50, 5.65, 20)
    lons = np.linspace(-0.35, -0.15, 20)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    grid_v = engine.interpolate_velocity_grid(decomp, lat_grid, lon_grid)

    assert grid_v.shape == (20, 20)
    assert not np.isnan(grid_v).any()
