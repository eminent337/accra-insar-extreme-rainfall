"""
Unit Tests: Coupled Hazard ML Module
====================================
"""

import numpy as np
import pandas as pd
import pytest

from accra_engine.coupled_hazard_ml import (
    CoupledHazardEngine,
    HazardClassificationResult,
)


@pytest.fixture
def sample_insar_decomp_df():
    rng = np.random.default_rng(42)
    n = 250
    df = pd.DataFrame({
        "point_id": [f"PS_{i:04d}" for i in range(n)],
        "latitude": rng.uniform(5.50, 5.65, n),
        "longitude": rng.uniform(-0.35, -0.15, n),
        "v_vert": rng.uniform(-35.0, 2.0, n),
        "temporal_coherence": rng.uniform(0.40, 0.95, n),
    })
    return df


def test_dhii_computation():
    # Test 1: Sinking ground gives higher hazard than stable ground
    dhii_sinking = CoupledHazardEngine.compute_dhii(
        rainfall_return_mm=180.0,
        subsidence_rate_mm_yr=-30.0,
        slope_deg=1.0,
        elevation_m=5.0,
        twi=14.0,
        dist_to_drainage_km=0.2,
    )

    dhii_stable = CoupledHazardEngine.compute_dhii(
        rainfall_return_mm=180.0,
        subsidence_rate_mm_yr=0.0,
        slope_deg=1.0,
        elevation_m=5.0,
        twi=14.0,
        dist_to_drainage_km=0.2,
    )

    assert dhii_sinking > dhii_stable

    # Test 2: Flat slope gives higher hazard than steep slope
    dhii_flat = CoupledHazardEngine.compute_dhii(
        rainfall_return_mm=150.0,
        subsidence_rate_mm_yr=-10.0,
        slope_deg=0.5,
        elevation_m=10.0,
        twi=12.0,
        dist_to_drainage_km=0.5,
    )

    dhii_steep = CoupledHazardEngine.compute_dhii(
        rainfall_return_mm=150.0,
        subsidence_rate_mm_yr=-10.0,
        slope_deg=22.0,
        elevation_m=10.0,
        twi=12.0,
        dist_to_drainage_km=0.5,
    )

    assert dhii_flat > dhii_steep

    # Test 3: Array inputs and bounds [0, 100]
    r_arr = np.array([80.0, 120.0, 200.0])
    s_arr = np.array([-5.0, -20.0, -35.0])
    slope_arr = np.array([10.0, 5.0, 0.5])
    elev_arr = np.array([50.0, 20.0, 2.0])
    twi_arr = np.array([6.0, 10.0, 15.0])
    dist_arr = np.array([3.0, 1.0, 0.1])

    dhii_res = CoupledHazardEngine.compute_dhii(
        r_arr, s_arr, slope_arr, elev_arr, twi_arr, dist_arr
    )
    assert len(dhii_res) == 3
    assert (dhii_res >= 0.0).all() and (dhii_res <= 100.0).all()
    assert dhii_res[0] < dhii_res[1] < dhii_res[2]


def test_build_coupled_dataset(sample_insar_decomp_df):
    engine = CoupledHazardEngine()
    coupled_df = engine.build_coupled_dataset(
        insar_df=sample_insar_decomp_df,
        rain_return_50yr=185.0,
        seed=42,
    )

    assert "dhii_index" in coupled_df.columns
    assert "hazard_class" in coupled_df.columns
    assert "hazard_label" in coupled_df.columns
    assert "v_vert_mm_yr" in coupled_df.columns
    assert set(coupled_df["hazard_class"].unique()).issubset({0, 1, 2})


def test_model_training_and_evaluation(sample_insar_decomp_df):
    engine = CoupledHazardEngine(model_type="hist_gradient_boosting", random_state=42)
    coupled_df = engine.build_coupled_dataset(sample_insar_decomp_df, rain_return_50yr=185.0)

    eval_res = engine.train_and_evaluate(coupled_df, test_size=0.3)

    assert isinstance(eval_res, HazardClassificationResult)
    assert 0.0 <= eval_res.accuracy <= 1.0
    assert -1.0 <= eval_res.mcc <= 1.0
    assert 0.0 <= eval_res.auc_pr <= 1.0
    assert 0.0 <= eval_res.auc_roc <= 1.0
    assert eval_res.confusion_matrix.shape == (3, 3)
    assert len(eval_res.feature_importances) == len(engine.FEATURE_NAMES)


def test_random_forest_training_and_evaluation(sample_insar_decomp_df):
    engine = CoupledHazardEngine(model_type="random_forest", random_state=42)
    coupled_df = engine.build_coupled_dataset(sample_insar_decomp_df, rain_return_50yr=185.0)

    eval_res = engine.train_and_evaluate(coupled_df, test_size=0.3)

    assert eval_res.model_name == "random_forest"
    assert eval_res.accuracy > 0.5
    assert sum(eval_res.feature_importances.values()) == pytest.approx(1.0, abs=1e-2)


def test_prediction_pipeline(sample_insar_decomp_df):
    engine = CoupledHazardEngine()
    coupled_df = engine.build_coupled_dataset(sample_insar_decomp_df, rain_return_50yr=185.0)
    engine.train_and_evaluate(coupled_df)

    preds = engine.predict(coupled_df.iloc[:10])
    probas = engine.predict_proba(coupled_df.iloc[:10])

    assert len(preds) == 10
    assert set(preds).issubset({0, 1, 2})
    assert probas.shape == (10, 3)
    assert np.allclose(probas.sum(axis=1), 1.0)
