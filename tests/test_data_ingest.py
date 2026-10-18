"""
Unit Tests: Data Ingest & Hydrology Grid Module
===============================================
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from accra_engine.data_ingest import (
    AccraDataIngest,
    PrecipitationRecord,
    InSARPointDataset,
    DEMHydrologyGrid,
)


@pytest.fixture
def temp_cache_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_precipitation_record_initialization():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    precip = np.random.exponential(scale=10.0, size=100)
    df = pd.DataFrame({"date": dates, "precipitation_mm": precip})

    rec = PrecipitationRecord(
        df=df,
        source="Test_GMet",
        station_name="Accra KIA",
        latitude=5.60,
        longitude=-0.17,
    )

    assert rec.n_records == 100
    assert rec.start_date == pd.Timestamp("2020-01-01")
    assert rec.end_date == pd.Timestamp("2020-04-09")
    assert rec.max_daily_precip >= 0.0
    assert len(rec.annual_maximums) == 1


def test_precipitation_record_validation():
    invalid_df = pd.DataFrame({"timestamp": [1, 2], "rain": [10.0, 20.0]})
    with pytest.raises(ValueError, match="must contain 'date' and 'precipitation_mm'"):
        PrecipitationRecord(
            df=invalid_df,
            source="Test",
            station_name="Test",
            latitude=5.0,
            longitude=-0.1,
        )


def test_data_ingest_fetch_rainfall(temp_cache_dir):
    ingest = AccraDataIngest(cache_dir=temp_cache_dir)
    rec = ingest.fetch_historical_rainfall(start_year=2015, end_year=2016)

    assert isinstance(rec, PrecipitationRecord)
    assert rec.n_records == 731  # 2016 is a leap year (365 + 366 = 731)
    assert rec.max_daily_precip >= 150.0  # June 3, 2015 disaster calibration

    # Verify local disk caching
    cache_files = list(temp_cache_dir.glob("*.csv"))
    assert len(cache_files) == 1

    # Second call should load cached file directly
    rec2 = ingest.fetch_historical_rainfall(start_year=2015, end_year=2016)
    assert rec2.n_records == rec.n_records


def test_insar_dataset_structure_and_filtering():
    n_pts = 100
    df = pd.DataFrame({
        "point_id": [f"PS_{i:04d}" for i in range(n_pts)],
        "latitude": np.random.uniform(5.50, 5.65, n_pts),
        "longitude": np.random.uniform(-0.30, -0.15, n_pts),
        "v_los": np.random.normal(-15.0, 5.0, n_pts),
        "v_los_std": np.random.uniform(0.5, 2.0, n_pts),
        "incidence_angle_deg": np.full(n_pts, 37.5),
        "temporal_coherence": np.linspace(0.1, 0.9, n_pts),
    })

    ds = InSARPointDataset(
        df=df,
        sensor="Sentinel-1_C-band",
        track_type="Descending",
        wavelength_cm=5.55,
    )

    assert ds.n_points == 100
    assert ds.wavelength_cm == 5.55

    # Filter coherence at 0.35
    filtered = ds.filter_coherence(min_coherence=0.35)
    assert filtered.n_points < ds.n_points
    assert (filtered.df["temporal_coherence"] >= 0.35).all()


def test_data_ingest_load_or_generate_insar(temp_cache_dir):
    ingest = AccraDataIngest(cache_dir=temp_cache_dir)
    ds_c = ingest.load_or_generate_insar_data(n_points=200, sensor="Sentinel-1_C-band", seed=99)
    assert ds_c.n_points == 200
    assert ds_c.wavelength_cm == 5.55
    assert (ds_c.df["latitude"] >= ingest.ACCRA_BBOX["min_lat"]).all()
    assert (ds_c.df["latitude"] <= ingest.ACCRA_BBOX["max_lat"]).all()

    ds_l = ingest.load_or_generate_insar_data(n_points=100, sensor="NISAR_L-band", seed=100)
    assert ds_l.n_points == 100
    assert ds_l.wavelength_cm == 24.0


def test_dem_grid_generation(temp_cache_dir):
    ingest = AccraDataIngest(cache_dir=temp_cache_dir)
    dem = ingest.load_or_generate_dem_grid(grid_resolution=0.02)

    assert isinstance(dem, DEMHydrologyGrid)
    assert dem.elevation_m.shape == dem.slope_deg.shape
    assert dem.elevation_m.min() >= 0.0
    assert dem.slope_deg.min() >= 0.0
    assert dem.twi.min() > 0.0
    assert dem.dist_to_drainage_km.min() >= 0.0


def test_infrastructure_catalog():
    ingest = AccraDataIngest()
    infra = ingest.get_key_infrastructure_assets()

    assert isinstance(infra, pd.DataFrame)
    assert len(infra) >= 8
    expected_cols = ["id", "name", "latitude", "longitude", "category", "criticality"]
    for col in expected_cols:
        assert col in infra.columns
    assert (infra["criticality"] >= 1).all() and (infra["criticality"] <= 10).all()
