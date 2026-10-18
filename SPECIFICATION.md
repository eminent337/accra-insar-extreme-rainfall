# 🌧️ Accra Hydro-InSAR Engine (v1.0.0 Specification)
### Coupled Multi-Temporal Satellite InSAR Ground Subsidence & Extreme Value Theory (EVT) Rainfall Hazard Intelligence Engine for Greater Accra, Ghana

## 1. Scientific & Commercial Mandate
Build a commercial-grade, production-quality geospatial intelligence package that couples satellite radar interferometry (Sentinel-1 C-band & NISAR L-band ready) with statistical Extreme Value Theory (EVT) and Machine Learning to map urban flood inundation and infrastructure vulnerability in Greater Accra, Ghana.
- High scientific rigor: Exact POT-GPD return level estimation, real atmospheric reanalysis data, multi-temporal InSAR displacement fields.
- 100% reproducible: Pure Python scientific stack (NumPy, SciPy, Pandas, Matplotlib, Folium).
- Complete automated unit tests (>90% coverage with pytest).
- Interactive Leaflet Web GIS visualization + 300 DPI publication diagnostic plots.
- Standalone CLI execution (`run_engine.py`).

## 2. Core Modules in `src/accra_engine/`:
1. `data_ingest.py`:
   - Automated retrieval of historical precipitation records for Greater Accra (ERA5 / CHIRPS / NASA GPM).
   - Multi-sensor SAR displacement interface: Sentinel-1 C-band PS/SBAS displacement grids + NISAR L-band ingest schema.
   - High-resolution Digital Elevation Model (DEM) and drainage vector handler.
2. `evt_extremes.py`:
   - Rigorous Extreme Value Theory (EVT) implementation:
     * Mean Residual Life (MRL) plot computation for optimal threshold selection ($u$).
     * Peaks-Over-Threshold (POT) sampling.
     * Generalized Pareto Distribution (GPD) maximum likelihood estimation (MLE):
       $$F(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}$$
     * Computation of 5, 10, 25, 50, and 100-year return level precipitation events ($x_T$).
     * Kolmogorov-Smirnov and Anderson-Darling goodness-of-fit testing.
3. `insar_deformation.py`:
   - Multi-temporal radar deformation modeling:
     * LOS velocity ($v_{\text{LOS}}$) to vertical velocity ($v_{\text{vert}}$) decomposition: $v_{\text{vert}} = v_{\text{LOS}} / \cos(\theta_{\text{inc}})$.
     * Temporal coherence filtering ($\gamma \ge 0.35$).
     * Kinematic subsidence mapping across key Accra vulnerability zones (Odaw River, Kwame Nkrumah Circle, Weija, Gbawe, Glefe, Korle Lagoon).
     * Identification of localized "subsidence bowl" depressions.
4. `coupled_hazard_ml.py`:
   - Multi-hazard vulnerability engine:
     * Dynamic Hydraulic Inundation Index ($DHII$):
       $$DHII = f(\text{Rainfall Return Level}, \; \Delta z_{\text{InSAR}}, \; \text{Slope}, \; \text{Topographic Wetness Index})$$
     * Machine Learning Classification (HistGradientBoosting / Random Forest) trained on historical flood events.
     * Class-imbalanced rare-event evaluation (AUC-PR, Matthews Correlation Coefficient).
     * Feature importance analysis.
5. `visualizer.py`:
   - Interactive Folium Web GIS map:
     * Layer 1: InSAR subsidence velocity contours (mm/yr).
     * Layer 2: Extreme rainfall inundation flood risk polygons (High/Moderate/Low).
     * Layer 3: Critical infrastructure exposure (bridges, major arterial roads, clinics).
   - Publication-quality 4-panel diagnostic plot (300 DPI):
     * Panel A: Historical rainfall time series & POT-GPD threshold fit.
     * Panel B: InSAR mean vertical subsidence velocity map ($v_{\text{vert}}$).
     * Panel C: EVT Return Level curves (5 to 100-year return periods).
     * Panel D: Coupled flood risk classification map across Greater Accra.
6. `report_generator.py`:
   - Automated engineering flood vulnerability report (Markdown and CSV) detailing station return levels, maximum subsidence hotspots, and exposed infrastructure assets.

## 3. Package Structure:
```
accra-insar-extreme-rainfall/
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
├── run_engine.py
├── SPECIFICATION.md
├── src/
│   └── accra_engine/
│       ├── __init__.py
│       ├── data_ingest.py
│       ├── evt_extremes.py
│       ├── insar_deformation.py
│       ├── coupled_hazard_ml.py
│       ├── visualizer.py
│       └── report_generator.py
├── tests/
│   ├── __init__.py
│   ├── test_data_ingest.py
│   ├── test_evt_extremes.py
│   ├── test_insar_deformation.py
│   └── test_coupled_hazard.py
├── notebooks/
│   └── 01_accra_hazard_walkthrough.ipynb
└── output/
```
