# 🌧️ Coupled Satellite InSAR Subsidence & Extreme Rainfall Hazard Engine: Greater Accra, Ghana
**Author & Principal Investigator:** Peter Aryee  
**Focus:** Fusing Extreme Value Theory (EVT) Rainfall Extremes with Multi-Temporal InSAR Ground Subsidence to Predict Flood Inundation & Infrastructure Vulnerability in Greater Accra.

---

## 1. The Core Scientific Premise
Accra experiences catastrophic annual flash floods (e.g. June 3 disasters, Odaw River basin, Kwame Nkrumah Circle, Weija/Gbawe drainage corridors). 

Standard flood hazard models fail because they make a false assumption: **they assume the ground elevation is static (fixed)**.
In reality:
1. **Severe Ground Subsidence:** Rapid urbanization, illegal reclamation of wetlands, and compaction of soft coastal soils cause parts of Accra to subside at rates of **-10 to -35 mm/year**!
2. **Extreme Convective Rainfall Bursts:** Intense tropical downpours exceed drainage capacity.
3. **The Coupled Hazard Mechanism:**
   $$\text{Flood Vulnerability Index} = f\Big(\text{Rainfall Extreme Exceedance (EVT)}, \; \text{Cumulative InSAR Subsidence } (\Delta z_{\text{InSAR}}), \; \text{Terrain Runoff (DEM)}\Big)$$
   When sinking ground (measured by Sentinel-1 InSAR) coincides with extreme rainfall events (modeled by Extreme Value Theory GPD), storm runoff becomes trapped, submerging roads, bridges, and communities.

---

## 2. Data Sources (100% Free & Open-Access):
1. **Satellite Radar (InSAR):**
   * **Sensor:** Sentinel-1 C-band SAR (Copernicus / ESA).
   * **Coverage:** Greater Accra Metropolitan Area (GAMA) / Odaw River Basin (Track 131 / Ascending & Descending).
   * **Output:** Mean velocity ($v_{\text{LOS}}$ in mm/yr) and displacement time-series (2018–2026).
2. **Rainfall & Atmospheric Records for Accra:**
   * **GMet (Ghana Meteorological Agency) station data:** Kotoka International Airport (KIA), Tema, Pokuase.
   * **NASA GPM (Global Precipitation Measurement) & CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data):** 0.05° high-resolution daily/hourly precipitation grid over Accra (1981–2026).
   * **ECMWF ERA5 Reanalysis:** Atmospheric moisture flux, total precipitable water, convective available potential energy (CAPE).

---

## 3. Methodological Architecture:
```
   [ Atmospheric & Rainfall Data ]             [ Satellite Radar InSAR Data ]
     (CHIRPS / GMet / ERA5)                         (Sentinel-1 SLC)
                │                                           │
                ▼                                           ▼
┌────────────────────────────────┐          ┌────────────────────────────────┐
│  Extreme Value Theory (EVT)    │          │  Multi-Temporal InSAR Pipeline │
│  • Peaks-Over-Threshold (POT)  │          │  • PS / SBAS Interferometry    │
│  • Generalized Pareto (GPD)    │          │  • Phase Unwrapping (SNAPHU)   │
│  • 10, 25, 50-Yr Return Periods│          │  • Ground Subsidence (mm/yr)   │
└────────────────┬───────────────┘          └────────────────┬───────────────┘
                 │                                           │
                 └───────────────────┬───────────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │ Coupled Machine Learning Hazard AI  │
                  │ • XGBoost / HistGradientBoosting    │
                  │ • High-Resolution DEM Topography    │
                  │ • Dynamic Flood Inundation Index    │
                  │ • SHAP Explainability Features      │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                [ Interactive Web GIS & Risk Decision Map ]
                     (Odaw River, Circle, Ga North)
```

---

## 4. Why This Project is an Academic Masterstroke for Peter:
1. **Solves a Real National Crisis:** Solves the flooding crisis in your home city (Accra), directly leveraging your field surveying at **Ga North Municipal Assembly**!
2. **Fuses Your Two Superpowers:**
   * Your **Chirano InSAR radar processing** experience.
   * Your **London Machine Learning & Extreme Value Theory (EVT)** experience.
3. **Publication Potential:** This is a direct target for top-tier journals (*Remote Sensing of Environment*, *Natural Hazards and Earth System Sciences - NHESS*, or *Science of the Total Environment*).
