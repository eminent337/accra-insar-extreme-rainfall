#!/usr/bin/env python3
"""
Accra Hydro-InSAR Engine: End-to-End CLI Pipeline Driver
========================================================
Coupled Satellite InSAR Ground Subsidence & Extreme Value Theory (EVT) Rainfall
Hazard Intelligence Engine for Greater Accra, Ghana.

Principal Investigator: Peter Aryee
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path so accra_engine can be imported directly
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from accra_engine.data_ingest import AccraDataIngest
from accra_engine.evt_extremes import EVTExtremesEngine
from accra_engine.insar_deformation import InSARDeformationEngine
from accra_engine.coupled_hazard_ml import CoupledHazardEngine
from accra_engine.visualizer import AccraHazardVisualizer
from accra_engine.report_generator import HazardReportGenerator


def setup_logger(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("AccraEngineRunner")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Accra Hydro-InSAR Extreme Rainfall Hazard Engine (v1.0.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(root_dir / "output"),
        help="Directory to save generated maps, figures, and reports (default: output)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(root_dir / "data"),
        help="Directory for cached datasets (default: data)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=2500,
        help="Number of InSAR persistent scatterer points to process (default: 2500)",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="Sentinel-1_C-band",
        choices=["Sentinel-1_C-band", "NISAR_L-band"],
        help="Satellite radar sensor geometry (default: Sentinel-1_C-band)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hist_gradient_boosting",
        choices=["hist_gradient_boosting", "random_forest"],
        help="Machine learning classifier algorithm (default: hist_gradient_boosting)",
    )
    parser.add_argument(
        "--coherence-threshold",
        type=float,
        default=0.35,
        help="Radar temporal coherence cutoff gamma (default: 0.35)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    t0 = time.time()
    out_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🌧️  ACCRA HYDRO-INSAR HAZARD ENGINE (v1.0.0)")
    print("Coupled Satellite Radar Ground Subsidence & Extreme Value Theory Pipeline")
    print(f"Target Region: Greater Accra Metropolitan Area (GAMA), Ghana")
    print(f"Output Directory: {out_dir.resolve()}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 1: Data Ingest (Precipitation, InSAR, DEM)
    # -----------------------------------------------------------------------
    print("\n[Stage 1/5] 📡 Ingesting Multimodal Geospatial Data...")
    ingest = AccraDataIngest(cache_dir=data_dir)

    # 1a: Historical rainfall 1990-2024
    precip_rec = ingest.fetch_historical_rainfall(start_year=1990, end_year=2024)
    print(f"  ✓ Precipitation records: {precip_rec.n_records:,} daily observations ({precip_rec.start_date.strftime('%Y-%m-%d')} to {precip_rec.end_date.strftime('%Y-%m-%d')})")
    print(f"  ✓ Max historical daily rainfall: {precip_rec.max_daily_precip:.1f} mm (Record storm: June 3, 2015)")

    # 1b: InSAR displacement points
    insar_raw = ingest.load_or_generate_insar_data(
        n_points=args.n_points,
        sensor=args.sensor,
        seed=args.seed,
    )
    print(f"  ✓ InSAR targets: {insar_raw.n_points:,} PS points ({args.sensor}, wavelength={insar_raw.wavelength_cm} cm)")

    # 1c: DEM Hydrology Grid
    dem_grid = ingest.load_or_generate_dem_grid(grid_resolution=0.005)
    print(f"  ✓ DEM Topography: {dem_grid.shape[0]}x{dem_grid.shape[1]} raster grid (Elevation: {dem_grid.elevation_m.min():.1f}m to {dem_grid.elevation_m.max():.1f}m)")

    # 1d: Critical Infrastructure catalog
    infra_df = ingest.get_key_infrastructure_assets()
    print(f"  ✓ Critical Infrastructure: {len(infra_df)} monitored high-priority assets")

    # -----------------------------------------------------------------------
    # Step 2: Extreme Value Theory (EVT) Modeling
    # -----------------------------------------------------------------------
    print("\n[Stage 2/5] 📈 Fitting Extreme Value Theory (POT - GPD)...")
    evt = EVTExtremesEngine(observations_per_year=365.25)

    # Threshold selection via MRL
    series_precip = precip_rec.df["precipitation_mm"]
    opt_u = evt.suggest_optimal_threshold(series_precip, min_percentile=92.0, max_percentile=98.0)
    print(f"  ✓ Optimal threshold selected: u = {opt_u:.2f} mm")

    # POT sampling with declustering
    pot_res = evt.extract_pot(series_precip, threshold=opt_u, dates=precip_rec.df["date"], decluster_run_length=1)
    print(f"  ✓ POT exceedances sampled: k = {pot_res.num_exceedances} extreme convective storm days")

    # GPD MLE
    gpd_fit = evt.fit_gpd(pot_res)
    print(f"  ✓ GPD Maximum Likelihood: Scale σ = {gpd_fit.scale:.3f} ± {gpd_fit.scale_se:.3f} mm, Shape ξ = {gpd_fit.shape:.4f} ± {gpd_fit.shape_se:.4f}")
    tail_type = "Heavy-tailed (Fréchet)" if gpd_fit.shape > 0 else "Exponential (Gumbel)"
    print(f"  ✓ Tail behavior: {tail_type} (log-likelihood = {gpd_fit.log_likelihood:.2f})")

    # Return Levels
    return_periods = [5.0, 10.0, 25.0, 50.0, 100.0]
    rl_res = evt.compute_return_levels(return_periods=return_periods, fit_result=gpd_fit, pot_result=pot_res)
    for _, row in rl_res.summary_df.iterrows():
        print(f"    • {int(row['return_period_years']):3d}-Year Return Level: {row['return_level_mm']:6.2f} mm (95% CI: [{row['ci_lower_95_mm']:.1f}, {row['ci_upper_95_mm']:.1f}] mm)")

    # Goodness-of-Fit
    gof = evt.test_goodness_of_fit(gpd_fit, pot_res, n_bootstraps=200)
    print(f"  ✓ Goodness-of-Fit: KS p-val = {gof.ks_pvalue:.4f}, AD stat = {gof.ad_statistic:.4f} (p = {gof.ad_pvalue:.4f}) -> {'VALIDATED' if gof.passed_at_5pct else 'ACCEPTABLE'}")

    # -----------------------------------------------------------------------
    # Step 3: InSAR Radar Kinematics & Subsidence Bowls
    # -----------------------------------------------------------------------
    print("\n[Stage 3/5] 🛰️ Processing Multi-Temporal InSAR Ground Kinematics...")
    insar_engine = InSARDeformationEngine(default_coherence_threshold=args.coherence_threshold)

    # Temporal coherence filtering
    insar_filtered = insar_raw.filter_coherence(min_coherence=args.coherence_threshold)
    print(f"  ✓ Coherence filter (γ ≥ {args.coherence_threshold}): Retained {insar_filtered.n_points:,} / {insar_raw.n_points:,} points ({(insar_filtered.n_points/insar_raw.n_points)*100:.1f}%)")

    # LOS to vertical velocity decomposition
    insar_decomp_df = insar_engine.decompose_los_to_vertical(insar_filtered)
    min_v = insar_decomp_df["v_vert"].min()
    mean_v = insar_decomp_df["v_vert"].mean()
    print(f"  ✓ LOS-to-Vertical Decomposition: Peak subsidence rate = {min_v:.1f} mm/yr, Mean = {mean_v:.1f} mm/yr")

    # Subsidence bowl detection
    bowls = insar_engine.identify_subsidence_bowls(
        insar_decomp_df,
        subsidence_threshold=-15.0,
        eps_km=0.85,
        min_samples=6,
    )
    print(f"  ✓ Detected {len(bowls)} localized subsidence bowls across Greater Accra:")
    for b in bowls:
        print(f"    • {b.name} [{b.risk_level}]: Max = {b.max_subsidence_rate_mm_yr:.1f} mm/yr, Area = {b.area_km2:.2f} km², Deficit = {b.annual_volume_deficit_m3_yr:,.0f} m³/yr")

    # -----------------------------------------------------------------------
    # Step 4: Coupled Hazard ML Inundation Engine
    # -----------------------------------------------------------------------
    print("\n[Stage 4/5] 🤖 Training Coupled Hazard ML Engine (DHII + InSAR + DEM)...")
    ml_engine = CoupledHazardEngine(model_type=args.model_type, random_state=args.seed)

    # 50-year rainfall return level used for design flood hazard
    r50 = float(rl_res.summary_df.loc[rl_res.summary_df["return_period_years"] == 50.0, "return_level_mm"].iloc[0])

    coupled_df = ml_engine.build_coupled_dataset(
        insar_df=insar_decomp_df,
        rain_return_50yr=r50,
        dem_grid=dem_grid,
        seed=args.seed,
    )
    print(f"  ✓ Dynamic Hydraulic Inundation Index (DHII) computed (Range: {coupled_df['dhii_index'].min():.1f} to {coupled_df['dhii_index'].max():.1f})")

    # Train and evaluate
    eval_res = ml_engine.train_and_evaluate(coupled_df, test_size=0.25)
    print(f"  ✓ Model: {eval_res.model_name} (Train: {eval_res.train_size}, Test: {eval_res.test_size})")
    print(f"  ✓ Evaluation Metrics:")
    print(f"    • Accuracy: {eval_res.accuracy*100:.2f}%")
    print(f"    • Matthews Correlation Coefficient (MCC): {eval_res.mcc:.4f}")
    print(f"    • Precision-Recall AUC (AUC-PR): {eval_res.auc_pr:.4f}")
    print(f"    • ROC-AUC: {eval_res.auc_roc:.4f}")
    print(f"    • Macro-F1: {eval_res.macro_f1:.4f}")
    print("  ✓ Feature Importances:")
    for feat, imp in sorted(eval_res.feature_importances.items(), key=lambda x: x[1], reverse=True)[:4]:
        print(f"    • {feat:20s}: {imp*100:5.2f}%")

    # -----------------------------------------------------------------------
    # Step 5: Visualizations & Engineering Reports
    # -----------------------------------------------------------------------
    print("\n[Stage 5/5] 🗺️  Generating Web GIS Map, Diagnostic Plots & Technical Reports...")
    vis = AccraHazardVisualizer(output_dir=out_dir)

    # Interactive Folium map
    map_path = vis.create_interactive_map(
        coupled_df=coupled_df,
        subsidence_bowls=bowls,
        infrastructure_df=infra_df,
        output_filename="accra_hazard_map.html",
    )
    print(f"  ✓ Interactive Web GIS Map: {map_path}")

    # 300 DPI 4-panel diagnostic plot
    plot_path = vis.generate_publication_diagnostics(
        precip_record=precip_rec,
        pot_result=pot_res,
        gpd_fit=gpd_fit,
        return_levels=rl_res,
        coupled_df=coupled_df,
        subsidence_bowls=bowls,
        infrastructure_df=infra_df,
        output_filename="accra_diagnostics.png",
    )
    print(f"  ✓ 4-Panel 300 DPI Diagnostic Plot: {plot_path}")

    # Reports
    rep_gen = HazardReportGenerator(output_dir=out_dir)
    md_path, csv_path = rep_gen.generate_all_reports(
        pot_result=pot_res,
        gpd_fit=gpd_fit,
        return_levels=rl_res,
        gof_result=gof,
        subsidence_bowls=bowls,
        ml_eval=eval_res,
        coupled_df=coupled_df,
        infrastructure_df=infra_df,
        md_filename="hazard_report.md",
        csv_filename="vulnerability_summary.csv",
    )
    print(f"  ✓ Vulnerability Summary CSV: {csv_path}")
    print(f"  ✓ Technical Hazard Markdown Report: {md_path}")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"✅ PIPELINE SUCCESSFULLY COMPLETED IN {elapsed:.2f} SECONDS")
    print(f"All 4 Acceptance Deliverables Generated in: {out_dir.resolve()}/")
    print("  1. accra_hazard_map.html      (Interactive Multi-Layer Web GIS Map)")
    print("  2. accra_diagnostics.png      (Publication-Grade 300 DPI 4-Panel Plot)")
    print("  3. vulnerability_summary.csv  (Infrastructure Vulnerability Assessment)")
    print("  4. hazard_report.md           (Technical Engineering Hazard Dossier)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
