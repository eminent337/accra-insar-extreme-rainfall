"""
Visualizer Module: Accra Hydro-InSAR Engine
===========================================
Generates publication-quality visualizations:
1. Interactive Folium Web GIS Map with multi-layer overlays (subsidence velocity,
   flood hazard zones, critical infrastructure, subsidence bowls).
2. 4-Panel 300 DPI diagnostic scientific figure for publication and decision-makers.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import folium
from folium import plugins
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from .data_ingest import DEMHydrologyGrid, PrecipitationRecord
from .evt_extremes import GPDFitResult, POTResult, ReturnLevelResult
from .insar_deformation import SubsidenceBowl

logger = logging.getLogger(__name__)


class AccraHazardVisualizer:
    """
    Geospatial visualization and scientific plotting engine for Accra hazard intelligence.
    """

    ACCRA_CENTER = [5.565, -0.215]

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Interactive Folium Web GIS Map
    # -----------------------------------------------------------------------
    def create_interactive_map(
        self,
        coupled_df: pd.DataFrame,
        subsidence_bowls: List[SubsidenceBowl],
        infrastructure_df: pd.DataFrame,
        output_filename: str = "accra_hazard_map.html",
    ) -> Path:
        """
        Build an interactive Leaflet/Folium multi-layer GIS map of Greater Accra:
        - Layer 1: InSAR Vertical Subsidence Velocity (mm/yr)
        - Layer 2: Extreme Flood Hazard Zones (Low, Moderate, Critical)
        - Layer 3: Localized Subsidence Bowl Depressions
        - Layer 4: Critical Infrastructure Exposure & Threat Assessment
        """
        out_path = self.output_dir / output_filename

        # Base Map
        m = folium.Map(
            location=self.ACCRA_CENTER,
            zoom_start=12,
            tiles=None,
            control_scale=True,
        )

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Esri Satellite Imagery",
            overlay=False,
            control=True,
            show=True
        ).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Topo Map",
            name="Esri Topographic Map",
            overlay=False,
            control=True,
            show=False
        ).add_to(m)

        # ----------------- Layer 1: InSAR Subsidence Points -----------------
        fg_insar = folium.FeatureGroup(name="InSAR Ground Subsidence (v_vert mm/yr)", show=True)

        # Color mapping for vertical velocity
        def get_v_color(v: float) -> str:
            if v <= -25.0:
                return "#800026"  # Dark red: severe subsidence
            elif v <= -15.0:
                return "#e31a1c"  # Red
            elif v <= -8.0:
                return "#fd8d3c"  # Orange
            elif v <= -3.0:
                return "#fecc5c"  # Yellow
            else:
                return "#2b83ba"  # Blue: stable / uplift

        # Sample points to keep HTML lightweight if point count > 2000
        sample_df = coupled_df.sample(min(len(coupled_df), 1800), random_state=42)

        for _, row in sample_df.iterrows():
            v_val = row["v_vert"]
            coherence = row.get("temporal_coherence", 0.75)
            pt_id = row.get("point_id", "PS")
            color = get_v_color(v_val)

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; width: 200px;">
                <h4 style="margin: 0 0 5px 0; color: #1a5276;">InSAR Target: {pt_id}</h4>
                <hr style="margin: 2px 0 6px 0;">
                <b>Vertical Velocity:</b> <span style="color: {color}; font-weight: bold;">{v_val:.1f} mm/yr</span><br>
                <b>Temporal Coherence:</b> {coherence:.2f}<br>
                <b>Elevation:</b> {row.get('elevation_m', 0.0):.1f} m<br>
                <b>Lat/Lon:</b> {row['latitude']:.4f}, {row['longitude']:.4f}
            </div>
            """

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3.5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                weight=0.5,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(fg_insar)

        fg_insar.add_to(m)

        # ----------------- Layer 2: Flood Hazard Zones -----------------
        fg_hazard = folium.FeatureGroup(name="Coupled Flood Inundation Hazard", show=True)

        hazard_colors = {
            0: ("#2ca02c", "Low Hazard"),
            1: ("#ff7f0e", "Moderate Hazard"),
            2: ("#d62728", "Critical Hazard"),
        }

        # Subsample hazard points
        for _, row in sample_df.iterrows():
            h_class = int(row.get("hazard_class", 0))
            color, label = hazard_colors.get(h_class, ("#2ca02c", "Low"))
            if h_class == 0 and np.random.default_rng(42).random() > 0.35:
                continue  # thin out low-hazard points for visual clarity

            dhii = row.get("dhii_index", 50.0)
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; width: 220px;">
                <h4 style="margin: 0 0 5px 0; color: {color};">{label}</h4>
                <hr style="margin: 2px 0 6px 0;">
                <b>DHII Score:</b> <b>{dhii:.1f} / 100</b><br>
                <b>50-Yr Rain Pressure:</b> {row.get('rainfall_50yr_mm', 0):.1f} mm<br>
                <b>Subsidence Rate:</b> {row['v_vert']:.1f} mm/yr<br>
                <b>Slope:</b> {row.get('slope_deg', 0):.1f}&deg; | <b>TWI:</b> {row.get('twi', 0):.1f}
            </div>
            """

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5.0 if h_class == 2 else 3.5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.65 if h_class == 2 else 0.45,
                weight=1,
                popup=folium.Popup(popup_html, max_width=260),
            ).add_to(fg_hazard)

        fg_hazard.add_to(m)

        # ----------------- Layer 3: Subsidence Bowls -----------------
        fg_bowls = folium.FeatureGroup(name="Identified Subsidence Bowls", show=True)

        for bowl in subsidence_bowls:
            b_color = "#800026" if bowl.risk_level == "Critical" else "#d7301f"
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; width: 260px;">
                <h4 style="margin: 0 0 4px 0; color: #900C3F;">🚨 {bowl.name}</h4>
                <hr style="margin: 2px 0 6px 0;">
                <b>Risk Category:</b> <span style="color: red; font-weight: bold;">{bowl.risk_level}</span><br>
                <b>Max Subsidence Rate:</b> <b>{bowl.max_subsidence_rate_mm_yr:.1f} mm/yr</b><br>
                <b>Mean Subsidence Rate:</b> {bowl.mean_subsidence_rate_mm_yr:.1f} mm/yr<br>
                <b>Depression Area:</b> {bowl.area_km2:.2f} km&sup2;<br>
                <b>Effective Radius:</b> {bowl.radius_km:.2f} km<br>
                <b>Annual Volume Deficit:</b> {bowl.annual_volume_deficit_m3_yr:,.0f} m&sup3;/yr<br>
                <b>Monitored Radar Points:</b> {bowl.point_count}
            </div>
            """

            # Depression boundary circle
            folium.Circle(
                location=[bowl.centroid_lat, bowl.centroid_lon],
                radius=bowl.radius_km * 1000.0,
                color=b_color,
                weight=2.5,
                dash_array="6, 6",
                fill=True,
                fill_color=b_color,
                fill_opacity=0.20,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(fg_bowls)

            # Center marker
            folium.Marker(
                location=[bowl.centroid_lat, bowl.centroid_lon],
                icon=folium.Icon(color="red", icon="warning-sign"),
                tooltip=f"{bowl.name} ({bowl.max_subsidence_rate_mm_yr:.1f} mm/yr)",
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(fg_bowls)

        fg_bowls.add_to(m)

        # ----------------- Layer 4: Critical Infrastructure -----------------
        fg_infra = folium.FeatureGroup(name="Critical Infrastructure Exposure", show=True)

        for _, asset in infrastructure_df.iterrows():
            a_lat = asset["latitude"]
            a_lon = asset["longitude"]
            a_name = asset["name"]
            a_cat = asset.get("category", "Infrastructure")
            crit = asset.get("criticality", 8)

            # Match local subsidence rate from nearest coupled point
            dists = np.sqrt(
                (coupled_df["latitude"] - a_lat) ** 2 +
                (coupled_df["longitude"] - a_lon) ** 2
            )
            nearest_idx = dists.idxmin()
            near_row = coupled_df.loc[nearest_idx]
            local_sub = near_row["v_vert"]
            local_hazard = near_row.get("hazard_label", "Moderate Hazard")
            local_dhii = near_row.get("dhii_index", 55.0)

            icon_color = "darkred" if local_hazard == "Critical Hazard" else "orange" if local_hazard == "Moderate Hazard" else "blue"

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px; width: 250px;">
                <h4 style="margin: 0 0 4px 0; color: #154360;">🏛️ {a_name}</h4>
                <hr style="margin: 2px 0 6px 0;">
                <b>Category:</b> {a_cat}<br>
                <b>Criticality Rating:</b> {crit} / 10<br>
                <b>Ground Subsidence Rate:</b> <b>{local_sub:.1f} mm/yr</b><br>
                <b>Hazard Zone:</b> <span style="font-weight: bold; color: {icon_color};">{local_hazard}</span><br>
                <b>DHII Vulnerability Score:</b> {local_dhii:.1f} / 100
            </div>
            """

            folium.Marker(
                location=[a_lat, a_lon],
                icon=folium.Icon(color=icon_color, icon="info-sign"),
                tooltip=f"{a_name} [{local_hazard}]",
                popup=folium.Popup(popup_html, max_width=290),
            ).add_to(fg_infra)

        fg_infra.add_to(m)

        # Layer Control
        folium.LayerControl(collapsed=False).add_to(m)

        # Legend HTML Overlay
        legend_html = """
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            width: 250px;
            height: auto;
            background-color: white;
            z-index: 9999;
            font-size: 12px;
            border: 2px solid grey;
            border-radius: 6px;
            padding: 10px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
            font-family: Arial, sans-serif;
        ">
            <b>Accra Hazard GIS Layers</b><br>
            <hr style="margin: 5px 0;">
            <b>InSAR Subsidence (v_vert):</b><br>
            <i style="background: #800026; width: 12px; height: 12px; display: inline-block;"></i> &lt; -25 mm/yr (Severe)<br>
            <i style="background: #e31a1c; width: 12px; height: 12px; display: inline-block;"></i> -15 to -25 mm/yr<br>
            <i style="background: #fd8d3c; width: 12px; height: 12px; display: inline-block;"></i> -8 to -15 mm/yr<br>
            <i style="background: #2b83ba; width: 12px; height: 12px; display: inline-block;"></i> &gt; -3 mm/yr (Stable)<br>
            <hr style="margin: 5px 0;">
            <b>Coupled Hazard Risk:</b><br>
            <i style="background: #d62728; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> Critical Hazard<br>
            <i style="background: #ff7f0e; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> Moderate Hazard<br>
            <i style="background: #2ca02c; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> Low Hazard<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        m.save(str(out_path))
        logger.info("Saved interactive Folium map to %s", out_path)
        return out_path

    # -----------------------------------------------------------------------
    # 2. Publication-Quality 4-Panel 300 DPI Diagnostic Plot
    # -----------------------------------------------------------------------
    def generate_publication_diagnostics(
        self,
        precip_record: PrecipitationRecord,
        pot_result: POTResult,
        gpd_fit: GPDFitResult,
        return_levels: ReturnLevelResult,
        coupled_df: pd.DataFrame,
        subsidence_bowls: List[SubsidenceBowl],
        infrastructure_df: pd.DataFrame,
        output_filename: str = "accra_diagnostics.png",
    ) -> Path:
        """
        Generate a 4-panel publication-grade 300 DPI scientific diagnostic plot:
        - Panel A: Historical Daily Precipitation & Peaks-Over-Threshold (POT) Exceedances
        - Panel B: InSAR Vertical Velocity Field (mm/yr) & Subsidence Bowl Contours
        - Panel C: EVT Return Level Curve (5 to 100-Year Return Periods) with 95% CIs
        - Panel D: Coupled Hazard Classification Map & Critical Infrastructure Exposure
        """
        out_path = self.output_dir / output_filename

        # Set matplotlib style
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 14,
        })

        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
        fig.suptitle(
            "Coupled Satellite InSAR Subsidence & EVT Extreme Rainfall Hazard Intelligence\nGreater Accra Metropolitan Area, Ghana",
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------------
        # Panel A: Historical Rainfall & POT Fit
        # -------------------------------------------------------------------
        ax_a = axes[0, 0]
        dates = precip_record.df["date"]
        p_vals = precip_record.df["precipitation_mm"]

        ax_a.plot(dates, p_vals, color="#2980b9", lw=0.6, alpha=0.75, label="Daily Precipitation (ERA5 / GMet)")
        ax_a.axhline(
            pot_result.threshold,
            color="#c0392b",
            linestyle="--",
            lw=1.5,
            label=f"POT Threshold u = {pot_result.threshold:.1f} mm",
        )

        # Highlight extreme exceedances
        if pot_result.exceedance_dates is not None:
            ax_a.scatter(
                pot_result.exceedance_dates,
                pot_result.exceedance_values,
                color="#e74c3c",
                s=20,
                zorder=5,
                edgecolors="black",
                linewidths=0.4,
                label=f"POT Exceedances (k = {pot_result.num_exceedances})",
            )

        # Annotate June 3, 2015 disaster
        mask_2015 = (dates.dt.year == 2015) & (dates.dt.month == 6) & (dates.dt.day == 3)
        if mask_2015.any():
            p_2015 = float(p_vals[mask_2015].iloc[0])
            d_2015 = dates[mask_2015].iloc[0]
            ax_a.annotate(
                f"3 June 2015 Disaster\n({p_2015:.1f} mm)",
                xy=(d_2015, p_2015),
                xytext=(d_2015 - pd.Timedelta(days=1200), p_2015 - 35),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
                fontweight="bold",
                color="#78281f",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fadbd8", ec="#e74c3c", lw=1),
            )

        ax_a.set_title("A: Historical Precipitation & POT Extreme Sampling (1990–2024)", fontweight="bold")
        ax_a.set_xlabel("Observation Year")
        ax_a.set_ylabel("Daily Precipitation (mm / day)")
        ax_a.set_ylim(bottom=0, top=max(p_vals) * 1.15)
        ax_a.grid(True, linestyle=":", alpha=0.6)
        ax_a.legend(loc="upper left")

        # -------------------------------------------------------------------
        # Panel B: InSAR Mean Vertical Velocity Field
        # -------------------------------------------------------------------
        ax_b = axes[0, 1]
        sc_b = ax_b.scatter(
            coupled_df["longitude"],
            coupled_df["latitude"],
            c=coupled_df["v_vert"],
            cmap="turbo_r",  # Dark red/orange = high subsidence, blue = stable
            s=12,
            alpha=0.85,
            edgecolors="none",
        )
        cbar_b = fig.colorbar(sc_b, ax=ax_b, fraction=0.046, pad=0.04)
        cbar_b.set_label("InSAR Vertical Velocity $v_{\\text{vert}}$ (mm / yr)")

        # Plot subsidence bowls
        for bowl in subsidence_bowls:
            circle = plt.Circle(
                (bowl.centroid_lon, bowl.centroid_lat),
                radius=bowl.radius_km / 111.32,  # approx degrees
                color="black",
                fill=False,
                lw=1.6,
                ls="--",
            )
            ax_b.add_patch(circle)
            ax_b.text(
                bowl.centroid_lon,
                bowl.centroid_lat,
                f" {bowl.name.split()[0]}\n ({bowl.max_subsidence_rate_mm_yr:.0f} mm/yr)",
                fontsize=7.5,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="black", lw=0.5),
            )

        ax_b.set_title("B: Sentinel-1 / NISAR Vertical Subsidence Velocity ($v_{\\text{vert}}$)", fontweight="bold")
        ax_b.set_xlabel("Longitude (&deg;E)")
        ax_b.set_ylabel("Latitude (&deg;N)")
        ax_b.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------------
        # Panel C: EVT Return Level Curves
        # -------------------------------------------------------------------
        ax_c = axes[1, 0]
        t_vals = np.array(return_levels.return_periods)
        rl_vals = np.array(return_levels.return_levels)
        ci_low = np.array(return_levels.ci_lower)
        ci_high = np.array(return_levels.ci_upper)

        # Plot curve and 95% confidence band
        ax_c.plot(t_vals, rl_vals, color="#1b4f72", lw=2.2, marker="o", label="Fitted GPD Return Level $x_T$")
        ax_c.fill_between(
            t_vals,
            ci_low,
            ci_high,
            color="#85c1e9",
            alpha=0.35,
            label="Delta-Method 95% Confidence Band",
        )

        for t, val in zip(t_vals, rl_vals):
            ax_c.annotate(
                f"{val:.1f} mm",
                xy=(t, val),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#1b4f72",
            )

        # Subtext with GPD parameters
        param_text = (
            f"GPD MLE Parameters:\n"
            f"Scale $\\sigma$: {gpd_fit.scale:.2f} &plusmn; {gpd_fit.scale_se:.2f} mm\n"
            f"Shape $\\xi$: {gpd_fit.shape:.3f} &plusmn; {gpd_fit.shape_se:.3f}\n"
            f"Threshold $u$: {gpd_fit.threshold:.1f} mm\n"
            f"Excess Count $k$: {gpd_fit.num_exceedances}"
        )
        ax_c.text(
            0.05, 0.62,
            param_text,
            transform=ax_c.transAxes,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#eaf2f8", ec="#2980b9", lw=0.8),
        )

        ax_c.set_title("C: EVT Return Level Curve & Design Rainstorms ($T$ = 5 to 100 yr)", fontweight="bold")
        ax_c.set_xlabel("Return Period $T$ (Years)")
        ax_c.set_ylabel("Extreme Rainfall Return Level $x_T$ (mm)")
        ax_c.set_xticks(t_vals)
        ax_c.set_xticklabels([f"{int(t)} yr" for t in t_vals])
        ax_c.grid(True, linestyle=":", alpha=0.6)
        ax_c.legend(loc="lower right")

        # -------------------------------------------------------------------
        # Panel D: Coupled Hazard Classification Map
        # -------------------------------------------------------------------
        ax_d = axes[1, 1]
        palette = {0: "#27ae60", 1: "#e67e22", 2: "#c0392b"}
        labels_map = {0: "Low Hazard", 1: "Moderate Hazard", 2: "Critical Hazard"}

        for cls_id in [0, 1, 2]:
            sub_cls = coupled_df[coupled_df["hazard_class"] == cls_id]
            ax_d.scatter(
                sub_cls["longitude"],
                sub_cls["latitude"],
                c=palette[cls_id],
                label=f"{labels_map[cls_id]} (n={len(sub_cls)})",
                s=10 if cls_id == 0 else 16,
                alpha=0.75,
                edgecolors="none",
            )

        # Plot critical infrastructure
        ax_d.scatter(
            infrastructure_df["longitude"],
            infrastructure_df["latitude"],
            c="#f1c40f",
            s=75,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
            label="Critical Infrastructure",
        )

        # Label key infrastructure
        for _, asset in infrastructure_df.iterrows():
            if asset["criticality"] >= 9:
                short_name = asset["name"].split()[0]
                ax_d.text(
                    asset["longitude"] + 0.005,
                    asset["latitude"] + 0.003,
                    short_name,
                    fontsize=7,
                    fontweight="bold",
                    color="#1a5276",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85, ec="#1a5276", lw=0.5),
                )

        ax_d.set_title("D: Coupled Multi-Hazard Classification (DHII + InSAR + DEM)", fontweight="bold")
        ax_d.set_xlabel("Longitude (&deg;E)")
        ax_d.set_ylabel("Latitude (&deg;N)")
        ax_d.grid(True, linestyle=":", alpha=0.6)
        ax_d.legend(loc="upper left")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(str(out_path), dpi=300)
        plt.close(fig)
        logger.info("Saved 300 DPI publication diagnostic plot to %s", out_path)
        return out_path
