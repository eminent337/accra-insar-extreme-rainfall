"""
Accra Hydro-InSAR Engine (v1.0.0)
=================================
Coupled Multi-Temporal Satellite InSAR Ground Subsidence & Extreme Value Theory (EVT)
Rainfall Hazard Intelligence Engine for Greater Accra, Ghana.

Principal Investigator: Peter Aryee
"""

__version__ = "1.0.0"
__author__ = "Peter Aryee"
__all__ = [
    # Data ingest
    "AccraDataIngest",
    "PrecipitationRecord",
    "InSARPointDataset",
    "DEMHydrologyGrid",
    # EVT Extremes
    "EVTExtremesEngine",
    "POTResult",
    "ReturnLevelResult",
    "GoodnessOfFitResult",
    # InSAR Deformation
    "InSARDeformationEngine",
    "SubsidenceBowl",
    # Coupled Hazard ML
    "CoupledHazardEngine",
    "HazardClassificationResult",
    # Visualizer
    "AccraHazardVisualizer",
    # Report generator
    "HazardReportGenerator",
]

from .data_ingest import (
    AccraDataIngest,
    PrecipitationRecord,
    InSARPointDataset,
    DEMHydrologyGrid,
)
from .evt_extremes import (
    EVTExtremesEngine,
    POTResult,
    ReturnLevelResult,
    GoodnessOfFitResult,
)
from .insar_deformation import (
    InSARDeformationEngine,
    SubsidenceBowl,
)
from .coupled_hazard_ml import (
    CoupledHazardEngine,
    HazardClassificationResult,
)
from .visualizer import (
    AccraHazardVisualizer,
)
from .report_generator import (
    HazardReportGenerator,
)
