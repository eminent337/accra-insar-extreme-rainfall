"""
Physics-Informed Neural Network (PINN) Module: Accra Hydro-InSAR Engine
========================================================================
Implements a Physics-Informed Neural Network that solves the coupled 2D
Saint-Venant Shallow Water Continuity Equation driven by:
  1. Extreme Value Theory (EVT) design rainfall flux R(t)
  2. Multi-temporal InSAR vertical subsidence velocity v_vert(x, y) = dz_bed/dt
  3. Dynamic topographic terrain slopes from DEM

Governing PDE Residual (Mass Conservation on Deforming Terrain):
  R_pde = d(h_w)/dt + d(h_w * u)/dx + d(h_w * v)/dy - R(t) - v_vert(x, y)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PINN Architecture (Multi-Layer Perceptron with Autograd)
# ---------------------------------------------------------------------------

class HydroInSARPINN(nn.Module):
    """
    Physics-Informed Neural Network mapping (x, y, t) -> (h_w, u, v)
    where:
      h_w : Water inundation depth (m)
      u   : Eastward depth-averaged flow velocity (m/s)
      v   : Northward depth-averaged flow velocity (m/s)
    """

    def __init__(
        self,
        input_dim: int = 3,       # (x, y, t)
        hidden_dim: int = 64,     # Neurons per hidden layer
        num_layers: int = 4,      # Deep representation depth
        output_dim: int = 3,      # (h_w, u, v)
    ):
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())  # Tanh provides smooth continuous derivatives for autograd

        # Hidden representation layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting state vector [h_w, u, v].
        Ensures water depth h_w >= 0 via softplus activation on output index 0.
        """
        raw_out = self.network(x_tensor)
        h_w = torch.nn.functional.softplus(raw_out[:, 0:1])  # Physical non-negativity constraint
        u = raw_out[:, 1:2]                                 # Velocity u can be negative
        v = raw_out[:, 2:3]                                 # Velocity v can be negative
        return torch.cat([h_w, u, v], dim=1)


# ---------------------------------------------------------------------------
# PINN Training & Physics Residual Engine
# ---------------------------------------------------------------------------

@dataclass
class PINNTrainingResult:
    """Telemetry report for PINN convergence and physics loss minimization."""
    final_loss: float
    data_loss: float
    pde_residual_loss: float
    iterations: int
    mcc_score: float
    history_loss: List[float]


class PhysicsInformedHazardSolver:
    """
    High-level solver managing spatial collocation points, InSAR velocity priors,
    and automatic differentiation of shallow water PDE residuals.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        learning_rate: float = 1e-3,
        lambda_physics: float = 0.5,
        random_seed: int = 42,
    ):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        self.device = torch.device("cpu")
        self.model = HydroInSARPINN(hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.lambda_physics = lambda_physics

    def compute_pde_residual(
        self,
        coords: torch.Tensor,       # [N, 3] representing (x_norm, y_norm, t_norm)
        rainfall_flux: torch.Tensor,# [N, 1] rainfall source term R(t) in m/s
        insar_subsidence: torch.Tensor, # [N, 1] vertical deformation velocity dz/dt in m/s
    ) -> torch.Tensor:
        """
        Computes the physics loss via PyTorch Autograd:
          Residual = d(h_w)/dt + d(h_w * u)/dx + d(h_w * v)/dy - R(t) - v_vert(x, y)
        """
        coords.requires_grad_(True)
        pred = self.model(coords)
        
        h_w = pred[:, 0:1]
        u = pred[:, 1:2]
        v = pred[:, 2:3]

        flux_x = h_w * u
        flux_y = h_w * v

        # Compute spatial and temporal gradients via automatic differentiation
        grads_h = torch.autograd.grad(
            h_w, coords,
            grad_outputs=torch.ones_like(h_w),
            create_graph=True, retain_graph=True
        )[0]
        dh_dt = grads_h[:, 2:3]

        grads_qx = torch.autograd.grad(
            flux_x, coords,
            grad_outputs=torch.ones_like(flux_x),
            create_graph=True, retain_graph=True
        )[0]
        dqx_dx = grads_qx[:, 0:1]

        grads_qy = torch.autograd.grad(
            flux_y, coords,
            grad_outputs=torch.ones_like(flux_y),
            create_graph=True, retain_graph=True
        )[0]
        dqy_dy = grads_qy[:, 1:2]

        # 2D Shallow Water Mass Conservation Residual coupled with InSAR deforming bed:
        # Note: Subsidence (sinking) creates a sink in bed elevation, increasing accommodation space!
        residual = dh_dt + dqx_dx + dqy_dy - (rainfall_flux + torch.abs(insar_subsidence))
        
        return residual

    def train_step(
        self,
        collocation_points: np.ndarray, # [N, 3] (x, y, t)
        observed_depths: np.ndarray,    # [N, 1]
        rainfall_rates: np.ndarray,     # [N, 1] m/s
        insar_rates_mm_yr: np.ndarray,  # [N, 1] mm/yr
        epochs: int = 150,
    ) -> PINNTrainingResult:
        """
        Executes gradient descent minimizing combined data fidelity and PDE physical laws.
        """
        # Convert inputs to PyTorch Tensors
        pts_tensor = torch.tensor(collocation_points, dtype=torch.float32, device=self.device)
        depth_target = torch.tensor(observed_depths, dtype=torch.float32, device=self.device)
        rain_tensor = torch.tensor(rainfall_rates, dtype=torch.float32, device=self.device)
        
        # Convert InSAR velocity from mm/year to m/second for physical consistency
        # 1 mm/yr = 1e-3 / (365.25 * 86400) m/s
        v_insar_ms = (insar_rates_mm_yr * 1e-3) / (365.25 * 86400.0)
        insar_tensor = torch.tensor(v_insar_ms, dtype=torch.float32, device=self.device)

        mse_criterion = nn.MSELoss()
        loss_history = []

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # 1. Data fidelity loss (matching observed gauge/sensor water depths)
            predictions = self.model(pts_tensor)
            loss_data = mse_criterion(predictions[:, 0:1], depth_target)
            
            # 2. Physics PDE conservation residual loss
            pde_residuals = self.compute_pde_residual(pts_tensor, rain_tensor, insar_tensor)
            loss_pde = torch.mean(pde_residuals ** 2)

            # Combined Loss Function
            total_loss = loss_data + self.lambda_physics * loss_pde
            
            total_loss.backward()
            self.optimizer.step()

            loss_val = total_loss.item()
            loss_history.append(loss_val)

        # Compute empirical correlation
        with torch.no_grad():
            final_pred_depth = self.model(pts_tensor)[:, 0].numpy()
            corr = np.corrcoef(observed_depths.ravel(), final_pred_depth)[0, 1]
            mcc = max(0.0, float(corr))  # Approximation of continuous correlation fidelity

        return PINNTrainingResult(
            final_loss=loss_history[-1],
            data_loss=loss_data.item(),
            pde_residual_loss=loss_pde.item(),
            iterations=epochs,
            mcc_score=mcc,
            history_loss=loss_history,
        )

    def predict_inundation(
        self,
        coords: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate trained PINN model to predict water depth and velocities.
        """
        self.model.eval()
        with torch.no_grad():
            t_input = torch.tensor(coords, dtype=torch.float32, device=self.device)
            out = self.model(t_input).numpy()
            h_w = out[:, 0]
            u = out[:, 1]
            v = out[:, 2]
        return h_w, u, v
