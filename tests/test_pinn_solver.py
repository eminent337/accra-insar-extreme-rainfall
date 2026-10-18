import pytest
import numpy as np
import torch
from accra_engine.pinn_solver import HydroInSARPINN, PhysicsInformedHazardSolver

def test_pinn_architecture_forward():
    model = HydroInSARPINN(input_dim=3, hidden_dim=32, num_layers=3, output_dim=3)
    batch_size = 16
    x_test = torch.randn(batch_size, 3)
    
    out = model(x_test)
    assert out.shape == (batch_size, 3)
    
    # Assert physical constraint: water depth h_w >= 0 everywhere
    h_w = out[:, 0]
    assert torch.all(h_w >= 0.0), "Physical constraint violated: negative water depth predicted!"

def test_pinn_autograd_pde_residuals():
    solver = PhysicsInformedHazardSolver(hidden_dim=32, num_layers=3)
    n_pts = 20
    coords = torch.rand(n_pts, 3)
    rain = torch.full((n_pts, 1), 0.0001) # 0.1 mm/s rain
    insar = torch.full((n_pts, 1), -0.000001) # Sinking ground
    
    residual = solver.compute_pde_residual(coords, rain, insar)
    assert residual.shape == (n_pts, 1)
    assert not torch.isnan(residual).any(), "NaN values found in PDE automatic differentiation!"

def test_pinn_training_convergence():
    solver = PhysicsInformedHazardSolver(hidden_dim=32, num_layers=3, learning_rate=0.01)
    n_samples = 50
    collocation_pts = np.random.rand(n_samples, 3)
    observed_depths = np.random.uniform(0.1, 2.5, (n_samples, 1))
    rain_rates = np.full((n_samples, 1), 1e-5)
    insar_rates = np.full((n_samples, 1), -30.0) # -30 mm/yr
    
    res = solver.train_step(collocation_pts, observed_depths, rain_rates, insar_rates, epochs=25)
    assert res.iterations == 25
    assert res.final_loss < res.history_loss[0], "PINN failed to minimize loss during training!"
    assert res.final_loss >= 0.0
