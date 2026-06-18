"""Simulation script for running a trained Mamba inverse controller on the
Chemostat plant model.

This script initializes the plant, loads a trained controller and scalers,
generates reference trajectories (dynamic and constant), and runs the
simulation routine.
"""

import matplotlib.pyplot as plt
from src.sample.classes.plants.PenicillinPlantBirol2002 import PenicillinPlantBirol2002
from src.sample.config import get_run_description, log_message
plt.style.use("src/sample/style.mplstyle")
import torch

# Utility functions and classes for simulation
from src.sample.utils.general_utils import *
from src.hyperparam_config import *
from src.sample.classes.plants.IdiophasePlant import IdiophasePlant
from src.sample.classes.plants.MassSpringDamperPlant import MassSpringDamperPlant
from src.sample.classes.plants.TrophophasePlant import TrophophasePlant
from src.sample.classes.plants.ChemostatPlant import ChemostatPlant
from src.sample.classes.plants.YeastFermentation import FedBatchYeastPlant
from src.sample.classes.controllers.MambaInverseController import *

# Device configuration (printed for visibility)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    # Optional: log high-level run description (disabled by default)
    # run_description = get_run_description()
    # log_message(f"RUN DESCRIPTION: {run_description}")

    hyperparam_config = hyperparam_config_ChemostatPlant   # Initialize the plant model using hyperparameters
    plant = ChemostatPlant(hyperparam_config=hyperparam_config)

    #plant = TrophophasePlant(hyperparam_config_TrophophasePlant)
    

    # Path to the trained controller checkpoint (keep as configured)
    controller_path = (
        "models/2026-06-18/2026-06-18_15-42-38/ChemostatPlant_training/fold_1/2026-06-18_15-42-38_best_fold_model.pt"
    )

    # Load the trained inverse controller
    loaded_controller = load_model(MambaInverseController, controller_path)

    scaler_x = load_scaler(
        "results/2026-06-18/2026-06-18_15-42-38/ChemostatPlant_training/fold_1/scalers/2026-06-18_15-42-38_scaler_x.pkl"
    )

    scaler_y = load_scaler(
        "results/2026-06-18/2026-06-18_15-42-38/ChemostatPlant_training/fold_1/scalers/2026-06-18_15-42-38_scaler_y.pkl"
        )
    
    # Example: Separate sine and cosine trajectories for y1 and y2

    # Generate a dynamic reference trajectory (time-varying target)
    r_dynamic = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["signal"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="dynamic",
        gain=1.2,      # sharper transition edges
        period=5.0    # period for cyclic dynamics
    )

    r_static_y_1 = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["signal"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="constant",
        constant_val=0.25,
    )

    r_static_y_2 = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["signal"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="constant",
        constant_val=50,
    )

    r_smooth = generate_smooth_profile_trajectory(
        time_axis=torch.arange(hyperparam_config["simulate"]["seq_len"]) * hyperparam_config["signal"]["dt"],
        config={
            'y_floor': 0.25,
            'y_peak': 0.15,
            't_rise_center': 10.0,
            'k_rise': 1.0,
            't_sink_center': 25.0,
            'k_sink': 1.0
        }
    )

    simulate_tracking_torchdiffeq(
        model=loaded_controller,
        plant=plant,
        r_trajectories=[r_dynamic.squeeze()],
        hyperparam_config=hyperparam_config,
        x_scaler=scaler_x,
        y_scaler=scaler_y,
        dirname=plant.__class__.__name__,
        plot_individual_plots = False
    )
    

    # Generate a constant reference trajectory (steady target)   


if __name__ == "__main__":
    main()

    