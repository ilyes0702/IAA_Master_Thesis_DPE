"""Simulation script for running a trained Mamba inverse controller on the
Chemostat plant model.

This script initializes the plant, loads a trained controller and scalers,
generates reference trajectories (dynamic and constant), and runs the
simulation routine.
"""

import matplotlib.pyplot as plt
from src.sample.config import get_run_description, log_message
plt.style.use("src/sample/style.mplstyle")
import torch

# Utility functions and classes for simulation
from src.sample.utils.general_utils import *
from src.hyperparam_config import *
from src.sample.classes.ChemostatPlant import ChemostatPlant
from src.sample.classes.TrophophasePlant import TrophophasePlant
from src.sample.classes.MassSpringDamperPlant import MassSpringDamperPlant
from src.sample.classes.MambaInverseController import MambaInverseController

# Device configuration (printed for visibility)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    # Optional: log high-level run description (disabled by default)
    # run_description = get_run_description()
    # log_message(f"RUN DESCRIPTION: {run_description}")

    hyperparam_config = hyperparam_config_MassSpringDamperPlant
    # Initialize the plant model using hyperparameters
    plant = MassSpringDamperPlant(hyperparam_config=hyperparam_config_MassSpringDamperPlant)

    #plant = TrophophasePlant(hyperparam_config_TrophophasePlant)
    

    # Path to the trained controller checkpoint (keep as configured)
    controller_path = (
        "models/2026-06-01/2026-06-01_09-30-01/MassSpringDamperPlant_training/fold_1/2026-06-01_09-30-01_best_fold_model.pt"
    )

    # Load the trained inverse controller
    loaded_controller = load_model(MambaInverseController, controller_path)

    # Load input/output scalers from training run directory
    scaler_x = load_scaler(
        "results/2026-06-01/2026-06-01_09-30-01/MassSpringDamperPlant_training/fold_1/scalers/2026-06-01_09-30-01_scaler_x.pkl"
    )

    scaler_y = load_scaler(
        "results/2026-06-01/2026-06-01_09-30-01/MassSpringDamperPlant_training/fold_1/scalers/2026-06-01_09-30-01_scaler_y.pkl"
        )

    # Generate a dynamic reference trajectory (time-varying target)
    r_dynamic = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["signal"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="dynamic",
        gain=1.2,      # sharper transition edges
        period=5.0    # period for cyclic dynamics
    )

    # Generate a constant reference trajectory (steady target)
    r_static = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["signal"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="constant",
        constant_val=0.15,
    )

    # Run the simulation using the dynamic reference by default
    simulate_tracking_sakura(
        model=loaded_controller,
        plant=plant,
        r_trajectory=r_dynamic,
        hyperparam_config=hyperparam_config,
        x_scaler=scaler_x,
        y_scaler=scaler_y,
        dirname=plant.__class__.__name__,
    )


if __name__ == "__main__":
    main()

    