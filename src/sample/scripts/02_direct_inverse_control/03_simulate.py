"""Simulation script for running a trained Mamba inverse controller on the
Chemostat plant model.

This script initializes the plant, loads a trained controller and scalers,
generates reference trajectories (dynamic and constant), and runs the
simulation routine.
"""
import torch.nn as nn
import matplotlib.pyplot as plt
from src.sample.classes.plants.PenicillinPlantBirol2002 import PenicillinPlantBirol2002
from src.sample.config import get_run_description, log_message
plt.style.use("src/sample/style.mplstyle")
import torch

# Utility functions and classes for simulation
from src.sample.utils.general_utils import *
from src.hyperparam_config import *
from src.sample.classes.plants.IdiophasePlant import *
from src.sample.classes.plants.MassSpringDamperPlant import *
from src.sample.classes.plants.TrophophasePlant import *
from src.sample.classes.plants.ChemostatPlant import *
from src.sample.classes.controllers.MambaInverseController import *
from src.sample.classes.controllers.ESNInverseController import *
from src.sample.classes.controllers.LSTMInverseController import *
from src.sample.classes.controllers.TransformerInverseController import *

# Tell PyTorch this class is safe to unpickle
torch.serialization.add_safe_globals([MambaInverseController])

# Device configuration (printed for visibility)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    # Optional: log high-level run description (disabled by default)
    # run_description = get_run_description()
    # log_message(f"RUN DESCRIPTION: {run_description}")

    # Initialize the plant model
    dirname = "results/run_1"
    hyperparam_config = hyperparam_config_TrophophasePlant   
    plant = TrophophasePlant(hyperparam_config=hyperparam_config)

    # controller_path = "models/2026-07-29/2026-07-29_22-53-39/TrophophasePlant_training/fold_1/2026-07-29_22-53-39_best_fold_model.pt"
    # Load the trained inverse controller
    #loaded_controller = load_model(LSTMInverseController, controller_path)

    # #scaler_x = load_scaler(
    #     "results/2026-07-29/2026-07-29_22-53-39/TrophophasePlant_training/fold_1/scalers/2026-07-29_22-53-39_scaler_x.pkl"
    # )

    # #scaler_y = load_scaler(
    #     "results/2026-07-29/2026-07-29_22-53-39/TrophophasePlant_training/fold_1/scalers/2026-07-29_22-53-39_scaler_y.pkl"
    #     )

    models_dict_1 = {
        "MIC_cl_ol" : {"model": load_model(LSTMInverseController, "models/2026-08-11/2026-08-11_19-02-53/ChemostatPlant_training/fold_2/2026-08-11_19-02-53_best_fold_model.pt"),
                     "x_scaler" : load_scaler("results/2026-08-11/2026-08-11_19-02-53/ChemostatPlant_training/fold_2/scalers/2026-08-11_19-02-53_scaler_x.pkl"),
                     "y_scaler" : load_scaler("results/2026-08-11/2026-08-11_19-02-53/ChemostatPlant_training/fold_2/scalers/2026-08-11_19-02-53_scaler_y.pkl")
                      }
    }    


    # models_dict = {
    #     "LSTMIC_1" : {"model": load_model(LSTMInverseController, "models/2026-07-30/2026-07-30_11-42-17/TrophophasePlant_training/fold_1/2026-07-30_11-42-17_best_fold_model.pt"),
    #                "x_scaler" : load_scaler("results/2026-07-30/2026-07-30_11-42-17/TrophophasePlant_training/fold_1/scalers/2026-07-30_11-42-17_scaler_x.pkl"),
    #                "y_scaler" : load_scaler("results/2026-07-30/2026-07-30_11-42-17/TrophophasePlant_training/fold_1/scalers/2026-07-30_11-42-17_scaler_y.pkl")
    #                 },
    #     "MIC_1" : {"model": load_model(MambaInverseController, "models/2026-07-30/2026-07-30_15-01-40/TrophophasePlant_training/fold_1/2026-07-30_15-01-40_best_fold_model.pt"),
    #                        "x_scaler" : load_scaler("results/2026-07-30/2026-07-30_15-01-40/TrophophasePlant_training/fold_1/scalers/2026-07-30_15-01-40_scaler_x.pkl"),
    #                        "y_scaler" : load_scaler("results/2026-07-30/2026-07-30_15-01-40/TrophophasePlant_training/fold_1/scalers/2026-07-30_15-01-40_scaler_y.pkl")
    #                         },
    #     "TIC_1" : {"model": load_model(TransformerInverseController, "models/2026-07-30/2026-07-30_11-50-42/TrophophasePlant_training/fold_1/2026-07-30_11-50-42_best_fold_model.pt"),
    #                        "x_scaler" : load_scaler("results/2026-07-30/2026-07-30_11-50-42/TrophophasePlant_training/fold_1/scalers/2026-07-30_11-50-42_scaler_x.pkl"),
    #                        "y_scaler" : load_scaler("results/2026-07-30/2026-07-30_11-50-42/TrophophasePlant_training/fold_1/scalers/2026-07-30_11-50-42_scaler_y.pkl")
    #                         },
    #     "ESNIC_1" : {"model": load_model_esn(ESNInverseController, "TrophophasePlant_training/fold_1/best_fold_model.pkl"),
    #                        "x_scaler" : load_scaler("results/2026-08-04/2026-08-04_10-03-05/TrophophasePlant_training/fold_1/scalers/2026-08-04_10-03-05_scaler_x.pkl"),
    #                        "y_scaler" : load_scaler("results/2026-08-04/2026-08-04_10-03-05/TrophophasePlant_training/fold_1/scalers/2026-08-04_10-03-05_scaler_y.pkl")
    #                         }
    # }
    
    # Generate a dynamic reference trajectory (time-varying target)
    r_dynamic = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["training_data_cfg"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="dynamic",
        gain=1.2,      # sharper transition edges
        period=5.0     # period for cyclic dynamics
    )

    r_static_y_1 = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["training_data_cfg"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="constant",
        constant_val=0.2,
    )

    r_static_y_2 = generate_reference_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["training_data_cfg"]["dt"],
        device=hyperparam_config["train"]["device"],
        mode="constant",
        constant_val=50,
    )

    r_smooth = generate_smooth_profile_trajectory(
        time_axis=torch.arange(hyperparam_config["simulate"]["seq_len"]) * hyperparam_config["training_data_cfg"]["dt"],
        config={
            'y_floor': 0.25,
            'y_peak': 0.15,
            't_rise_center': 10.0,
            'k_rise': 1.0,
            't_sink_center': 25.0,
            'k_sink': 1.0
        }
    )
 

    r_smooth_decay = generate_exponential_decay_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["training_data_cfg"]["dt"],
        y_start=0.12,       # Starts here
        y_target=0.015,     # Smoothly descends and turns to this constant value
        tau=0.1,            # Governs speed (lower = faster drop)
        device="cuda"
    )


    r_smooth_decay = generate_exponential_decay_trajectory(
        steps=hyperparam_config["simulate"]["seq_len"],
        dt=hyperparam_config["training_data_cfg"]["dt"],
        y_start=0.12,       # Starts here
        y_target=0.02,      # Smoothly descends and turns to this constant value
        tau=0.1,            # Governs speed (lower = faster drop)
        device="cuda"
    )

    # simulate_tracking_stateful(
    #     model=loaded_controller,
    #     plant=plant,
    #     r_trajectories=[r_static_y_1.squeeze()],
    #     hyperparam_config=hyperparam_config,
    #     x_scaler=scaler_x,
    #     y_scaler=scaler_y,
    #     dirname=plant.__class__.__name__,
    #     plot_individual_plots = False
    # ) 
    val_data = torch.load("results/2026-08-17/2026-08-17_23-08-14/results/run_1/dataset/2026-08-17_23-08-14_val_data.pt", weights_only=False)
    model = load_model(MambaInverseController, "results/2026-08-17/2026-08-17_23-08-14/results/run_1/2026-08-17_23-08-14_best_model.pt")
    x_scaler = load_scaler("results/2026-08-17/2026-08-17_23-08-14/results/run_1/scalers/2026-08-17_23-08-14_scaler_x.pkl")
    y_scaler = load_scaler("results/2026-08-17/2026-08-17_23-08-14/results/run_1/scalers/2026-08-17_23-08-14_scaler_y.pkl")
    simulate_tracking_stateful(
        model=model,
        plant=plant,
        val_data=val_data,  # Tuple (Y_val, U_val, [X_val]) or Dict {"y": ..., "u": ...}
        hyperparam_config=hyperparam_config,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        dirname=dirname,
        mode="closed_loop",  # Options: "open_loop" or "closed_loop"
        plot_individual_plots=True,
    )

    # comparison_summary = simulate_tracking_stateful_multi_model(
    #     models_dict=models_dict_1,
    #     plant=ChemostatPlant,
    #     r_trajectories=[r_static_y_1.squeeze()],
    #     hyperparam_config=hyperparam_config,
    #     dirname="results/model_benchmark_comparison"
    # )


if __name__ == "__main__":
    main()   