"""
Train the Mamba inverse controller for the Chemostat plant.

This script loads a prepared training dataset (features X and targets Y),
initializes the plant and controller using the shared hyperparameter
configuration, and runs the training routine to produce a trained controller
and any associated artifacts (saved models, logs, plots).

Usage: run this file as a script. Adjust dataset_path below if needed.
"""

import torch

from src.sample.classes.plants.PenicillinPlantBirol2002 import PenicillinPlantBirol2002
from src.sample.config import *

from src.sample.classes.plants.IndForProteinProductionPlant import *
from src.sample.classes.plants.ChemostatPlant import *
from src.sample.classes.plants.MassSpringDamperPlant import MassSpringDamperPlant
from src.sample.classes.plants.TrophophasePlant import *
from src.sample.classes.plants.IdiophasePlant import *
from src.sample.classes.plants.CocultivationPlant import CoCultivationPlant
from src.sample.classes.controllers.MambaInverseController import *
from src.sample.classes.controllers.ESNInverseController import *

from src.hyperparam_config import *

from src.sample.utils.training_utils import * 
from src.sample.utils.saving_utils import *
from src.sample.utils.plotting_utils import *

# Device configuration: use GPU if available, otherwise fallback to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import optuna

def objective(trial, model_class, Y_trajectories, U_trajectories, X_states, base_config, plant, dirname):
        # --- 1. SAMPLE HYPERPARAMETERS WITH OPTUNA ---
        # Create a deep copy of base_config so we don't mutate global state
        config = copy.deepcopy(base_config)

        # Example hyperparameter search spaces (customize as needed):
        #config["train"]["n_y"] = trial.suggest_int("n_y", 1, 10)
        #config["train"]["n_u"] = trial.suggest_int("n_u", 1, 10)

        config["mamba"]["d_conv"] = trial.suggest_int("d_conv", 1, 10)
        config["mamba"]["d_state"] = trial.suggest_int("d_state", 1, 64)
        config["mamba"]["expand"] = trial.suggest_int("expand", 1, 10)
        config["mamba"]["n_layer"] = trial.suggest_int("n_layer", 1, 5)
        
        # Optional model-specific hyperparameters (e.g., hidden dims, layers)
        # config["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])

        # Dynamic directory for each trial to prevent file collision
        trial_dirname = f"{dirname}/trial_{trial.number}"

        # Re-instantiate a fresh model instance for the trial
        # (Assuming model_class takes relevant dimension/hidden args)
        model = model_class(config)

        # --- 2. EXECUTE CONTROLLER TRAINING ---
        # Turn off plotting and full simulation rollout during tuning to speed up trials
        fold_histories, mean_cv_val_loss = train_controller(
            model=model,
            Y_trajectories=Y_trajectories,
            U_trajectories=U_trajectories,
            X_states=X_states,
            hyperparam_config=config,
            plant=plant,
            dirname=trial_dirname,
            show_plots=False,
            run_simulation=False,  # Set to True if optimizing directly for simulation MSE
            run_sim_with_plots=False
        )

        # Return metric to MINIMIZE
        return mean_cv_val_loss


if __name__ == "__main__":
    # Instantiate plant using shared hyperparameters.
    hyperparam_config = hyperparam_config_ChemostatPlant
    
    plant = ChemostatPlant(hyperparam_config=hyperparam_config)
    
    dataset_path = (
       "results/2026-07-22/2026-07-22_16-29-07/ChemostatPlant_training_data/dataset/2026-07-22_16-29-07_training_data.pt"
    )
    
    # Load the dataset from disk. The file is expected to be a dict-like object containing 'x' and 'y' keys. If loading fails, inspect the path first.
    dataset = torch.load(dataset_path, weights_only=True)
    Y_trajectories=dataset["y"]      # Passes [Num_Trajectories, Seq_Len, input_dim]
    U_trajectories=dataset["u"]
    X_states=dataset["states"]
    # Initialize the inverse controller and move it to the selected device.

    controller = MambaInverseController(hyperparam_config=hyperparam_config)

    # Directory name to store training artifacts (models, plots, logs).
    dirname = f"{plant.__class__.__name__}_training"

    save_to_json(hyperparam_config, dataset_path, "training_data_path")
    save_to_json(hyperparam_config, dirname,"hyperparam_config")
    # Run the training routine. show_plots can be toggled for interactive use.
    # train_controller(
    #     controller,
    #     Y_trajectories=dataset["y"],      # Passes [Num_Trajectories, Seq_Len, input_dim]
    #     U_trajectories=dataset["u"],
    #     X_states=dataset["states"],
    #     hyperparam_config=hyperparam_config,
    #     plant=plant,
    #     dirname=dirname,
    #     run_simulation=True,
    #     run_sim_with_plots=True
    # )   

    # Create Optuna study using Tree-structured Parzen Estimator (TPE)
    study = optuna.create_study(
        study_name="controller_hyperparam_tuning",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Run Bayesian optimization
    n_trials = 100  # Total combinations to evaluate
    study.optimize(
        lambda trial: objective(
            trial=trial,
            model_class=MambaInverseController,  # Replace with your actual PyTorch model class
            Y_trajectories=Y_trajectories,
            U_trajectories=U_trajectories,
            X_states=X_states,
            base_config=hyperparam_config,
            plant=plant,
            dirname="./optuna_trials"
        ),
        n_trials=n_trials
    )

    # Print results
    print("\n==========================================")
    print("🏆 OPTUNA HYPERPARAMETER TUNING COMPLETE")
    print("==========================================")
    print(f"Best Trial Number: {study.best_trial.number}")
    print(f"Best Validation Loss: {study.best_value:.6f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")


    # Generate and export the heatmap image
    plot_param_heatmap(
        study=study,
        param_x="d_state",
        param_y="expand",
        filename="optuna_heatmap_d_state_expand",
        dirname=dirname,
    )

    plot_param_heatmap(
            study=study,
            param_x="d_state",
            param_y="d_conv",
            filename="optuna_heatmap_d_state_d_conv",
            dirname=dirname,
        )
    