"""
Train the Mamba inverse controller for the Chemostat plant.

This script loads a prepared training dataset (features X and targets Y),
initializes the plant and controller using the shared hyperparameter
configuration, and runs the training routine to produce a trained controller
and any associated artifacts (saved models, logs, plots).

Usage: run this file as a script. Adjust dataset_path below if needed.
"""

import torch

from src.sample.config import *

from src.sample.classes.plants.IndForProteinProductionPlant import *
from src.sample.classes.plants.ChemostatPlant import *
from src.sample.classes.plants.MassSpringDamperPlant import MassSpringDamperPlant
from src.sample.classes.plants.TrophophasePlant import *
from src.sample.classes.plants.IdiophasePlant import *
from src.sample.classes.plants.CocultivationPlant import CoCultivationPlant


from src.sample.classes.controllers.MambaInverseController import *
from src.sample.classes.controllers.ESNInverseController import *
from src.sample.classes.controllers.LSTMInverseController import *
from src.sample.classes.controllers.TransformerInverseController import *

from src.hyperparam_config import *

from src.sample.utils.training_utils import * 
from sample.utils.saving_and_loading_utils import *
from src.sample.utils.plotting_utils import *

# Device configuration: use GPU if available, otherwise fallback to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import optuna

import copy

def set_nested_value(d, keys, value):
    """Helper to set a value in a nested dictionary given a list/path of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def objective(trial, model_class, Y_trajectories, U_trajectories, X_states, base_config, plant, dirname, param_space):
    """
    Modular Optuna objective function sampling hyperparameters from a space dict.
    """
    config = copy.deepcopy(base_config)

    # --- 1. DYNAMICALLY SAMPLE HYPERPARAMETERS FROM PARAM_SPACE ---
    for param_path, spec in param_space.items():
        # Split path string (e.g., "mamba.d_state") into list of keys
        keys = param_path.split(".") if isinstance(param_path, str) else param_path
        
        # Determine parameter name for Optuna logs
        param_name = keys[-1] if isinstance(keys, list) else keys
        
        param_type = spec["type"]
        
        # Sample using appropriate Optuna method
        if param_type == "int":
            val = trial.suggest_int(param_name, spec["low"], spec["high"], step=spec.get("step", 1), log=spec.get("log", False))
        elif param_type == "float":
            val = trial.suggest_float(param_name, spec["low"], spec["high"], step=spec.get("step", None), log=spec.get("log", False))
        elif param_type == "categorical":
            val = trial.suggest_categorical(param_name, spec["choices"])
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")

        # Insert sampled value into the nested config dictionary
        set_nested_value(config, keys, val)

    # Dynamic directory for each trial
    trial_dirname = f"{dirname}/trial_{trial.number}"

    # Re-instantiate model with updated config
    model = model_class(config)

    # --- 2. EXECUTE CONTROLLER TRAINING ---
    fold_histories, dict, mean_cv_val_loss = train_controller_open_loop(
        model=model,
        Y_trajectories=Y_trajectories,
        U_trajectories=U_trajectories,
        X_states=X_states,
        hyperparam_config=config,
        plant=plant,
        dirname=trial_dirname,
        show_plots=False,
        run_simulation=False,
        run_sim_with_plots=False
    )

    return mean_cv_val_loss


def run_optuna_study():
    # Setup directories
    optuna_dir = "./optuna_trials"
    final_dir = "./final_production_model"
    n_trials = 1000  # Set desired trials limit
    
    print("🛠️ Initializing Optuna Study (TPE Sampler)...")
    study = optuna.create_study(
        study_name="controller_hyperparam_tuning",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # 1. Run Bayesian Optimization via Optuna
    study.optimize(
        lambda trial: objective(
            trial=trial,
            model_class=MambaInverseController, 
            Y_trajectories=Y_trajectories,
            U_trajectories=U_trajectories,
            X_states=X_states,
            base_config=hyperparam_config,
            plant=plant,
            param_space=hyperparam_config["mamba_param_space"],
            dirname=optuna_dir
        ),
        n_trials=n_trials
    )

    # 2. Print Summary Results
    print("\n==========================================")
    print("🏆 OPTUNA HYPERPARAMETER TUNING COMPLETE")
    print("==========================================")
    print(f"Best Trial Number   : {study.best_trial.number}")
    print(f"Best Mean CV Loss   : {study.best_value:.6f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   - {key}: {value}")

    # 3. Export Visualizations
    plot_param_heatmap(
        study=study,
        param_x="d_state",
        param_y="expand",
        filename="optuna_heatmap_d_state_expand",
        dirname=optuna_dir
    )

    # 4. Retrain Final Model on Full Dataset
    # Prepare best configuration dictionary
    best_config = copy.deepcopy(hyperparam_config)
    best_config["mamba"].update(study.best_params)

    print("best config", best_config)

    # Instantiate fresh model with the best parameters
    best_model = MambaInverseController(best_config)

    # Train on complete dataset
    final_model = train_full_dataset(
        model=best_model,
        Y_trajectories=Y_trajectories,
        U_trajectories=U_trajectories,
        hyperparam_config=best_config,
        dirname=final_dir
    )
    
    return study, final_model
            #
def evaluate_dataset_size_scaling(
        fractions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        dataset_path="results/2026-07-27/2026-07-27_13-38-34/TrophophasePlant_training_data/dataset/2026-07-27_13-38-34_training_data.pt",
        base_output_dir="dataset_size_ablation_study"
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load hyperparameter configuration & instantiate plant
        hyperparam_config = copy.deepcopy(hyperparam_config_TrophophasePlant)
        plant = TrophophasePlant(hyperparam_config=hyperparam_config)
        
        # Load dataset
        print(f"📦 Loading dataset from: {dataset_path}")
        dataset = torch.load(dataset_path, weights_only=True)
        Y_trajectories = dataset["y"]       # Shape: [Num_Trajectories, Seq_Len, output_dim]
        U_trajectories = dataset["u"]       # Shape: [Num_Trajectories, Seq_Len, input_dim]
        X_states = dataset["states"]        # Shape: [Num_Trajectories, Seq_Len, state_dim]

        total_trajectories = Y_trajectories.shape[0]
        results_records = []

        print(f"🚀 Starting ablation study over dataset fractions: {fractions}")
        print(f"Total available trajectories: {total_trajectories}")

        for frac in fractions:
            print(f"\n==================================================")
            print(f"🔬 RUNNING EVALUATION FOR DATASET FRACTION: {int(frac * 100)}%")
            print(f"==================================================")

            # 1. Slice dataset by fraction of trajectories
            num_sampled_trajectories = max(1, int(total_trajectories * frac))
            
            # Deterministic slice (or use np.random.choice for random subsetting)
            indices = np.arange(num_sampled_trajectories)
            
            Y_sub = Y_trajectories[indices]
            U_sub = U_trajectories[indices]
            X_sub = X_states[indices]

            # 2. Setup directory structure for this run
            run_dirname = os.path.join(base_output_dir, f"frac_{int(frac*100):03d}_percent")
            os.makedirs(run_dirname, exist_ok=True)
            
            # 3. Instantiate fresh model controller
            controller = TransformerInverseController(hyperparam_config=hyperparam_config)

            # 4. Train model on subsetted dataset
            _, metrics = train_controller_open_loop(
                model=controller,
                Y_trajectories=Y_sub,
                U_trajectories=U_sub,
                X_states=X_sub,
                hyperparam_config=hyperparam_config,
                plant=plant,
                dirname=run_dirname,
                show_plots=False,
                run_simulation=False,
                run_sim_with_plots=False
            )

            # 5. Record results
            record = {
                "fraction": frac,
                "dataset_percentage": f"{int(frac * 100)}%",
                "num_trajectories": num_sampled_trajectories,
                "avg_train_loss": metrics["train_loss"],
                "avg_val_loss": metrics["val_loss"],
                "avg_closed_loop_mse": metrics["closed_loop_mse"]
            }
            results_records.append(record)
            
            print(f"\n✅ Finished {int(frac * 100)}% Fraction | "
                f"Train Loss: {metrics['train_loss']:.6f} | "
                f"Val Loss: {metrics['val_loss']:.6f} | "
                f"Closed-Loop MSE: {metrics['closed_loop_mse']:.6f}")

        # Compile into DataFrame
        results_df = pd.DataFrame(results_records)

        # Save summary results CSV
        os.makedirs(base_output_dir, exist_ok=True)
        summary_path = os.path.join(base_output_dir, "dataset_size_ablation_summary.csv")
        results_df.to_csv(summary_path, index=False)
        
        print("\n==================================================")
        print("📊 FINAL DATASET SIZE ABLATION SUMMARY")
        print("==================================================")
        print(results_df.to_string(index=False))

        fractions_to_test = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        df_results = evaluate_dataset_size_scaling(fractions=fractions_to_test)
        x_trajectories = df_results["num_trajectories"].to_numpy()
        train_losses = df_results["avg_train_loss"].to_numpy()
        val_losses = df_results["avg_val_loss"].to_numpy()
        
        plot_signals(
            t=x_trajectories,
            signals=[train_losses, val_losses],
            labels=["Avg Train Loss", "Avg Val Loss"],
            title="Controller Loss vs. Dataset Size",
            xlabel="Number of Training Sequences",
            ylabel="Loss (Normalized RMSE / MSE)",
            figsize=(6, 5),
            filename="dataset_size_vs_loss",
            dirname="dataset_size_ablation_study",
            show=True
        )
        
        return results_df

if __name__ == "__main__":

    # Instantiate plant using hyperparameters.
    dirname = "results/run_1"
    hyperparam_config = hyperparam_config_TrophophasePlant
    
    plant = TrophophasePlant(hyperparam_config=hyperparam_config)
    
    dataset_path = (
       "results/2026-08-17/2026-08-17_22-44-04/TrophophasePlant_training_data_run_1/dataset/2026-08-17_22-44-04_training_data.pt"
    )
    
    # Load the dataset from disk. 
    dataset = torch.load(dataset_path, weights_only=True)
    # Extract the trajectories
    Y_trajectories=dataset["y"]      
    U_trajectories=dataset["u"]
    X_states=dataset["states"]

    # Set seed for reproducibility
    np.random.seed(42)
    N_total = len(Y_trajectories)
    indices = np.random.permutation(N_total)

    # Calculate split sizes (70% train, 15% test, 15% val)
    n_train = int(0.70 * N_total)
    n_test  = int(0.15 * N_total)

    train_idx = indices[:n_train]
    test_idx  = indices[n_train : n_train + n_test]
    val_idx   = indices[n_train + n_test:]

    # Slice data along axis 0
    train_data = (Y_trajectories[train_idx], U_trajectories[train_idx])
    save_training_dataset(train_data, dirname, "train_data")
    test_data  = (Y_trajectories[test_idx],  U_trajectories[test_idx])
    save_training_dataset(test_data, dirname, "test_data")

    X_val = X_states[val_idx] if X_states is not None else None
    val_data   = (Y_trajectories[val_idx],   U_trajectories[val_idx], X_val)
    save_training_dataset(val_data, dirname, "val_data")
    # Initialize the inverse controller.
    controller = MambaInverseController(hyperparam_config=hyperparam_config)

    # Directory name to store training artifacts (models, plots, logs).
    dirname_open_loop = f"{plant.__class__.__name__}_training_open_loop"
    dirname_closed_loop_offline = f"{plant.__class__.__name__}_training_closed_loop_offline"
    dirname_closed_loop_online = f"{plant.__class__.__name__}_training_closed_loop_online"

    save_to_json(dataset_path, dirname_open_loop, "training_data_path")
    save_to_json(hyperparam_config, dirname_open_loop,"hyperparam_config") 

    save_to_json(dataset_path, dirname_closed_loop_offline, "training_data_path")
    save_to_json(hyperparam_config, dirname_closed_loop_offline,"hyperparam_config")    

    save_to_json(dataset_path, dirname_closed_loop_online, "training_data_path")
    save_to_json(hyperparam_config, dirname_closed_loop_online,"hyperparam_config")   

    # run_optuna_study()

    train_controller_closed_loop_offline(
        model=controller,
        train_data=train_data,
        test_data=test_data,
        hyperparam_config=hyperparam_config,
        plant=plant,
        dirname="results/run_1"
    )


    
    # train_controller_sanem(
    #     controller,
    #     Y_trajectories=dataset["y"],      
    #     U_trajectories=dataset["u"],
    #     X_states=dataset["states"],
    #     hyperparam_config=hyperparam_config,
    #     plant=plant,
    #     dirname=dirname,
    #     run_simulation=True,
    #     run_sim_with_plots=False
    # )  


    # train_controller_closed_loop_offline(
    #             controller,
    #             Y_trajectories=dataset["y"],      
    #             U_trajectories=dataset["u"],
    #             X_states=dataset["states"],
    #             hyperparam_config=hyperparam_config,
    #             plant=plant,
    #             dirname=dirname_closed_loop_offline
    #         )  


    # train_controller_closed_loop_online(
    #             controller,
    #             Y_trajectories=dataset["y"],      
    #             U_trajectories=dataset["u"],
    #             X_states=dataset["states"],
    #             hyperparam_config=hyperparam_config,
    #             plant=plant,
    #             dirname=dirname_closed_loop_online
    #         )  


    # ESN
    # def prepare_inverse_control_dataset_esn(data_dict, seq_len=None):
    #     """
    #     Slices raw sequence data into X (features) and Y (targets) for Direct Inverse Control.
        
    #     Args:
    #         data_dict (dict): Dictionary with keys "u", "y", "states"
    #                         where "u" has shape [N, T, num_inputs]
    #                         and "y" has shape [N, T, num_outputs]
    #         seq_len (int, optional): Truncates or sub-sequences the trajectory to a fixed length.
            
    #     Returns:
    #         dict: {"x": X, "y": Y} where
    #             X shape: [total_sequences, sequence_length, 2 * num_outputs]
    #             Y shape: [total_sequences, sequence_length, num_inputs]
    #     """
    #     u = data_dict["u"]  # Shape: [N, T, num_inputs]
    #     y = data_dict["y"]  # Shape: [N, T, num_outputs]

    #     # 1. Align time steps for transition: (y_t -> y_t+1) using u_t
    #     # y_t    : step 0 to T-1
    #     # y_next : step 1 to T
    #     y_t = y[:, :-1, :]       # [N, T-1, num_outputs]
    #     y_next = y[:, 1:, :]      # [N, T-1, num_outputs]
    #     u_control = u[:, :-1, :]  # [N, T-1, num_inputs]

    #     # 2. Concatenate y_t and y_next along the feature dimension (dim=-1)
    #     # Resulting feature shape: [N, T-1, 2 * num_outputs]
    #     X = torch.cat([y_t, y_next], dim=-1)
        
    #     # Target shape: [N, T-1, num_inputs]
    #     Y = u_control

    #     # 3. (Optional) Crop sequence length if requested
    #     if seq_len is not None and seq_len < X.shape[1]:
    #         X = X[:, :seq_len, :]
    #         Y = Y[:, :seq_len, :]

    #     return {
    #         "x": X,
    #         "y": Y
    #     }

    # prepared_data = prepare_inverse_control_dataset_esn(dataset, seq_len=2000)
    # X = prepared_data["x"]
    # Y = prepared_data["y"]
    # controller = ESNInverseController(hyperparam_config=hyperparam_config)
    # train_controller_esn(
    #         controller,
    #         X,
    #         Y,
    #         hyperparam_config,
    #         plant,
    #         dirname=dirname,
    #         run_simulation=False,
    #     )

    
    
    