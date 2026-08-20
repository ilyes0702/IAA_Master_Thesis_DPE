# Import standard libraries
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.sample.config import *
from src.sample.decorators.general_decorators import *
from src.sample.utils.loss_utils import *
from src.sample.utils.plotting_utils import *
from sample.utils.saving_and_loading_utils import *
from src.sample.utils.general_utils import *

plt.style.use("src/sample/style.mplstyle")


import copy
import os




def train_controller_open_loop(
    model,
    train_data,
    test_data,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_simulation=True,
    run_sim_with_plots=False,
):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    device = train_cfg["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]

    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    patience = train_cfg.get("val_patience_epochs", 10)
    min_delta = train_cfg.get("val_min_delta", 1e-4)

    plant_cfg = hyperparam_config["plant"]
    input_dim = plant_cfg["input_dim"]
    output_dim = plant_cfg["output_dim"]

    # Helper function to extract or slice dataset arrays
    def _prepare_split(split_data, split_name):
        arg1, arg2 = split_data
        
        print(
            f"🔄 Slicing {split_name} trajectories with sliding windows (n_y={n_y}, n_u={n_u})..."
            )
        return create_inverse_controller_dataset(
            Y_trajectories=arg1, U_trajectories=arg2, n_y=n_y, n_u=n_u
        )
            
    # --- 1. PREPARE SPLITS ---
    train_x_raw, train_y_raw = _prepare_split(train_data, "train")
    test_x_raw, test_y_raw = _prepare_split(test_data, "test")

    print(
        f"📊 Dataset shapes -> Train: {train_x_raw.shape} | Test: {test_x_raw.shape}"
    )

    # --- 2. FIT SCALERS ON TRAIN DATA ONLY ---
    print("⚖️ Fitting StandardScalers on Training Data...")
    N_train, seq_len_train, dim_x = train_x_raw.shape
    dim_y = train_y_raw.shape[-1]

    train_x_flat = train_x_raw.reshape(-1, dim_x)
    train_y_flat = train_y_raw.reshape(-1, dim_y)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(train_x_flat)
    scaler_y.fit(train_y_flat)

    # Transform Train
    train_x = torch.tensor(
        scaler_x.transform(train_x_flat).reshape(N_train, seq_len_train, dim_x),
        dtype=torch.float32,
    )
    train_y = torch.tensor(
        scaler_y.transform(train_y_flat).reshape(N_train, seq_len_train, dim_y),
        dtype=torch.float32,
    )

    # Transform Test
    N_test, seq_len_test, _ = test_x_raw.shape
    test_x_flat = test_x_raw.reshape(-1, dim_x)
    test_y_flat = test_y_raw.reshape(-1, dim_y)
    test_x = torch.tensor(
        scaler_x.transform(test_x_flat).reshape(N_test, seq_len_test, dim_x),
        dtype=torch.float32,
    )
    test_y = torch.tensor(
        scaler_y.transform(test_y_flat).reshape(N_test, seq_len_test, dim_y),
        dtype=torch.float32,
    )

    # Save Config and Scalers
    save_to_json(hyperparam_config, dirname, "hyperparam_config")
    save_scaler_object(scaler_x, dirname=dirname, filename="scaler_x")
    save_scaler_object(scaler_y, dirname=dirname, filename="scaler_y")

    # --- SETUP MODEL, OPTIMIZER, & LOSS ---
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=train_cfg["lr_decay_rate"]
    )

    loss_name = train_cfg["loss_function"].replace("()", "")
    if loss_name == "NormalizedRMSELoss":
        criterion = NormalizedRMSELoss(reduction="none")
    elif loss_name == "MSELoss":
        criterion = torch.nn.MSELoss(reduction="none")

    train_epoch_history = []
    test_epoch_history = []
    train_batch_loss_history = []

    best_test_loss = float("inf")
    patience_counter = 0

    # --- 3. TRAINING LOOP (TESTING PERFORMANCE EVERY EPOCH) ---
    for epoch in range(epochs):
        model.train()
        epoch_train_loss_accum = 0.0

        print(f"\n🎬 Starting Epoch {epoch+1}/{epochs}")
        shuffled_train_indices = torch.randperm(N_train)

        # Batch Training Pass
        for i in range(0, N_train, mini_batch_size):
            batch_indices = shuffled_train_indices[i : i + mini_batch_size]
            current_batch_size = len(batch_indices)

            batch_x = train_x[batch_indices].to(device)
            batch_y = train_y[batch_indices].to(device)

            if hasattr(model, "reset_memory"):
                model.reset_memory(
                    batch_size=current_batch_size, device=device
                )

            optimizer.zero_grad()
            u_pred_batch = model(batch_x)
            raw_loss = criterion(u_pred_batch, batch_y)
            loss = raw_loss.mean()

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            epoch_train_loss_accum += current_loss_val * current_batch_size
            train_batch_loss_history.append(current_loss_val)

        scheduler.step()
        mean_train_loss = epoch_train_loss_accum / N_train
        train_epoch_history.append(mean_train_loss)

        # --- TEST PASS EVERY EPOCH ---
        model.eval()
        epoch_test_loss_accum = 0.0

        with torch.no_grad():
            for i in range(0, N_test, mini_batch_size):
                batch_test_x = test_x[i : i + mini_batch_size].to(device)
                batch_test_y = test_y[i : i + mini_batch_size].to(device)
                current_test_batch_size = len(batch_test_x)

                if hasattr(model, "reset_memory"):
                    model.reset_memory(
                        batch_size=current_test_batch_size, device=device
                    )

                u_test_pred = model(batch_test_x)
                raw_test_loss = criterion(u_test_pred, batch_test_y)
                epoch_test_loss_accum += (
                    raw_test_loss.mean().item() * current_test_batch_size
                )

        mean_test_loss = (
            epoch_test_loss_accum / N_test if N_test > 0 else 0.0
        )
        test_epoch_history.append(mean_test_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"✨ Epoch {epoch+1} Summary | LR: {current_lr:.6e} | Train Loss: {mean_train_loss:.6f} | Test Loss: {mean_test_loss:.6f}"
        )

        # Checkpoint / Early Stopping on Test Set Performance
        if mean_test_loss < (best_test_loss - min_delta):
            best_test_loss = mean_test_loss
            patience_counter = 0
            save_model(
                model,
                dirname=dirname,
                hyperparam_config=hyperparam_config,
                filename="best_model",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered at Epoch {epoch+1}.")
                break

    # --- 4. LOSS PLOTS ---
    plot_signals(
        t=np.arange(len(train_batch_loss_history)),
        signals=[np.array(train_batch_loss_history)],
        labels=["Batch Training Loss"],
        xlabel="Optimization Steps",
        ylabel="Loss",
        dirname=dirname,
        filename="granular_training_loss",
    )

    plot_signals(
        t=np.arange(1, len(train_epoch_history) + 1),
        signals=[np.array(train_epoch_history), np.array(test_epoch_history)],
        labels=["Avg Train Loss", "Avg Test Loss"],
        xlabel="Epochs",
        ylabel="Loss",
        dirname=dirname,
        filename="epoch_loss_curves",
    )

    # --- SUMMARY METRICS ---
    summary_metrics = {
        "Metric": ["Best Training Loss", "Best Testing Loss"],
        "Value": [min(train_epoch_history), best_test_loss],
    }
    summary_df = pd.DataFrame(summary_metrics)

    print("\n==========================================")
    print("📊 TRAINING SUMMARY")
    print("==========================================")
    print(summary_df.to_string(index=False))
    print("==========================================\n")

    save_df_to_csv(summary_df, dirname=dirname, filename="training_summary")

    history = {
        "train_loss": train_epoch_history,
        "test_loss": test_epoch_history,
    }

    return model, history, summary_df

def train_controller_sanem(
    model,
    Y_trajectories,
    U_trajectories,
    X_states,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_simulation=True,
    run_sim_with_plots=False
):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    
    device = train_cfg["device"]
    
    dt = hyperparam_config["training_data_cfg"]["dt"]
    k_folds = train_cfg["k_folds"]

    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    val_patience = train_cfg["val_patience_epochs"]
    min_delta = train_cfg["val_min_delta"]


    plant_cfg = hyperparam_config["plant"]
    input_dim = plant_cfg["input_dim"]
    output_dim = plant_cfg["output_dim"]

    # Place this right before: for seq_idx in range(val_size):
    sim_metrics_records = []

    # --- 1. GENERATE SLIDING WINDOW DATASET ---
    print(f"🔄 Slicing trajectories with sliding windows (n_y={n_y}, n_u={n_u})...")
    X_raw, Y_raw = create_inverse_controller_dataset(
        Y_trajectories=Y_trajectories,
        U_trajectories=U_trajectories,
        n_y=n_y,
        n_u=n_u
    )
    print("X_raw", X_raw.shape)
    print("Y_raw", Y_raw.shape)

    total_sequences = X_raw.shape[0]
    sliding_seq_len = X_raw.shape[1]
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    initial_model_state = copy.deepcopy(model.state_dict())
    fold_histories = {}

    # --- K-FOLD CROSS VALIDATION LOOP ---
    # Loop across folds. In case of 5-fold cross validation, the code in this for loop will run 5 times for different partitions of the data into training and validation data.
    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING FOLD {fold + 1} / {k_folds}")
        print(f"==========================================")

        model.load_state_dict(initial_model_state)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg["lr_decay_rate"])

        loss_name = train_cfg["loss_function"].replace("()", "")
        if loss_name == "NormalizedRMSELoss":
            criterion = NormalizedRMSELoss(reduction='None')
        elif loss_name == "MSELoss":
            criterion = torch.nn.MSELoss(reduction='none')

        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)

        train_x_raw, val_x_raw = X_raw[train_idx_arr], X_raw[val_idx_arr]
        train_y_raw, val_y_raw = Y_raw[train_idx_arr], Y_raw[val_idx_arr]

        # --- INTERNAL FOLD STANDARD SCALING ---
        print(f"⚖️ Fitting independent StandardScalers for Fold {fold + 1}...")

        N_train, seq_len, dim_x = train_x_raw.shape
        dim_y = train_y_raw.shape[-1]

        train_x_flat = train_x_raw.reshape(-1, dim_x)
        train_y_flat = train_y_raw.reshape(-1, dim_y)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        scaler_x.fit(train_x_flat)
        scaler_y.fit(train_y_flat)

        train_x = torch.tensor(scaler_x.transform(train_x_flat).reshape(N_train, seq_len, dim_x), dtype=torch.float32)
        train_y = torch.tensor(scaler_y.transform(train_y_flat).reshape(N_train, seq_len, dim_y), dtype=torch.float32)

        N_val = val_x_raw.shape[0]
        val_x_flat = val_x_raw.reshape(-1, dim_x)
        val_y_flat = val_y_raw.reshape(-1, dim_y)

        val_x = torch.tensor(scaler_x.transform(val_x_flat).reshape(N_val, seq_len, dim_x), dtype=torch.float32)
        val_y = torch.tensor(scaler_y.transform(val_y_flat).reshape(N_val, seq_len, dim_y), dtype=torch.float32)

        fold_dir = f"{dirname}/fold_{fold+1}"
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")

        if show_plots:
            curves_dir = f"{fold_dir}/transformed_data_curves"
            sample_x = train_x[0].numpy()
            sample_y = train_y[0].numpy()
            t_axis = np.arange(seq_len) * dt
            for out_idx in range(output_dim):
                plot_signals(t=t_axis, signals=[sample_y[:, out_idx]], labels=[f"Scaled u_{out_idx+1}"],
                             xlabel="Time (s)", ylabel="Standardized Units",
                             title=f"Fold {fold+1} | Transformed Input u_{out_idx+1}",
                             dirname=curves_dir, filename=f"scaled_u{out_idx+1}_curve")

        train_size = train_x.shape[0]
        val_size = val_x.shape[0]

        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        fold_train_channel_batch_loss = {ch: [] for ch in range(output_dim)}
        fold_train_channel_epoch_history = {ch: [] for ch in range(output_dim)}
        fold_val_channel_epoch_history = {ch: [] for ch in range(output_dim)}
        fold_val_epoch_history = []
        fold_train_epoch_history = []

        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False

        for epoch in range(epochs):
            model.train()
            epoch_train_loss_accum = 0.0
            epoch_train_channel_accum = {ch: 0.0 for ch in range(output_dim)}

            print(f"\n🎬 Fold {fold+1} | Starting Epoch {epoch+1}/{epochs}")
            shuffled_train_indices = torch.randperm(train_size)

            for i in range(0, train_size, mini_batch_size):
                batch_indices = shuffled_train_indices[i : i + mini_batch_size]
                current_batch_size = len(batch_indices)

                batch_x = train_x[batch_indices].to(device)
                batch_y = train_y[batch_indices].to(device)

                if hasattr(model, 'reset_memory'):
                    model.reset_memory(batch_size=current_batch_size, device=device)

                optimizer.zero_grad()
                u_pred_batch = model(batch_x)
                raw_loss = criterion(u_pred_batch, batch_y)
                # Reduce the loss using sum or mean
                loss = raw_loss.mean()

                loss.backward()
                optimizer.step()

                current_loss_val = loss.item()
                epoch_train_loss_accum += current_loss_val * current_batch_size
                fold_train_batch_loss.append(current_loss_val)
                fold_train_batch_indices.append(global_batch_counter)

                for ch in range(input_dim):
                    ch_loss_val = raw_loss[:, :, ch].mean().item()
                    epoch_train_channel_accum[ch] += ch_loss_val * current_batch_size
                    fold_train_channel_batch_loss[ch].append(ch_loss_val)

                global_batch_counter += 1

            scheduler.step()

            # --- 3. VALIDATION PASS ---
            model.eval()
            epoch_val_loss_accum = 0.0
            epoch_val_channel_accum = {ch: 0.0 for ch in range(output_dim)}

            all_val_preds = []
            all_val_trues = []

            with torch.no_grad():
                for i in range(0, val_size, mini_batch_size):
                    batch_val_x = val_x[i : i + mini_batch_size].to(device)
                    batch_val_y = val_y[i : i + mini_batch_size].to(device)
                    current_val_batch_size = len(batch_val_x)

                    if hasattr(model, 'reset_memory'):
                        model.reset_memory(batch_size=current_val_batch_size, device=device)

                    u_val_pred = model(batch_val_x)
                    raw_val_loss = criterion(u_val_pred, batch_val_y)

                    val_loss = raw_val_loss.mean()
                    epoch_val_loss_accum += val_loss.item() * current_val_batch_size

                    for ch in range(input_dim):
                        ch_val_loss_val = raw_val_loss[:, :, ch].mean().item()
                        epoch_val_channel_accum[ch] += ch_val_loss_val * current_val_batch_size

                    all_val_preds.append(u_val_pred.cpu().numpy())
                    all_val_trues.append(batch_val_y.cpu().numpy())

            val_all_preds_arr = np.concatenate(all_val_preds, axis=0)
            val_all_trues_arr = np.concatenate(all_val_trues, axis=0)

            mean_train_loss = epoch_train_loss_accum / train_size
            mean_val_loss = (epoch_val_loss_accum / val_size) if val_size > 0 else 0.0

            fold_val_epoch_history.append(mean_val_loss)
            fold_train_epoch_history.append(mean_train_loss)

            if fold not in fold_histories:
                fold_histories[fold] = {
                    "train_loss": [], "val_loss": [], "val_epochs": [],
                    **{f"train_loss_ch{ch+1}": [] for ch in range(output_dim)},
                    **{f"val_loss_ch{ch+1}": [] for ch in range(output_dim)}
                }

            fold_histories[fold]["train_loss"].append(mean_train_loss)
            fold_histories[fold]["val_loss"].append(mean_val_loss)
            fold_histories[fold]["val_epochs"].append(epoch + 1)

            for ch in range(output_dim):
                mean_train_ch = epoch_train_channel_accum[ch] / train_size
                mean_val_ch = (epoch_val_channel_accum[ch] / val_size) if val_size > 0 else 0.0

                fold_train_channel_epoch_history[ch].append(mean_train_ch)
                fold_val_channel_epoch_history[ch].append(mean_val_ch)

                fold_histories[fold][f"train_loss_ch{ch+1}"].append(mean_train_ch)
                fold_histories[fold][f"val_loss_ch{ch+1}"].append(mean_val_ch)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"✨ [Fold {fold+1}] Epoch {epoch+1} Summary:")
            print(f"   ↳ LR: {current_lr:.6e} | Total Train Loss: {mean_train_loss:.6f} | Total Val Loss: {mean_val_loss:.6f}")

            if mean_val_loss < (best_val_loss - min_delta):
                best_val_loss = mean_val_loss
                patience_counter = 0
                save_model(model, dirname=fold_dir, hyperparam_config=hyperparam_config, filename="best_fold_model")
            else:
                patience_counter += 1
                if patience_counter >= val_patience:
                    print(f"🛑 Early stopping fold {fold+1} at Epoch {epoch+1}.")
                    early_stopped = True
                    break
        
        # --- 4. PLOT EXTENDED LOSS CURVES ---
        fold_title_suffix = " (Early Stopped)" if early_stopped else " (Full Run)"
        epoch_axis = np.array(fold_histories[fold]["val_epochs"])

        plot_signals(t=np.array(fold_train_batch_indices),
                     signals=[np.array(fold_train_batch_loss)],
                     labels=[f"Fold {fold+1} Total Loss"],
                     xlabel="Optimization Steps",
                     ylabel="Loss",
                     dirname=fold_dir,
                     filename="granular_training_loss")

        plot_signals(t=epoch_axis,
                     signals=[np.array(fold_train_epoch_history), np.array(fold_val_epoch_history)],
                     labels=["Avg Train Loss", "Avg Val Loss"],
                     xlabel="Epochs",
                     ylabel="Loss",
                     dirname=fold_dir,
                     filename="epoch_validation_loss")

        # --- PLOT VALIDATION PREDICTION SAMPLE FOR THIS FOLD ---
        print(f"📈 Plotting sample validation prediction for Fold {fold + 1}...")
        pred_curves_dir = f"{fold_dir}/validation_tracking_curves"
        t_axis_val = np.arange(seq_len) * dt

        sample_seq_idx = 0
        sample_pred_scaled = val_all_preds_arr[sample_seq_idx]
        sample_true_scaled = val_all_trues_arr[sample_seq_idx]

        sample_pred_unscaled = scaler_y.inverse_transform(sample_pred_scaled)
        sample_true_unscaled = scaler_y.inverse_transform(sample_true_scaled)

        for ch in range(input_dim):
            plot_signals(
                t=t_axis_val,
                signals=[sample_true_unscaled[:, ch], sample_pred_unscaled[:, ch]],
                labels=[f"True u_{ch+1}", f"Predicted u_{ch+1}"],
                xlabel="Time (h)",
                ylabel="Control Units",
                #title=f"Fold {fold+1} | Validation Sample Performance - Channel u_{ch+1}",
                dirname=pred_curves_dir,
                filename=f"val_prediction_sample_u{ch+1}_0"
            )

            # --- 5. TRUE CLOSED-LOOP PLANT SIMULATION ROLLOUT ---
            if run_simulation:
                print(f"📊 Running TRUE CLOSED-LOOP plant simulation for Fold {fold + 1}...")

                if isinstance(plant, type):
                    plant_instance = plant(hyperparam_config)
                else:
                    plant_instance = plant

                plot_configs = plant_instance.get_plot_config() if hasattr(plant_instance, "get_plot_config") else []
                u_config = next((c for c in plot_configs if any(col.startswith("u") for col in c["cols"])), None)
                y_config = next((c for c in plot_configs if any(col.startswith("y") for col in c["cols"])), None)
                x_config = next((c for c in plot_configs if any(col.startswith("x") for col in c["cols"])), None)

                total_trajectory_len = Y_trajectories.shape[1]
                t_axis_full = np.arange(total_trajectory_len) * dt
                lookback_offset = hyperparam_config["train"]["lookback_offset"] #max(n_y, n_u)
                sliding_seq_len = total_trajectory_len - 1 - lookback_offset

                # We must evaluate the model in eval mode
                model.eval()

                for seq_idx in range(val_size):
                    val_traj_idx = val_idx_arr[seq_idx]
                    target_y_trajectory = Y_trajectories[val_traj_idx].cpu().numpy() if hasattr(Y_trajectories, "cpu") else Y_trajectories[val_traj_idx]
                    target_u_trajectory = U_trajectories[val_traj_idx].cpu().numpy() if hasattr(U_trajectories, "cpu") else U_trajectories[val_traj_idx]
                    
                    # 🌟 NEW: Extract the TRUE ground-truth states for this trace
                    target_state_trajectory = X_states[val_traj_idx].cpu().numpy() if hasattr(X_states, "cpu") else X_states[val_traj_idx]

                    # 🌟 NEW: Initialize simulator state with the EXACT true physical initial state
                    true_initial_state = target_state_trajectory[0].copy()
                    current_sim_state = torch.tensor(true_initial_state, device=device, dtype=torch.float32).unsqueeze(0)
                    
                    state_dim = current_sim_state.shape[-1]

                    # Pre-allocate tracking arrays for simulation
                    simulated_states = np.zeros((total_trajectory_len, state_dim))
                    simulated_outputs = np.zeros((total_trajectory_len, output_dim))
                    simulated_controls = np.zeros((total_trajectory_len, input_dim))

                    # Step 0 Initialization
                    simulated_outputs[0] = target_y_trajectory[0].copy()
                    simulated_controls[0] = target_u_trajectory[0].copy()
                    simulated_states[0] = true_initial_state.copy()  # Use the true initial state

                    # To feed a sequence model (like Mamba) step-by-step without stateful memory hacks,
                    # we maintain an active history of our scaled inputs.
                    # Shape: [1, sliding_seq_len, feature_dim]
                    dim_x = scaler_x.mean_.shape[0]
                    scaled_inputs_seq = np.zeros((1, sliding_seq_len, dim_x))

                    # Step through the plant simulator
                    for k in range(total_trajectory_len - 1):
                        
                        # 1. HAND-OFF PHASE: Before we have enough lookback history
                        if k < lookback_offset:
                            u_action = target_u_trajectory[k].copy()
                            
                            # Step the plant using ground-truth actions to build up initial history
                            u_action_tensor = torch.tensor(u_action.reshape(1, -1), device=device, dtype=torch.float32)
                            current_sim_state, y_next_sim = plant_instance.step(
                                state=current_sim_state, u=u_action_tensor, t=k * dt, dt=dt
                            )
                            
                            simulated_controls[k + 1] = u_action
                            simulated_outputs[k + 1] = y_next_sim.squeeze(0).cpu().numpy()
                            simulated_states[k + 1] = current_sim_state[0].cpu().numpy()
                            
                        # 2. AUTONOMOUS CLOSED-LOOP PHASE: Model takes the wheel
                        else:
                            # Dynamic Closed-Loop Slicing of Simulated History:
                            # Reference target output we want to reach next: r_{k+1}
                            y_next_ref = target_y_trajectory[k + 1] 
                            
                            # Simulated plant output history (reversed: y_k, y_{k-1}, ... y_{k-n_y})
                            y_hist = simulated_outputs[k - n_y : k + 1]
                            y_hist_reversed = y_hist[::-1].flatten()
                            
                            # Simulated control history (reversed: u_{k-1}, u_{k-2}, ... u_{k-n_u})
                            u_hist = simulated_controls[k - n_u : k]
                            u_hist_reversed = u_hist[::-1].flatten()
                            
                            # Assemble raw feature vector v_k
                            v_k = np.concatenate([y_next_ref, y_hist_reversed, u_hist_reversed])
                            
                            # Scale the step feature using the fold's scaler
                            v_k_scaled = scaler_x.transform(v_k.reshape(1, -1)).squeeze(0)
                            
                            # Place it into our sequence buffer
                            seq_buffer_idx = k - lookback_offset
                            scaled_inputs_seq[0, seq_buffer_idx] = v_k_scaled
                            
                            # Forward pass through the model using the sequence gathered up to this step
                            with torch.no_grad():
                                input_tensor = torch.tensor(scaled_inputs_seq[:, :seq_buffer_idx + 1, :], device=device, dtype=torch.float32)
                                
                                if hasattr(model, 'reset_memory'):
                                    model.reset_memory(batch_size=1, device=device)
                                    
                                u_pred_scaled = model(input_tensor)
                                # Extract the prediction for the current step (the last item in the sequence dimension)
                                u_pred_step_scaled = u_pred_scaled[0, -1, :].cpu().numpy()
                            
                            # Unscale prediction back to physical control units
                            u_action = scaler_y.inverse_transform(u_pred_step_scaled.reshape(1, -1)).squeeze(0)
                            u_action = np.clip(u_action, hyperparam_config["plant"]["u_1_hard_min"], hyperparam_config["plant"]["u_1_hard_max"])
                            # Step the physical plant simulator using the predicted action!
                            u_action_tensor = torch.tensor(u_action.reshape(1, -1), device=device, dtype=torch.float32)
                            current_sim_state, y_next_sim = plant_instance.step(
                                state=current_sim_state, u=u_action_tensor, t=k * dt, dt=dt
                            )
                            
                            # Log actual simulated consequences for the NEXT step (k+1)
                            simulated_controls[k + 1] = u_action
                            simulated_outputs[k + 1] = y_next_sim.squeeze(0).cpu().numpy()
                            simulated_states[k + 1] = current_sim_state[0].cpu().numpy()

                    # Extract trajectories for plotting/saving
                    original_y = target_y_trajectory
                    original_u = target_u_trajectory
                    original_states = target_state_trajectory

                    # --- TRACKING ERROR CALCULATIONS ---
                    # Timestep-by-timestep squared error per output channel
                    # Shape: (total_trajectory_len, output_dim)
                    output_squared_error = (simulated_outputs - original_y) ** 2

                    # Overall Mean Squared Error across the entire sequence
                    seq_mse_total = float(np.mean(output_squared_error))

                    # Per-channel Mean Squared Errors for this sequence
                    seq_mse_per_channel = {
                        f"MSE_y{out_idx+1}": float(np.mean(output_squared_error[:, out_idx]))
                        for out_idx in range(output_dim)
                    }

                    # Record metrics for the summary file
                    metrics_record = {
                        "fold": fold + 1,
                        "seq_idx": seq_idx,
                        "val_traj_idx": int(val_traj_idx),
                        "total_mse": seq_mse_total,
                        **seq_mse_per_channel
                    }
                    sim_metrics_records.append(metrics_record)

                    log_data = {"Time (s)": t_axis_full}

                    # Logging Desired vs Simulated outputs + Per-timestep Squared Error
                    for out_idx in range(output_dim):
                        log_data[f"Desired_y{out_idx+1}"] = original_y[:, out_idx]
                        log_data[f"Simulated_Output_y{out_idx+1}"] = simulated_outputs[:, out_idx]
                        log_data[f"Squared_Error_y{out_idx+1}"] = output_squared_error[:, out_idx]

                    for ch in range(input_dim):
                        log_data[f"Actual_u{ch+1}"] = original_u[:, ch]
                        log_data[f"Predicted_u{ch+1}_ClosedLoop"] = simulated_controls[:, ch]

                    for st in range(state_dim):
                        log_data[f"Simulated_State_x{st+1}"] = simulated_states[:, st]
                        log_data[f"Original_State_x{st+1}"] = original_states[:, st]

                    val_profile_df = pd.DataFrame(log_data)
                    save_df_to_csv(
                        val_profile_df,
                        dirname=pred_curves_dir,
                        filename=f"val_plant_simulation_fold_{fold+1}_seq_{seq_idx+1}"
                    )
                    # Place this immediately after the `for seq_idx in range(val_size):` loop finishes
                    if sim_metrics_records:
                        summary_df = pd.DataFrame(sim_metrics_records)
                        save_df_to_csv(
                            summary_df,
                            dirname=pred_curves_dir,
                            filename=f"val_closed_loop_mse_summary_fold_{fold+1}"
                        )
                        
                        mean_fold_mse = summary_df["total_mse"].mean()
                        print(f"📊 Fold {fold+1} Closed-Loop Mean MSE across validation set: {mean_fold_mse:.6f}")
                    # ---------------------------------------------------------
                    # PLOT AND LOG WRITING
                    # ---------------------------------------------------------
                    if run_sim_with_plots:
                        state_signals = []
                        state_labels = []
                        state_ylabels = []

                        for st in range(state_dim):
                            state_signals.append([original_states[:, st], simulated_states[:, st]])
                            state_labels.append(["Original", "Simulated"])
                            if x_config and st < len(x_config["labels"]):
                                state_ylabels.append(x_config["labels"][st])
                            else:
                                state_ylabels.append(rf"State $x_{{{st+1}}}$")

                        state_asp = [0.33] * len(state_signals)
                        plot_stacked(
                            t=t_axis_full,
                            signals=state_signals,
                            labels=state_labels,
                            xlabel=rf"$t \; / \; \mathrm{{s}}$",
                            ylabel=state_ylabels,
                            asp=state_asp,
                            dirname=pred_curves_dir,
                            filename=f"val_simulation_states_fold_{fold+1}_seq_{seq_idx+1}.png",
                            show=False
                        )

                        io_signals = []
                        io_labels = []
                        io_ylabels = []

                        for ch in range(input_dim):
                            io_signals.append([original_u[:, ch], simulated_controls[:, ch]])
                            io_labels.append(["Original", "Predicted (Closed Loop)"])
                            if u_config and ch < len(u_config["labels"]):
                                io_ylabels.append(u_config["labels"][ch])
                            else:
                                io_ylabels.append(rf"Input $u_{{{ch+1}}}$")

                        for out_idx in range(output_dim):
                            io_signals.append([original_y[:, out_idx], simulated_outputs[:, out_idx]])
                            io_labels.append(["Desired", "Simulated"])
                            if y_config and out_idx < len(y_config["labels"]):
                                io_ylabels.append(y_config["labels"][out_idx])
                            else:
                                io_ylabels.append(rf"Output $y_{{{out_idx+1}}}$")

                        io_asp = [0.33] * len(io_signals)
                        plot_stacked(
                            t=t_axis_full,
                            signals=io_signals,
                            labels=io_labels,
                            xlabel=rf"$t \; / \; \mathrm{{s}}$",
                            ylabel=io_ylabels,
                            asp=io_asp,
                            dirname=pred_curves_dir,
                            filename=f"val_simulation_io_fold_{fold+1}_seq_{seq_idx+1}.png",
                            show=False
                        )

                    log_data = {"Time (s)": t_axis_full}
                    for out_idx in range(output_dim):
                        log_data[f"Desired_y{out_idx+1}"] = original_y[:, out_idx]
                        log_data[f"Simulated_Output_y{out_idx+1}"] = simulated_outputs[:, out_idx]
                    for ch in range(input_dim):
                        log_data[f"Actual_u{ch+1}"] = original_u[:, ch]
                        log_data[f"Predicted_u{ch+1}_ClosedLoop"] = simulated_controls[:, ch]
                    for st in range(state_dim):
                        log_data[f"Simulated_State_x{st+1}"] = simulated_states[:, st]
                        log_data[f"Original_State_x{st+1}"] = original_states[:, st]

                    val_profile_df = pd.DataFrame(log_data)
                    save_df_to_csv(
                        val_profile_df,
                        dirname=pred_curves_dir,
                        filename=f"val_plant_simulation_fold_{fold+1}_seq_{seq_idx+1}"
                    )
    # Calculate average best validation loss across all folds
    all_best_val_losses = []
    for fold_id in fold_histories:
        best_loss_in_fold = min(fold_histories[fold_id]["val_loss"])
        all_best_val_losses.append(best_loss_in_fold)
        
    mean_cv_loss = float(np.mean(all_best_val_losses))

    # --- METRIC AGGREGATION ACROSS FOLDS ---
    fold_best_train_losses = []
    fold_best_val_losses = []

    for f in range(k_folds):
        best_val_epoch_idx = np.argmin(fold_histories[f]["val_loss"])
        fold_best_val_losses.append(fold_histories[f]["val_loss"][best_val_epoch_idx])
        fold_best_train_losses.append(fold_histories[f]["train_loss"][best_val_epoch_idx])

    avg_best_train_loss = float(np.mean(fold_best_train_losses))
    avg_best_val_loss = float(np.mean(fold_best_val_losses))
    mean_cv_loss = avg_best_val_loss  # Open-loop mean CV loss

    # Calculate average closed-loop error across all folds
    if run_simulation and sim_metrics_records:
        sim_df = pd.DataFrame(sim_metrics_records)
        avg_closed_loop_mse = float(sim_df["total_mse"].mean())
    else:
        avg_closed_loop_mse = np.nan

    # --- DATAFRAME CREATION & PRINTING ---


    summary_metrics = {
        "Metric": [
            "Avg Best Open-Loop Train Loss",
            "Avg Best Open-Loop Val Loss (Mean CV)",
            "Avg Closed-Loop MSE",
        ],
        "Value": [
            avg_best_train_loss,
            mean_cv_loss,
            avg_closed_loop_mse,
        ],
    }

    summary_results_df = pd.DataFrame(summary_metrics)

    print("\n==========================================")
    print("📊 K-FOLD CROSS-VALIDATION SUMMARY")
    print("==========================================")
    print(summary_results_df.to_string(index=False))
    print("==========================================\n")

    return (
        fold_histories,
        {
            "train_loss": avg_best_train_loss,
            "val_loss": avg_best_val_loss,
            "closed_loop_mse": avg_closed_loop_mse,
        },
        mean_cv_loss,
        summary_results_df,
    )

import copy
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

import copy
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

import copy
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


import copy
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

def train_controller_closed_loop_online(
    model,
    Y_trajectories,
    U_trajectories,
    X_states,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_sim_with_plots=False
):
    """
    Trains an inverse neural controller online timestep-by-timestep using closed-loop tracking loss.
    
    At each step k, the controller forecasts u_k, steps the plant to compute y_{k+1}, 
    evaluates step loss against y_ref, backpropagates gradients, and updates model weights immediately.
    """
    # --- 1. HYPERPARAMETER & CONFIG EXTRACTION ---
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]
    
    device = train_cfg["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    k_folds = train_cfg["k_folds"]
    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    val_patience = train_cfg["val_patience_epochs"]
    min_delta = train_cfg["val_min_delta"]
    lookback_offset = train_cfg["lookback_offset"]

    input_dim = plant_cfg["input_dim"]    # Control signal dimension (u)
    output_dim = plant_cfg["output_dim"]  # Measured output dimension (y)
    
    u_min = torch.tensor(plant_cfg["u_1_hard_min"], device=device, dtype=torch.float32)
    u_max = torch.tensor(plant_cfg["u_1_hard_max"], device=device, dtype=torch.float32)

    total_trajectories = Y_trajectories.shape[0]
    total_trajectory_len = Y_trajectories.shape[1]

    # Feature dimension formula matching model input_size: (1 target + 1 current + n_y past)*dim_y + n_u*dim_u
    feature_dim = (2 + n_y) * output_dim + n_u * input_dim

    # --- 2. K-FOLD CV SPLIT ---
    all_indices = np.arange(total_trajectories)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    initial_model_state = copy.deepcopy(model.state_dict())
    fold_histories = {}

    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING CLOSED-LOOP ONLINE FOLD {fold + 1} / {k_folds}")
        print(f"==========================================")

        model.load_state_dict(initial_model_state)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg["lr_decay_rate"])

        loss_name = train_cfg["loss_function"].replace("()", "")
        if loss_name == "NormalizedRMSELoss":
            criterion = NormalizedRMSELoss(reduction='none')
        else:
            criterion = torch.nn.MSELoss(reduction='none')

        # Instantiate Plant for this fold
        plant_instance = plant(hyperparam_config) if isinstance(plant, type) else plant

        # --- 3. FIT STANDARD SCALERS ON TRAINING SPLIT ---
        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)

        Y_train_flat = Y_trajectories[train_idx_arr].reshape(-1, output_dim)
        U_train_flat = U_trajectories[train_idx_arr].reshape(-1, input_dim)

        scaler_y = StandardScaler().fit(Y_train_flat)
        scaler_u = StandardScaler().fit(U_train_flat)

        fold_dir = f"{dirname}/fold_{fold+1}"
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")
        save_scaler_object(scaler_u, dirname=fold_dir, filename="scaler_u")

        # Scaler PyTorch tensors for differentiable graph operations
        scaler_y_mean_t = torch.tensor(scaler_y.mean_, device=device, dtype=torch.float32)
        scaler_y_std_t = torch.tensor(scaler_y.scale_, device=device, dtype=torch.float32)
        scaler_u_mean_t = torch.tensor(scaler_u.mean_, device=device, dtype=torch.float32)
        scaler_u_std_t = torch.tensor(scaler_u.scale_, device=device, dtype=torch.float32)

        # Tile scaler tensors for concatenated feature vector (1 target + 1 current + n_y past y's)
        y_next_y_hist_mean = scaler_y_mean_t.repeat(2 + n_y)
        y_next_y_hist_std = scaler_y_std_t.repeat(2 + n_y)

        u_hist_mean = scaler_u_mean_t.repeat(n_u)
        u_hist_std = scaler_u_std_t.repeat(n_u)

        best_val_loss = float('inf')
        patience_counter = 0
import copy
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def train_controller_closed_loop_online(
    model,
    Y_trajectories,
    U_trajectories,
    X_states,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_sim_with_plots=False,
):
    """
    Trains an inverse controller online in a CLOSED-LOOP setting.
    Parameters are updated at every time step using the instant MSE output loss.
    
    Gradient path per step:
      Loss(y_sim_{k+1}, y_ref_{k+1}) -> Plant Dynamics -> Denormalization (scaler_y) 
      -> Controller (model) -> Normalization (scaler_x)
    """
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]

    device = train_cfg["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    k_folds = train_cfg["k_folds"]

    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    val_patience = train_cfg["val_patience_epochs"]
    min_delta = train_cfg["val_min_delta"]
    lookback_offset = train_cfg.get("lookback_offset", max(n_y, n_u))

    input_dim = plant_cfg["input_dim"]    # Control u dimension
    output_dim = plant_cfg["output_dim"]  # Output y dimension

    # --- 1. GENERATE SLIDING WINDOW DATASET ---
    print(f"🔄 Slicing trajectories with sliding windows (n_y={n_y}, n_u={n_u})...")
    X_raw, Y_raw = create_inverse_controller_dataset(
        Y_trajectories=Y_trajectories,
        U_trajectories=U_trajectories,
        n_y=n_y,
        n_u=n_u,
    )
    print("X_raw shape:", X_raw.shape)
    print("Y_raw shape:", Y_raw.shape)

    total_sequences = X_raw.shape[0]
    total_trajectory_len = Y_trajectories.shape[1]
    
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    initial_model_state = copy.deepcopy(model.state_dict())
    fold_histories = {}

    # Initialize plant instance
    if isinstance(plant, type):
        plant_instance = plant(hyperparam_config)
    else:
        plant_instance = plant

    # --- K-FOLD CROSS VALIDATION LOOP ---
    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING ONLINE CLOSED-LOOP FOLD {fold + 1} / {k_folds}")
        print(f"==========================================")

        model.load_state_dict(initial_model_state)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=train_cfg.get("lr_decay_rate", 0.99)
        )

        loss_name = train_cfg["loss_function"].replace("()", "")
        if loss_name == "NormalizedRMSELoss":
            criterion = NormalizedRMSELoss(reduction="none")
        else:
            criterion = torch.nn.MSELoss(reduction="none")

        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)

        train_x_raw, val_x_raw = X_raw[train_idx_arr], X_raw[val_idx_arr]
        train_y_raw, val_y_raw = Y_raw[train_idx_arr], Y_raw[val_idx_arr]

        # --- INTERNAL FOLD STANDARD SCALING ---
        print(f"⚖️ Fitting independent StandardScalers for Fold {fold + 1}...")

        dim_x = train_x_raw.shape[-1]
        dim_y = train_y_raw.shape[-1]

        train_x_flat = train_x_raw.reshape(-1, dim_x)
        train_y_flat = train_y_raw.reshape(-1, dim_y)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaler_x.fit(train_x_flat)
        scaler_y.fit(train_y_flat)

        fold_dir = f"{dirname}/fold_{fold+1}"
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")

        mean_x_t = torch.tensor(scaler_x.mean_, dtype=torch.float32, device=device)
        scale_x_t = torch.tensor(scaler_x.scale_, dtype=torch.float32, device=device)
        mean_y_t = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device)
        scale_y_t = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)

        train_size = len(train_idx_arr)
        val_size = len(val_idx_arr)

        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        fold_val_epoch_history = []
        fold_train_epoch_history = []

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            epoch_train_loss_accum = 0.0
            shuffled_train_indices = np.random.permutation(train_size)

            print(f"\n🎬 Online Fold {fold+1} | Starting Epoch {epoch+1}/{epochs}")

            for i in range(0, train_size, mini_batch_size):
                batch_train_seq_idx = train_idx_arr[
                    shuffled_train_indices[i : i + mini_batch_size]
                ]
                current_batch_size = len(batch_train_seq_idx)

                target_y_batch = Y_trajectories[batch_train_seq_idx].to(device)
                target_u_batch = U_trajectories[batch_train_seq_idx].to(device)
                init_states_batch = X_states[batch_train_seq_idx, 0].to(device)

                sim_outputs = torch.zeros(
                    current_batch_size, total_trajectory_len, output_dim, device=device
                )
                sim_controls = torch.zeros(
                    current_batch_size, total_trajectory_len, input_dim, device=device
                )

                sim_outputs[:, 0] = target_y_batch[:, 0]
                sim_controls[:, 0] = target_u_batch[:, 0]
                current_sim_state = init_states_batch.clone()

                if hasattr(model, "reset_memory"):
                    model.reset_memory(batch_size=current_batch_size, device=device)

                batch_accumulated_seq_loss = 0.0
                active_steps_count = 0

                # --- ONLINE CLOSED-LOOP UNROLLING & STEP-BY-STEP GRADIENT UPDATE ---
                for k in range(total_trajectory_len - 1):
                    if k < lookback_offset:
                        # Warm-up phase: ground-truth actions to build historical context
                        u_action = target_u_batch[:, k]
                        current_sim_state, y_next_sim = plant_instance.step(
                            state=current_sim_state, u=u_action, t=k * dt, dt=dt
                        )
                        sim_controls[:, k + 1] = u_action
                        sim_outputs[:, k + 1] = y_next_sim
                    else:
                        optimizer.zero_grad()

                        # 1. Construct raw feature vector
                        y_next_ref = target_y_batch[:, k + 1]
                        
                        y_hist = sim_outputs[:, k - n_y : k + 1]
                        y_hist_rev = torch.flip(y_hist, dims=[1]).reshape(current_batch_size, -1)
                        
                        u_hist = sim_controls[:, k - n_u : k]
                        u_hist_rev = torch.flip(u_hist, dims=[1]).reshape(current_batch_size, -1)

                        v_k_raw = torch.cat([y_next_ref, y_hist_rev, u_hist_rev], dim=-1)

                        # 2. Normalize input features
                        v_k_scaled = (v_k_raw - mean_x_t) / scale_x_t
                        v_k_input = v_k_scaled.unsqueeze(1)

                        # 3. Forecast control u
                        u_pred_norm = model(v_k_input).squeeze(1)
                        u_pred_raw = u_pred_norm * scale_y_t + mean_y_t

                        # 4. Saturation limits
                        u_action = torch.clamp(
                            u_pred_raw,
                            plant_cfg["u_1_hard_min"],
                            plant_cfg["u_1_hard_max"],
                        )

                        # 5. Forward step through plant dynamics
                        current_sim_state, y_next_sim = plant_instance.step(
                            state=current_sim_state, u=u_action, t=k * dt, dt=dt
                        )

                        # 6. Immediate single-step loss computation
                        step_loss_raw = criterion(y_next_sim, y_next_ref)
                        step_loss = step_loss_raw.mean()

                        # 7. ONLINE GRADIENT UPDATE STEP
                        step_loss.backward()
                        optimizer.step()

                        # 8. Accumulate loss for epoch statistics
                        batch_accumulated_seq_loss += step_loss.item()
                        active_steps_count += 1

                        # 9. Store and detach variables for truncated BPTT at step k
                        current_sim_state = current_sim_state.detach()
                        sim_controls[:, k + 1] = u_action.detach()
                        sim_outputs[:, k + 1] = y_next_sim.detach()

                avg_seq_loss = batch_accumulated_seq_loss / max(1, active_steps_count)
                epoch_train_loss_accum += avg_seq_loss * current_batch_size
                fold_train_batch_loss.append(avg_seq_loss)
                fold_train_batch_indices.append(global_batch_counter)
                global_batch_counter += 1

            scheduler.step()

            # --- VALIDATION PASS (INFERENCE / EVALUATION MODE) ---
            model.eval()
            epoch_val_loss_accum = 0.0

            with torch.no_grad():
                for i in range(0, val_size, mini_batch_size):
                    batch_val_seq_idx = val_idx_arr[i : i + mini_batch_size]
                    current_val_batch_size = len(batch_val_seq_idx)

                    target_y_val = Y_trajectories[batch_val_seq_idx].to(device)
                    target_u_val = U_trajectories[batch_val_seq_idx].to(device)
                    init_states_val = X_states[batch_val_seq_idx, 0].to(device)

                    sim_outputs_val = torch.zeros(
                        current_val_batch_size, total_trajectory_len, output_dim, device=device
                    )
                    sim_controls_val = torch.zeros(
                        current_val_batch_size, total_trajectory_len, input_dim, device=device
                    )

                    sim_outputs_val[:, 0] = target_y_val[:, 0]
                    sim_controls_val[:, 0] = target_u_val[:, 0]
                    current_sim_state_val = init_states_val.clone()

                    if hasattr(model, "reset_memory"):
                        model.reset_memory(batch_size=current_val_batch_size, device=device)

                    for k in range(total_trajectory_len - 1):
                        if k < lookback_offset:
                            u_action = target_u_val[:, k]
                            current_sim_state_val, y_next_sim = plant_instance.step(
                                state=current_sim_state_val, u=u_action, t=k * dt, dt=dt
                            )
                            sim_controls_val[:, k + 1] = u_action
                            sim_outputs_val[:, k + 1] = y_next_sim
                        else:
                            y_next_ref = target_y_val[:, k + 1]
                            y_hist = sim_outputs_val[:, k - n_y : k + 1]
                            y_hist_rev = torch.flip(y_hist, dims=[1]).reshape(current_val_batch_size, -1)
                            u_hist = sim_controls_val[:, k - n_u : k]
                            u_hist_rev = torch.flip(u_hist, dims=[1]).reshape(current_val_batch_size, -1)

                            v_k_raw = torch.cat([y_next_ref, y_hist_rev, u_hist_rev], dim=-1)

                            v_k_scaled = (v_k_raw - mean_x_t) / scale_x_t
                            v_k_input = v_k_scaled.unsqueeze(1)

                            u_pred_norm = model(v_k_input).squeeze(1)
                            u_pred_raw = u_pred_norm * scale_y_t + mean_y_t

                            u_action = torch.clamp(
                                u_pred_raw,
                                plant_cfg["u_1_hard_min"],
                                plant_cfg["u_1_hard_max"],
                            )

                            current_sim_state_val, y_next_sim = plant_instance.step(
                                state=current_sim_state_val, u=u_action, t=k * dt, dt=dt
                            )
                            sim_controls_val[:, k + 1] = u_action
                            sim_outputs_val[:, k + 1] = y_next_sim

                    val_raw_loss = criterion(
                        sim_outputs_val[:, lookback_offset:],
                        target_y_val[:, lookback_offset:],
                    )
                    epoch_val_loss_accum += val_raw_loss.mean().item() * current_val_batch_size

            mean_train_loss = epoch_train_loss_accum / train_size
            mean_val_loss = (epoch_val_loss_accum / val_size) if val_size > 0 else 0.0

            fold_val_epoch_history.append(mean_val_loss)
            fold_train_epoch_history.append(mean_train_loss)

            if fold not in fold_histories:
                fold_histories[fold] = {"train_loss": [], "val_loss": [], "val_epochs": []}

            fold_histories[fold]["train_loss"].append(mean_train_loss)
            fold_histories[fold]["val_loss"].append(mean_val_loss)
            fold_histories[fold]["val_epochs"].append(epoch + 1)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"✨ [Fold {fold+1}] Epoch {epoch+1} Summary:")
            print(f"   ↳ LR: {current_lr:.6e} | Online Train Loss (MSE_y): {mean_train_loss:.6f} | Closed-Loop Val Loss: {mean_val_loss:.6f}")

            if mean_val_loss < (best_val_loss - min_delta):
                best_val_loss = mean_val_loss
                patience_counter = 0
                save_model(model, dirname=fold_dir, hyperparam_config=hyperparam_config, filename="best_fold_model")
            else:
                patience_counter += 1
                if patience_counter >= val_patience:
                    print(f"🛑 Early stopping fold {fold+1} at Epoch {epoch+1}.")
                    break

        # --- PLOT LOSS CURVES ---
        epoch_axis = np.array(fold_histories[fold]["val_epochs"])

        plot_signals(
            t=np.array(fold_train_batch_indices),
            signals=[np.array(fold_train_batch_loss)],
            labels=[f"Fold {fold+1} Total Loss"],
            xlabel="Optimization Steps",
            ylabel="Loss",
            dirname=fold_dir,
            filename="granular_training_loss"
        )

        plot_signals(
            t=epoch_axis,
            signals=[np.array(fold_train_epoch_history), np.array(fold_val_epoch_history)],
            labels=["Avg Train Loss", "Avg Val Loss"],
            xlabel="Epochs",
            ylabel="Loss",
            dirname=fold_dir,
            filename="epoch_validation_loss"
        )

    # --- AGGREGATE RESULTS ACROSS ALL FOLDS ---
    fold_best_train_losses = []
    fold_best_val_losses = []

    for f in range(k_folds):
        best_val_epoch_idx = np.argmin(fold_histories[f]["val_loss"])
        fold_best_val_losses.append(fold_histories[f]["val_loss"][best_val_epoch_idx])
        fold_best_train_losses.append(fold_histories[f]["train_loss"][best_val_epoch_idx])

    avg_best_train_loss = float(np.mean(fold_best_train_losses))
    avg_best_val_loss = float(np.mean(fold_best_val_losses))

    summary_metrics = {
        "Metric": [
            "Avg Best Online Closed-Loop Train Loss (MSE_y)",
            "Avg Best Closed-Loop Val Loss (MSE_y)",
        ],
        "Value": [
            avg_best_train_loss,
            avg_best_val_loss,
        ],
    }

    summary_results_df = pd.DataFrame(summary_metrics)

    print("\n==========================================")
    print("📊 ONLINE CLOSED-LOOP K-FOLD CROSS-VALIDATION SUMMARY")
    print("==========================================")
    print(summary_results_df.to_string(index=False))
    print("==========================================\n")

    save_df_to_csv(
        summary_results_df,
        dirname=dirname,
        filename="k_fold_cross_validation_summary"
    )

    return (
        fold_histories,
        {
            "train_loss": avg_best_train_loss,
            "val_loss": avg_best_val_loss,
            "closed_loop_mse": avg_best_val_loss,
        },
        avg_best_val_loss,
        summary_results_df,
    )
    
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

def train_controller_closed_loop_offline(
    model,
    train_data,
    test_data,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_simulation=True,
    run_sim_with_plots=False,
):
    """Entraîne un contrôleur en boucle fermée hors-ligne (Offline Closed-Loop Training).

    Le contrôleur interagit étape par étape avec la plante (TrophophasePlant)
    sur l'ensemble de la séquence temporelle. La perte (loss) est calculée entre
    les commandes prédites et les commandes cibles sur toute la trajectoire.
    """
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    device = train_cfg["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]

    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    patience = train_cfg.get("val_patience_epochs", 10)
    min_delta = train_cfg.get("val_min_delta", 1e-4)

    plant_cfg = hyperparam_config["plant"]
    input_dim = plant_cfg["input_dim"]
    output_dim = plant_cfg["output_dim"]

    # Helper function to extract dataset split
    def _prepare_split(split_data, split_name):
        arg1, arg2 = split_data
        print(
            f"🔄 Slicing {split_name} trajectories with sliding windows"
            f" (n_y={n_y}, n_u={n_u})..."
        )
        return create_inverse_controller_dataset(
            Y_trajectories=arg1, U_trajectories=arg2, n_y=n_y, n_u=n_u
        )

    # --- 1. PREPARE SPLITS ---
    train_x_raw, train_y_raw = _prepare_split(train_data, "train")
    test_x_raw, test_y_raw = _prepare_split(test_data, "test")

    print(
        f"📊 Dataset shapes -> Train: {train_x_raw.shape} | Test:"
        f" {test_x_raw.shape}"
    )

    # --- 2. FIT SCALERS ON TRAIN DATA ONLY ---
    print("⚖️ Fitting StandardScalers on Training Data...")
    N_train, seq_len_train, dim_x = train_x_raw.shape
    dim_y = train_y_raw.shape[-1]

    train_x_flat = train_x_raw.reshape(-1, dim_x)
    train_y_flat = train_y_raw.reshape(-1, dim_y)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(train_x_flat)
    scaler_y.fit(train_y_flat)

    # Transform Train
    train_x = torch.tensor(
        scaler_x.transform(train_x_flat).reshape(
            N_train, seq_len_train, dim_x
        ),
        dtype=torch.float32,
    )
    train_y = torch.tensor(
        scaler_y.transform(train_y_flat).reshape(
            N_train, seq_len_train, dim_y
        ),
        dtype=torch.float32,
    )

    # Transform Test
    N_test, seq_len_test, _ = test_x_raw.shape
    test_x_flat = test_x_raw.reshape(-1, dim_x)
    test_y_flat = test_y_raw.reshape(-1, dim_y)
    test_x = torch.tensor(
        scaler_x.transform(test_x_flat).reshape(N_test, seq_len_test, dim_x),
        dtype=torch.float32,
    )
    test_y = torch.tensor(
        scaler_y.transform(test_y_flat).reshape(N_test, seq_len_test, dim_y),
        dtype=torch.float32,
    )

    # Save Config and Scalers
    save_to_json(hyperparam_config, dirname, "hyperparam_config")
    save_scaler_object(scaler_x, dirname=dirname, filename="scaler_x")
    save_scaler_object(scaler_y, dirname=dirname, filename="scaler_y")

    # --- SETUP MODEL, OPTIMIZER, & LOSS ---
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=train_cfg["lr_decay_rate"]
    )

    loss_name = train_cfg["loss_function"].replace("()", "")
    if loss_name == "NormalizedRMSELoss":
        criterion = NormalizedRMSELoss(reduction="none")
    elif loss_name == "MSELoss":
        criterion = torch.nn.MSELoss(reduction="none")

    train_epoch_history = []
    test_epoch_history = []
    train_batch_loss_history = []

    best_test_loss = float("inf")
    patience_counter = 0

    # --- HELPER FUNCTION FOR CLOSED-LOOP BATCH RUN ---
    def _run_closed_loop_batch(batch_x, batch_y_target, is_training=True):
        """Simule la boucle fermée pas à pas et calcule la perte sur la séquence."""
        batch_size, seq_len, _ = batch_x.shape

        if hasattr(model, "reset_memory"):
            model.reset_memory(batch_size=batch_size, device=device)

        # Initialisation de l'état de la plante [batch_size, 2] -> [x1, x2]
        state = plant.get_initial_state(
            batch_size=batch_size, randomize=is_training
        )

        u_preds = []

        # Boucle temporelle étape par étape
        for t_step in range(seq_len):
            current_time = t_step * dt
            x_t = batch_x[
                :, t_step : t_step + 1, :
            ]  # Entrée au temps t: [batch_size, 1, dim_x]

            # 1. Le contrôleur prédit u_t: [batch_size, 1, dim_u]
            u_t = model(x_t)
            u_preds.append(u_t)

            # Formatage de u_t pour la plante: [batch_size, dim_u]
            u_t_step = u_t.squeeze(1) if u_t.dim() == 3 else u_t

            # 2. Intégration RK45 de la plante: step(state, u, t, dt)
            state, _ = plant.step(
                state=state, u=u_t_step, t=current_time, dt=dt
            )

        # Reconstitution de la trajectoire de commande prédite [batch_size, seq_len, dim_u]
        u_pred_seq = torch.cat(u_preds, dim=1)

        # Calcul de la perte globale sur la séquence
        raw_loss = criterion(u_pred_seq, batch_y_target)
        return raw_loss.mean()

    # --- 3. TRAINING LOOP ---
    for epoch in range(epochs):
        model.train()
        epoch_train_loss_accum = 0.0

        print(f"\n🎬 Starting Epoch {epoch+1}/{epochs}")
        shuffled_train_indices = torch.randperm(N_train)

        # Batch Training Pass
        for i in range(0, N_train, mini_batch_size):
            batch_indices = shuffled_train_indices[i : i + mini_batch_size]
            current_batch_size = len(batch_indices)

            batch_x = train_x[batch_indices].to(device)
            batch_y = train_y[batch_indices].to(device)

            optimizer.zero_grad()

            loss = _run_closed_loop_batch(batch_x, batch_y, is_training=True)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            epoch_train_loss_accum += current_loss_val * current_batch_size
            train_batch_loss_history.append(current_loss_val)

        scheduler.step()
        mean_train_loss = epoch_train_loss_accum / N_train
        train_epoch_history.append(mean_train_loss)

        # --- TEST PASS EVERY EPOCH ---
        model.eval()
        epoch_test_loss_accum = 0.0

        with torch.no_grad():
            for i in range(0, N_test, mini_batch_size):
                batch_test_x = test_x[i : i + mini_batch_size].to(device)
                batch_test_y = test_y[i : i + mini_batch_size].to(device)
                current_test_batch_size = len(batch_test_x)

                test_loss = _run_closed_loop_batch(
                    batch_test_x, batch_test_y, is_training=False
                )
                epoch_test_loss_accum += (
                    test_loss.item() * current_test_batch_size
                )

        mean_test_loss = (
            epoch_test_loss_accum / N_test if N_test > 0 else 0.0
        )
        test_epoch_history.append(mean_test_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"✨ Epoch {epoch+1} Summary | LR: {current_lr:.6e} | Train Loss:"
            f" {mean_train_loss:.6f} | Test Loss: {mean_test_loss:.6f}"
        )

        # Checkpoint / Early Stopping
        if mean_test_loss < (best_test_loss - min_delta):
            best_test_loss = mean_test_loss
            patience_counter = 0
            save_model(
                model,
                dirname=dirname,
                hyperparam_config=hyperparam_config,
                filename="best_model",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered at Epoch {epoch+1}.")
                break

    # --- 4. LOSS PLOTS ---
    plot_signals(
        t=np.arange(len(train_batch_loss_history)),
        signals=[np.array(train_batch_loss_history)],
        labels=["Batch Training Loss"],
        xlabel="Optimization Steps",
        ylabel="Loss",
        dirname=dirname,
        filename="granular_training_loss",
    )

    plot_signals(
        t=np.arange(1, len(train_epoch_history) + 1),
        signals=[np.array(train_epoch_history), np.array(test_epoch_history)],
        labels=["Avg Train Loss", "Avg Test Loss"],
        xlabel="Epochs",
        ylabel="Loss",
        dirname=dirname,
        filename="epoch_loss_curves",
    )

    # --- SUMMARY METRICS ---
    summary_metrics = {
        "Metric": ["Best Training Loss", "Best Testing Loss"],
        "Value": [min(train_epoch_history), best_test_loss],
    }
    summary_df = pd.DataFrame(summary_metrics)

    print("\n==========================================")
    print("📊 TRAINING SUMMARY")
    print("==========================================")
    print(summary_df.to_string(index=False))
    print("==========================================\n")

    save_df_to_csv(summary_df, dirname=dirname, filename="training_summary")

    history = {
        "train_loss": train_epoch_history,
        "test_loss": test_epoch_history,
    }

    return model, history, summary_df

def train_controller_closed_loop_offline_paola(
    model,
    Y_trajectories,
    U_trajectories,
    X_states,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_sim_with_plots=False,
):
    """
    Trains an inverse controller offline in a CLOSED-LOOP end-to-end setting
    using the sliding-window dataset structure and scaler_x / scaler_y workflow.
    
    Gradient path:
      Loss(y_sim, y_target) -> Plant Dynamics -> Denormalization (scaler_y) 
      -> Controller (model) -> Normalization (scaler_x)
    """
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]

    device = train_cfg["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]
    k_folds = train_cfg["k_folds"]

    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    val_patience = train_cfg["val_patience_epochs"]
    min_delta = train_cfg["val_min_delta"]
    lookback_offset = train_cfg.get("lookback_offset", max(n_y, n_u))

    input_dim = plant_cfg["input_dim"]    # Control u dimension
    output_dim = plant_cfg["output_dim"]  # Output y dimension

    # --- 1. GENERATE SLIDING WINDOW DATASET (Matching SANEM setup) ---
    print(f"🔄 Slicing trajectories with sliding windows (n_y={n_y}, n_u={n_u})...")
    X_raw, Y_raw = create_inverse_controller_dataset(
        Y_trajectories=Y_trajectories,
        U_trajectories=U_trajectories,
        n_y=n_y,
        n_u=n_u,
    )
    print("X_raw shape:", X_raw.shape)
    print("Y_raw shape:", Y_raw.shape)

    total_sequences = X_raw.shape[0]
    total_trajectory_len = Y_trajectories.shape[1]
    
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    initial_model_state = copy.deepcopy(model.state_dict())
    fold_histories = {}

    # Initialize plant instance
    if isinstance(plant, type):
        plant_instance = plant(hyperparam_config)
    else:
        plant_instance = plant

    # --- K-FOLD CROSS VALIDATION LOOP ---
    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING CLOSED-LOOP FOLD {fold + 1} / {k_folds}")
        print(f"==========================================")

        model.load_state_dict(initial_model_state)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=train_cfg.get("lr_decay_rate", 0.99)
        )

        loss_name = train_cfg["loss_function"].replace("()", "")
        if loss_name == "NormalizedRMSELoss":
            criterion = NormalizedRMSELoss(reduction="none")
        else:
            criterion = torch.nn.MSELoss(reduction="none")

        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)

        train_x_raw, val_x_raw = X_raw[train_idx_arr], X_raw[val_idx_arr]
        train_y_raw, val_y_raw = Y_raw[train_idx_arr], Y_raw[val_idx_arr]

        # --- INTERNAL FOLD STANDARD SCALING ---
        print(f"⚖️ Fitting independent StandardScalers for Fold {fold + 1}...")

        N_train, win_len, dim_x = train_x_raw.shape
        dim_y = train_y_raw.shape[-1]  # Target control u dimension

        train_x_flat = train_x_raw.reshape(-1, dim_x)
        train_y_flat = train_y_raw.reshape(-1, dim_y)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaler_x.fit(train_x_flat)
        scaler_y.fit(train_y_flat)

        fold_dir = f"{dirname}/fold_{fold+1}"
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")

        # PyTorch Tensors for differentiable scaling during closed-loop unrolling
        mean_x_t = torch.tensor(scaler_x.mean_, dtype=torch.float32, device=device)
        scale_x_t = torch.tensor(scaler_x.scale_, dtype=torch.float32, device=device)
        mean_y_t = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device)
        scale_y_t = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)

        train_size = len(train_idx_arr)
        val_size = len(val_idx_arr)

        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        fold_val_epoch_history = []
        fold_train_epoch_history = []

        best_val_loss = float("inf")
        patience_counter = 0

        # --- TRAINING LOOP ---
        for epoch in range(epochs):
            model.train()
            epoch_train_loss_accum = 0.0
            shuffled_train_indices = np.random.permutation(train_size)

            print(f"\n🎬 Closed-Loop Fold {fold+1} | Starting Epoch {epoch+1}/{epochs}")

            for i in range(0, train_size, mini_batch_size):
                batch_train_seq_idx = train_idx_arr[
                    shuffled_train_indices[i : i + mini_batch_size]
                ]
                current_batch_size = len(batch_train_seq_idx)

                # Reference trajectories and initial physical state
                target_y_batch = Y_trajectories[batch_train_seq_idx].to(device)
                target_u_batch = U_trajectories[batch_train_seq_idx].to(device)
                init_states_batch = X_states[batch_train_seq_idx, 0].to(device)

                sim_outputs = torch.zeros(
                    current_batch_size, total_trajectory_len, output_dim, device=device
                )
                sim_controls = torch.zeros(
                    current_batch_size, total_trajectory_len, input_dim, device=device
                )

                sim_outputs[:, 0] = target_y_batch[:, 0]
                sim_controls[:, 0] = target_u_batch[:, 0]
                current_sim_state = init_states_batch.clone()

                if hasattr(model, "reset_memory"):
                    model.reset_memory(batch_size=current_batch_size, device=device)

                optimizer.zero_grad()

                # --- CLOSED-LOOP UNROLLING OVER TIME ---
                for k in range(total_trajectory_len - 1):
                    if k < lookback_offset:
                        # Warm-up phase: feed ground truth actions to establish history
                        u_action = target_u_batch[:, k]
                        current_sim_state, y_next_sim = plant_instance.step(
                            state=current_sim_state, u=u_action, t=k * dt, dt=dt
                        )
                        sim_controls[:, k + 1] = u_action
                        sim_outputs[:, k + 1] = y_next_sim
                    else:
                        # Construct raw feature vector: [y_{k+1}^{ref}, y_k, ..., u_{k-1}, ...]
                        y_next_ref = target_y_batch[:, k + 1]
                        
                        y_hist = sim_outputs[:, k - n_y : k + 1]
                        y_hist_rev = torch.flip(y_hist, dims=[1]).reshape(current_batch_size, -1)
                        
                        u_hist = sim_controls[:, k - n_u : k]
                        u_hist_rev = torch.flip(u_hist, dims=[1]).reshape(current_batch_size, -1)

                        v_k_raw = torch.cat([y_next_ref, y_hist_rev, u_hist_rev], dim=-1)

                        # Normalize features using scaler_x
                        v_k_scaled = (v_k_raw - mean_x_t) / scale_x_t
                        v_k_input = v_k_scaled.unsqueeze(1)

                        # Predict normalized control action
                        u_pred_norm = model(v_k_input).squeeze(1)

                        # Denormalize predicted control action using scaler_y
                        u_pred_raw = u_pred_norm * scale_y_t + mean_y_t

                        # Physical boundary saturation
                        u_action = torch.clamp(
                            u_pred_raw,
                            plant_cfg["u_1_hard_min"],
                            plant_cfg["u_1_hard_max"],
                        )

                        # Step differentiable plant
                        current_sim_state, y_next_sim = plant_instance.step(
                            state=current_sim_state, u=u_action, t=k * dt, dt=dt
                        )

                        sim_controls[:, k + 1] = u_action
                        sim_outputs[:, k + 1] = y_next_sim

                # --- COMPUTE CLOSED-LOOP OUTPUT LOSS ---
                raw_loss = criterion(
                    sim_outputs[:, lookback_offset:],
                    target_y_batch[:, lookback_offset:],
                )
                loss = raw_loss.mean()

                loss.backward()
                optimizer.step()

                current_loss_val = loss.item()
                epoch_train_loss_accum += current_loss_val * current_batch_size
                fold_train_batch_loss.append(current_loss_val)
                fold_train_batch_indices.append(global_batch_counter)
                global_batch_counter += 1

            scheduler.step()

            # --- VALIDATION PASS ---
            model.eval()
            epoch_val_loss_accum = 0.0

            with torch.no_grad():
                for i in range(0, val_size, mini_batch_size):
                    batch_val_seq_idx = val_idx_arr[i : i + mini_batch_size]
                    current_val_batch_size = len(batch_val_seq_idx)

                    target_y_val = Y_trajectories[batch_val_seq_idx].to(device)
                    target_u_val = U_trajectories[batch_val_seq_idx].to(device)
                    init_states_val = X_states[batch_val_seq_idx, 0].to(device)

                    sim_outputs_val = torch.zeros(
                        current_val_batch_size, total_trajectory_len, output_dim, device=device
                    )
                    sim_controls_val = torch.zeros(
                        current_val_batch_size, total_trajectory_len, input_dim, device=device
                    )

                    sim_outputs_val[:, 0] = target_y_val[:, 0]
                    sim_controls_val[:, 0] = target_u_val[:, 0]
                    current_sim_state_val = init_states_val.clone()

                    if hasattr(model, "reset_memory"):
                        model.reset_memory(batch_size=current_val_batch_size, device=device)

                    for k in range(total_trajectory_len - 1):
                        if k < lookback_offset:
                            u_action = target_u_val[:, k]
                            current_sim_state_val, y_next_sim = plant_instance.step(
                                state=current_sim_state_val, u=u_action, t=k * dt, dt=dt
                            )
                            sim_controls_val[:, k + 1] = u_action
                            sim_outputs_val[:, k + 1] = y_next_sim
                        else:
                            y_next_ref = target_y_val[:, k + 1]
                            y_hist = sim_outputs_val[:, k - n_y : k + 1]
                            y_hist_rev = torch.flip(y_hist, dims=[1]).reshape(current_val_batch_size, -1)
                            u_hist = sim_controls_val[:, k - n_u : k]
                            u_hist_rev = torch.flip(u_hist, dims=[1]).reshape(current_val_batch_size, -1)

                            v_k_raw = torch.cat([y_next_ref, y_hist_rev, u_hist_rev], dim=-1)

                            v_k_scaled = (v_k_raw - mean_x_t) / scale_x_t
                            v_k_input = v_k_scaled.unsqueeze(1)

                            u_pred_norm = model(v_k_input).squeeze(1)
                            u_pred_raw = u_pred_norm * scale_y_t + mean_y_t

                            u_action = torch.clamp(
                                u_pred_raw,
                                plant_cfg["u_1_hard_min"],
                                plant_cfg["u_1_hard_max"],
                            )

                            current_sim_state_val, y_next_sim = plant_instance.step(
                                state=current_sim_state_val, u=u_action, t=k * dt, dt=dt
                            )
                            sim_controls_val[:, k + 1] = u_action
                            sim_outputs_val[:, k + 1] = y_next_sim

                    val_raw_loss = criterion(
                        sim_outputs_val[:, lookback_offset:],
                        target_y_val[:, lookback_offset:],
                    )
                    epoch_val_loss_accum += val_raw_loss.mean().item() * current_val_batch_size

            mean_train_loss = epoch_train_loss_accum / train_size
            mean_val_loss = (epoch_val_loss_accum / val_size) if val_size > 0 else 0.0

            fold_val_epoch_history.append(mean_val_loss)
            fold_train_epoch_history.append(mean_train_loss)

            if fold not in fold_histories:
                fold_histories[fold] = {"train_loss": [], "val_loss": [], "val_epochs": []}

            fold_histories[fold]["train_loss"].append(mean_train_loss)
            fold_histories[fold]["val_loss"].append(mean_val_loss)
            fold_histories[fold]["val_epochs"].append(epoch + 1)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"✨ [Fold {fold+1}] Epoch {epoch+1} Summary:")
            print(f"   ↳ LR: {current_lr:.6e} | Closed-Loop Train Loss (MSE_y): {mean_train_loss:.6f} | Closed-Loop Val Loss: {mean_val_loss:.6f}")

            if mean_val_loss < (best_val_loss - min_delta):
                best_val_loss = mean_val_loss
                patience_counter = 0
                save_model(model, dirname=fold_dir, hyperparam_config=hyperparam_config, filename="best_fold_model")
            else:
                patience_counter += 1
                if patience_counter >= val_patience:
                    print(f"🛑 Early stopping fold {fold+1} at Epoch {epoch+1}.")
                    break

        # --- PLOT LOSS CURVES ---
        epoch_axis = np.array(fold_histories[fold]["val_epochs"])

        plot_signals(
            t=np.array(fold_train_batch_indices),
            signals=[np.array(fold_train_batch_loss)],
            labels=[f"Fold {fold+1} Total Loss"],
            xlabel="Optimization Steps",
            ylabel="Loss",
            dirname=fold_dir,
            filename="granular_training_loss"
        )

        plot_signals(
            t=epoch_axis,
            signals=[np.array(fold_train_epoch_history), np.array(fold_val_epoch_history)],
            labels=["Avg Train Loss", "Avg Val Loss"],
            xlabel="Epochs",
            ylabel="Loss",
            dirname=fold_dir,
            filename="epoch_validation_loss"
        )

    # --- AGGREGATE RESULTS ACROSS ALL FOLDS ---
    fold_best_train_losses = []
    fold_best_val_losses = []

    for f in range(k_folds):
        best_val_epoch_idx = np.argmin(fold_histories[f]["val_loss"])
        fold_best_val_losses.append(fold_histories[f]["val_loss"][best_val_epoch_idx])
        fold_best_train_losses.append(fold_histories[f]["train_loss"][best_val_epoch_idx])

    avg_best_train_loss = float(np.mean(fold_best_train_losses))
    avg_best_val_loss = float(np.mean(fold_best_val_losses))

    summary_metrics = {
        "Metric": [
            "Avg Best Closed-Loop Train Loss (MSE_y)",
            "Avg Best Closed-Loop Val Loss (MSE_y)",
        ],
        "Value": [
            avg_best_train_loss,
            avg_best_val_loss,
        ],
    }

    summary_results_df = pd.DataFrame(summary_metrics)

    print("\n==========================================")
    print("📊 CLOSED-LOOP K-FOLD CROSS-VALIDATION SUMMARY")
    print("==========================================")
    print(summary_results_df.to_string(index=False))
    print("==========================================\n")

    save_df_to_csv(
        summary_results_df,
        dirname=dirname,
        filename="k_fold_cross_validation_summary"
    )

    return (
        fold_histories,
        {
            "train_loss": avg_best_train_loss,
            "val_loss": avg_best_val_loss,
            "closed_loop_mse": avg_best_val_loss,
        },
        avg_best_val_loss,
        summary_results_df,
    )

def train_full_dataset(model, Y_trajectories, U_trajectories, hyperparam_config, dirname="./final_model"):
    train_cfg = hyperparam_config["train"]
    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    batch_size = train_cfg["mini_batch_size"]
    
    plant_cfg = hyperparam_config["plant"]
    input_dim = plant_cfg["input_dim"]
    output_dim = plant_cfg["output_dim"]
    
    os.makedirs(dirname, exist_ok=True)
    
    print("\n==========================================")
    print("🚀 RETRAINING FINAL MODEL ON FULL DATASET")
    print("==========================================")
    
    # 1. Slicing full dataset
    X_raw, Y_raw = create_inverse_controller_dataset(
        Y_trajectories=Y_trajectories,
        U_trajectories=U_trajectories,
        n_y=n_y,
        n_u=n_u
    )
    
    N_total, seq_len, dim_x = X_raw.shape
    dim_y = Y_raw.shape[-1]
    
    # 2. Fit global scalers on entire dataset
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_flat = X_raw.reshape(-1, dim_x)
    Y_flat = Y_raw.reshape(-1, dim_y)
    
    scaler_x.fit(X_flat)
    scaler_y.fit(Y_flat)
    
    train_x = torch.tensor(scaler_x.transform(X_flat).reshape(N_total, seq_len, dim_x), dtype=torch.float32)
    train_y = torch.tensor(scaler_y.transform(Y_flat).reshape(N_total, seq_len, dim_y), dtype=torch.float32)
    
    # Save production scalers
    save_scaler_object(scaler_x, dirname=dirname, filename="final_scaler_x")
    save_scaler_object(scaler_y, dirname=dirname, filename="final_scaler_y")
    save_to_json(hyperparam_config, dirname, "best_hyperparam_config")
    
    # 3. Setup training components
    device = "cuda"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg["lr_decay_rate"])
    
    loss_name = train_cfg["loss_function"].replace("()", "")
    criterion = NormalizedRMSELoss(reduction='none') if loss_name == "NormalizedRMSELoss" else getattr(nn, loss_name)(reduction='none')
    
    # 4. Optimization loop across whole dataset
    for epoch in range(epochs):
        model.train()
        epoch_loss_accum = 0.0
        shuffled_indices = torch.randperm(N_total)
        
        for i in range(0, N_total, batch_size):
            batch_indices = shuffled_indices[i : i + batch_size]
            current_bs = len(batch_indices)
            
            batch_x = train_x[batch_indices].to(device)
            batch_y = train_y[batch_indices].to(device)
            
            if hasattr(model, 'reset_memory'):
                model.reset_memory(batch_size=current_bs, device=device)
                
            optimizer.zero_grad()
            u_pred = model(batch_x)
            loss = criterion(u_pred, batch_y).mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss_accum += loss.item() * current_bs
            
        scheduler.step()
        mean_epoch_loss = epoch_loss_accum / N_total
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"🔥 [Full Retrain] Epoch {epoch+1}/{epochs} | Loss: {mean_epoch_loss:.6f}")
            
    # Save fully retrained model
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="final_retrained_model")
    print(f"✅ Production Model & Scalers saved successfully in: {dirname}")
    return model


def save_esn_parameters_to_csv(model, fold_dir):
    """
    Helper function to safely extract and save ESN weight matrices 
    (both trained readout weights and untrained reservoir weights) to CSV.
    """
    print(f"💾 Exporting ESN weight matrices to CSV in: {fold_dir}")
    
    # 1. Handle ReservoirPy Model or Node structure
    if hasattr(model, "nodes"):
        for node in model.nodes:
            node_name = node.name.lower()
            # Untrained Reservoir Parameters
            if "reservoir" in node_name:
                if hasattr(node, "Win") and node.Win is not None:
                    pd.DataFrame(node.Win).to_csv(os.path.join(fold_dir, "esn_W_in_untrained.csv"), index=False, header=False)
                if hasattr(node, "W") and node.W is not None:
                    pd.DataFrame(node.W).to_csv(os.path.join(fold_dir, "esn_W_reservoir_untrained.csv"), index=False, header=False)
                if hasattr(node, "bias") and node.bias is not None:
                    pd.DataFrame(node.bias).to_csv(os.path.join(fold_dir, "esn_bias_untrained.csv"), index=False, header=False)
            
            # Trained Readout Parameters
            elif "ridge" in node_name or "readout" in node_name:
                # ReservoirPy stores readout weights either in 'Wout' or 'W' depending on version/configuration
                w_out = getattr(node, "Wout", getattr(node, "W", None))
                if w_out is not None:
                    pd.DataFrame(w_out).to_csv(os.path.join(fold_dir, "esn_W_out_trained.csv"), index=False, header=False)
                if hasattr(node, "bias") and node.bias is not None:
                    pd.DataFrame(node.bias).to_csv(os.path.join(fold_dir, "esn_readout_bias_trained.csv"), index=False, header=False)

    # 2. Fallback for custom or flat objects (e.g., model.W_in, model.W, model.W_out)
    else:
        # Untrained parameters
        for attr_name, file_name in [("W_in", "esn_W_in_untrained.csv"), 
                                     ("Win", "esn_W_in_untrained.csv"),
                                     ("W", "esn_W_reservoir_untrained.csv"), 
                                     ("W_res", "esn_W_reservoir_untrained.csv"),
                                     ("bias", "esn_bias_untrained.csv")]:
            if hasattr(model, attr_name):
                weights = getattr(model, attr_name)
                if weights is not None:
                    pd.DataFrame(np.asarray(weights)).to_csv(os.path.join(fold_dir, file_name), index=False, header=False)
        
        # Trained parameters
        for attr_name, file_name in [("W_out", "esn_W_out_trained.csv"), 
                                     ("Wout", "esn_W_out_trained.csv")]:
            if hasattr(model, attr_name):
                weights = getattr(model, attr_name)
                if weights is not None:
                    pd.DataFrame(np.asarray(weights)).to_csv(os.path.join(fold_dir, file_name), index=False, header=False)


def train_controller_esn(
    model,
    X_raw,          # Shape: [Total_Seqs, Seq_Len, input_dim * 2] (y_t and y_next)
    Y_raw,          # Shape: [Total_Seqs, Seq_Len, output_dim]
    hyperparam_config,
    plant,
    dirname,
    run_simulation=False
):
    # --- EXTRACT HYPERPARAMETERS ---
    dt = hyperparam_config["training_data_cfg"]["dt"]
    k_folds = hyperparam_config["train"]["k_folds"]

    # --- MIMO-SPECIFIC CONFIG ---
    esn_cfg = hyperparam_config["plant"]
    input_dim = esn_cfg["input_dim"]    # Number of plant outputs
    output_dim = esn_cfg["output_dim"]  # Number of control inputs

    # --- SET UP K-FOLD INDICES ---
    total_sequences = X_raw.shape[0]
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    fold_histories = {}

    # --- K-FOLD CROSS VALIDATION LOOP ---
    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING FOLD {fold + 1} / {k_folds} (Dedicated ESN Analytical Fit)")
        print(f"==========================================")

        # Clear internal memory states for the new fold
        model.load_state_dict(None)

        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)

        # Isolate raw splits for this specific fold
        train_x_raw, val_x_raw = X_raw[train_idx_arr], X_raw[val_idx_arr]
        train_y_raw, val_y_raw = Y_raw[train_idx_arr], Y_raw[val_idx_arr]

        # --- INTERNAL FOLD STANDARD SCALING ---
        print(f"⚖️ Fitting independent StandardScalers for Fold {fold + 1}...")
        N_train, seq_len, dim_x = train_x_raw.shape
        dim_y = train_y_raw.shape[-1]

        train_x_flat = train_x_raw.reshape(-1, dim_x)
        train_y_flat = train_y_raw.reshape(-1, dim_y)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaler_x.fit(train_x_flat)
        scaler_y.fit(train_y_flat)

        # Keep everything native NumPy for ReservoirPy processing
        train_x = scaler_x.transform(train_x_flat).reshape(N_train, seq_len, dim_x)
        train_y = scaler_y.transform(train_y_flat).reshape(N_train, seq_len, dim_y)

        N_val = val_x_raw.shape[0]
        val_x_flat = val_x_raw.reshape(-1, dim_x)
        val_y_flat = val_y_raw.reshape(-1, dim_y)

        val_x = scaler_x.transform(val_x_flat).reshape(N_val, seq_len, dim_x)
        val_y = scaler_y.transform(val_y_flat).reshape(N_val, seq_len, dim_y)

        fold_dir = f"{dirname}/fold_{fold+1}"
        os.makedirs(fold_dir, exist_ok=True)
        save_to_json(hyperparam_config, fold_dir, f"hyperparam_config_fold_{fold+1}")
        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")

        # --- ⚡ ANALYTICAL RIDGE REGRESSION TRAINING ---
        print(f"⚡ Executing instant weight computation via Ridge Regression...")

        # Convert full array matrices into sequence lists for ReservoirPy compatibility
        X_train_list = [train_x[i] for i in range(N_train)]
        Y_train_list = [train_y[i] for i in range(N_train)]

        # Train linear readout matrix instantly
        model.fit(X_train_list, Y_train_list)

        # --- 📊 EVALUATION METRICS COLLECTION ---
        # Generate predictions across train traces
        train_all_preds = np.array([model.forward(train_x[i]) for i in range(N_train)])

        # Generate predictions across validation traces
        val_all_preds_arr = np.array([model.forward(val_x[i]) for i in range(N_val)])
        val_all_trues_arr = val_y

        # Calculate Mean Squared Error performance evaluation bounds
        mean_train_loss = np.mean((train_all_preds - train_y) ** 2)
        mean_val_loss = np.mean((val_all_preds_arr - val_all_trues_arr) ** 2)

        # Populate history dictionaries to match validation summary targets
        fold_histories[fold] = {
            "train_loss": [mean_train_loss],
            "val_loss": [mean_val_loss],
            "val_epochs": [1],
            **{f"train_loss_ch{ch+1}": [np.mean((train_all_preds[..., ch] - train_y[..., ch]) ** 2)] for ch in range(output_dim)},
            **{f"val_loss_ch{ch+1}": [np.mean((val_all_preds_arr[..., ch] - val_all_trues_arr[..., ch]) ** 2)] for ch in range(output_dim)}
        }
     
        print(f"✨ [Fold {fold+1}] Performance Complete:")
        model.save_parameters(f"{fold_dir}/parameters")
        print(f"   ↳ Total Train MSE: {mean_train_loss:.6f} | Total Val MSE: {mean_val_loss:.6f}")

        # Persist trained parameters to disk
        save_model_esn(model, dirname=fold_dir, hyperparam_config=hyperparam_config, filename="best_fold_model")

        # --- 📊 VALIDATION PLOTS: CONTROL INPUTS + OUTPUT SIGNALS (FIRST 5 SEQUENCES) ---
        if N_val > 0:
            print(f"📊 Generating validation plots for first 5 sequences of Fold {fold+1}...")

            t_axis_val = np.arange(seq_len) * dt
            plots_dir = f"{fold_dir}/validation_control_and_output_plots"
            os.makedirs(plots_dir, exist_ok=True)

            for seq_idx in range(min(5, N_val)):
                # Unscaled predictions and ground truth for control inputs
                seq_pred_unscaled = scaler_y.inverse_transform(val_all_preds_arr[seq_idx])
                seq_true_unscaled = scaler_y.inverse_transform(val_all_trues_arr[seq_idx])

                # Unscaled output signals (y_t and y_next) from X_raw
                seq_x_unscaled = scaler_x.inverse_transform(val_x[seq_idx])
                y_t = seq_x_unscaled[:, :input_dim]  # First half: y_t
                y_next = seq_x_unscaled[:, input_dim:]  # Second half: y_next

                # --- PLOT 1: CONTROL INPUTS (u) ---
                for ch in range(output_dim):
                    actual_u = seq_true_unscaled[:, ch]
                    predicted_u = seq_pred_unscaled[:, ch]

                    plot_signals(
                        t=t_axis_val,
                        signals=[actual_u, predicted_u],
                        labels=[
                            rf"Actual Control ($u_{ch+1}$)",
                            rf"Predicted Control ($\hat{{u}}_{ch+1}$)"
                        ],
                        title=f"Fold {fold+1} - Seq {seq_idx+1}: Control Input (Channel {ch+1})",
                        xlabel="Time [s]",
                        ylabel="Control Input",
                        figsize=(7, 5),
                        filename=f"control_tracking_fold_{fold+1}_seq_{seq_idx+1}_ch{ch+1}",
                        dirname=plots_dir
                    )

                # --- PLOT 2: OUTPUT SIGNALS (y) ---
                for out_ch in range(input_dim):
                    plot_signals(
                        t=t_axis_val,
                        signals=[y_t[:, out_ch], y_next[:, out_ch]],
                        labels=[
                            rf"Original Output ($y_{out_ch+1}$)",
                            rf"Next Output ($y_{out_ch+1,next}$)"
                        ],
                        title=f"Fold {fold+1} - Seq {seq_idx+1}: Output Signal (Channel {out_ch+1})",
                        xlabel="Time [s]",
                        ylabel="Output Signal",
                        figsize=(7, 5),
                        filename=f"output_tracking_fold_{fold+1}_seq_{seq_idx+1}_ch{out_ch+1}",
                        dirname=plots_dir
                    )

            print(f"✅ Validation plots (control + output) generated for first 5 sequences of Fold {fold+1}.")

        # --- ⏳ PLANT SIMULATION ROLLOUT FOR VALIDATION SEQUENCES (OPTIONAL) ---
        if run_simulation and N_val > 0:
            print(f"📊 Simulating plant dynamics across ALL ({N_val}) validation profiles...")

            t_axis_val = np.arange(seq_len) * dt
            pred_curves_dir = f"{fold_dir}/validation_tracking_curves"
            os.makedirs(pred_curves_dir, exist_ok=True)

            # Instantiate or use the plant instance
            plant_instance = plant(hyperparam_config) if isinstance(plant, type) else plant
            device = plant_instance.device

            for seq_idx in range(N_val):
                seq_pred_unscaled = scaler_y.inverse_transform(val_all_preds_arr[seq_idx])
                seq_true_unscaled = scaler_y.inverse_transform(val_all_trues_arr[seq_idx])
                seq_x_unscaled = scaler_x.inverse_transform(val_x[seq_idx])

                # Get starting state profile: [1, 2]
                current_sim_state = plant_instance.get_initial_state(batch_size=1)
                state_dim = current_sim_state.shape[-1]

                simulated_states_history = {st: [] for st in range(state_dim)}
                simulated_outputs_history = {out: [] for out in range(input_dim)}

                # 🌀 CRITICAL CORRECTION: Calculate and record initial output at t = 0
                y_init = plant_instance.get_y(current_sim_state, t_axis_val[0])
                for out in range(input_dim):
                    simulated_outputs_history[out].append(y_init[0, out].item())

                for step in range(seq_len):
                    # Record the current state components before stepping forward
                    for st in range(state_dim):
                        simulated_states_history[st].append(current_sim_state[0, st].item())

                    # Package predicted controller output u into a Tensor for the step
                    u_pred_step = torch.from_numpy(seq_pred_unscaled[step:step+1]).to(device=device, dtype=torch.float32)
                    t_curr = t_axis_val[step]

                    # Execute plant step -> advances state to t + dt
                    current_sim_state, y_next_pred = plant_instance.step(
                        current_sim_state,
                        u_pred_step,
                        t_curr,
                        dt
                    )

                    # Only capture the subsequent steps up to step < seq_len - 1 to match timeline bounds
                    if step < (seq_len - 1):
                        for out in range(input_dim):
                            simulated_outputs_history[out].append(y_next_pred[0, out].item())

                # --- PLOT 3: ORIGINAL VS. SIMULATED OUTPUTS (ONLY FOR FIRST 5 SEQUENCES) ---
                if seq_idx < 5:
                    for out_ch in range(input_dim):
                        # Ensure arrays match length exactly
                        sim_y_track = np.array(simulated_outputs_history[out_ch])
                        original_y_track = seq_x_unscaled[:, out_ch] # y_t from dataset

                        plot_signals(
                            t=t_axis_val,
                            signals=[
                                original_y_track,  # Original ground truth path
                                sim_y_track        # Pure output driven by ESN predicted control sequence
                            ],
                            labels=[
                                rf"Original Dataset Output ($y_{out_ch+1}$)",
                                rf"Simulated Output from Predicted $u$ ($\hat{{y}}_{out_ch+1}$)"
                            ],
                            title=f"Fold {fold+1} - Seq {seq_idx+1}: Dataset vs. Predicted Control Output (Channel {out_ch+1})",
                            xlabel="Time [s]",
                            ylabel="Output Signal [Growth Rate]",
                            figsize=(7, 5),
                            filename=f"output_comparison_fold_{fold+1}_seq_{seq_idx+1}_ch{out_ch+1}",
                            dirname=pred_curves_dir
                        )

                # Save simulation data to CSV
                log_data = {"Time (s)": t_axis_val}
                for out_idx in range(input_dim):
                    log_data[f"Target_y{out_idx+1}_t"] = seq_x_unscaled[:, out_idx]
                    log_data[f"Target_y{out_idx+1}_next"] = seq_x_unscaled[:, input_dim + out_idx]
                    log_data[f"Simulated_Output_y{out_idx+1}"] = simulated_outputs_history[out_idx]

                for ch in range(output_dim):
                    log_data[f"Actual_u{ch+1}"] = seq_true_unscaled[:, ch]
                    log_data[f"Predicted_u{ch+1}"] = seq_pred_unscaled[:, ch]
                    log_data[f"Control_Error_u{ch+1}"] = seq_true_unscaled[:, ch] - seq_pred_unscaled[:, ch]

                for st in range(state_dim):
                    log_data[f"Simulated_State_x{st+1}"] = simulated_states_history[st]

                val_profile_df = pd.DataFrame(log_data)
                save_df_to_csv(val_profile_df, dirname=pred_curves_dir, filename=f"val_plant_simulation_fold_{fold+1}_seq_{seq_idx+1}")

            print(f"✅ All {N_val} validation trajectory simulation logs dumped. Sample diagrams generated for Fold {fold + 1}.")

    # --- FINAL SUMMARY RECORD GENERATION ---
    print("\n💾 Packing overarching metadata curves...")
    summary_records = []
    for f in fold_histories:
        record = {"fold": f+1, "best_recorded_total_val_loss": fold_histories[f]["val_loss"][0]}
        for ch in range(output_dim):
            record[f"best_val_loss_u{ch+1}"] = fold_histories[f][f"val_loss_ch{ch+1}"][0]
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    save_df_to_csv(summary_df, dirname=dirname, filename="kfold_cross_validation_summary")
    print("✅ Dedicated ESN optimization finalized successfully.")

    return fold_histories