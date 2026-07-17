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
from src.sample.utils.saving_utils import *

plt.style.use("src/sample/style.mplstyle")

# Assuming your save utilities are imported correctly
# from src.sample.utils.saving_utils import save_scaler_object, save_model, save_df_to_csv
# from src.sample.utils.plotting_utils import plot_signals

import copy
import os

import numpy as np
import torch

import numpy as np
import torch

def create_inverse_controller_dataset(Y_trajectories, U_trajectories, n_y, n_u):
    """
    Slices raw batch continuous MIMO trajectories into history-windowed features 
    and targets for an inverse controller.
    
    Parameters:
        Y_trajectories: Tensor or NumPy array of shape [Num_Traces, Seq_Len, input_dim] (Plant Outputs)
        U_trajectories: Tensor or NumPy array of shape [Num_Traces, Seq_Len, output_dim] (Control Inputs)
        n_y: Number of past plant output lookbacks (excluding current y_k)
        n_u: Number of past control action lookbacks
        
    Returns:
        X_raw: NumPy array of shape [Num_Traces, Sliding_Seq_Len, Feature_Dim]
        Y_raw: NumPy array of shape [Num_Traces, Sliding_Seq_Len, output_dim]
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if torch.is_tensor(Y_trajectories):
        Y_trajectories = Y_trajectories.detach().cpu().numpy()
    if torch.is_tensor(U_trajectories):
        U_trajectories = U_trajectories.detach().cpu().numpy()
        
    num_traces, total_seq_len, input_dim = Y_trajectories.shape
    output_dim = U_trajectories.shape[-1]
    
    start_idx = max(n_y, n_u)
    end_idx = total_seq_len - 1
    sliding_seq_len = end_idx - start_idx
    
    # Calculate total feature dimension for verification
    # y_{k+1} (input_dim) + y_k...y_{k-n_y} (input_dim * (n_y + 1)) + u_{k-1}...u_{k-n_u} (output_dim * n_u)
    feature_dim = input_dim + (input_dim * (n_y + 1)) + (output_dim * n_u)
    
    print(f"📦 Slicing {num_traces} traces. Window metrics:")
    print(f"   ↳ Clean Rollout Steps per Trace: {sliding_seq_len}")
    print(f"   ↳ Total Feature vector size (dim_v): {feature_dim}")

    X_list = []
    Y_list = []
    
    for t_idx in range(num_traces):
        y_trace = Y_trajectories[t_idx]  # Shape: [Total_Seq_Len, input_dim]
        u_trace = U_trajectories[t_idx]  # Shape: [Total_Seq_Len, output_dim]
        
        trace_features = []
        trace_targets = []
        
        for k in range(start_idx, end_idx):
            # 1. Future target trajectory point: y_{k+1}
            y_next = y_trace[k + 1]
            
            # 2. Plant output history: [y_k, y_{k-1}, ..., y_{k-n_y}]
            # We fetch from k down to k-n_y (inclusive), then flip to reverse chronological order
            y_hist = y_trace[k - n_y : k + 1] 
            y_hist_reversed = y_hist[::-1].flatten()
            
            # 3. Control input history: [u_{k-1}, u_{k-2}, ..., u_{k-n_u}]
            # We fetch from k-n_u up to k-1, then flip to reverse chronological order
            u_hist = u_trace[k - n_u : k]
            u_hist_reversed = u_hist[::-1].flatten()
            
            # Combine into a single feature row v_k
            v_k = np.concatenate([y_next, y_hist_reversed, u_hist_reversed])
            
            trace_features.append(v_k)
            trace_targets.append(u_trace[k])  # Target is the control action u_k
            
        X_list.append(np.array(trace_features))  # Shape: [Sliding_Seq_Len, feature_dim]
        Y_list.append(np.array(trace_targets))   # Shape: [Sliding_Seq_Len, output_dim]
        
    # Stack back to 3D arrays matching your train_controller layout expectations
    X_raw = np.stack(X_list, axis=0)  # [Num_Traces, Sliding_Seq_Len, feature_dim]
    Y_raw = np.stack(Y_list, axis=0)  # [Num_Traces, Sliding_Seq_Len, output_dim]
    
    return X_raw, Y_raw

def train_controller(
    model,
    Y_trajectories,
    U_trajectories,
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_simulation=True,
    run_sim_with_plots=False
):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]
    k_folds = train_cfg.get("k_folds", 5)

    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    batch_size = 64
    val_patience = train_cfg.get("val_patience_epochs", 3)
    min_delta = train_cfg.get("val_min_delta", 0.001)

    mamba_cfg = hyperparam_config.get("mamba", {})
    input_dim = mamba_cfg["input_dim"]
    output_dim = mamba_cfg["output_dim"]

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
            criterion = NormalizedRMSELoss(reduction='none')
        else:
            criterion = getattr(nn, loss_name)(reduction='none')

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

            for i in range(0, train_size, batch_size):
                batch_indices = shuffled_train_indices[i : i + batch_size]
                current_batch_size = len(batch_indices)

                batch_x = train_x[batch_indices].to(device)
                batch_y = train_y[batch_indices].to(device)

                if hasattr(model, 'reset_memory'):
                    model.reset_memory(batch_size=current_batch_size, device=device)

                optimizer.zero_grad()
                u_pred_batch = model(batch_x)
                raw_loss = criterion(u_pred_batch, batch_y)
                loss = raw_loss.mean()
                loss.backward()
                optimizer.step()

                current_loss_val = loss.item()
                epoch_train_loss_accum += current_loss_val * current_batch_size
                fold_train_batch_loss.append(current_loss_val)
                fold_train_batch_indices.append(global_batch_counter)

                for ch in range(output_dim):
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
                for i in range(0, val_size, batch_size):
                    batch_val_x = val_x[i : i + batch_size].to(device)
                    batch_val_y = val_y[i : i + batch_size].to(device)
                    current_val_batch_size = len(batch_val_x)

                    if hasattr(model, 'reset_memory'):
                        model.reset_memory(batch_size=current_val_batch_size, device=device)

                    u_val_pred = model(batch_val_x)
                    raw_val_loss = criterion(u_val_pred, batch_val_y)

                    val_loss = raw_val_loss.mean()
                    epoch_val_loss_accum += val_loss.item() * current_val_batch_size

                    for ch in range(output_dim):
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

        for ch in range(output_dim):
            plot_signals(
                t=t_axis_val,
                signals=[sample_true_unscaled[:, ch], sample_pred_unscaled[:, ch]],
                labels=[f"True u_{ch+1}", f"Predicted u_{ch+1}"],
                xlabel="Time (s)",
                ylabel="Control Units",
                title=f"Fold {fold+1} | Validation Sample Performance - Channel u_{ch+1}",
                dirname=pred_curves_dir,
                filename=f"val_prediction_sample_u{ch+1}_0"
            )

        # --- 5. CLOSED-LOOP PLANT SIMULATION ROLLOUT (ORIGINAL VERSION) ---
        if run_simulation:
            if 'val_all_preds_arr' in locals() and len(val_all_preds_arr) > 0:
                print(f"📊 Simulating plant dynamics across ALL ({val_size}) validation profiles for Fold {fold + 1}...")

                if isinstance(plant, type):
                    plant_instance = plant(hyperparam_config)
                else:
                    plant_instance = plant

                plot_configs = plant_instance.get_plot_config() if hasattr(plant_instance, "get_plot_config") else []
                u_config = next((c for c in plot_configs if any(col.startswith("u") for col in c["cols"])), None)
                y_config = next((c for c in plot_configs if any(col.startswith("y") for col in c["cols"])), None)
                x_config = next((c for c in plot_configs if any(col.startswith("x") for col in c["cols"])), None)

                # Pure trajectory configuration
                total_trajectory_len = Y_trajectories.shape[1]
                t_axis_full = np.arange(total_trajectory_len) * dt

                for seq_idx in range(val_size):
                    val_traj_idx = val_idx_arr[seq_idx]
                    target_y_trajectory = Y_trajectories[val_traj_idx].cpu().numpy() if hasattr(Y_trajectories, "cpu") else Y_trajectories[val_traj_idx]
                    target_u_trajectory = U_trajectories[val_traj_idx].cpu().numpy() if hasattr(U_trajectories, "cpu") else U_trajectories[val_traj_idx]

                    # Extract the pre-computed validation predictions for this specific sequence
                    val_seq_preds = val_all_preds_arr[seq_idx] 

                    # 🌟 FIX: Calculate prediction window offset (e.g., n_y or max(n_y, n_u))
                    # Because of the sliding window, you have fewer predictions than raw trajectory length.
                    pred_len = len(val_seq_preds)
                    lookback_offset = total_trajectory_len - pred_len

                    current_sim_state = plant_instance.get_initial_state(batch_size=1)
                    state_dim = current_sim_state.shape[-1]

                    # Pre-allocate tracking arrays
                    simulated_states = np.zeros((total_trajectory_len, state_dim))
                    simulated_outputs = np.zeros((total_trajectory_len, input_dim))
                    simulated_controls = np.zeros((total_trajectory_len, output_dim))

                    # Initialize step 0
                    simulated_outputs[0] = target_y_trajectory[0].copy()
                    simulated_controls[0] = target_u_trajectory[0].copy()
                    simulated_states[0] = current_sim_state[0].cpu().numpy()

                    # Step through the plant simulator using pre-computed model outputs
                    for k in range(total_trajectory_len - 1):
                        # 🌟 FIX: If we are in the initial lookback phase (where no prediction exists),
                        # use the target/ground-truth control input.
                        if k < lookback_offset:
                            u_action = target_u_trajectory[k]
                        else:
                            # Safely map k to the prediction array by subtracting the offset
                            u_action = val_seq_preds[k - lookback_offset]

                        u_action_tensor = torch.tensor(u_action.reshape(1, -1), device=device, dtype=torch.float32)
                        current_sim_state, y_next_sim = plant_instance.step(
                            state=current_sim_state,
                            u1=u_action_tensor,
                            t=k * dt,
                            dt=dt
                        )
                        
                        # Log the physical consequences
                        simulated_controls[k + 1] = u_action
                        simulated_outputs[k + 1] = y_next_sim.squeeze(0).cpu().numpy()
                        simulated_states[k + 1] = current_sim_state[0].cpu().numpy()

                    # Extract full trajectories for plotting
                    original_y = target_y_trajectory
                    original_u = target_u_trajectory
                    original_states = simulated_states.copy()

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

                        for ch in range(output_dim):
                            io_signals.append([original_u[:, ch], simulated_controls[:, ch]])
                            io_labels.append(["Original", "Predicted (Closed Loop)"])
                            if u_config and ch < len(u_config["labels"]):
                                io_ylabels.append(u_config["labels"][ch])
                            else:
                                io_ylabels.append(rf"Input $u_{{{ch+1}}}$")

                        for out_idx in range(input_dim):
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
                    for out_idx in range(input_dim):
                        log_data[f"Desired_y{out_idx+1}"] = original_y[:, out_idx]
                        log_data[f"Simulated_Output_y{out_idx+1}"] = simulated_outputs[:, out_idx]
                    for ch in range(output_dim):
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

    return fold_histories

def train_controller_marcia(
    model,
    X_raw,          # Clean Shape: [Total_Seqs, Seq_Len, input_dim * 2] (y_t and y_next)
    Y_raw,          # Clean Shape: [Total_Seqs, Seq_Len, output_dim]
    hyperparam_config,
    plant,
    dirname,
    show_plots=False,
    run_simulation = True,
    run_sim_with_plots = False
):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]
    k_folds = train_cfg.get("k_folds", 5)
    delay_steps = train_cfg["delay_steps"]

    batch_size = 64
    val_patience = train_cfg.get("val_patience_epochs", 3)
    min_delta = train_cfg.get("val_min_delta", 0.001)

    # --- MIMO-SPECIFIC CONFIG ---
    mamba_cfg = hyperparam_config.get("mamba", {})
    input_dim = mamba_cfg["input_dim"]    # Number of plant outputs (y1, y2)
    output_dim = mamba_cfg["output_dim"]  # Number of control inputs (u1, u2)

    # --- IDENTIFY TOTAL SEQUENCES & SET UP K-FOLD INDICES ---
    total_sequences = X_raw.shape[0]
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    initial_model_state = copy.deepcopy(model.state_dict())
    fold_histories = {}

    # --- K-FOLD CROSS VALIDATION LOOP ---
    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING FOLD {fold + 1} / {k_folds}")
        print(f"==========================================")

        model.load_state_dict(initial_model_state)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg["lr_decay_rate"])

        # 🔴 CRITICAL: Force reduction='none' so we can split losses per channel
        loss_name = train_cfg["loss_function"].replace("()", "")
        if loss_name == "NormalizedRMSELoss":
         criterion = NormalizedRMSELoss(reduction='none')
        else:
            criterion = getattr(nn, loss_name)(reduction='none')

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

        means_x = scaler_x.mean_
        stds_x = scaler_x.scale_

        print(f"Means x: {means_x}")
        print(f"Standard Deviations x: {stds_x}")

        means_y = scaler_y.mean_
        stds_y = scaler_y.scale_

        print(f"Means y: {means_y}")
        print(f"Standard Deviations y: {stds_y}")

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

        # Plot transformed data curves if show_plots is enabled
        if show_plots:
            curves_dir = f"{fold_dir}/transformed_data_curves"
            sample_x = train_x[0].numpy()
            sample_y = train_y[0].numpy()
            t_axis = np.arange(seq_len) * dt
            
            for out_idx in range(output_dim):
                plot_signals(t=t_axis, signals=[sample_y[:, out_idx]], labels=[f"Scaled u_{out_idx+1}"],
                             xlabel="Time (s)", ylabel="Standardized Units", title=f"Fold {fold+1} | Transformed Input u_{out_idx+1}",
                             dirname=curves_dir, filename=f"scaled_u{out_idx+1}_curve")
            for in_idx in range(input_dim):
                plot_signals(t=t_axis, signals=[sample_x[:, in_idx], sample_x[:, input_dim + in_idx]],
                             labels=[f"Scaled y_{in_idx+1}_t", f"Scaled y_{in_idx+1}_next"], xlabel="Time (s)", ylabel="Standardized Units",
                             title=f"Fold {fold+1} | Transformed State & Target (Channel {in_idx+1})",
                             dirname=curves_dir, filename=f"scaled_y{in_idx+1}_and_next_curve")

        train_size = train_x.shape[0]
        val_size = val_x.shape[0]

        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        
        # --- NEW LOGGING STRUCTURES FOR CHANNELS ---
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

            # --- 1. MINI-BATCH TRAINING PASS ---
            for i in range(0, train_size, batch_size):
                batch_indices = shuffled_train_indices[i : i + batch_size]
                current_batch_size = len(batch_indices)

                batch_x = train_x[batch_indices].to(device)
                batch_y = train_y[batch_indices].to(device)

                y_t_split = batch_x[:, :, :input_dim]
                y_next_split = batch_x[:, :, input_dim:]

                if hasattr(model, 'reset_memory'):
                    model.reset_memory(batch_size=current_batch_size, device=device)

                optimizer.zero_grad()
                u_pred_batch = model(y_t_split, y_next_split)

                # Raw element-wise loss: shape [Batch, Seq_Len, Output_Dim]
                raw_loss = criterion(u_pred_batch, batch_y)
                
                # Global backward tracking scalar loss
                loss = raw_loss.mean() 
                loss.backward()
                optimizer.step()

                # Extract granular channel-level scalar values from this forward pass
                current_loss_val = loss.item()
                epoch_train_loss_accum += current_loss_val * current_batch_size
                fold_train_batch_loss.append(current_loss_val)
                fold_train_batch_indices.append(global_batch_counter)

                for ch in range(output_dim):
                    ch_loss_val = raw_loss[:, :, ch].mean().item()
                    epoch_train_channel_accum[ch] += ch_loss_val * current_batch_size
                    fold_train_channel_batch_loss[ch].append(ch_loss_val)

                global_batch_counter += 1

            scheduler.step()

            # --- 2. VALIDATION PASS ---
            model.eval()
            epoch_val_loss_accum = 0.0
            epoch_val_channel_accum = {ch: 0.0 for ch in range(output_dim)}
            
            all_val_preds = []
            all_val_trues = []

            with torch.no_grad():
                for i in range(0, val_size, batch_size):
                    batch_val_x = val_x[i : i + batch_size].to(device)
                    batch_val_y = val_y[i : i + batch_size].to(device)
                    current_val_batch_size = len(batch_val_x)

                    y_val_t_split = batch_val_x[:, :, :input_dim]
                    y_val_next_split = batch_val_x[:, :, input_dim:]

                    if hasattr(model, 'reset_memory'):
                        model.reset_memory(batch_size=current_val_batch_size, device=device)

                    u_val_pred = model(y_val_t_split, y_val_next_split)
                    raw_val_loss = criterion(u_val_pred, batch_val_y)
                    
                    val_loss = raw_val_loss.mean()
                    epoch_val_loss_accum += val_loss.item() * current_val_batch_size
                    
                    for ch in range(output_dim):
                        ch_val_loss_val = raw_val_loss[:, :, ch].mean().item()
                        epoch_val_channel_accum[ch] += ch_val_loss_val * current_val_batch_size

                    all_val_preds.append(u_val_pred.cpu().numpy())
                    all_val_trues.append(batch_val_y.cpu().numpy())

            val_all_preds_arr = np.concatenate(all_val_preds, axis=0)
            val_all_trues_arr = np.concatenate(all_val_trues, axis=0)

            # Compute normalized averages
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

            # Append historical logs for individual tracking channels
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
            ch_summary_str = " | ".join([f"u{ch+1} (Tr: {epoch_train_channel_accum[ch]/train_size:.4f}, Val: {epoch_val_channel_accum[ch]/val_size:.4f})" for ch in range(output_dim)])
            print(f"   ↳ Channels -> {ch_summary_str}")

            # --- 3. EVALUATE VALIDATION EARLY STOPPING ---
            if mean_val_loss < (best_val_loss - min_delta):
                best_val_loss = mean_val_loss
                patience_counter = 0
                print("dattebayo")
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
        
        # --- NEW: PLOT VALIDATION PREDICTION SAMPLE FOR THIS FOLD ---
        
        print(f"📈 Plotting sample validation prediction for Fold {fold + 1}...")
        pred_curves_dir = f"{fold_dir}/validation_tracking_curves"
        t_axis_val = np.arange(seq_len) * dt
        
        # Extract a single sample sequence (index 0) from the validation predictions
        sample_seq_idx = 0
        sample_pred_scaled = val_all_preds_arr[sample_seq_idx]  # Shape: [Seq_Len, output_dim]
        sample_true_scaled = val_all_trues_arr[sample_seq_idx]  # Shape: [Seq_Len, output_dim]
        
        # Inverse transform to return back to real physical control units
        sample_pred_unscaled = scaler_y.inverse_transform(sample_pred_scaled)
        sample_true_unscaled = scaler_y.inverse_transform(sample_true_scaled)
        
        # Plot tracking performance for each control channel (u1, u2, etc.)
        for ch in range(output_dim):
            plot_signals(
                t=t_axis_val, 
                signals=[sample_true_unscaled[:, ch], sample_pred_unscaled[:, ch]],
                labels=[f"True u_{ch+1}", f"Predicted u_{ch+1}"], 
                xlabel="Time (s)", 
                ylabel="Control Units", 
                title=f"Fold {fold+1} | Validation Sample Performance - Channel u_{ch+1}",
                dirname=pred_curves_dir, 
                filename=f"val_prediction_sample_u{ch+1}"
            )

        # --- 5. PLANT SIMULATION ROLLOUT FOR ALL VALIDATION SEQUENCES ---
        if run_simulation:
            if 'val_all_preds_arr' in locals() and len(val_all_preds_arr) > 0:
                print(f"📊 Simulating plant dynamics across ALL ({val_size}) validation profiles for Fold {fold + 1}...")
                
                t_axis_val = np.arange(seq_len) * dt
                pred_curves_dir = f"{fold_dir}/validation_tracking_curves"
                
                # Handle plant class vs instance instantiation
                if isinstance(plant, type):
                    plant_instance = plant(hyperparam_config)
                else:
                    plant_instance = plant

                # Fetch plot configs for extracting labels if available
                plot_configs = plant_instance.get_plot_config() if hasattr(plant_instance, "get_plot_config") else []
                u_config = next((c for c in plot_configs if any(col.startswith("u") for col in c["cols"])), None)
                y_config = next((c for c in plot_configs if any(col.startswith("y") for col in c["cols"])), None)
                x_config = next((c for c in plot_configs if any(col.startswith("x") for col in c["cols"])), None)

                # Loop through every single validation sequence index
                for seq_idx in range(val_size):
                    
                    # 1. Isolate and unscale control signals (Inputs) for this specific sequence
                    seq_pred_scaled = val_all_preds_arr[seq_idx]  # Shape: [Seq_Len, output_dim]
                    seq_true_scaled = val_all_trues_arr[seq_idx]  # Shape: [Seq_Len, output_dim]
                    
                    seq_pred_unscaled = scaler_y.inverse_transform(seq_pred_scaled) # Predicted 'u'
                    seq_true_unscaled = scaler_y.inverse_transform(seq_true_scaled) # Original 'u'
                    
                    # Isolate and unscale outputs (y_t, y_next) for this specific sequence
                    seq_x_scaled = val_x[seq_idx].cpu().numpy()
                    seq_x_unscaled = scaler_x.inverse_transform(seq_x_scaled)
                    
                    # original_y shape: [Seq_Len, input_dim]
                    # Note: seq_x_unscaled contains [y_t, y_next]. We take the first 'input_dim' columns for y_t
                    original_y = seq_x_unscaled[:, :input_dim] 
                    
                    # 2. Reset the plant to its initial state (Batch size = 1)
                    current_sim_state = plant_instance.get_initial_state(batch_size=1)
                    state_dim = current_sim_state.shape[-1]
                    
                    simulated_states_history = {st: [] for st in range(state_dim)}
                    simulated_outputs_history = {out: [] for out in range(input_dim)}

                    # --- 3. Step-by-step native plant integration rollout loop ---
                    # Here we feed the PREDICTED control inputs (seq_pred_unscaled) back to the plant
                    for step in range(seq_len):
                        for st in range(state_dim):
                            simulated_states_history[st].append(current_sim_state[0, st].item())
                        
                        # Grab the step control input predicted by the controller
                        u_pred_step = torch.tensor(
                            seq_pred_unscaled[step:step+1], 
                            device=device, 
                            dtype=torch.float32
                        )
                        t_start = t_axis_val[step]
                        
                        # Step the plant forward using the predicted control action
                        current_sim_state, y_next_pred = plant_instance.step(
                            state=current_sim_state, 
                            u1=u_pred_step[:, 0:1],  # Adjust slice depending on MIMO setup
                            t=t_start, 
                            dt=dt
                        )
                        
                        for out in range(input_dim):
                            simulated_outputs_history[out].append(y_next_pred[0, out].item())

                    # Reorganize simulated histories into structured arrays
                    simulated_states = np.zeros((seq_len, state_dim))
                    for st in range(state_dim):
                        simulated_states[:, st] = simulated_states_history[st]
                        
                    simulated_outputs = np.zeros((seq_len, input_dim))
                    for out in range(input_dim):
                        simulated_outputs[:, out] = simulated_outputs_history[out]
                
                    # ---------------------------------------------------------
                    # 4. PLOT 1: STACKED PLOT OF STATE VARIABLES (True vs. Simulated)
                    # ---------------------------------------------------------
                    state_signals = []
                    state_labels = []
                    state_ylabels = []

                    # If your original/true states are stored somewhere, replace 'None' with the array.
                    # Otherwise, we will step a secondary parallel loop with original inputs to reconstruct true states.
                    # For safety, let's step the plant using TRUE inputs to reconstruct the 'original state' trajectory.
                    original_sim_state = plant_instance.get_initial_state(batch_size=1)
                    original_states = np.zeros((seq_len, state_dim))
                    for step in range(seq_len):
                        for st in range(state_dim):
                            original_states[step, st] = original_sim_state[0, st].item()
                        u_true_step = torch.tensor(seq_true_unscaled[step:step+1], device=device, dtype=torch.float32)
                        original_sim_state, _ = plant_instance.step(
                            state=original_sim_state, 
                            u1=u_true_step[:, 0:1], 
                            t=t_axis_val[step], 
                            dt=dt
                        )
                    if run_sim_with_plots:
                        for st in range(state_dim):
                            # Pack [Original State, Simulated State] into the row
                            state_signals.append([original_states[:, st], simulated_states[:, st]])
                            state_labels.append(["Original", "Simulated"])
                            
                            if x_config and st < len(x_config["labels"]):
                                state_ylabels.append(x_config["labels"][st])
                            else:
                                state_ylabels.append(rf"State $x_{{{st+1}}}$")
                    
                        state_asp = [0.33] * len(state_signals)
                        plot_stacked(
                            t=t_axis_val,
                            signals=state_signals,
                            labels=state_labels,
                            xlabel=rf"$t \; / \; \mathrm{{s}}$",
                            ylabel=state_ylabels,
                            asp=state_asp,
                            dirname=pred_curves_dir,
                            filename=f"val_simulation_states_fold_{fold+1}_seq_{seq_idx+1}.png",
                            show=False  # Keeps loop running quickly without blocking popups
                        )

                        # ---------------------------------------------------------
                        # 5. PLOT 2: STACKED PLOT OF INPUTS & OUTPUTS (True vs. Predicted/Simulated)
                        # ---------------------------------------------------------
                        io_signals = []
                        io_labels = []
                        io_ylabels = []

                        # A. Control Inputs (u): Original vs. Predicted
                        for ch in range(output_dim):
                            io_signals.append([seq_true_unscaled[:, ch], seq_pred_unscaled[:, ch]])
                            io_labels.append(["Original", "Predicted"])
                            
                            if u_config and ch < len(u_config["labels"]):
                                io_ylabels.append(u_config["labels"][ch])
                            else:
                                io_ylabels.append(rf"Input $u_{{{ch+1}}}$")

                        # B. Plant Outputs (y): Original vs. Simulated
                        for out_idx in range(input_dim):
                            io_signals.append([original_y[:, out_idx], simulated_outputs[:, out_idx]])
                            io_labels.append(["Original", "Simulated"])
                            
                            if y_config and out_idx < len(y_config["labels"]):
                                io_ylabels.append(y_config["labels"][out_idx])
                            else:
                                io_ylabels.append(rf"Output $y_{{{out_idx+1}}}$")

                        io_asp = [0.33] * len(io_signals)
                        plot_stacked(
                            t=t_axis_val,
                            signals=io_signals,
                            labels=io_labels,
                            xlabel=rf"$t \; / \; \mathrm{{s}}$",
                            ylabel=io_ylabels,
                            asp=io_asp,
                            dirname=pred_curves_dir,
                            filename=f"val_simulation_io_fold_{fold+1}_seq_{seq_idx+1}.png",
                            show=False
                        )

                    # 6. Save traditional CSV log database
                    log_data = {"Time (s)": t_axis_val}
                    for out_idx in range(input_dim):
                        log_data[f"Target_y{out_idx+1}_t"] = original_y[:, out_idx]
                        log_data[f"Simulated_Output_y{out_idx+1}"] = simulated_outputs[:, out_idx]
                    for ch in range(output_dim):
                        log_data[f"Actual_u{ch+1}"] = seq_true_unscaled[:, ch]
                        log_data[f"Predicted_u{ch+1}"] = seq_pred_unscaled[:, ch]
                    for st in range(state_dim):
                        log_data[f"Simulated_State_x{st+1}"] = simulated_states[:, st]
                        log_data[f"Original_State_x{st+1}"] = original_states[:, st]

                    val_profile_df = pd.DataFrame(log_data)
                    save_df_to_csv(
                        val_profile_df, 
                        dirname=pred_curves_dir, 
                        filename=f"val_plant_simulation_fold_{fold+1}_seq_{seq_idx+1}"
                    )
                                
                print(f"✅ All {val_size} validation trajectory plots and CSV logs safely saved for Fold {fold + 1}.")

    # --- FINAL CROSS VALIDATION SUMMARY ---
    print("\n💾 Complete Cross-Validation run finished. Packing overarching metadata curves...")
    summary_records = []
    for f in fold_histories:
        final_best = min(fold_histories[f]["val_loss"])
        record = {"fold": f+1, "best_recorded_total_val_loss": final_best}
        for ch in range(output_dim):
            best_epoch_idx = np.argmin(fold_histories[f]["val_loss"])
            record[f"best_val_loss_u{ch+1}"] = fold_histories[f][f"val_loss_ch{ch+1}"][best_epoch_idx]
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    save_df_to_csv(summary_df, dirname=dirname, filename="kfold_cross_validation_summary")
    print("\n✅ K-Fold optimization execution finalized.")
    return fold_histories


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

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
    dt = hyperparam_config["signal"]["dt"]
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