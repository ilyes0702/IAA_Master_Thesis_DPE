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
from src.sample.utils.general_utils import *

plt.style.use("src/sample/style.mplstyle")

# Assuming your save utilities are imported correctly
# from src.sample.utils.saving_utils import save_scaler_object, save_model, save_df_to_csv
# from src.sample.utils.plotting_utils import plot_signals

import copy
import os

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
        
    # --- FIXED DIMENSION UNPACKING HERE ---
    num_traces, total_seq_len, output_dim = Y_trajectories.shape
    input_dim = U_trajectories.shape[-1]
    
    start_idx = max(n_y, n_u)
    end_idx = total_seq_len - 1
    sliding_seq_len = end_idx - start_idx
    
    # Calculate total feature dimension for verification
    # y_{k+1} (input_dim) + y_k...y_{k-n_y} (input_dim * (n_y + 1)) + u_{k-1}...u_{k-n_u} (output_dim * n_u)
    feature_dim = n_u * input_dim + (n_y+2) * output_dim
    
    print(f"📦 Slicing {num_traces} traces. Window metrics:")
    print(f"   ↳ Clean Rollout Steps per Trace: {sliding_seq_len}")
    print(f"   ↳ Total Feature vector size (dim_v): {feature_dim}")

    X_list = []
    Y_list = []
    
    for t_idx in range(num_traces):
        y_trace = Y_trajectories[t_idx]  # Shape: [Total_Seq_Len, input_dim]
        #print("y_trace.shape", y_trace.shape)
        u_trace = U_trajectories[t_idx]  # Shape: [Total_Seq_Len, output_dim]
        #print("u_trace.shape", u_trace.shape)
        trace_features = []
        trace_targets = []
        
        for k in range(start_idx, end_idx):
            # 1. Future target trajectory point: y_{k+1}
            y_next = y_trace[k + 1]
            #print("y_next: ", y_next.shape)
            # 2. Plant output history: [y_k, y_{k-1}, ..., y_{k-n_y}]
            y_hist = y_trace[k - n_y : k + 1].flatten()
            #print("y_hist: ", y_hist.shape) 
            #y_hist_reversed = y_hist[::-1].flatten()
            
            # 3. Control input history: [u_{k-1}, u_{k-2}, ..., u_{k-n_u}]
            u_hist = u_trace[k - n_u : k].flatten()
            #print("u_hist: ", u_hist.shape) 
            #u_hist_reversed = u_hist[::-1].flatten()
            
            # Combine into a single feature row v_k
            v_k = np.concatenate([y_next, y_hist, u_hist])

            #print("v_k: ", v_k.shape)
            
            trace_features.append(v_k)
            trace_targets.append(u_trace[k])  # Target is the control action u_k
            
        X_list.append(np.array(trace_features))  # Shape: [Sliding_Seq_Len, feature_dim]
        Y_list.append(np.array(trace_targets))   # Shape: [Sliding_Seq_Len, output_dim]
        
    # Stack back to 3D arrays matching your train_controller layout expectations
    X_raw = np.stack(X_list, axis=0)  # [Num_Traces, Sliding_Seq_Len, feature_dim]
    Y_raw = np.stack(Y_list, axis=0)  # [Num_Traces, Sliding_Seq_Len, output_dim]

    print("X_raw shape after slicing:", X_raw.shape)
    
    return X_raw, Y_raw


def evaluate_closed_loop_validation(
    model,
    val_idx_arr,
    Y_trajectories,
    U_trajectories,
    X_states,
    scaler_x,
    scaler_y,
    plant,
    hyperparam_config,
    save_dir,
    fold_num,
    device,
    dt,
    run_sim_with_plots=False,
):
    """Executes closed-loop simulation on validation trajectories for a given fold model

    and logs tracking metrics, plots, and CSV profiles.
    """
    print(f"📊 Running TRUE CLOSED-LOOP plant simulation for Fold {fold_num}...")

    # Instantiation or handle plant instance
    if isinstance(plant, type):
        plant_instance = plant(hyperparam_config)
    else:
        plant_instance = plant

    plant_cfg = hyperparam_config["plant"]
    train_cfg = hyperparam_config["train"]

    input_dim = plant_cfg["input_dim"]
    output_dim = plant_cfg["output_dim"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    lookback_offset = train_cfg["lookback_offset"]

    # Extract plot configs if available
    plot_configs = (
        plant_instance.get_plot_config()
        if hasattr(plant_instance, "get_plot_config")
        else []
    )
    u_config = next(
        (
            c
            for c in plot_configs
            if any(col.startswith("u") for col in c["cols"])
        ),
        None,
    )
    y_config = next(
        (
            c
            for c in plot_configs
            if any(col.startswith("y") for col in c["cols"])
        ),
        None,
    )
    x_config = next(
        (
            c
            for c in plot_configs
            if any(col.startswith("x") for col in c["cols"])
        ),
        None,
    )

    total_trajectory_len = Y_trajectories.shape[1]
    t_axis_full = np.arange(total_trajectory_len) * dt
    sliding_seq_len = total_trajectory_len - 1 - lookback_offset
    dim_x = scaler_x.mean_.shape[0]

    model.eval()
    fold_metrics_records = []

    val_size = len(val_idx_arr)

    for seq_idx in range(val_size):
        val_traj_idx = val_idx_arr[seq_idx]

        target_y_trajectory = (
            Y_trajectories[val_traj_idx].cpu().numpy()
            if hasattr(Y_trajectories, "cpu")
            else Y_trajectories[val_traj_idx]
        )
        target_u_trajectory = (
            U_trajectories[val_traj_idx].cpu().numpy()
            if hasattr(U_trajectories, "cpu")
            else U_trajectories[val_traj_idx]
        )
        target_state_trajectory = (
            X_states[val_traj_idx].cpu().numpy()
            if hasattr(X_states, "cpu")
            else X_states[val_traj_idx]
        )

        # Initialize simulator state with the exact initial state
        true_initial_state = target_state_trajectory[0].copy()
        current_sim_state = torch.tensor(
            true_initial_state, device=device, dtype=torch.float32
        ).unsqueeze(0)
        state_dim = current_sim_state.shape[-1]

        # Pre-allocate simulation history
        simulated_states = np.zeros((total_trajectory_len, state_dim))
        simulated_outputs = np.zeros((total_trajectory_len, output_dim))
        simulated_controls = np.zeros((total_trajectory_len, input_dim))

        # Step 0 Initialization
        simulated_outputs[0] = target_y_trajectory[0].copy()
        simulated_controls[0] = target_u_trajectory[0].copy()
        simulated_states[0] = true_initial_state.copy()

        scaled_inputs_seq = np.zeros((1, sliding_seq_len, dim_x))

        # --- Closed-Loop Step Simulation ---
        for k in range(total_trajectory_len - 1):
            # 1. HAND-OFF PHASE: Before we have enough lookback history
            if k < lookback_offset:
                u_action = target_u_trajectory[k].copy()
                u_action_tensor = torch.tensor(
                    u_action.reshape(1, -1), device=device, dtype=torch.float32
                )

                current_sim_state, y_next_sim = plant_instance.step(
                    state=current_sim_state,
                    u=u_action_tensor,
                    t=k * dt,
                    dt=dt,
                )

                simulated_controls[k + 1] = u_action
                simulated_outputs[k + 1] = y_next_sim.squeeze(0).cpu().numpy()
                simulated_states[k + 1] = current_sim_state[0].cpu().numpy()

            # 2. AUTONOMOUS CLOSED-LOOP PHASE: Model takes over
            else:
                y_next_ref = target_y_trajectory[k + 1]

                # Extract simulated history (reversed)
                y_hist = simulated_outputs[k - n_y : k + 1]
                y_hist_reversed = y_hist[::-1].flatten()

                u_hist = simulated_controls[k - n_u : k]
                u_hist_reversed = u_hist[::-1].flatten()

                v_k = np.concatenate(
                    [y_next_ref, y_hist_reversed, u_hist_reversed]
                )
                v_k_scaled = scaler_x.transform(v_k.reshape(1, -1)).squeeze(0)

                seq_buffer_idx = k - lookback_offset
                scaled_inputs_seq[0, seq_buffer_idx] = v_k_scaled

                # Model Forward Pass
                with torch.no_grad():
                    input_tensor = torch.tensor(
                        scaled_inputs_seq[:, : seq_buffer_idx + 1, :],
                        device=device,
                        dtype=torch.float32,
                    )

                    if hasattr(model, "reset_memory"):
                        model.reset_memory(batch_size=1, device=device)

                    u_pred_scaled = model(input_tensor)
                    u_pred_step_scaled = (
                        u_pred_scaled[0, -1, :].cpu().numpy()
                    )

                u_action = scaler_y.inverse_transform(
                    u_pred_step_scaled.reshape(1, -1)
                ).squeeze(0)
                u_action = np.clip(
                    u_action,
                    plant_cfg["u_1_hard_min"],
                    plant_cfg["u_1_hard_max"],
                )

                u_action_tensor = torch.tensor(
                    u_action.reshape(1, -1), device=device, dtype=torch.float32
                )
                current_sim_state, y_next_sim = plant_instance.step(
                    state=current_sim_state,
                    u=u_action_tensor,
                    t=k * dt,
                    dt=dt,
                )

                simulated_controls[k + 1] = u_action
                simulated_outputs[k + 1] = y_next_sim.squeeze(0).cpu().numpy()
                simulated_states[k + 1] = current_sim_state[0].cpu().numpy()

        # --- Calculate Metrics ---
        output_squared_error = (simulated_outputs - target_y_trajectory) ** 2
        seq_mse_total = float(np.mean(output_squared_error))
        seq_mse_per_channel = {
            f"MSE_y{out_idx+1}": float(
                np.mean(output_squared_error[:, out_idx])
            )
            for out_idx in range(output_dim)
        }

        metrics_record = {
            "fold": fold_num,
            "seq_idx": seq_idx,
            "val_traj_idx": int(val_traj_idx),
            "total_mse": seq_mse_total,
            **seq_mse_per_channel,
        }
        fold_metrics_records.append(metrics_record)

        # --- CSV Logging ---
        log_data = {"Time (s)": t_axis_full}
        for out_idx in range(output_dim):
            log_data[f"Desired_y{out_idx+1}"] = target_y_trajectory[:, out_idx]
            log_data[f"Simulated_Output_y{out_idx+1}"] = simulated_outputs[
                :, out_idx
            ]
            log_data[f"Squared_Error_y{out_idx+1}"] = output_squared_error[
                :, out_idx
            ]

        for ch in range(input_dim):
            log_data[f"Actual_u{ch+1}"] = target_u_trajectory[:, ch]
            log_data[f"Predicted_u{ch+1}_ClosedLoop"] = simulated_controls[
                :, ch
            ]

        for st in range(state_dim):
            log_data[f"Simulated_State_x{st+1}"] = simulated_states[:, st]
            log_data[f"Original_State_x{st+1}"] = target_state_trajectory[
                :, st
            ]

        val_profile_df = pd.DataFrame(log_data)
        save_df_to_csv(
            val_profile_df,
            dirname=save_dir,
            filename=f"val_plant_simulation_fold_{fold_num}_seq_{seq_idx+1}",
        )

        # --- Optional Plotting ---
        if run_sim_with_plots:
            # Plot States
            state_signals = [
                [target_state_trajectory[:, st], simulated_states[:, st]]
                for st in range(state_dim)
            ]
            state_labels = [["Original", "Simulated"]] * state_dim
            state_ylabels = [
                (
                    x_config["labels"][st]
                    if x_config and st < len(x_config["labels"])
                    else rf"State $x_{{{st+1}}}$"
                )
                for st in range(state_dim)
            ]

            plot_stacked(
                t=t_axis_full,
                signals=state_signals,
                labels=state_labels,
                xlabel=rf"$t \; / \; \mathrm{{s}}$",
                ylabel=state_ylabels,
                asp=[0.33] * state_dim,
                dirname=save_dir,
                filename=f"val_simulation_states_fold_{fold_num}_seq_{seq_idx+1}.png",
                show=False,
            )

            # Plot Inputs & Outputs
            io_signals, io_labels, io_ylabels = [], [], []
            for ch in range(input_dim):
                io_signals.append(
                    [target_u_trajectory[:, ch], simulated_controls[:, ch]]
                )
                io_labels.append(["Original", "Predicted (Closed Loop)"])
                io_ylabels.append(
                    u_config["labels"][ch]
                    if u_config and ch < len(u_config["labels"])
                    else rf"Input $u_{{{ch+1}}}$"
                )

            for out_idx in range(output_dim):
                io_signals.append(
                    [
                        target_y_trajectory[:, out_idx],
                        simulated_outputs[:, out_idx],
                    ]
                )
                io_labels.append(["Desired", "Simulated"])
                io_ylabels.append(
                    y_config["labels"][out_idx]
                    if y_config and out_idx < len(y_config["labels"])
                    else rf"Output $y_{{{out_idx+1}}}$"
                )

            plot_stacked(
                t=t_axis_full,
                signals=io_signals,
                labels=io_labels,
                xlabel=rf"$t \; / \; \mathrm{{s}}$",
                ylabel=io_ylabels,
                asp=[0.33] * len(io_signals),
                dirname=save_dir,
                filename=f"val_simulation_io_fold_{fold_num}_seq_{seq_idx+1}.png",
                show=False,
            )

    # Save summary dataframe for this fold
    if fold_metrics_records:
        summary_df = pd.DataFrame(fold_metrics_records)
        save_df_to_csv(
            summary_df,
            dirname=save_dir,
            filename=f"val_closed_loop_mse_summary_fold_{fold_num}",
        )
        mean_fold_mse = summary_df["total_mse"].mean()
        print(
            f"📊 Fold {fold_num} Closed-Loop Mean MSE across validation set: {mean_fold_mse:.6f}"
        )

    return fold_metrics_records

import copy
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def train_controller_ol_and_cl(
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
    all_fold_sim_metrics = []

    open_loop_records = []
    closed_loop_records = []
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
            for out_idx in range(input_dim):
                plot_signals(
                    t=t_axis, signals=[sample_y[:, out_idx]], labels=[f"Scaled u_{out_idx+1}"],
                    xlabel="Time (s)", ylabel="Standardized Units",
                    title=f"Fold {fold+1} | Transformed Input u_{out_idx+1}",
                    dirname=curves_dir, filename=f"scaled_u{out_idx+1}_curve"
                )

        train_size = train_x.shape[0]
        val_size = val_x.shape[0]

        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        fold_train_channel_batch_loss = {ch: [] for ch in range(input_dim)}
        fold_train_channel_epoch_history = {ch: [] for ch in range(input_dim)}
        fold_val_channel_epoch_history = {ch: [] for ch in range(input_dim)}
        fold_val_epoch_history = []
        fold_train_epoch_history = []

        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False

        # --- 2. TRAINING LOOP ---
        for epoch in range(epochs):
            model.train()
            epoch_train_loss_accum = 0.0
            epoch_train_channel_accum = {ch: 0.0 for ch in range(input_dim)}

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
            epoch_val_channel_accum = {ch: 0.0 for ch in range(input_dim)}

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
                    **{f"train_loss_ch{ch+1}": [] for ch in range(input_dim)},
                    **{f"val_loss_ch{ch+1}": [] for ch in range(input_dim)}
                }

            fold_histories[fold]["train_loss"].append(mean_train_loss)
            fold_histories[fold]["val_loss"].append(mean_val_loss)
            fold_histories[fold]["val_epochs"].append(epoch + 1)

            for ch in range(input_dim):
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

        # --- OPEN-LOOP TRAIN METRICS (End of Epoch) ---
        #best_model_path = f"{fold_dir}/best_fold_model.pt"
        #model.load_state_dict(torch.load(best_model_path))
        model.eval()
        all_train_preds, all_train_trues = [], []

        with torch.no_grad():
            for i in range(0, train_size, mini_batch_size):
                b_x = train_x[i : i + mini_batch_size].to(device)
                b_y = train_y[i : i + mini_batch_size].to(device)
                
                if hasattr(model, 'reset_memory'):
                    model.reset_memory(batch_size=len(b_x), device=device)
                    
                u_p = model(b_x)
                all_train_preds.append(u_p.cpu().numpy())
                all_train_trues.append(b_y.cpu().numpy())

        # Reshape & Unscale
        train_preds_flat = np.concatenate(all_train_preds, axis=0).reshape(-1, dim_y)
        train_trues_flat = np.concatenate(all_train_trues, axis=0).reshape(-1, dim_y)

        u_train_pred_unscaled = scaler_y.inverse_transform(train_preds_flat).reshape(N_train, seq_len, dim_y)
        u_train_true_unscaled = scaler_y.inverse_transform(train_trues_flat).reshape(N_train, seq_len, dim_y)

        train_open_metrics = compute_trajectory_metrics(u_train_true_unscaled, u_train_pred_unscaled, dt=dt)
        open_loop_records.append({
            "fold": fold + 1,
            "epoch": epoch + 1,
            "split": "train",
            **train_open_metrics
        })

        # --- OPEN-LOOP VALIDATION METRICS ---
        val_preds_flat = val_all_preds_arr.reshape(-1, dim_y)
        val_trues_flat = val_all_trues_arr.reshape(-1, dim_y)

        u_val_pred_unscaled = scaler_y.inverse_transform(val_preds_flat).reshape(N_val, seq_len, dim_y)
        u_val_true_unscaled = scaler_y.inverse_transform(val_trues_flat).reshape(N_val, seq_len, dim_y)

        val_open_metrics = compute_trajectory_metrics(u_val_true_unscaled, u_val_pred_unscaled, dt=dt)
        open_loop_records.append({
            "fold": fold + 1,
            "epoch": epoch + 1,
            "split": "val",
            **val_open_metrics
        })

        # Save open-loop metrics DataFrame for all folds
        open_loop_df = pd.DataFrame(open_loop_records)
        save_df_to_csv(open_loop_df, dirname=dirname, filename="open_loop_trajectory_metrics_all_folds")

        # Optional: Aggregate mean metrics across validation folds
        val_open_loop_summary = open_loop_df[open_loop_df["split"] == "val"].groupby("fold").mean(numeric_only=True)
        save_df_to_csv(val_open_loop_summary, dirname=dirname, filename="open_loop_val_metrics_summary")

        # --- 4. PLOT LOSS CURVES ---
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

        # --- PLOT VALIDATION PREDICTION SAMPLES ---
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
                dirname=pred_curves_dir,
                filename=f"val_prediction_sample_u{ch+1}_0"
            )

        # --- 5. TRUE CLOSED-LOOP PLANT SIMULATION ROLLOUT ---
        if run_simulation:
            print(f"📊 Running TRUE CLOSED-LOOP plant simulation for Fold {fold + 1}...")

            sim_metrics_records = []  # Initialize fresh for this fold

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
            lookback_offset = hyperparam_config["train"]["lookback_offset"]
            sliding_seq_len = total_trajectory_len - 1 - lookback_offset

            model.eval()

            for seq_idx in range(val_size):
                val_traj_idx = val_idx_arr[seq_idx]
                target_y_trajectory = Y_trajectories[val_traj_idx].cpu().numpy() if hasattr(Y_trajectories, "cpu") else Y_trajectories[val_traj_idx]
                target_u_trajectory = U_trajectories[val_traj_idx].cpu().numpy() if hasattr(U_trajectories, "cpu") else U_trajectories[val_traj_idx]
                target_state_trajectory = X_states[val_traj_idx].cpu().numpy() if hasattr(X_states, "cpu") else X_states[val_traj_idx]

                true_initial_state = target_state_trajectory[0].copy()
                current_sim_state = torch.tensor(true_initial_state, device=device, dtype=torch.float32).unsqueeze(0)
                state_dim = current_sim_state.shape[-1]

                simulated_states = np.zeros((total_trajectory_len, state_dim))
                simulated_outputs = np.zeros((total_trajectory_len, output_dim))
                simulated_controls = np.zeros((total_trajectory_len, input_dim))

                simulated_outputs[0] = target_y_trajectory[0].copy()
                simulated_controls[0] = target_u_trajectory[0].copy()
                simulated_states[0] = true_initial_state.copy()

                dim_x = scaler_x.mean_.shape[0]
                scaled_inputs_seq = np.zeros((1, sliding_seq_len, dim_x))

                for k in range(total_trajectory_len - 1):
                    # 1. HAND-OFF PHASE
                    if k < lookback_offset:
                        u_action = target_u_trajectory[k].copy()
                        u_action_tensor = torch.tensor(u_action.reshape(1, -1), device=device, dtype=torch.float32)
                        current_sim_state, y_next_sim = plant_instance.step(
                            state=current_sim_state, u=u_action_tensor, t=k * dt, dt=dt
                        )

                        simulated_controls[k + 1] = u_action
                        simulated_outputs[k + 1] = y_next_sim.squeeze(0).cpu().numpy()
                        simulated_states[k + 1] = current_sim_state[0].cpu().numpy()

                    # 2. AUTONOMOUS CLOSED-LOOP PHASE
                    else:
                        y_next_ref = target_y_trajectory[k + 1]
                        y_hist = simulated_outputs[k - n_y : k + 1]
                        y_hist_reversed = y_hist[::-1].flatten()

                        u_hist = simulated_controls[k - n_u : k]
                        u_hist_reversed = u_hist[::-1].flatten()

                        v_k = np.concatenate([y_next_ref, y_hist_reversed, u_hist_reversed])
                        v_k_scaled = scaler_x.transform(v_k.reshape(1, -1)).squeeze(0)

                        seq_buffer_idx = k - lookback_offset
                        scaled_inputs_seq[0, seq_buffer_idx] = v_k_scaled

                        with torch.no_grad():
                            input_tensor = torch.tensor(scaled_inputs_seq[:, :seq_buffer_idx + 1, :], device=device, dtype=torch.float32)

                            if hasattr(model, 'reset_memory'):
                                model.reset_memory(batch_size=1, device=device)

                            u_pred_scaled = model(input_tensor)
                            u_pred_step_scaled = u_pred_scaled[0, -1, :].cpu().numpy()

                        u_action = scaler_y.inverse_transform(u_pred_step_scaled.reshape(1, -1)).squeeze(0)
                        u_action = np.clip(u_action, hyperparam_config["plant"]["u_1_hard_min"], hyperparam_config["plant"]["u_1_hard_max"])

                        u_action_tensor = torch.tensor(u_action.reshape(1, -1), device=device, dtype=torch.float32)
                        current_sim_state, y_next_sim = plant_instance.step(
                            state=current_sim_state, u=u_action_tensor, t=k * dt, dt=dt
                        )

                        simulated_controls[k + 1] = u_action
                        simulated_outputs[k + 1] = y_next_sim.squeeze(0).cpu().numpy()
                        simulated_states[k + 1] = current_sim_state[0].cpu().numpy()

                original_y = target_y_trajectory
                original_u = target_u_trajectory
                original_states = target_state_trajectory

                output_squared_error = (simulated_outputs - original_y) ** 2
                seq_mse_total = float(np.mean(output_squared_error))

                seq_mse_per_channel = {
                    f"MSE_y{out_idx+1}": float(np.mean(output_squared_error[:, out_idx]))
                    for out_idx in range(output_dim)
                }

                metrics_record = {
                    "fold": fold + 1,
                    "seq_idx": seq_idx,
                    "val_traj_idx": int(val_traj_idx),
                    "total_mse": seq_mse_total,
                    **seq_mse_per_channel
                }
                sim_metrics_records.append(metrics_record)

                log_data = {"Time (s)": t_axis_full}
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

                    cl_output_metrics = compute_trajectory_metrics(original_y, simulated_outputs, dt=dt)
                    cl_control_metrics = compute_trajectory_metrics(original_u, simulated_controls, dt=dt)
        
                    # Record all trajectory metrics (MSE, RMSE, MAE, MAPE, IAE, ISE)
                    metrics_record = {
                        "fold": fold + 1,
                        "seq_idx": seq_idx,
                        "val_traj_idx": int(val_traj_idx),
                        # Closed-loop tracking metrics for output (y)
                        **{f"y_{k}": v for k, v in cl_output_metrics.items()},
                        # Closed-loop metrics for input control (u)
                        **{f"u_{k}": v for k, v in cl_control_metrics.items()},
                    }
                    sim_metrics_records.append(metrics_record)

            # --- SAVE SUMMARY FOR THIS FOLD ---
            if sim_metrics_records:
                summary_df = pd.DataFrame(sim_metrics_records)
                save_df_to_csv(
                    summary_df,
                    dirname=pred_curves_dir,
                    filename=f"val_closed_loop_mse_summary_fold_{fold+1}"
                )
                mean_fold_mse = summary_df["total_mse"].mean()
                print(f"📊 Fold {fold+1} Closed-Loop Mean MSE across validation set: {mean_fold_mse:.6f}")
                all_fold_sim_metrics.extend(sim_metrics_records)
            # Replace manual MSE with compute_trajectory_metrics call
            
    # --- AGGREGATED METRICS SUMMARY ---
    cl_summary_df = pd.DataFrame(all_fold_sim_metrics) if run_simulation and all_fold_sim_metrics else None

    if cl_summary_df is not None:
        # Save detailed closed-loop metrics for all trajectories
        save_df_to_csv(cl_summary_df, dirname=dirname, filename="closed_loop_all_trajectories_metrics")

        # Aggregate closed-loop metrics across all folds
        cl_mean_metrics = cl_summary_df.mean(numeric_only=True).to_dict()
    else:
        cl_mean_metrics = {}

        # Expand final summary dataframe
        summary_data = {
            "Metric": [
                "Avg Open-Loop Train MSE",
                "Avg Open-Loop Val MSE",
                "Avg Open-Loop Val RMSE",
                "Avg Open-Loop Val MAE",
                "Avg Closed-Loop Output MSE (y)",
                "Avg Closed-Loop Output RMSE (y)",
                "Avg Closed-Loop Output MAE (y)",
                "Avg Closed-Loop Output IAE (y)",
                "Avg Closed-Loop Output ISE (y)",
            ],
            "Value": [
                avg_best_train_loss,
                avg_best_val_loss,
                open_loop_df[open_loop_df["split"] == "val"]["RMSE"].mean() if not open_loop_df.empty else np.nan,
                open_loop_df[open_loop_df["split"] == "val"]["MAE"].mean() if not open_loop_df.empty else np.nan,
                cl_mean_metrics.get("y_MSE", np.nan),
                cl_mean_metrics.get("y_RMSE", np.nan),
                cl_mean_metrics.get("y_MAE", np.nan),
                cl_mean_metrics.get("y_IAE", np.nan),
                cl_mean_metrics.get("y_ISE", np.nan),
            ]
        }

        summary_results_df = pd.DataFrame(summary_data)
        save_df_to_csv(summary_results_df, dirname=dirname, filename="k_fold_cross_validation_summary")
    # --- METRIC AGGREGATION ACROSS ALL FOLDS ---
    fold_best_train_losses = []
    fold_best_val_losses = []

    for f in range(k_folds):
        best_val_epoch_idx = np.argmin(fold_histories[f]["val_loss"])
        fold_best_val_losses.append(fold_histories[f]["val_loss"][best_val_epoch_idx])
        fold_best_train_losses.append(fold_histories[f]["train_loss"][best_val_epoch_idx])

    avg_best_train_loss = float(np.mean(fold_best_train_losses))
    avg_best_val_loss = float(np.mean(fold_best_val_losses))
    mean_cv_loss = avg_best_val_loss

    if run_simulation and all_fold_sim_metrics:
        sim_df = pd.DataFrame(all_fold_sim_metrics)
        avg_closed_loop_mse = float(sim_df["total_mse"].mean())
    else:
        avg_closed_loop_mse = np.nan

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
            "closed_loop_mse": avg_closed_loop_mse,
        },
        mean_cv_loss,
        summary_results_df,
    )

import torch
import numpy as np
import pandas as pd

def simulate_closed_loop(
    model,
    plant,
    y_ref_trajectory,      # Shape: (T, dim_y)
    u_gt_trajectory=None,  # Shape: (T, dim_u) - Ground truth control for warmup
    x_initial=None,        # Initial plant state
    scaler_x=None,
    scaler_y=None,
    n_y=1,
    n_u=1,
    dt=0.01,
    u_min=None,
    u_max=None,
    device="cpu"
):
    """
    Closed-loop plant simulation with warm-up buffer initialization and scaler feature matching.
    
    Feature Vector Layout per step k:
        X(k) = [ y(k), y(k-1), ..., y(k - n_y + 1),   (n_y features)
                 y_ref(k + 1),                        (1 target feature)
                 u(k-1), u(k-2), ..., u(k - n_u) ]    (n_u features)
        Total features = n_y + 1 + n_u
    """
    T, dim_y = y_ref_trajectory.shape
    dim_u = u_gt_trajectory.shape[-1] if u_gt_trajectory is not None else 1
    
    # Minimum steps required before model takes over closed-loop control
    k_warmup = max(n_y, n_u, n_y + n_u)

    # Initialize plant state
    if x_initial is not None:
        plant_state = x_initial
    elif hasattr(plant, 'get_initial_state'):
        plant_state = plant.get_initial_state()
    else:
        plant_state = np.zeros_like(x_initial) if x_initial is not None else None

    # Storage arrays for trajectory tracking
    y_sim_hist = []
    u_sim_hist = []
    x_sim_hist = []

    # Obtain initial output y(0)
    if hasattr(plant, 'get_y'):
        y_curr = plant.get_y(plant_state, 0)
    else:
        y_curr = y_ref_trajectory[0].copy()

    for k in range(T):
        # ----------------------------------------------------
        # 1. WARM-UP PHASE (k < k_warmup)
        # Apply ground truth u_gt to seed history buffers
        # ----------------------------------------------------
        if k < k_warmup:
            if u_gt_trajectory is not None:
                u_apply = u_gt_trajectory[k].copy()
            else:
                u_apply = np.zeros(dim_u)
        
        # ----------------------------------------------------
        # 2. CLOSED-LOOP NEURAL CONTROL PHASE (k >= k_warmup)
        # ----------------------------------------------------
        else:
            # Construct Past Output Window: [y(k), y(k-1), ..., y(k - n_y + 1)]
            past_y_list = [y_sim_hist[k - i] for i in range(n_y)]
            past_y_flat = np.concatenate(past_y_list, axis=-1)  # (n_y * dim_y,)

            # Target reference output for next step: y_ref(k + 1)
            k_ref = min(k + 1, T - 1)
            target_y_flat = y_ref_trajectory[k_ref].flatten()    # (1 * dim_y,)

            # Construct Past Input Window: [u(k-1), u(k-2), ..., u(k - n_u)]
            past_u_list = [u_sim_hist[k - 1 - i] for i in range(n_u)]
            past_u_flat = np.concatenate(past_u_list, axis=-1)  # (n_u * dim_u,)

            # Concatenate features into exact shape expected by scaler_x:
            # (n_y * dim_y) + (1 * dim_y) + (n_u * dim_u)
            feat_vector = np.hstack([past_y_flat, target_y_flat, past_u_flat]).reshape(1, -1)

            # Apply scaler transform
            if scaler_x is not None:
                feat_scaled = scaler_x.transform(feat_vector)
            else:
                feat_scaled = feat_vector

            # Reshape into (batch_size=1, seq_len=1, feature_dim)
            x_tensor = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            # Neural network prediction
            with torch.no_grad():
                if hasattr(model, 'reset_memory'):
                    model.reset_memory(batch_size=1, device=device)
                u_pred_scaled = model(x_tensor).cpu().numpy().reshape(1, -1)

            # Inverse scale control output
            if scaler_y is not None:
                u_pred_unscaled = scaler_y.inverse_transform(u_pred_scaled).flatten()
            else:
                u_pred_unscaled = u_pred_scaled.flatten()

            # Actuator output saturation / clipping
            if u_min is not None or u_max is not None:
                u_apply = np.clip(u_pred_unscaled, a_min=u_min, a_max=u_max)
            else:
                u_apply = u_pred_unscaled

        # ----------------------------------------------------
        # 3. STEP PLANT DYNAMICS FORWARD
        # ----------------------------------------------------
        if hasattr(plant, 'step'):
            plant_state = plant.step(plant_state, u_apply, dt)
            y_curr = plant.get_y(plant_state) if hasattr(plant, 'get_y') else plant_state[:dim_y]
        else:
            y_curr = y_ref_trajectory[k]  # Fallback dummy step

        # Append step results to history
        y_sim_hist.append(np.atleast_1d(y_curr))
        u_sim_hist.append(np.atleast_1d(u_apply))
        x_sim_hist.append(np.atleast_1d(plant_state) if plant_state is not None else np.array([]))

    return np.array(y_sim_hist), np.array(u_sim_hist), np.array(x_sim_hist)


import copy
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def train_and_validate_controller(
    model,
    train_data,       # Tuple: (Y_train, U_train)
    test_data,        # Tuple: (Y_test, U_test)
    val_data,         # Tuple: (Y_val, U_val, X_val)
    hyperparam_config,
    plant=None,       # Plant object implementing get_initial_state, get_y, and step
    dirname=".",
    show_plots=False,
    run_closed_loop_sim=True,
    u_min=None,
    u_max=None
):
    """
    Plant-agnostic inverse neural controller training and dual-mode validation pipeline
    with 3-row stacked plotting (Inputs u, Outputs y, States x).
    """
    # --- EXTRACT CONFIGURATION ---
    train_cfg = hyperparam_config["train"]
    device = train_cfg["device"]
    dt = hyperparam_config["training_data_cfg"]["dt"]

    lr = train_cfg["lr"]
    epochs = train_cfg["epochs"]
    n_y = train_cfg["n_y"]
    n_u = train_cfg["n_u"]
    mini_batch_size = train_cfg["mini_batch_size"]
    val_patience = train_cfg["val_patience_epochs"]
    min_delta = train_cfg["val_min_delta"]

    Y_train, U_train = train_data
    Y_test, U_test   = test_data
    Y_val, U_val, X_val = val_data

    # ==========================================
    # 1. CREATE SLIDING WINDOW DATASETS
    # ==========================================
    print(f"🔄 Creating sliding window datasets (n_y={n_y}, n_u={n_u})...")
    X_train_raw, Y_train_raw = create_inverse_controller_dataset(Y_trajectories=Y_train, U_trajectories=U_train, n_y=n_y, n_u=n_u)
    X_test_raw,  Y_test_raw  = create_inverse_controller_dataset(Y_trajectories=Y_test,  U_trajectories=U_test,  n_y=n_y, n_u=n_u)
    X_val_raw,   Y_val_raw   = create_inverse_controller_dataset(Y_trajectories=Y_val,   U_trajectories=U_val,   n_y=n_y, n_u=n_u)

    N_train, seq_len, dim_x = X_train_raw.shape
    dim_y = Y_train_raw.shape[-1]
    N_test = X_test_raw.shape[0]
    N_val  = X_val_raw.shape[0]

    # ==========================================
    # 2. FIT SCALERS (TRAINING SET ONLY)
    # ==========================================
    print("⚖️ Fitting StandardScalers on Training dataset...")
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    train_x_flat = X_train_raw.reshape(-1, dim_x)
    train_y_flat = Y_train_raw.reshape(-1, dim_y)

    scaler_x.fit(train_x_flat)
    scaler_y.fit(train_y_flat)

    save_scaler_object(scaler_x, dirname=dirname, filename="scaler_x")
    save_scaler_object(scaler_y, dirname=dirname, filename="scaler_y")

    # Transform all datasets
    train_x = torch.tensor(scaler_x.transform(train_x_flat).reshape(N_train, seq_len, dim_x), dtype=torch.float32)
    train_y = torch.tensor(scaler_y.transform(train_y_flat).reshape(N_train, seq_len, dim_y), dtype=torch.float32)

    test_x = torch.tensor(scaler_x.transform(X_test_raw.reshape(-1, dim_x)).reshape(N_test, seq_len, dim_x), dtype=torch.float32)
    test_y = torch.tensor(scaler_y.transform(Y_test_raw.reshape(-1, dim_y)).reshape(N_test, seq_len, dim_y), dtype=torch.float32)

    val_x = torch.tensor(scaler_x.transform(X_val_raw.reshape(-1, dim_x)).reshape(N_val, seq_len, dim_x), dtype=torch.float32)
    val_y = torch.tensor(scaler_y.transform(Y_val_raw.reshape(-1, dim_y)).reshape(N_val, seq_len, dim_y), dtype=torch.float32)

    # ==========================================
    # 3. TRAINING LOOP
    # ==========================================
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg.get("lr_decay_rate", 1.0))

    loss_name = train_cfg.get("loss_function", "MSELoss").replace("()", "")
    criterion = NormalizedRMSELoss(reduction='none') if loss_name == "NormalizedRMSELoss" else torch.nn.MSELoss(reduction='none')

    best_test_loss = float('inf')
    patience_counter = 0
    train_epoch_history, test_epoch_history = [], []

    print("\n==========================================")
    print("🎬 STARTING MODEL TRAINING")
    print("==========================================")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        shuffled_indices = torch.randperm(N_train)

        for i in range(0, N_train, mini_batch_size):
            batch_idx = shuffled_indices[i : i + mini_batch_size]
            b_x, b_y = train_x[batch_idx].to(device), train_y[batch_idx].to(device)

            if hasattr(model, 'reset_memory'):
                model.reset_memory(batch_size=len(b_x), device=device)

            optimizer.zero_grad()
            u_pred = model(b_x)
            loss = criterion(u_pred, b_y).mean()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * len(b_x)

        scheduler.step()
        mean_train_loss = epoch_train_loss / N_train

        # Evaluation on test set
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for i in range(0, N_test, mini_batch_size):
                b_x, b_y = test_x[i : i + mini_batch_size].to(device), test_y[i : i + mini_batch_size].to(device)
                if hasattr(model, 'reset_memory'):
                    model.reset_memory(batch_size=len(b_x), device=device)

                u_pred = model(b_x)
                epoch_test_loss += criterion(u_pred, b_y).mean().item() * len(b_x)

        mean_test_loss = epoch_test_loss / N_test
        train_epoch_history.append(mean_train_loss)
        test_epoch_history.append(mean_test_loss)

        print(f"Epoch {epoch+1:03d}/{epochs:03d} | Train Loss: {mean_train_loss:.6f} | Test Loss: {mean_test_loss:.6f}")

        # Checkpointing & Early Stopping
        if mean_test_loss < (best_test_loss - min_delta):
            best_test_loss = mean_test_loss
            patience_counter = 0
            save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="best_model")
        else:
            patience_counter += 1
            if patience_counter >= val_patience:
                print(f"🛑 Early stopping triggered at Epoch {epoch+1}.")
                break

    # Save Loss Curve
    plot_signals(
        t=np.arange(1, len(train_epoch_history) + 1),
        signals=[np.array(train_epoch_history), np.array(test_epoch_history)],
        labels=["Train Loss", "Test Loss"],
        xlabel="Epochs", ylabel="Loss", dirname=dirname, filename="train_test_loss_curve"
    )

    model.eval()

    # ==========================================
    # 4. MODE 1: OPEN-LOOP VALIDATION
    # ==========================================
    print("\n==========================================")
    print("🔍 EVALUATING MODE 1: OPEN-LOOP VALIDATION")
    print("==========================================")

    val_preds = []
    with torch.no_grad():
        for i in range(0, N_val, mini_batch_size):
            b_x = val_x[i : i + mini_batch_size].to(device)
            if hasattr(model, 'reset_memory'):
                model.reset_memory(batch_size=len(b_x), device=device)
            val_preds.append(model(b_x).cpu().numpy())

    val_preds_flat = np.concatenate(val_preds, axis=0).reshape(-1, dim_y)
    val_trues_flat = val_y.numpy().reshape(-1, dim_y)

    u_val_pred_unscaled = scaler_y.inverse_transform(val_preds_flat).reshape(N_val, seq_len, dim_y)
    u_val_true_unscaled = scaler_y.inverse_transform(val_trues_flat).reshape(N_val, seq_len, dim_y)

    open_loop_metrics = compute_trajectory_metrics(u_val_true_unscaled, u_val_pred_unscaled, dt=dt)
    open_loop_df = pd.DataFrame([{"split": "val", **open_loop_metrics}])
    save_df_to_csv(open_loop_df, dirname=dirname, filename="open_loop_validation_metrics")

    # --- MODE 1 STACKED PLOTS (Sample Trajectories) ---
    if show_plots:
        ol_curves_dir = f"{dirname}/open_loop_curves"
        num_ol_samples = min(3, len(Y_val))

        for traj_idx in range(num_ol_samples):
            t_axis = np.arange(len(U_val[traj_idx])) * dt

            # Row 1: Controls (u_true vs u_pred)
            u_true_i = U_val[traj_idx]
            u_pred_i = u_val_pred_unscaled[traj_idx] if traj_idx < len(u_val_pred_unscaled) else u_true_i

            # Row 2: Reference Outputs (y_ref)
            y_ref_i = Y_val[traj_idx]

            # Row 3: True Plant States (x_true)
            x_true_i = X_val[traj_idx] if X_val is not None else np.zeros((len(t_axis), 1))

            signals_ol = [
                [u_true_i, u_pred_i],  # Subplot 1: Control inputs
                [y_ref_i],              # Subplot 2: Reference outputs
                [x_true_i]              # Subplot 3: States
            ]
            labels_ol = [
                ["True Control u", "Open-Loop Pred u"],
                ["Reference Output y"],
                ["Plant State x"]
            ]
            ylabels_ol = ["Control u", "Output y", "State x"]

            plot_stacked(
                t=t_axis,
                signals=signals_ol,
                labels=labels_ol,
                ylabel=ylabels_ol,
                title=f"Open-Loop Validation (Sample {traj_idx+1})",
                xlabel="Time (s)",
                dirname=ol_curves_dir,
                filename=f"ol_stacked_sample_{traj_idx+1}",
                show=False
            )

    # ==========================================
    # 5. MODE 2: CLOSED-LOOP VALIDATION
    # ==========================================
    closed_loop_df = pd.DataFrame()
    if run_closed_loop_sim and plant is not None:
        print("\n==========================================")
        print("🔄 EVALUATING MODE 2: CLOSED-LOOP VALIDATION")
        print("==========================================")

        closed_loop_records = []
        cl_curves_dir = f"{dirname}/closed_loop_tracking_curves"

        for traj_idx in range(len(Y_val)):
            y_ref_traj = Y_val[traj_idx]
            u_ref_traj = U_val[traj_idx] if U_val is not None else None
            x_ref_traj = X_val[traj_idx] if X_val is not None else None
            x_init     = x_ref_traj[0] if x_ref_traj is not None else None

            # Closed-loop simulation with warm-up support
            y_sim, u_sim, x_sim = simulate_closed_loop(
                model=model,
                plant=plant,
                y_ref_trajectory=y_ref_traj,
                u_gt_trajectory=u_ref_traj,  # <-- Seeding ground-truth for warmup steps
                x_initial=x_init,
                scaler_x=scaler_x,
                scaler_y=scaler_y,
                n_y=n_y,
                n_u=n_u,
                dt=dt,
                u_min=u_min,
                u_max=u_max,
                device=device
            )

            # Compute tracking metrics (Y_ref vs Y_sim)
            cl_metrics = compute_trajectory_metrics(
                y_ref_traj.reshape(1, -1, y_sim.shape[-1]),
                y_sim.reshape(1, -1, y_sim.shape[-1]),
                dt=dt
            )
            closed_loop_records.append({"trajectory_id": traj_idx, **cl_metrics})

            # --- MODE 2 STACKED PLOTS (Inputs u, Outputs y, States x) ---
            if show_plots and traj_idx < 3:
                t_axis = np.arange(len(y_ref_traj)) * dt

                # Prepare signal components
                u_signals = [u_sim]
                u_labels  = ["Simulated Control u"]
                if u_ref_traj is not None:
                    u_signals.insert(0, u_ref_traj)
                    u_labels.insert(0, "Ground Truth u")

                y_signals = [y_ref_traj, y_sim]
                y_labels  = ["Reference y_ref", "Closed-Loop Output y_sim"]

                x_signals = [x_sim]
                x_labels  = ["Simulated State x"]
                if x_ref_traj is not None:
                    x_signals.insert(0, x_ref_traj)
                    x_labels.insert(0, "Ground Truth State x")

                # Stack into [Inputs, Outputs, States]
                signals_cl = [u_signals, y_signals, x_signals]
                labels_cl  = [u_labels, y_labels, x_labels]
                ylabels_cl = ["Control u", "Output y", "State x"]

                plot_stacked(
                    t=t_axis,
                    signals=signals_cl,
                    labels=labels_cl,
                    ylabel=ylabels_cl,
                    title=f"Closed-Loop Tracking (Trajectory {traj_idx+1})",
                    xlabel="Time (s)",
                    dirname=cl_curves_dir,
                    filename=f"cl_stacked_sample_{traj_idx+1}",
                    show=False
                )

        closed_loop_df = pd.DataFrame(closed_loop_records)
        save_df_to_csv(closed_loop_df, dirname=dirname, filename="closed_loop_validation_metrics")

        print("Closed-Loop Validation Summary (Mean across trajectories):")
        print(closed_loop_df.mean(numeric_only=True).to_frame().T.to_string(index=False))

    # ==========================================
    # 6. SUMMARY METRICS REPORT
    # ==========================================
    summary_metrics = {
        "Metric": [
            "Best Train Loss",
            "Best Test Loss",
            "Open-Loop Control MSE",
            "Closed-Loop Output Tracking MSE"
        ],
        "Value": [
            train_epoch_history[np.argmin(test_epoch_history)],
            best_test_loss,
            open_loop_metrics.get("mse", np.nan),
            closed_loop_df["mse"].mean() if not closed_loop_df.empty else np.nan
        ]
    }
    summary_df = pd.DataFrame(summary_metrics)
    save_df_to_csv(summary_df, dirname=dirname, filename="final_evaluation_summary")

    print("\n==========================================")
    print("📊 FINAL EVALUATION SUMMARY")
    print("==========================================")
    print(summary_df.to_string(index=False))

    return model, scaler_x, scaler_y, summary_df

def train_controller_open_loop_no_split(
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
    all_fold_sim_metrics = []

    open_loop_records = []
    closed_loop_records = []
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
            for out_idx in range(input_dim):
                plot_signals(
                    t=t_axis, signals=[sample_y[:, out_idx]], labels=[f"Scaled u_{out_idx+1}"],
                    xlabel="Time (s)", ylabel="Standardized Units",
                    title=f"Fold {fold+1} | Transformed Input u_{out_idx+1}",
                    dirname=curves_dir, filename=f"scaled_u{out_idx+1}_curve"
                )

        train_size = train_x.shape[0]
        val_size = val_x.shape[0]

        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        fold_train_channel_batch_loss = {ch: [] for ch in range(input_dim)}
        fold_train_channel_epoch_history = {ch: [] for ch in range(input_dim)}
        fold_val_channel_epoch_history = {ch: [] for ch in range(input_dim)}
        fold_val_epoch_history = []
        fold_train_epoch_history = []

        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False

        # --- 2. TRAINING LOOP ---
        for epoch in range(epochs):
            model.train()
            epoch_train_loss_accum = 0.0
            epoch_train_channel_accum = {ch: 0.0 for ch in range(input_dim)}

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
            epoch_val_channel_accum = {ch: 0.0 for ch in range(input_dim)}

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
                    **{f"train_loss_ch{ch+1}": [] for ch in range(input_dim)},
                    **{f"val_loss_ch{ch+1}": [] for ch in range(input_dim)}
                }

            fold_histories[fold]["train_loss"].append(mean_train_loss)
            fold_histories[fold]["val_loss"].append(mean_val_loss)
            fold_histories[fold]["val_epochs"].append(epoch + 1)

            for ch in range(input_dim):
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

        # --- OPEN-LOOP TRAIN METRICS (End of Epoch) ---
        #best_model_path = f"{fold_dir}/best_fold_model.pt"
        #model.load_state_dict(torch.load(best_model_path))
        model.eval()
        all_train_preds, all_train_trues = [], []

        with torch.no_grad():
            for i in range(0, train_size, mini_batch_size):
                b_x = train_x[i : i + mini_batch_size].to(device)
                b_y = train_y[i : i + mini_batch_size].to(device)
                
                if hasattr(model, 'reset_memory'):
                    model.reset_memory(batch_size=len(b_x), device=device)
                    
                u_p = model(b_x)
                all_train_preds.append(u_p.cpu().numpy())
                all_train_trues.append(b_y.cpu().numpy())

        # Reshape & Unscale
        train_preds_flat = np.concatenate(all_train_preds, axis=0).reshape(-1, dim_y)
        train_trues_flat = np.concatenate(all_train_trues, axis=0).reshape(-1, dim_y)

        u_train_pred_unscaled = scaler_y.inverse_transform(train_preds_flat).reshape(N_train, seq_len, dim_y)
        u_train_true_unscaled = scaler_y.inverse_transform(train_trues_flat).reshape(N_train, seq_len, dim_y)

        train_open_metrics = compute_trajectory_metrics(u_train_true_unscaled, u_train_pred_unscaled, dt=dt)
        open_loop_records.append({
            "fold": fold + 1,
            "epoch": epoch + 1,
            "split": "train",
            **train_open_metrics
        })

        # --- OPEN-LOOP VALIDATION METRICS ---
        val_preds_flat = val_all_preds_arr.reshape(-1, dim_y)
        val_trues_flat = val_all_trues_arr.reshape(-1, dim_y)

        u_val_pred_unscaled = scaler_y.inverse_transform(val_preds_flat).reshape(N_val, seq_len, dim_y)
        u_val_true_unscaled = scaler_y.inverse_transform(val_trues_flat).reshape(N_val, seq_len, dim_y)

        val_open_metrics = compute_trajectory_metrics(u_val_true_unscaled, u_val_pred_unscaled, dt=dt)
        open_loop_records.append({
            "fold": fold + 1,
            "epoch": epoch + 1,
            "split": "val",
            **val_open_metrics
        })

        # Save open-loop metrics DataFrame for all folds
        open_loop_df = pd.DataFrame(open_loop_records)
        save_df_to_csv(open_loop_df, dirname=dirname, filename="open_loop_trajectory_metrics_all_folds")

        # Optional: Aggregate mean metrics across validation folds
        val_open_loop_summary = open_loop_df[open_loop_df["split"] == "val"].groupby("fold").mean(numeric_only=True)
        save_df_to_csv(val_open_loop_summary, dirname=dirname, filename="open_loop_val_metrics_summary")

        # --- 4. PLOT LOSS CURVES ---
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

        # --- PLOT VALIDATION PREDICTION SAMPLES ---
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
                dirname=pred_curves_dir,
                filename=f"val_prediction_sample_u{ch+1}_0"
            )
            
    
    # --- METRIC AGGREGATION ACROSS ALL FOLDS ---
    fold_best_train_losses = []
    fold_best_val_losses = []

    for f in range(k_folds):
        best_val_epoch_idx = np.argmin(fold_histories[f]["val_loss"])
        fold_best_val_losses.append(fold_histories[f]["val_loss"][best_val_epoch_idx])
        fold_best_train_losses.append(fold_histories[f]["train_loss"][best_val_epoch_idx])

    avg_best_train_loss = float(np.mean(fold_best_train_losses))
    avg_best_val_loss = float(np.mean(fold_best_val_losses))
    mean_cv_loss = avg_best_val_loss

    if run_simulation and all_fold_sim_metrics:
        sim_df = pd.DataFrame(all_fold_sim_metrics)
        avg_closed_loop_mse = float(sim_df["total_mse"].mean())
    else:
        avg_closed_loop_mse = np.nan

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
            "closed_loop_mse": avg_closed_loop_mse,
        },
        mean_cv_loss,
        summary_results_df,
    )

#sara

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
    k_folds = train_cfg["k_folds"]
    delay_steps = train_cfg["delay_steps"]

    batch_size = 64
    val_patience = train_cfg["val_patience_epochs"]
    min_delta = train_cfg["val_min_delta"]

    # --- MIMO-SPECIFIC CONFIG ---
    
    input_dim = train_cfg["input_dim"]    # Number of plant outputs (y1, y2)
    output_dim = train_cfg["output_dim"]  # Number of control inputs (u1, u2)

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