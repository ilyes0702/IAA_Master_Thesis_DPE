# Import standard libraries
import copy
import os
import pickle

from src.sample.utils.data_generation_utils import TorchDiffeqPlantWrapper
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.sample.config import *
from src.sample.decorators.general_decorators import *
from src.sample.utils.loss_utils import *
from src.sample.utils.plotting_utils import plot_signals
from src.sample.utils.saving_utils import *

plt.style.use("src/sample/style.mplstyle")

def train_controller_siso_old(
    model, 
    X_raw,          # Pass RAW un-normalized arrays now! Shape: [Total_Seqs, Seq_Len, 2]
    Y_raw,          # Pass RAW un-normalized targets!    Shape: [Total_Seqs, Seq_Len, 1]
    hyperparam_config, 
    dirname="name_directory",  
    show_plots=False  
):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]
    k_folds = train_cfg["k_folds"]

    input_dim = hyperparam_config["mamba"]["input_dim"]
    output_dim = hyperparam_config["mamba"]["output_dim"]
    batch_size = 64
    val_patience = train_cfg.get("val_patience_epochs", 3) 
    min_delta = train_cfg.get("val_min_delta", 0)         

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
        
        loss_name = train_cfg["loss_function"].replace("()", "")
        criterion = getattr(nn, loss_name)()
        
        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)
        
        # Isolate raw splits for this specific fold
        train_x_raw, val_x_raw = X_raw[train_idx_arr], X_raw[val_idx_arr]
        train_y_raw, val_y_raw = Y_raw[train_idx_arr], Y_raw[val_idx_arr]

        # --- ⚡ INTERNAL FOLD STANDARD SCALING (NO LEAKAGE) ---
        print(f"⚖️ Fitting independent StandardScalers for Fold {fold + 1}...")
        
        # 1. Reshape 3D sequential data to 2D matrices for scikit-learn
        N_train, seq_len, dim_x = train_x_raw.shape
        dim_y = train_y_raw.shape[-1]
        
        train_x_flat = train_x_raw.reshape(-1, dim_x)
        train_y_flat = train_y_raw.reshape(-1, dim_y)
        
        # 2. Instantiate and Fit exclusively on Training splits
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        scaler_x.fit(train_x_flat)
        scaler_y.fit(train_y_flat)
        
        # 3. Transform Training arrays and rebuild 3D Tensors
        train_x = torch.tensor(scaler_x.transform(train_x_flat).reshape(N_train, seq_len, dim_x), dtype=torch.float32)
        train_y = torch.tensor(scaler_y.transform(train_y_flat).reshape(N_train, seq_len, dim_y), dtype=torch.float32)
        
        # 4. Transform Validation arrays using the FROZEN training parameters
        N_val = val_x_raw.shape[0]
        val_x_flat = val_x_raw.reshape(-1, dim_x)
        val_y_flat = val_y_raw.reshape(-1, dim_y)
        
        val_x = torch.tensor(scaler_x.transform(val_x_flat).reshape(N_val, seq_len, dim_x), dtype=torch.float32)
        val_y = torch.tensor(scaler_y.transform(val_y_flat).reshape(N_val, seq_len, dim_y), dtype=torch.float32)
        
        # --- 💾 SAVE SCALER OBJECTS FOR THIS FOLD ---
        fold_dir = f"{dirname}/fold_{fold+1}"
        
        # --- 💾 SAVE SCALER OBJECTS FOR THIS FOLD ---
        # This isolates each fold into its own sub-directory inside the scalers block

        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")
        print(f"💾 Scalers saved to disk at: {fold_dir}/scaler_*.pkl")

        

        train_size = train_x.shape[0]
        val_size = val_x.shape[0]
        
        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        fold_val_epoch_history = []
        fold_train_epoch_history = []  
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False

        for epoch in range(epochs):
            model.train()
            epoch_train_loss_accum = 0.0
            print(f"\n🎬 Fold {fold+1} | Starting Epoch {epoch+1}/{epochs}")
            
            shuffled_train_indices = torch.randperm(train_size)
            
            # --- 1. TRUE MINI-BATCH TRAINING PASS ---
            for i in range(0, train_size, batch_size):
                batch_indices = shuffled_train_indices[i : i + batch_size]
                current_batch_size = len(batch_indices)
                
                batch_x = train_x[batch_indices].to(device)
                batch_y = train_y[batch_indices].to(device)
                
                y_t_split = batch_x[:, :, 0:1]
                y_next_split = batch_x[:, :, 1:2]
                
                if hasattr(model, 'reset_memory'):
                    model.reset_memory(batch_size=current_batch_size, device=device)
                
                optimizer.zero_grad()
                u_pred_batch = model(y_t_split, y_next_split)
                
                loss = criterion(u_pred_batch, batch_y)
                loss.backward()
                optimizer.step()
                
                current_loss_val = loss.item()
                epoch_train_loss_accum += current_loss_val * current_batch_size
                
                fold_train_batch_loss.append(current_loss_val)
                fold_train_batch_indices.append(global_batch_counter)
                global_batch_counter += 1

                if i == 0:
                    u_p_np = u_pred_batch[0].detach().cpu().numpy().flatten()
                    u_t_np = batch_y[0].detach().cpu().numpy().flatten()
                    t_axis = np.arange(len(u_p_np)) * dt

                    comparison_df = pd.DataFrame({
                        "t": t_axis, "u_train": u_t_np, "u_pred": u_p_np, "abs_error": np.abs(u_t_np - u_p_np)
                    })
                    pred_dir = f"{fold_dir}/predictions/epoch_{epoch+1}"
                    save_df_to_csv(comparison_df, dirname=pred_dir, filename=f"batch_step_{global_batch_counter}_sample_preds")

                    # if show_plots:
                    #     plot_signals(
                    #         t=t_axis,
                    #         signals=[u_t_np, u_p_np],
                    #         labels=["Target (u_train)", "Predicted (u_pred)"],
                    #         xlabel="Time (s)", ylabel="Signal Value",
                    #         title=f"Fold {fold+1} | Epoch {epoch+1} | Train Step {i} (Batch {global_batch_counter})",
                    #         dirname=pred_dir,
                    #         filename=f"train_step_{i}_batch_{global_batch_counter}_plot"
                    #     )

            scheduler.step()

            # --- 2. VALIDATION PASS (BATCHED CONTEXT) ---
            model.eval()
            epoch_val_loss_accum = 0.0
            
            with torch.no_grad():
                for i in range(0, val_size, batch_size):
                    batch_val_x = val_x[i : i + batch_size].to(device)
                    batch_val_y = val_y[i : i + batch_size].to(device)
                    current_val_batch_size = len(batch_val_x)
                    
                    y_val_t_split = batch_val_x[:, :, 0:1]   
                    y_val_next_split = batch_val_x[:, :, 1:2]
                    
                    if hasattr(model, 'reset_memory'):
                        model.reset_memory(batch_size=current_val_batch_size, device=device)
                    
                    u_val_pred = model(y_val_t_split, y_val_next_split)
                    val_loss = criterion(u_val_pred, batch_val_y)
                    
                    epoch_val_loss_accum += val_loss.item() * current_val_batch_size

                    # --- 📈 2b. GRANULAR PREDICTION LOGGING FOR VALIDATION SAMPLES ---
                    if show_plots:
                        u_v_p_np = u_val_pred[0].detach().cpu().numpy().flatten()
                        u_v_t_np = batch_val_y[0].detach().cpu().numpy().flatten()
                        t_val_axis = np.arange(len(u_v_p_np)) * dt
                        val_fold_dir = f"{fold_dir}/validation/epoch_{epoch+1}"
                        
                        # if show_plots:
                        #     plot_signals(
                        #         t=t_val_axis,
                        #         signals=[u_v_t_np, u_v_p_np],
                        #         labels=["Target (u_val)", "Predicted (u_val_pred)"],
                        #         xlabel="Time (s)", ylabel="Signal Value",
                        #         title=f"Fold {fold+1} | Epoch {epoch+1} | Val Step {i}",
                        #         dirname=val_fold_dir,
                        #         filename=f"val_step_{i}_plot"
                        #     )
            
            mean_train_loss = epoch_train_loss_accum / train_size
            mean_val_loss = (epoch_val_loss_accum / val_size) if val_size > 0 else 0.0
            
            fold_val_epoch_history.append(mean_val_loss)
            fold_train_epoch_history.append(mean_train_loss)  

            # --- FIX: Populate fold_histories so it exists when plotting ---
            if fold not in fold_histories:
                fold_histories[fold] = {
                    "train_loss": [],
                    "val_loss": [],
                    "val_epochs": []
                }
            
            fold_histories[fold]["train_loss"].append(mean_train_loss)
            fold_histories[fold]["val_loss"].append(mean_val_loss)
            fold_histories[fold]["val_epochs"].append(epoch + 1) # Tracks the epoch numbers (1, 2, 3...)
            
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"✨ [Fold {fold+1}] Epoch {epoch+1} Summary:")
            print(f"   ↳ LR: {current_lr:.6e} | Avg Train Loss: {mean_train_loss:.6f} | Avg Val Loss: {mean_val_loss:.6f}")

            # --- 3. EVALUATE VALIDATION EARLY STOPPING ---
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
                    
        # --- 📉 4. PLOT INDIVIDUAL FOLD LOSS CURVES (EXECUTED PER FOLD) ---
        fold_title_suffix = " (Early Stopped)" if early_stopped else " (Full Run)"
        
        # Granular Batch Iteration Curve
        plot_signals(
            t=np.array(fold_train_batch_indices),
            signals=[np.array(fold_train_batch_loss)],
            labels=[f"Fold {fold+1} Batch Loss"],
            xlabel="Total Training Batch Optimization Iterations",
            ylabel="Loss Value",
            title=f"Fold {fold+1} Granular Step Loss{fold_title_suffix}",
            dirname=fold_dir, filename="granular_training_loss"
        )
        
        # Epoch-level Summary Tracking Curve
        plot_signals(
            t=np.array(fold_histories[fold]["val_epochs"]),
            signals=[
                np.array(fold_train_epoch_history), 
                np.array(fold_val_epoch_history)
            ],
            labels=[
                f"Fold {fold+1} Avg Train Loss", 
                f"Fold {fold+1} Avg Val Loss"
            ],
            xlabel="Epochs", ylabel="Loss Value",
            title=f"Fold {fold+1} Epoch-level Loss Progress",
            dirname=fold_dir, filename="epoch_validation_loss"
        )

    # --- FINAL CROSS VALIDATION LOGGING SUMMARY ---
    print("\n💾 Complete Cross-Validation run finished. Packing overarching metadata curves...")
    
    summary_records = []
    for f in fold_histories:
        final_best = min(fold_histories[f]["val_loss"])
        summary_records.append({"fold": f+1, "best_recorded_val_loss": final_best})
    
    summary_df = pd.DataFrame(summary_records)
    save_df_to_csv(summary_df, dirname=dirname, filename="kfold_cross_validation_summary")
    
    print("\n✅ K-Fold optimization execution finalized.")
    return fold_histories


import os
import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Assuming your save utilities are imported correctly
# from src.sample.utils.saving_utils import save_scaler_object, save_model, save_df_to_csv
# from src.sample.utils.plotting_utils import plot_signals

def train_controller(
    model,
    X_raw,          # Clean Shape: [Total_Seqs, Seq_Len, input_dim * 2] (y_t and y_next)
    Y_raw,          # Clean Shape: [Total_Seqs, Seq_Len, output_dim]
    hyperparam_config,
    plant,
    dirname="name_directory",
    show_plots=False
):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]
    k_folds = train_cfg.get("k_folds", 5)

    batch_size = 64
    val_patience = train_cfg.get("val_patience_epochs", 3)
    min_delta = train_cfg.get("val_min_delta", 0.001)

    # --- MIMO-SPECIFIC CONFIG ---
    mamba_cfg = hyperparam_config.get("mamba", {})
    input_dim = mamba_cfg.get("input_dim", 2)    # Number of plant outputs (y1, y2)
    output_dim = mamba_cfg.get("output_dim", 2)  # Number of control inputs (u1, u2)

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
            
            # 🟢 CHANGED: Storage to accumulate ALL predictions/trues for tracking at the final epoch
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

                    # 🟢 CHANGED: Collect all batch items into CPU numpy arrays
                    all_val_preds.append(u_val_pred.cpu().numpy())
                    all_val_trues.append(batch_val_y.cpu().numpy())

            # 🟢 NEW: Concatenate list of batches into complete matrices of shape [N_val, Seq_Len, Dim]
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

        # Plot global overarching baseline losses
        plot_signals(t=np.array(fold_train_batch_indices), signals=[np.array(fold_train_batch_loss)],
                     labels=[f"Fold {fold+1} Total Loss"], xlabel="Optimization Steps", ylabel="Loss",
                     title=f"Fold {fold+1} Total Step Loss{fold_title_suffix}", dirname=fold_dir, filename="granular_training_loss")

        plot_signals(t=epoch_axis, signals=[np.array(fold_train_epoch_history), np.array(fold_val_epoch_history)],
                     labels=["Avg Train Loss", "Avg Val Loss"], xlabel="Epochs", ylabel="Loss",
                     title=f"Fold {fold+1} Total Epoch Loss Progress", dirname=fold_dir, filename="epoch_validation_loss")

       # --- 5. PLANT SIMULATION ROLLOUT FOR ALL VALIDATION SEQUENCES ---
        if 'val_all_preds_arr' in locals() and len(val_all_preds_arr) > 0:
            print(f"📊 Simulating plant dynamics across ALL ({val_size}) validation profiles for Fold {fold + 1}...")
            
            t_axis_val = np.arange(seq_len) * dt
            pred_curves_dir = f"{fold_dir}/validation_tracking_curves"
            
            # Handle plant class vs instance instantiation
            if isinstance(plant, type):
                plant_instance = plant(hyperparam_config)
            else:
                plant_instance = plant

            # Loop through every single validation sequence index
            for seq_idx in range(val_size):
                
                # 1. Isolate and unscale control signals for this specific sequence
                seq_pred_scaled = val_all_preds_arr[seq_idx] # [Seq_Len, output_dim]
                seq_true_scaled = val_all_trues_arr[seq_idx] # [Seq_Len, output_dim]
                
                seq_pred_unscaled = scaler_y.inverse_transform(seq_pred_scaled)
                seq_true_unscaled = scaler_y.inverse_transform(seq_true_scaled)
                
                # Isolate and unscale inputs/targets (y_t, y_next) for this specific sequence
                seq_x_scaled = val_x[seq_idx].cpu().numpy()
                seq_x_unscaled = scaler_x.inverse_transform(seq_x_scaled)
                
                # 2. Reset the plant to its initial state for the new rollout sequence
                current_sim_state = plant_instance.get_initial_state(batch_size=1)
                state_dim = current_sim_state.shape[-1]
                
                simulated_states_history = {st: [] for st in range(state_dim)}
                simulated_outputs_history = {out: [] for out in range(input_dim)}

                # --- 3. Step-by-step physical integration rollout loop ---
                # Wrap your plant instance dynamically for this fold
                gpu_solver = TorchDiffeqPlantWrapper(plant_instance, hyperparam_config).to(device)
                
                for step in range(seq_len):
                    for st in range(state_dim):
                        simulated_states_history[st].append(current_sim_state[0, st].item())
                    
                    u_pred_step = torch.from_numpy(seq_pred_unscaled[step:step+1]).to(device=device, dtype=torch.float32)
                    t_start = t_axis_val[step]
                    t_end = t_start + dt
                    
                    # Construct the micro-time interval for torchdiffeq
                    t_span = torch.tensor([t_start, t_end], device=device, dtype=torch.float32)
                    
                    # Update the current control input in the wrapper
                    gpu_solver.current_u = u_pred_step
                    
                    # 🟢 RUN TORCHDIFFEQ INSTEAD OF plant_instance.step
                    solution = odeint(gpu_solver, current_sim_state, t_span, method='dopri5', rtol=1e-5, atol=1e-7)
                    
                    # Extract final state and apply dynamic boundary clamping
                    current_sim_state = torch.clamp(
                        solution[1].detach(),
                        min=gpu_solver.state_min_bounds,
                        max=gpu_solver.state_max_bounds
                    )
                    
                    # Calculate output tracking (y_next) using your plant's native output equation
                    y_next_pred = plant_instance.get_y(current_sim_state, t_end)
                    
                    for out in range(input_dim):
                        simulated_outputs_history[out].append(y_next_pred[0, out].item())

                # 4. Construct log dictionary dynamically for this specific sequence item
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
                    
                # 5. Convert to DataFrame and save with a unique identifier per sequence
                val_profile_df = pd.DataFrame(log_data)
                save_df_to_csv(
                    val_profile_df, 
                    dirname=pred_curves_dir, 
                    filename=f"val_plant_simulation_fold_{fold+1}_seq_{seq_idx+1}"
                )
                
                # Optional: Only plot the very first sequence to keep folder clean of thousands of images
                if seq_idx == 0:
                    for ch in range(output_dim):
                        plot_signals(
                            t=t_axis_val, signals=[seq_true_unscaled[:, ch], seq_pred_unscaled[:, ch]],
                            labels=[f"Actual u_{ch+1}", f"Predicted u_{ch+1}"],
                            xlabel="Time (s)", ylabel="Physical Units",
                            title=f"Fold {fold+1} | Control Input Profile u_{ch+1} (Seq 1)",
                            dirname=pred_curves_dir, filename=f"val_tracking_u{ch+1}_sample"
                        )
                        
            print(f"✅ All {val_size} validation trajectory logs safely dumped to CSV files for Fold {fold + 1}.")

    # --- FINAL CROSS VALIDATION SUMMARY ---
    print("\n💾 Complete Cross-Validation run finished. Packing overarching metadata curves...")
    summary_records = []
    for f in fold_histories:
        final_best = min(fold_histories[f]["val_loss"])
        record = {"fold": f+1, "best_recorded_total_val_loss": final_best}
        for ch in range(output_dim):
            # Find channel loss at the epoch where total validation loss was minimized
            best_epoch_idx = np.argmin(fold_histories[f]["val_loss"])
            record[f"best_val_loss_u{ch+1}"] = fold_histories[f][f"val_loss_ch{ch+1}"][best_epoch_idx]
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)
    save_df_to_csv(summary_df, dirname=dirname, filename="kfold_cross_validation_summary")
    print("\n✅ K-Fold optimization execution finalized.")
    return fold_histories


def train_controller_avg_loss(
    model,
    X_raw,          # Clean Shape: [Total_Seqs, Seq_Len, input_dim * 2] (y_t and y_next)
    Y_raw,          # Clean Shape: [Total_Seqs, Seq_Len, output_dim]
    hyperparam_config,
    dirname="name_directory",
    show_plots=False
):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]
    k_folds = train_cfg.get("k_folds", 5)

    batch_size = 64
    val_patience = train_cfg.get("val_patience_epochs", 3)
    min_delta = train_cfg.get("val_min_delta", 0)

    # --- MIMO-SPECIFIC CONFIG ---
    mamba_cfg = hyperparam_config.get("mamba", {})
    input_dim = mamba_cfg.get("input_dim", 2)    # Number of plant outputs (y1, y2)
    output_dim = mamba_cfg.get("output_dim", 2)  # Number of control inputs (u1, u2)

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

        loss_name = train_cfg["loss_function"].replace("()", "")
        criterion = getattr(nn, loss_name)()

        val_idx_arr = folds[fold]
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)

        # Isolate raw splits for this specific fold
        train_x_raw, val_x_raw = X_raw[train_idx_arr], X_raw[val_idx_arr]
        train_y_raw, val_y_raw = Y_raw[train_idx_arr], Y_raw[val_idx_arr]

        # --- ⚡ INTERNAL FOLD STANDARD SCALING (NO LEAKAGE) ---
        print(f"⚖️ Fitting independent StandardScalers for Fold {fold + 1}...")

        N_train, seq_len, dim_x = train_x_raw.shape
        dim_y = train_y_raw.shape[-1]

        train_x_flat = train_x_raw.reshape(-1, dim_x)
        train_y_flat = train_y_raw.reshape(-1, dim_y)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaler_x.fit(train_x_flat)
        scaler_y.fit(train_y_flat)

        # Transform arrays and rebuild clean 3D Tensors
        train_x = torch.tensor(scaler_x.transform(train_x_flat).reshape(N_train, seq_len, dim_x), dtype=torch.float32)
        train_y = torch.tensor(scaler_y.transform(train_y_flat).reshape(N_train, seq_len, dim_y), dtype=torch.float32)

        N_val = val_x_raw.shape[0]
        val_x_flat = val_x_raw.reshape(-1, dim_x)
        val_y_flat = val_y_raw.reshape(-1, dim_y)

        val_x = torch.tensor(scaler_x.transform(val_x_flat).reshape(N_val, seq_len, dim_x), dtype=torch.float32)
        val_y = torch.tensor(scaler_y.transform(val_y_flat).reshape(N_val, seq_len, dim_y), dtype=torch.float32)

        # --- 💾 SAVE SCALER OBJECTS FOR THIS FOLD ---
        fold_dir = f"{dirname}/fold_{fold+1}"

        save_scaler_object(scaler_x, dirname=fold_dir, filename="scaler_x")
        save_scaler_object(scaler_y, dirname=fold_dir, filename="scaler_y")
        print(f"💾 Scalers saved to disk at: {fold_dir}/scaler_*.pkl")

        # --- 📈 NEW CODE: PLOT TRANSFORMED TRAINING CURVES ---
        print(f"📊 Plotting scaled data signals for Fold {fold + 1}...")
        
        # Extract the first sequence from the training partition [Seq_Len, Dimension]
        sample_x = train_x[0].numpy()
        sample_y = train_y[0].numpy()
        t_axis = np.arange(seq_len) * dt
        
        # Setup directories for this specific fold's data curves
        curves_dir = f"{fold_dir}/transformed_data_curves"

        # 1. Plot Control Inputs (u1, u2)
        for out_idx in range(output_dim):
            plot_signals(
                t=t_axis,
                signals=[sample_y[:, out_idx]],
                labels=[f"Scaled u_{out_idx+1}"],
                xlabel="Time (s)",
                ylabel="Standardized Units",
                title=f"Fold {fold+1} | Transformed Control Input u_{out_idx+1}",
                dirname=curves_dir,
                filename=f"scaled_u{out_idx+1}_curve"
            )

        # 2. Plot Plant Outputs and Targets (y1, y1_next, y2, y2_next, ...)
        # Inside train_x, the features are ordered as: [y1, y2, ..., y1_next, y2_next, ...]
        for in_idx in range(input_dim):
            y_t_signal = sample_x[:, in_idx]
            y_next_signal = sample_x[:, input_dim + in_idx]
            
            plot_signals(
                t=t_axis,
                signals=[y_t_signal, y_next_signal],
                labels=[f"Scaled y_{in_idx+1}_t", f"Scaled y_{in_idx+1}_next"],
                xlabel="Time (s)",
                ylabel="Standardized Units",
                title=f"Fold {fold+1} | Transformed Plant State & Target (Channel {in_idx+1})",
                dirname=curves_dir,
                filename=f"scaled_y{in_idx+1}_and_next_curve"
            )
        # --- 📈 END OF NEW CODE ---

        train_size = train_x.shape[0]
        val_size = val_x.shape[0]

        global_batch_counter = 0
        fold_train_batch_loss = []
        fold_train_batch_indices = []
        fold_val_epoch_history = []
        fold_train_epoch_history = []

        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False

        for epoch in range(epochs):
            model.train()
            epoch_train_loss_accum = 0.0
            print(f"\n🎬 Fold {fold+1} | Starting Epoch {epoch+1}/{epochs}")

            shuffled_train_indices = torch.randperm(train_size)

            # --- 1. MINI-BATCH TRAINING PASS ---
            for i in range(0, train_size, batch_size):
                batch_indices = shuffled_train_indices[i : i + batch_size]
                current_batch_size = len(batch_indices)

                batch_x = train_x[batch_indices].to(device)
                batch_y = train_y[batch_indices].to(device)

                # --- Clean Slicing: Only extract current and next values ---
                y_t_split = batch_x[:, :, :input_dim]
                y_next_split = batch_x[:, :, input_dim:] # Takes exactly the remaining input_dim columns

                if hasattr(model, 'reset_memory'):
                    model.reset_memory(batch_size=current_batch_size, device=device)

                optimizer.zero_grad()
                u_pred_batch = model(y_t_split, y_next_split)

                loss = criterion(u_pred_batch, batch_y)
                loss.backward()
                optimizer.step()

                current_loss_val = loss.item()
                epoch_train_loss_accum += current_loss_val * current_batch_size

                fold_train_batch_loss.append(current_loss_val)
                fold_train_batch_indices.append(global_batch_counter)
                global_batch_counter += 1

                if i == 0 and show_plots:
                    u_p_np = u_pred_batch[0].detach().cpu().numpy()
                    u_t_np = batch_y[0].detach().cpu().numpy()
                    t_axis = np.arange(u_p_np.shape[0]) * dt

                    for out_dim in range(output_dim):
                        plot_signals(
                            t=t_axis,
                            signals=[u_t_np[:, out_dim], u_p_np[:, out_dim]],
                            labels=[f"Target (u_{out_dim+1})", f"Predicted (u_{out_dim+1})"],
                            xlabel="Time (s)",
                            ylabel="Signal Value",
                            title=f"Fold {fold+1} | Epoch {epoch+1} | Train Step {i} | Output {out_dim+1}",
                            dirname=f"{fold_dir}/predictions/epoch_{epoch+1}",
                            filename=f"train_step_{i}_output_{out_dim+1}_plot"
                        )

            scheduler.step()

            # --- 2. VALIDATION PASS ---
            model.eval()
            epoch_val_loss_accum = 0.0
            epoch_val_channel_accum = {ch: 0.0 for ch in range(output_dim)}

            # 🟢 FIX/CHANGED: Initialize lists to store ALL sequence profiles across all batches
            all_val_preds = []
            all_val_trues = []

            with torch.no_grad():
                for i in range(0, val_size, batch_size):
                    batch_val_x = val_x[i : i + batch_size].to(device)
                    batch_val_y = val_y[i : i + batch_size].to(device)
                    current_val_batch_size = len(batch_val_x)

                    # --- Clean Slicing matching train pass ---
                    y_val_t_split = batch_val_x[:, :, :input_dim]
                    y_val_next_split = batch_val_x[:, :, input_dim:]

                    if hasattr(model, 'reset_memory'):
                        model.reset_memory(batch_size=current_val_batch_size, device=device)

                    u_val_pred = model(y_val_t_split, y_val_next_split)
                    
                    # 🟢 FIX: Keep the output element-wise ('none') reduction active for per-channel logs
                    val_loss = criterion(u_val_pred, batch_val_y)

                    # Aggregate total validation scalar loss
                    epoch_val_loss_accum += val_loss.mean().item() * current_val_batch_size

                    for ch in range(output_dim):
                        # 🟢 FIX: Changed 'raw_val_loss' to 'val_loss' to prevent NameError
                        ch_val_loss_val = val_loss[:, :, ch].mean().item()
                        epoch_val_channel_accum[ch] += ch_val_loss_val * current_val_batch_size

                    # 🟢 NEW: Collect tracking targets and predictions into CPU storage arrays 
                    all_val_preds.append(u_val_pred.cpu().numpy())
                    all_val_trues.append(batch_val_y.cpu().numpy())

                    if i == 0 and show_plots:
                        u_v_p_np = u_val_pred[0].detach().cpu().numpy()
                        u_v_t_np = batch_val_y[0].detach().cpu().numpy()
                        t_val_axis = np.arange(u_v_p_np.shape[0]) * dt

                        for out_dim in range(output_dim):
                            plot_signals(
                                t=t_val_axis,
                                signals=[u_v_t_np[:, out_dim], u_v_p_np[:, out_dim]],
                                labels=[f"Target (u_{out_dim+1})", f"Predicted (u_{out_dim+1})"],
                                xlabel="Time (s)",
                                ylabel="Signal Value",
                                title=f"Fold {fold+1} | Epoch {epoch+1} | Val Step {i} | Output {out_dim+1}",
                                dirname=f"{fold_dir}/validation/epoch_{epoch+1}",
                                filename=f"val_step_{i}_output_{out_dim+1}_plot"
                            )

            # 🟢 NEW: Post-processing batch stack to form evaluation arrays of shape [val_size, Seq_Len, output_dim]
            val_all_preds_arr = np.concatenate(all_val_preds, axis=0)
            val_all_trues_arr = np.concatenate(all_val_trues, axis=0)

            mean_train_loss = epoch_train_loss_accum / train_size
            mean_val_loss = (epoch_val_loss_accum / val_size) if val_size > 0 else 0.0

            fold_val_epoch_history.append(mean_val_loss)
            fold_train_epoch_history.append(mean_train_loss)

            if fold not in fold_histories:
                fold_histories[fold] = {
                    "train_loss": [],
                    "val_loss": [],
                    "val_epochs": []
                }

            fold_histories[fold]["train_loss"].append(mean_train_loss)
            fold_histories[fold]["val_loss"].append(mean_val_loss)
            fold_histories[fold]["val_epochs"].append(epoch + 1)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"✨ [Fold {fold+1}] Epoch {epoch+1} Summary:")
            print(f"   ↳ LR: {current_lr:.6e} | Avg Train Loss: {mean_train_loss:.6f} | Avg Val Loss: {mean_val_loss:.6f}")

            # --- 3. EVALUATE VALIDATION EARLY STOPPING ---
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

        # --- 4. PLOT LOSS CURVES ---
        fold_title_suffix = " (Early Stopped)" if early_stopped else " (Full Run)"

        plot_signals(
            t=np.array(fold_train_batch_indices),
            signals=[np.array(fold_train_batch_loss)],
            labels=[f"Fold {fold+1} Batch Loss"],
            xlabel="Total Training Batch Optimization Iterations",
            ylabel="Loss Value",
            title=f"Fold {fold+1} Granular Step Loss{fold_title_suffix}",
            dirname=fold_dir,
            filename="granular_training_loss"
        )

        plot_signals(
            t=np.array(fold_histories[fold]["val_epochs"]),
            signals=[np.array(fold_train_epoch_history), np.array(fold_val_epoch_history)],
            labels=[f"Fold {fold+1} Avg Train Loss", f"Fold {fold+1} Avg Val Loss"],
            xlabel="Epochs",
            ylabel="Loss Value",
            title=f"Fold {fold+1} Epoch-level Loss Progress",
            dirname=fold_dir,
            filename="epoch_validation_loss"
        )

    # --- FINAL CROSS VALIDATION LOGGING SUMMARY ---
    print("\n💾 Complete Cross-Validation run finished. Packing overarching metadata curves...")

    summary_records = []
    for f in fold_histories:
        final_best = min(fold_histories[f]["val_loss"])
        summary_records.append({"fold": f+1, "best_recorded_val_loss": final_best})

    summary_df = pd.DataFrame(summary_records)
    save_df_to_csv(summary_df, dirname=dirname, filename="kfold_cross_validation_summary")

    print("\n✅ K-Fold optimization execution finalized.")
    return fold_histories