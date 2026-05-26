# Import standard libraries
import numpy as np
import pandas as pd

# import machine learning modules
from src.sample.utils.loss_utils import *
from src.sample.decorators.general_decorators import *
from src.sample.utils.saving_utils import *
from src.sample.config import *
from src.sample.utils.plotting_utils import plot_signals
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use("src/sample/style.mplstyle")

@track_resources
def train_controller(model, dataset_path, hyperparam_config, dirname="name_directory", num_sequences_to_use=None):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]

    # --- INITIALIZE ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #criterion = relative_huber_loss
    criterion = nn.MSELoss()  # You can switch to your custom loss if needed
    sequence_loss_history = []
    
    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=True) 
    batches_x = dataset["x"]
    batches_y = dataset["y"]

    # --- DATA SLICING LOGIC ---
    # Concatenate all batches into one large tensor: [Total_Seqs, Seq_Len, Features]
    full_x = torch.cat(dataset["x"], dim=0)
    full_y = torch.cat(dataset["y"], dim=0)
    # If num_sequences_to_use is provided, slice the data
    if num_sequences_to_use is not None:
        full_x = full_x[:num_sequences_to_use]
        full_y = full_y[:num_sequences_to_use]
        print(f"✂️ Sliced dataset to the first {num_sequences_to_use} sequences.")
    else:
        print(f"✅ Using full dataset ({full_x.shape[0]} sequences).")

    model.to(device)

    for epoch in range(epochs):
        for b_idx, (x_batch, y_batch) in enumerate(zip(batches_x, batches_y)):
            # Move batch to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.shape[0]

            model.train()

            for s_idx in range(full_x.shape[0]):
                optimizer.zero_grad()
                
                u_train_single = full_y[s_idx:s_idx+1, :, :].to(device)
                x_input_single = full_x[s_idx:s_idx+1, :, :].to(device)

                # 1. Prediction (Forward Pass)
                u_pred_single = model(x_batch[s_idx:s_idx+1, :, :])
                
                # 2. Optimization (Backward Pass)
                loss = criterion(u_pred_single, u_train_single)
                loss.backward()
                optimizer.step()
                
                sequence_loss_history.append(loss.item())

                # --- NEW: SAVE PREDICTIONS CSV ---
                

                # --- DATA PREPARATION for CSV and PLOT ---
                # detach() removes from graph, cpu() moves to RAM, numpy() converts, flatten() makes it 1D
                u_p_np = u_pred_single[0].detach().cpu().numpy().flatten()
                u_t_np = u_train_single[0].detach().cpu().numpy().flatten()
                t_axis = np.arange(len(u_p_np)) * dt

                # --- SAVE PREDICTIONS CSV ---
                comparison_df = pd.DataFrame({
                    "t": t_axis,
                    "u_train": u_t_np,
                    "u_pred": u_p_np,
                    "abs_error": np.abs(u_t_np - u_p_np)
                })

                # We use your custom save_df_to_csv function
                # This will land in: results/DATE/TIME/dirname/predictions/reports/
                csv_filename = f"epoch_{epoch}_batch_{b_idx}_seq_{s_idx}_preds"
                save_df_to_csv(comparison_df, 
                               dirname=f"{dirname}/predictions", 
                               filename=csv_filename)
                

                # plot_signals(
                #         t_axis, 
                #         [u_t_np, u_p_np], # Pass the flattened numpy arrays
                #         labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
                #         xlabel="Time", ylabel="Control Signal",
                #         title=f"Mamba Prediction Accuracy Epoch {epoch} Seq {s_idx}",
                #         dirname=dirname+"/sequences",
                #         filename=f"prediction_accuracy_epoch_{epoch}_batch_{b_idx}_seq_{s_idx}"
                #     )

                plot_signals(
                        t_axis, 
                        [u_t_np, u_p_np], # Pass the flattened numpy arrays
                        labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
                        xlabel="Time", ylabel="Control Signal",
                        title=f"Mamba Prediction Accuracy Epoch {epoch} Seq {s_idx}",
                        dirname=dirname+"/sequences",
                        filename=f"prediction_accuracy"
                    )

        print(f"🚀 Epoch {epoch+1}/{epochs} Finished | Final Seq Loss: {loss.item():.6f}")

    # Final model and loss plots
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="trained_controller_disk")
    
    # Save master loss history
    loss_df = pd.DataFrame({"sequence_index": range(len(sequence_loss_history)), "loss": sequence_loss_history})
    save_df_to_csv(loss_df, dirname=dirname, filename="total_sequence_loss")

    plot_signals(
        loss_df["sequence_index"].values, [loss_df["loss"].values],
        labels=["Hybrid Control Loss"], 
        xlabel="Total Sequences Trained", 
        ylabel="Loss",
        title="Learning Curve (Per Sequence) - From Disk",
        dirname=dirname, filename="sequence_loss_from_disk_plot"
    )

    return sequence_loss_history



#=== FUNCTION TO TRAIN CONTROLLER (INCREMENTAL SEQUENCE LEARNING) ===#
def train_controller_new(model, dataset_path, hyperparam_config, dirname="name_directory", num_sequences_to_use=None):
    """
    Trains the network by updating model weights incrementally after evaluating each individual sequence.

    Parameters:
    - model: The neural network controller model architecture to be trained.
    - dataset_path (str): The system file path locating the pre-generated serialized PyTorch training tensors.
    - hyperparam_config (dict): Configuration dictionary containing nested parameters for 'train' and 'signal'.
    - dirname (str): Base output directory pathway where performance logs and reports will be saved (default: "name_directory").
    - num_sequences_to_use (int, optional): Truncation bound to restrict training to a specific subset of sequences.

    Returns:
    - sequence_loss_history (list): A step-by-step historical tracker recording the loss value computed for every single sequence.

    The function loads a structural dataset and loops through training configurations. For every epoch, 
    it processes data patterns sequentially, running one isolated sequence through the model, computing its loss, 
    executing backpropagation to update the architecture's parameters immediately, and logging performance before 
    moving to the next individual sequence array.
    """
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]

    # --- INITIALIZE PIPELINES ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = mape_loss  # You can switch to your custom loss if needed
    sequence_loss_history = []
    
    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=True) 

    # --- DATA CONCATENATION & SLICING ---
    # Merge nested batches into single sequential matrix blocks: [Total_Sequences, Seq_Len, Features]
    full_x = torch.cat(dataset["x"], dim=0)
    full_y = torch.cat(dataset["y"], dim=0)
    
    if num_sequences_to_use is not None:
        full_x = full_x[:num_sequences_to_use]
        full_y = full_y[:num_sequences_to_use]
        print(f"✂️ Sliced dataset to the first {num_sequences_to_use} sequences.")
    else:
        print(f"✅ Using full dataset ({full_x.shape[0]} sequences).")

    model.to(device)
    total_sequences = full_x.shape[0]

    # --- NESTED EPOCH & SEQUENCE RUN LOOPS ---
    for epoch in range(epochs):
        model.train()
        
        for s_idx in range(total_sequences):
            # Isolate exactly ONE tracking sequence and push it to the execution device
            x_input_single = full_x[s_idx:s_idx+1, :, :].to(device)  # Shape: [1, Seq_Len, Input_Dim]
            u_target_single = full_y[s_idx:s_idx+1, :, :].to(device) # Shape: [1, Seq_Len, Output_Dim]
            
            # Clear historical gradients from the previous sequence's step
            optimizer.zero_grad()
            
            # 1. Prediction (Forward Pass for this single sequence)
            u_pred_single = model(x_input_single)
            
            # 2. Optimization (Evaluate loss, backpropagate, and update weights instantly)
            loss = criterion(u_pred_single, u_target_single)
            loss.backward()
            optimizer.step()
            
            # Save the localized raw sequence loss value to track rapid transitions
            sequence_loss_history.append(loss.item())

            # --- DATA PREPARATION FOR CSV LOGGING & PLOT ---
            u_p_np = u_pred_single[0].detach().cpu().numpy().flatten()
            u_t_np = u_target_single[0].detach().cpu().numpy().flatten()
            t_axis = np.arange(len(u_p_np)) * dt

            comparison_df = pd.DataFrame({
                "t": t_axis,
                "u_train": u_t_np,
                "u_pred": u_p_np,
                "abs_error": np.abs(u_t_np - u_p_np)
            })

            # Save the evaluation dataframe log for this isolated sequence update step
            csv_filename = f"epoch_{epoch}_seq_{s_idx}_preds"
            save_df_to_csv(comparison_df, 
                           dirname=f"{dirname}/predictions", 
                           filename=csv_filename)
            
            # Periodically generate validation accuracy tracking plots to avoid storage bloat
            # if s_idx == 0 or s_idx == (total_sequences - 1):
            #     plot_signals(
            #         t_axis, 
            #         [u_t_np, u_p_np], 
            #         labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
            #         xlabel="Time", ylabel="Control Signal",
            #         title=f"Mamba Prediction Accuracy Epoch {epoch} Seq {s_idx}",
            #         dirname=dirname + "/sequences",
            #         filename=f"prediction_accuracy_epoch_{epoch}_seq_{s_idx}"
            #     )
            # plot_signals(
            #                 t_axis, 
            #                 [u_t_np, u_p_np], # Pass the flattened numpy arrays
            #                 labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
            #                 xlabel="Time", ylabel="Control Signal",
            #                 title=f"Mamba Prediction Accuracy Epoch {epoch} Seq {s_idx}",
            #                 dirname=dirname+"/sequences",
            #                 filename=f"prediction_accuracy"
            #             )

        print(f"🚀 Epoch {epoch+1}/{epochs} Complete | Last Evaluated Sequence Loss: {loss.item():.6f}")

    # --- FINAL DATASET EXPORTS ---
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="trained_controller_disk")
    
    loss_df = pd.DataFrame({"sequence_index": range(len(sequence_loss_history)), "loss": sequence_loss_history})
    save_df_to_csv(loss_df, dirname=dirname, filename="total_sequence_loss")

    plot_signals(
        loss_df["sequence_index"].values, [loss_df["loss"].values],
        labels=["Incremental Sequence Loss"], 
        xlabel="Total Sequences Evaluated", 
        ylabel="Loss Value",
        title="Learning Curve (Sequence-by-Sequence Progression)",
        dirname=dirname, filename="sequence_loss_from_disk_plot"
    )

    return sequence_loss_history


from collections import deque

#=== FUNCTION TO TRAIN CONTROLLER (INCREMENTAL SEQUENCE LEARNING WITH DECAY) ===#
def train_controller_lr_decay(
    model, 
    dataset_path, 
    hyperparam_config, 
    dirname="name_directory", 
    num_sequences_to_use=None,
    criterion_type="max"
    ):
    """
    Trains the network incrementally with an active early-stopping convergence criterion.

    Parameters:
    - critical_value (float): The loss value ceiling defining acceptable convergence.
    - patience_steps (int): Number of consecutive sequence evaluations required below critical_value to trigger stop.
    - criterion_type (str): Options: "max" (all steps in window must be below threshold) 
                            or "mean" (rolling average must be below threshold).
    """

    critical_value = hyperparam_config["train"]["critical_loss_value"]
    patience_steps = hyperparam_config["train"]["patience_steps"]
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]

    # --- INITIALIZE PIPELINES ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyperparam_config["train"]["lr_decay_rate"])
    criterion = eval(hyperparam_config["train"]["loss_function"])  # You can switch to your custom loss if needed
    sequence_loss_history = []
    
    # --- CONVERGENCE TRACKER ---
    # Deque automatically drops old values when new ones are added past maxlen
    rolling_window = deque(maxlen=patience_steps)
    converged = False
    
    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=True) 

    # --- DATA CONCATENATION & SLICING ---
    full_x = torch.cat(dataset["x"], dim=0)
    full_y = torch.cat(dataset["y"], dim=0)
    
    if num_sequences_to_use is not None:
        full_x = full_x[:num_sequences_to_use]
        full_y = full_y[:num_sequences_to_use]
    
    model.to(device)
    total_sequences = full_x.shape[0]

    # --- NESTED EPOCH & SEQUENCE RUN LOOPS ---
    for epoch in range(epochs):
        if converged:
            break
            
        model.train()
        
        for s_idx in range(total_sequences):
            x_input_single = full_x[s_idx:s_idx+1, :, :].to(device)  
            u_target_single = full_y[s_idx:s_idx+1, :, :].to(device) 
            
            optimizer.zero_grad()
            
            # 1. Prediction (Forward Pass)
            u_pred_single = model(x_input_single)
            
            # 2. Optimization (Backward Pass)
            loss = criterion(u_pred_single, u_target_single)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            current_loss = loss.item()
            sequence_loss_history.append(current_loss)

            
            
            # --- UPDATE CONVERGENCE WINDOW ---
            rolling_window.append(current_loss)
            
            # 3. EVALUATE EARLY STOPPING CONVERGENCE CRITERION
            if len(rolling_window) == patience_steps:
                if criterion_type == "max":
                    # Absolute strict: Is the WORST loss in our recent history below the threshold?
                    current_metric = max(rolling_window)
                else:
                    # Smooth trend: Is the AVERAGE loss of our recent history below the threshold?
                    current_metric = sum(rolling_window) / patience_steps
                
                if current_metric < critical_value:
                    print(f"\n🎯 CONVERGENCE CRITERION MET at Epoch {epoch+1}, Sequence {s_idx}!")
                    print(f"📉 Rolling {criterion_type} loss stayed below {critical_value} for {patience_steps} steps.")
                    converged = True
                    break # Break out of the sequence loop

            # --- DATA PREPARATION FOR CSV LOGGING ---
            u_p_np = u_pred_single[0].detach().cpu().numpy().flatten()
            u_t_np = u_target_single[0].detach().cpu().numpy().flatten()
            t_axis = np.arange(len(u_p_np)) * dt

            # plot_signals(
            #             t_axis, 
            #             [u_t_np, u_p_np], # Pass the flattened numpy arrays
            #             labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
            #             xlabel="Time", ylabel="Control Signal",
            #             title=f"Mamba Prediction Accuracy Epoch {epoch} Seq {s_idx}",
            #             dirname=dirname+"/sequences",
            #             filename=f"prediction_accuracy_seq_{s_idx}"
            #         )
            # plot_signals(
            #                 t_axis, 
            #                 [u_t_np, u_p_np], # Pass the flattened numpy arrays
            #                 labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
            #                 xlabel="Time", ylabel="Control Signal",
            #                 title=f"Mamba Prediction Accuracy Epoch {epoch} Seq {s_idx}",
            #                 dirname=dirname+"/sequences",
            #                 filename=f"prediction_accuracy"
            #             )

            comparison_df = pd.DataFrame({
                "t": t_axis, "u_train": u_t_np, "u_pred": u_p_np, "abs_error": np.abs(u_t_np - u_p_np)
            })
            csv_filename = f"epoch_{epoch}_seq_{s_idx}_preds"
            save_df_to_csv(comparison_df, dirname=f"{dirname}/predictions", filename=csv_filename)

            print("Sequence: {}/{} | Current Sequence Loss: {:.6f}".format(s_idx+1, total_sequences, current_loss))
                  

        current_lr = optimizer.param_groups[0]['lr']
        print(f"🚀 Epoch {epoch+1}/{epochs} Complete | LR: {current_lr:.6e} | Last Sequence Loss: {current_loss:.6f}")

    # --- FINAL DATASET EXPORTS (Saves regardless of whether we hit the limit or early-stopped) ---
    print("💾 Saving finalized model and performance records...")
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="trained_controller_disk")
    
    loss_df = pd.DataFrame({"sequence_index": range(len(sequence_loss_history)), "loss": sequence_loss_history})
    save_df_to_csv(loss_df, dirname=dirname, filename="total_sequence_loss")

    stop_type_title = " (Early Stopped)" if converged else " (Full Run Completed)"
    plot_signals(
        loss_df["sequence_index"].values, [loss_df["loss"].values],
        labels=["Incremental Sequence Loss"], 
        xlabel="Total Sequences Evaluated", 
        ylabel="Loss Value",
        title=f"Learning Curve{stop_type_title}",
        dirname=dirname, filename="sequence_loss_from_disk_plot"
    )

    return sequence_loss_history


from collections import deque
import torch
import numpy as np
import pandas as pd
import os

from collections import deque
import torch
import numpy as np
import pandas as pd
import os
import copy


import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy

def train_controller_kfold_exp(
    model, 
    dataset_path, 
    hyperparam_config, 
    dirname="name_directory", 
    num_sequences_to_use=None,
    k_folds=1,  
    show_plots=False  
):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]
    
    val_patience = train_cfg.get("val_patience_epochs", 3) 
    min_delta = train_cfg.get("val_min_delta", 0)        

    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=True) 

    # --- DATA CONCATENATION & SLICING ---
    raw_x = dataset["x"]  
    raw_y = dataset["y"]  
    
    # 🔥 NEW: Calculate global normalization stats
    all_y_values = []
    all_u_values = []
    for x_seq, y_seq in zip(raw_x, raw_y):
        y_raw = x_seq.squeeze()
        u_raw = y_seq.squeeze()
        while y_raw.ndim > 1: y_raw = y_raw[..., 0]
        while u_raw.ndim > 1: u_raw = u_raw[..., 0]
        all_y_values.append(y_raw)
        all_u_values.append(u_raw)

    all_y_tensor = torch.cat(all_y_values)
    all_u_tensor = torch.cat(all_u_values)
    if num_sequences_to_use is not None:
        raw_x = raw_x[:num_sequences_to_use]
        raw_y = raw_y[:num_sequences_to_use]
    norm_stats = {
        'y_mean': all_y_tensor.mean().item(),
        'y_std': all_y_tensor.std().item(),
        'u_mean': all_u_tensor.mean().item(),
        'u_std': all_u_tensor.std().item()
    }

    # Save for simulation
    save_to_json(norm_stats, dirname=dirname, filename="normalization_stats")
    
    delta_steps = hyperparam_config["train"]["delay_steps"] 
    
    processed_x_sequences = []
    processed_y_sequences = []
    
    
    
    for x_seq, y_seq in zip(raw_x, raw_y):
        y_raw = x_seq.squeeze()
        u_raw = y_seq.squeeze()

        # 🔥 NEW: Reduce to 1D by taking first element along ALL extra dimensions
        while y_raw.ndim > 1:
            y_raw = y_raw[..., 0]  # Repeatedly take first element along last dim
        while u_raw.ndim > 1:
            u_raw = u_raw[..., 0]

        max_valid_idx = len(y_raw) - delta_steps

        # 🔥 NEW: Normalize before slicing
        y_raw_norm = (y_raw - norm_stats['y_mean']) / (norm_stats['y_std'] + 1e-8)
        u_raw_norm = (u_raw - norm_stats['u_mean']) / (norm_stats['u_std'] + 1e-8)

        y_t = y_raw_norm[:max_valid_idx]
        y_t_delta = y_raw_norm[delta_steps : max_valid_idx + delta_steps]
        u_t = u_raw_norm[:max_valid_idx]

        # Now y_t and y_t_delta are guaranteed 1D → stack to (seq_len, 2)
        triplet_inputs = torch.stack([y_t, y_t_delta], dim=-1)
        processed_x_sequences.append(triplet_inputs)
        processed_y_sequences.append(u_t.unsqueeze(-1))

    full_x = torch.stack(processed_x_sequences, dim=0) 
    full_y = torch.stack(processed_y_sequences, dim=0)

    # --- IDENTIFY TOTAL SEQUENCES & SET UP K-FOLD INDICES ---
    total_sequences = full_x.shape[0]
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
        
        train_x, val_x = full_x[train_idx_arr], full_x[val_idx_arr]
        train_y, val_y = full_y[train_idx_arr], full_y[val_idx_arr]
        
        train_size = train_x.shape[0]
        val_size = val_x.shape[0]
        
        global_sequence_counter = 0
        fold_train_seq_loss = []
        fold_train_seq_indices = []
        fold_val_epoch_history = []
        fold_train_epoch_history = []  
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False

        for epoch in range(epochs):
            model.train()
            epoch_train_losses = []
            print(f"\n🎬 Fold {fold+1} | Starting Epoch {epoch+1}/{epochs}")
            
            shuffled_train_indices = torch.randperm(train_size)
            
            # --- 1. TRAINING PASS ---
            for step, s_idx in enumerate(shuffled_train_indices):
                s_idx = s_idx.item()
                
                # Extracting raw sequence block: Shape (1, Sequence_Length, 2)
                x_input_single = train_x[s_idx:s_idx+1, :, :].to(device)  
                u_target_single = train_y[s_idx:s_idx+1, :, :].to(device) 
                
                # 🔥 REDESIGN FIX: Explicitly split data into the 2 inputs requested by the model
                # y_t sits at channel index 0, y_t_delta sits at channel index 1
                y_t_split = x_input_single[:, :, 0:1].to(device)
                y_t_delta_split = x_input_single[:, :, 1:2].to(device)
                
                optimizer.zero_grad()
                
                # Pass both explicit arguments into the model
                u_pred_single = model(y_t=y_t_split, y_t_delta=y_t_delta_split, use_memory=True)
                
                loss = criterion(u_pred_single, u_target_single)
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                epoch_train_losses.append(current_loss)
                
                fold_train_seq_loss.append(current_loss)
                fold_train_seq_indices.append(global_sequence_counter)
                global_sequence_counter += 1

                u_p_np = u_pred_single[0].detach().cpu().numpy().flatten()
                u_t_np = u_target_single[0].detach().cpu().numpy().flatten()
                t_axis = np.arange(len(u_p_np)) * dt

                comparison_df = pd.DataFrame({
                    "t": t_axis, "u_train": u_t_np, "u_pred": u_p_np, "abs_error": np.abs(u_t_np - u_p_np)
                })
                fold_dir = f"{dirname}/fold_{fold+1}/predictions/epoch_{epoch+1}"
                save_df_to_csv(comparison_df, dirname=fold_dir, filename=f"step_{step}_preds")

                if show_plots:
                    plot_signals(
                        t=t_axis,
                        signals=[u_t_np, u_p_np],
                        labels=["Target (u_train)", "Predicted (u_pred)"],
                        xlabel="Time (s)",
                        ylabel="Signal Value",
                        title=f"Fold {fold+1} | Epoch {epoch+1} | Train Step {step} Sequence Comparison",
                        dirname=fold_dir,
                        filename=f"step_sequence_plot"
                    )

            # Step learning rate once per complete training epoch loop
            scheduler.step()

            # --- 2. VALIDATION PASS ---
            model.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for v_idx in range(val_size):
                    x_val_single = val_x[v_idx:v_idx+1, :, :].to(device)
                    u_val_target = val_y[v_idx:v_idx+1, :, :].to(device)
                    
                    # 🔥 REDESIGN FIX: Explicitly split validation features into 2 arguments
                    y_val_t_split = x_val_single[:, :, 0:1].to(device)   
                    y_val_t_delta_split = x_val_single[:, :, 1:2].to(device)
                    
                    u_val_pred = model(y_t=y_val_t_split, y_t_delta=y_val_t_delta_split, use_memory=False)
                    
                    val_loss = criterion(u_val_pred, u_val_target)
                    epoch_val_losses.append(val_loss.item())

                    u_v_p_np = u_val_pred[0].detach().cpu().numpy().flatten()
                    u_v_t_np = u_val_target[0].detach().cpu().numpy().flatten()
                    t_val_axis = np.arange(len(u_v_p_np)) * dt
                    
                    val_fold_dir = f"{dirname}/fold_{fold+1}/validation/epoch_{epoch+1}"
                    
                    if show_plots:
                        plot_signals(
                            t=t_val_axis,
                            signals=[u_v_t_np, u_v_p_np],
                            labels=["Target (u_val)", "Predicted (u_val_pred)"],
                            xlabel="Time (s)",
                            ylabel="Signal Value",
                            title=f"Fold {fold+1} | Epoch {epoch+1} | Val Sequence {v_idx+1}",
                            dirname=val_fold_dir,
                            filename=f"val_seq_sequence_plot"
                        )
            
            mean_train_loss = np.mean(epoch_train_losses)
            mean_val_loss = np.mean(epoch_val_losses) if val_size > 0 else 0.0
            fold_val_epoch_history.append(mean_val_loss)
            fold_train_epoch_history.append(mean_train_loss)  
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"✨ [Fold {fold+1}] Epoch {epoch+1} Summary:")
            print(f"   ↳ LR: {current_lr:.6e} | Avg Train: {mean_train_loss:.6f} | Avg Val: {mean_val_loss:.6f}")

            # --- 3. EVALUATE VALIDATION EARLY STOPPING ---
            if mean_val_loss < (best_val_loss - min_delta):
                best_val_loss = mean_val_loss
                patience_counter = 0 
                save_model(model, dirname=f"{dirname}/fold_{fold+1}", hyperparam_config=hyperparam_config, filename="best_fold_model")
            else:
                patience_counter += 1
                if patience_counter >= val_patience:
                    print(f"🛑 Early stopping fold {fold+1} at Epoch {epoch+1}.")
                    early_stopped = True
                    break
                    
        fold_histories[fold] = {
            "train_seq_idx": fold_train_seq_indices,
            "train_seq_loss": fold_train_seq_loss,
            "val_epochs": list(range(1, len(fold_val_epoch_history) + 1)),
            "train_loss_epochs": fold_train_epoch_history,  
            "val_loss": fold_val_epoch_history
        }

        # --- PLOT INDIVIDUAL FOLD RESULTS ---
        fold_title_suffix = " (Early Stopped)" if early_stopped else " (Full Run)"
        
        plot_signals(
            t=np.array(fold_train_seq_indices),
            signals=[np.array(fold_train_seq_loss)],
            labels=[f"Fold {fold+1} Sequence Loss"],
            xlabel="Total Training Sequences Processed",
            ylabel="Loss Value",
            title=f"Fold {fold+1} Granular Sequence Loss{fold_title_suffix}",
            dirname=f"{dirname}/fold_{fold+1}", filename="granular_training_loss"
        )
        
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
            xlabel="Epochs",
            ylabel="Loss Value",
            title=f"Fold {fold+1} Epoch-level Loss Progress",
            dirname=f"{dirname}/fold_{fold+1}", filename="epoch_validation_loss"
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


def train_controller_kfold(
    model, 
    dataset_path, 
    hyperparam_config, 
    dirname="name_directory", 
    num_sequences_to_use=None,
    k_folds=5,  # Configures the number of cross-validation slices
    show_plots=False  # Option to display plots during training (can be turned off for faster runs)
    ):
    
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]
    
    val_patience = train_cfg.get("val_patience_epochs", 3) 
    min_delta = train_cfg.get("val_min_delta", 0)        

    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=True) 
    
    # --- DATA CONCATENATION & SLICING ---
    full_x = torch.cat(dataset["x"], dim=0)
    full_y = torch.cat(dataset["y"], dim=0)
    
    
    
    total_sequences = full_x.shape[0]
    print(f"📊 Total dataset size: {total_sequences} sequences. Preparing {k_folds}-Fold Split...")

    # --- GENERATE INDICES FOR SEAMLESS K-FOLD SPLITTING ---
    # We shuffle indices once globally so folds contain a balanced mixture of sequences
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, k_folds)

    # Save a clean template of initial model weights so we can reset it every fold
    initial_model_state = copy.deepcopy(model.state_dict())
    
    # Trackers for overarching cross-validation summary statistics
    fold_histories = {}

    # --- K-FOLD CROSS VALIDATION LOOP ---
    for fold in range(k_folds):
        print(f"\n==========================================")
        print(f"🌀 STARTING FOLD {fold + 1} / {k_folds}")
        print(f"==========================================")
        
        # Reset model weights & optimizer state to prevent data leakage from prior folds
        model.load_state_dict(initial_model_state)
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg["lr_decay_rate"])
        criterion = eval(train_cfg["loss_function"])
        
        # Extract validation indices for current fold
        val_idx_arr = folds[fold]
        # Train indices are everything else EXCEPT the current validation fold
        train_idx_arr = np.setdiff1d(all_indices, val_idx_arr)
        
        train_x, val_x = full_x[train_idx_arr], full_x[val_idx_arr]
        train_y, val_y = full_y[train_idx_arr], full_y[val_idx_arr]
        
        train_size = train_x.shape[0]
        val_size = val_x.shape[0]
        
        # Trackers for this specific fold run
        global_sequence_counter = 0
        fold_train_seq_loss = []
        fold_train_seq_indices = []
        fold_val_epoch_history = []
        fold_train_epoch_history = []  # <--- UPDATED: New tracker for epoch-level train loss
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopped = False

        # --- NESTED EPOCH & SEQUENCE RUN LOOPS ---
        for epoch in range(epochs):
            model.train()
            epoch_train_losses = []
            print(f"\n🎬 Fold {fold+1} | Starting Epoch {epoch+1}/{epochs}")
            
            # Shuffle indices locally within our training fold every epoch
            shuffled_train_indices = torch.randperm(train_size)
            
            # --- 1. TRAINING PASS ---
            for step, s_idx in enumerate(shuffled_train_indices):
                s_idx = s_idx.item()
                
                x_input_single = train_x[s_idx:s_idx+1, :, :].to(device)  
                u_target_single = train_y[s_idx:s_idx+1, :, :].to(device) 
                
                optimizer.zero_grad()
                u_pred_single = model(x_input_single)
                loss = criterion(u_pred_single, u_target_single)
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                current_loss = loss.item()
                epoch_train_losses.append(current_loss)
                
                # Record step-by-step sequential metric
                fold_train_seq_loss.append(current_loss)
                fold_train_seq_indices.append(global_sequence_counter)
                global_sequence_counter += 1

                # Export tracking data with safe split path naming
                u_p_np = u_pred_single[0].detach().cpu().numpy().flatten()
                u_t_np = u_target_single[0].detach().cpu().numpy().flatten()
                t_axis = np.arange(len(u_p_np)) * dt

                comparison_df = pd.DataFrame({
                    "t": t_axis, "u_train": u_t_np, "u_pred": u_p_np, "abs_error": np.abs(u_t_np - u_p_np)
                })
                fold_dir = f"{dirname}/fold_{fold+1}/predictions/epoch_{epoch+1}"
                save_df_to_csv(comparison_df, dirname=fold_dir, filename=f"step_{step}_preds")

                # --- NEW: Plot Single Sequence Training Prediction vs Target ---
                if show_plots:
                    plot_signals(
                        t=t_axis,
                        signals=[u_t_np, u_p_np],
                        labels=["Target (u_train)", "Predicted (u_pred)"],
                        xlabel="Time (s)",
                        ylabel="Signal Value",
                        title=f"Fold {fold+1} | Epoch {epoch+1} | Train Step {step} Sequence Comparison",
                        dirname=fold_dir,
                        # filename=f"step_{step}_sequence_plot"
                        filename=f"step_sequence_plot"
                    )

            # --- 2. VALIDATION PASS ---
            model.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for v_idx in range(val_size):
                    x_val_single = val_x[v_idx:v_idx+1, :, :].to(device)
                    u_val_target = val_y[v_idx:v_idx+1, :, :].to(device)
                    
                    u_val_pred = model(x_val_single)
                    val_loss = criterion(u_val_pred, u_val_target)
                    epoch_val_losses.append(val_loss.item())

                    # --- NEW: Process and Plot Validation Sequence ---
                    u_v_p_np = u_val_pred[0].detach().cpu().numpy().flatten()
                    u_v_t_np = u_val_target[0].detach().cpu().numpy().flatten()
                    t_val_axis = np.arange(len(u_v_p_np)) * dt
                    
                    val_fold_dir = f"{dirname}/fold_{fold+1}/validation/epoch_{epoch+1}"
                    
                    if show_plots:
                        plot_signals(
                            t=t_val_axis,
                            signals=[u_v_t_np, u_v_p_np],
                            labels=["Target (u_val)", "Predicted (u_val_pred)"],
                            xlabel="Time (s)",
                            ylabel="Signal Value",
                            title=f"Fold {fold+1} | Epoch {epoch+1} | Val Sequence {v_idx+1}",
                            dirname=val_fold_dir,
                            # filename=f"val_seq_{v_idx+1}_sequence_plot",
                            filename=f"val_seq_sequence_plot"
                        )
            
            mean_train_loss = np.mean(epoch_train_losses)
            mean_val_loss = np.mean(epoch_val_losses) if val_size > 0 else 0.0
            fold_val_epoch_history.append(mean_val_loss)
            fold_train_epoch_history.append(mean_train_loss)  # <--- UPDATED: Store mean train loss
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"✨ [Fold {fold+1}] Epoch {epoch+1} Summary:")
            print(f"   ↳ LR: {current_lr:.6e} | Avg Train: {mean_train_loss:.6f} | Avg Val: {mean_val_loss:.6f}")

            # --- 3. EVALUATE VALIDATION EARLY STOPPING ---
            if mean_val_loss < (best_val_loss - min_delta):
                best_val_loss = mean_val_loss
                patience_counter = 0 
                # Save the absolute best model checkpoint seen for this distinct fold slice
                save_model(model, dirname=f"{dirname}/fold_{fold+1}", hyperparam_config=hyperparam_config, filename="best_fold_model")
            else:
                patience_counter += 1
                if patience_counter >= val_patience:
                    print(f"🛑 Early stopping fold {fold+1} at Epoch {epoch+1}.")
                    early_stopped = True
                    break
                    
        # Save metrics collected explicitly out of this single fold run
        fold_histories[fold] = {
            "train_seq_idx": fold_train_seq_indices,
            "train_seq_loss": fold_train_seq_loss,
            "val_epochs": list(range(1, len(fold_val_epoch_history) + 1)),
            "train_loss_epochs": fold_train_epoch_history,  # <--- UPDATED: Package it for the final return
            "val_loss": fold_val_epoch_history
        }

        # --- PLOT INDIVIDUAL FOLD RESULTS ---
        fold_title_suffix = " (Early Stopped)" if early_stopped else " (Full Run)"
        
        # Plot continuous granular sequence execution for this fold
        plot_signals(
            t=np.array(fold_train_seq_indices),
            signals=[np.array(fold_train_seq_loss)],
            labels=[f"Fold {fold+1} Sequence Loss"],
            xlabel="Total Training Sequences Processed",
            ylabel="Loss Value",
            title=f"Fold {fold+1} Granular Sequence Loss{fold_title_suffix}",
            dirname=f"{dirname}/fold_{fold+1}", filename="granular_training_loss"
        )
        
        # Plot validation performance per epoch for this fold
        # --- UPDATED PLOT: Pass BOTH arrays and BOTH labels into the epoch plot ---
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
            xlabel="Epochs",
            ylabel="Loss Value",
            title=f"Fold {fold+1} Epoch-level Loss Progress",
            dirname=f"{dirname}/fold_{fold+1}", filename="epoch_validation_loss"
        )

    # --- FINAL CROSS VALIDATION LOGGING SUMMARY ---
    print("\n💾 Complete Cross-Validation run finished. Packing overarching metadata curves...")
    
    # Build summary logs tracking final validation results per fold to disk
    summary_records = []
    for f in fold_histories:
        final_best = min(fold_histories[f]["val_loss"])
        summary_records.append({"fold": f+1, "best_recorded_val_loss": final_best})
    
    summary_df = pd.DataFrame(summary_records)
    save_df_to_csv(summary_df, dirname=dirname, filename="kfold_cross_validation_summary")
    
    print("\n✅ K-Fold optimization execution finalized.")
    return fold_histories


def train_controller_es(
    model, 
    dataset_path, 
    hyperparam_config, 
    dirname="name_directory", 
    num_sequences_to_use=None,
    val_split=0.2
    ):
    
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]
    
    # Validation Early Stopping Config
    # If your config file doesn't have these specific keys yet, add them or use these fallbacks
    val_patience = train_cfg.get("val_patience_epochs", 3) 
    min_delta = train_cfg.get("val_min_delta", 1e-4)        

    # --- INITIALIZE PIPELINES ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg["lr_decay_rate"])
    criterion = eval(train_cfg["loss_function"])
    
    # Histories to save and track for plotting
    epoch_indices = []
    epoch_train_history = []
    epoch_val_history = []
    
    # --- EARLY STOPPING CONFIGURATION ---
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopped = False
    
    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=True) 

    # --- DATA CONCATENATION & SLICING ---
    full_x = torch.cat(dataset["x"], dim=0)
    full_y = torch.cat(dataset["y"], dim=0)
    
    if num_sequences_to_use is not None:
        full_x = full_x[:num_sequences_to_use]
        full_y = full_y[:num_sequences_to_use]
    
    total_sequences = full_x.shape[0]
    
    # --- TRAIN / VALIDATION SPLIT ---
    val_size = int(total_sequences * val_split)
    train_size = total_sequences - val_size
    
    train_x, val_x = full_x[:train_size], full_x[train_size:]
    train_y, val_y = full_y[:train_size], full_y[train_size:]
    
    print(f"📊 Dataset Split -> Training: {train_size} sequences | Validation: {val_size} sequences")
    model.to(device)

    # --- NESTED EPOCH & SEQUENCE RUN LOOPS ---
    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        print(f"\n🎬 Starting Epoch {epoch+1}/{epochs}")

        # --- NEW: SHUFFLE TRAINING INDICES EVERY EPOCH ---
        # Generates a random order of indices (e.g., [412, 73, 5901, ...])
        shuffled_indices = torch.randperm(train_size)
        
        # --- 1. TRAINING PASS ---
        for step, s_idx in enumerate(shuffled_indices):
            # Convert the tensor item to a standard python integer
            s_idx = s_idx.item()

            x_input_single = train_x[s_idx:s_idx+1, :, :].to(device)  
            u_target_single = train_y[s_idx:s_idx+1, :, :].to(device)
            
            optimizer.zero_grad()
            u_pred_single = model(x_input_single)
            loss = criterion(u_pred_single, u_target_single)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            epoch_train_losses.append(loss.item())

            # Export tracking CSV records for analysis
            u_p_np = u_pred_single[0].detach().cpu().numpy().flatten()
            u_t_np = u_target_single[0].detach().cpu().numpy().flatten()
            t_axis = np.arange(len(u_p_np)) * dt

            comparison_df = pd.DataFrame({
                "t": t_axis, "u_train": u_t_np, "u_pred": u_p_np, "abs_error": np.abs(u_t_np - u_p_np)
            })
            save_df_to_csv(comparison_df, dirname=f"{dirname}/predictions/epoch_{epoch+1}", filename=f"seq_{s_idx}_preds")

        # --- 2. VALIDATION PASS ---
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for v_idx in range(val_size):
                x_val_single = val_x[v_idx:v_idx+1, :, :].to(device)
                u_val_target = val_y[v_idx:v_idx+1, :, :].to(device)
                
                u_val_pred = model(x_val_single)
                val_loss = criterion(u_val_pred, u_val_target)
                epoch_val_losses.append(val_loss.item())
        
        # Aggregate Epoch Data
        mean_train_loss = np.mean(epoch_train_losses)
        mean_val_loss = np.mean(epoch_val_losses) if val_size > 0 else 0.0
        
        epoch_indices.append(epoch + 1)
        epoch_train_history.append(mean_train_loss)
        epoch_val_history.append(mean_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"✨ Epoch {epoch+1}/{epochs} Summary:")
        print(f"   ↳ LR: {current_lr:.6e} | Avg Train Loss: {mean_train_loss:.6f} | Avg Val Loss: {mean_val_loss:.6f}")

        # --- 3. EVALUATE VALIDATION EARLY STOPPING CRITERION ---
        # Did the validation loss drop significantly past our threshold?
        if mean_val_loss < (best_val_loss - min_delta):
            best_val_loss = mean_val_loss
            patience_counter = 0  # Reset counter since it is still improving
            # Option to save the "best" model checkpoint here if wanted
        else:
            patience_counter += 1
            print(f"⚠️ Validation loss stopped decreasing. Patience counter: {patience_counter}/{val_patience}")
            
            if patience_counter >= val_patience:
                print(f"\n🛑 EARLY STOPPING TRIGGERED! Validation loss plateaued for {val_patience} consecutive epochs.")
                early_stopped = True
                break

    # --- FINAL DATASET EXPORTS & DUAL PLOTTING ---
    print("\n💾 Saving finalized model and performance records...")
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="trained_controller_disk")
    
    # Save training progression metrics to a csv file
    epoch_metrics_df = pd.DataFrame({
        "epoch": epoch_indices, 
        "train_loss": epoch_train_history, 
        "val_loss": epoch_val_history
    })
    save_df_to_csv(epoch_metrics_df, dirname=dirname, filename="epoch_learning_curve_data")

    # Generate the requested comparative Training vs Validation learning curve
    stop_title_suffix = " (Early Stopped)" if early_stopped else " (Full Run Completed)"
    plot_signals(
        t=epoch_metrics_df["epoch"].values, 
        signals=[epoch_metrics_df["train_loss"].values, epoch_metrics_df["val_loss"].values],
        labels=["Average Training Loss", "Average Validation Loss"], 
        xlabel="Epochs", 
        ylabel="Loss Value",
        title=f"Mamba Controller Learning Curves{stop_title_suffix}",
        dirname=dirname, 
        filename="training_validation_losses_plot"
    )

    return epoch_train_history, epoch_val_history



import numpy as np
import pandas as pd
import torch

#=== FUNCTION TO TRAIN CONTROLLER WITH VALIDATION EARLY STOPPING ===#
def train_controller_w_validation_early_stopping(
    model, 
    dataset_path, 
    hyperparam_config, 
    dirname="name_directory", 
    num_sequences_to_use=None
):
    """
    Trains the network incrementally where each sequence represents a single epoch.
    Implements rolling look-ahead validation to calculate early-stopping patience.
    """
    # --- EXTRACT CONFIGURATION VALUES ---
    train_cfg = hyperparam_config["train"]
    patience_steps = train_cfg["patience_steps"]  # Number of sequences to look at
    critical_value = train_cfg["critical_loss_value"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]

    # --- INITIALIZE PIPELINES ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Decays the learning rate slightly after every single sequence (epoch)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=train_cfg.get("lr_decay_rate", 0.995) 
    )
    
    criterion = eval(train_cfg["loss_function"])  
    
    # Track metrics matching your single history array concept
    train_loss_history = []
    val_loss_history = []
    
    # --- CONVERGENCE TRACKERS ---
    rolling_val_window = deque(maxlen=patience_steps)
    converged = False
    
    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=True) 

    # --- DATA CONCATENATION & SLICING ---
    full_x = torch.cat(dataset["x"], dim=0)
    full_y = torch.cat(dataset["y"], dim=0)
    
    if num_sequences_to_use is not None:
        full_x = full_x[:num_sequences_to_use]
        full_y = full_y[:num_sequences_to_use]
    
    total_sequences = full_x.shape[0]
    print(f"📈 Starting online training across {total_sequences} individual sequences (epochs)...")
    
    model.to(device)

    # --- SINGLE SEQUENCE STREAM LOOP ---
    for s_idx in range(total_sequences):
        if converged:
            break
            
        x_input_single = full_x[s_idx:s_idx+1, :, :].to(device)  
        u_target_single = full_y[s_idx:s_idx+1, :, :].to(device) 
        
        # =====================================================================
        # PHASE 1: LOOK-AHEAD VALIDATION (Evaluate before learning)
        # =====================================================================
        model.eval()
        with torch.no_grad():
            u_val_pred = model(x_input_single)
            val_loss = criterion(u_val_pred, u_target_single)
            current_val_loss = val_loss.item()
            val_loss_history.append(current_val_loss)
            rolling_val_window.append(current_val_loss)

        # =====================================================================
        # PHASE 2: TRAINING UPDATE (Learn from the sequence)
        # =====================================================================
        model.train()
        optimizer.zero_grad()
        
        u_pred_single = model(x_input_single)
        loss = criterion(u_pred_single, u_target_single)
        loss.backward()
        optimizer.step()
        
        current_train_loss = loss.item()
        train_loss_history.append(current_train_loss)
        
        # Advance learning rate decay per sequence
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # =====================================================================
        # PHASE 3: EVALUATE EARLY STOPPING CONVERGENCE
        # =====================================================================
        if len(rolling_val_window) == patience_steps:
            # Check the average validation trend over our patience window
            avg_rolling_val = sum(rolling_val_window) / patience_steps
            
            if avg_rolling_val < critical_value:
                print(f"\n🎯 CONVERGENCE MET at Sequence/Epoch {s_idx+1}!")
                print(f"📉 Rolling validation loss average ({avg_rolling_val:.6f}) dropped below threshold ({critical_value}).")
                converged = True

        # --- DATA PREPARATION FOR CSV LOGGING (Every 10 sequences) ---
        if s_idx % 10 == 0 or converged:
            u_p_np = u_pred_single[0].detach().cpu().numpy().flatten()
            u_t_np = u_target_single[0].detach().cpu().numpy().flatten()
            t_axis = np.arange(len(u_p_np)) * dt

            comparison_df = pd.DataFrame({
                "t": t_axis, "u_train": u_t_np, "u_pred": u_p_np, "abs_error": np.abs(u_t_np - u_p_np)
            })
            csv_filename = f"seq_epoch_{s_idx}_predictions"
            save_df_to_csv(comparison_df, dirname=f"{dirname}/predictions", filename=csv_filename)

        if (s_idx + 1) % 10 == 0 or (s_idx + 1) == total_sequences:
            print(f"Epoch/Seq {s_idx+1}/{total_sequences} | LR: {current_lr:.4e} | Train Loss: {current_train_loss:.6f} | Look-Ahead Val Loss: {current_val_loss:.6f}")

    # --- FINAL DATASET EXPORTS ---
    print("💾 Saving finalized model and performance records...")
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="trained_controller_disk")
    
    # Save combined history log
    loss_df = pd.DataFrame({
        "sequence_epoch": range(1, len(train_loss_history) + 1), 
        "train_loss": train_loss_history,
        "look_ahead_val_loss": val_loss_history
    })
    save_df_to_csv(loss_df, dirname=dirname, filename="total_sequence_loss")

    stop_type_title = " (Early Stopped)" if converged else " (Full Run Completed)"
    plot_signals(
        loss_df["sequence_epoch"].values, 
        [loss_df["train_loss"].values, loss_df["look_ahead_val_loss"].values],
        labels=["Immediate Training Loss", "Look-Ahead Validation Loss"], 
        xlabel="Total Sequences Evaluated (Epochs)", 
        ylabel="Loss Value",
        title=f"Online Learning Curve{stop_type_title}",
        dirname=dirname, filename="sequence_loss_from_disk_plot"
    )

    return train_loss_history




