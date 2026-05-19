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




