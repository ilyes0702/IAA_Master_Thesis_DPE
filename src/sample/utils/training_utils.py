# Import standard libraries

import numpy as np
import pandas as pd

# import machine learning modules
from src.sample.utils.loss_utils import relative_huber_loss
from src.sample.decorators.general_decorators import *
from src.sample.utils.saving_utils import *
from src.sample.config import *
from src.sample.utils.plotting_utils import plot_signals
import pandas as pd
import os
import torch
plt.style.use("src/sample/style.mplstyle")

@track_resources
def GPUtrain_controllerFFT(model, plant, hyperparam_config, dirname="name_directory"):
    # --- EXTRACT HYPERPARAMETERS FROM CONFIG ---
    # Training params
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    device = train_cfg["device"]
    
    # Signal params
    sig_cfg =   hyperparam_config["signal"]
    seq_len = sig_cfg["seq_len"]
    dt = sig_cfg["dt"]
    lambd = sig_cfg.get("lambd")
    p = sig_cfg.get("p")

    # --- INITIALIZE TRAINING ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = relative_huber_loss
    loss_history = []
    all_data_frames = []
    all_D_centers = []
    sequence_loss_history = []  # Rename to clarify it's per sequence


    model.to(device)
    print(f"🚀 Training Mamba {plant.__class__.__name__} on {device}")

    for epoch in range(epochs):
        
        if hasattr(plant, 'reset_trajectory'):
            try:
                D_center = plant.reset_trajectory(seq_len=seq_len, dt=dt, lambd=lambd, p=p)
            except TypeError:
                D_center = plant.reset_trajectory()
            all_D_centers.append(D_center) 


        state = plant.get_initial_state(batch_size)
        all_y_t, all_y_next, all_u = [], [], []
        sequence_D_centers = []

        # --- SIMULATION PHASE ---
        # Inside GPUtrain_controllerFFT
        delta_steps = hyperparam_config["train"]["delay_steps"]  # For example, look 5 steps ahead (delta = 5 * dt)

        for t_idx in range(seq_len - delta_steps):
            t = t_idx * dt
            u_signal = plant.get_u_at_step(t_idx)
            
            # Current state
            y_t = plant.get_y(state)
            
            # Forward simulate k steps using the SAME u_signal
            # (Canaday logic: the input v_train is held constant over the interval delta)
            temp_state = state
            for _ in range(delta_steps):
                temp_state, _ = plant.step(temp_state, u_signal, t, dt)
            
            # The state after delta
            y_delta = plant.get_y(temp_state)
            
            all_y_t.append(y_t)
            all_y_next.append(y_delta) # This is now y(t + delta)
            all_u.append(u_signal)
            
            # IMPORTANT: Move the actual simulation forward only 1 step 
            # to keep the trajectory continuous, or skip delta_steps.
            state_next, _ = plant.step(state, u_signal, t, dt)
            state = state_next.detach()
            # If D_center is updated per sequence, store it here
            if hasattr(plant, 'current_D_center'):
                sequence_D_centers.append(plant.current_D_center)

        # --- DATA PREPARATION FOR CSV ---
        y_t_stack = torch.stack(all_y_t, dim=1).cpu().numpy()
        y_next_stack = torch.stack(all_y_next, dim=1).cpu().numpy()
        u_stack = torch.stack(all_u, dim=1).cpu().numpy()

        num_samples = y_t_stack.shape[1] 

        epoch_df = pd.DataFrame({
            "t": [i * dt for i in range(num_samples)],
            "y_t": y_t_stack[0, :, 0],
            "y_next": y_next_stack[0, :, 0],
            "u_control": u_stack[0, :, 0],
            "D_center": D_center  # Add D_center for this epoch
        })
        
        save_df_to_csv(epoch_df, 
                       dirname=dirname+"/sequences",
                       filename=f"sample_trajectory_epoch_{epoch}_seq_0.csv")

        # Save a signal plot for this sample sequence, including D_center as a horizontal line
        time_axis = epoch_df["t"].values
        u_signal_plot = epoch_df["u_control"].values
        y_t_plot = epoch_df["y_t"].values
        y_next_plot = epoch_df["y_next"].values
        d_center_plot = np.full_like(time_axis, D_center, dtype=float)

        plot_signals(
            time_axis,
            [u_signal_plot, y_t_plot, y_next_plot, d_center_plot],
            labels=["u_control", "y_t", "y_next", "D_center"],
            xlabel="Time",
            ylabel="Signal",
            title=f"Sample Sequence Signals Epoch {epoch}",
            dirname=dirname+"/sequences",
            filename=f"sample_trajectory_epoch_{epoch}_seq_0_plot"
        )

        all_data_frames.append(epoch_df)

        # --- TRAINING PHASE ---
        y_t_seq = torch.stack(all_y_t, dim=1)
        y_next_seq = torch.stack(all_y_next, dim=1)
        y_target = torch.stack(all_u, dim=1)
        x_tensor = torch.cat([y_t_seq, y_next_seq], dim=-1)

        model.train()

        # Loop through each individual trajectory in the batch
        for b in range(batch_size):
            optimizer.zero_grad()
            
            # Extract one trajectory: [1, seq_len, 2]
            u_pred_single = model(x_tensor[b:b+1, :, :])
            y_target_single = y_target[b:b+1, :, :]
            
            # Calculate loss for just this one trajectory
            loss = criterion(u_pred_single, y_target_single)
            
            # Update weights immediately
            loss.backward()
            optimizer.step()
            
            # Now this will append 32 rows to your CSV per epoch!
            sequence_loss_history.append(loss.item())


        
        
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1:04d} | Loss: {loss.item():.6f}")
                # We take index 0 of the batch
                u_truth_sample = y_target_single[0].detach().cpu().numpy().flatten()
                u_pred_sample = u_pred_single[0].detach().cpu().numpy().flatten()
                
                plot_signals(
                    time_axis[:len(u_truth_sample)], 
                    [u_truth_sample, u_pred_sample],
                    labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
                    xlabel="Time", ylabel="Control Signal",
                    title=f"Mamba Prediction Accuracy - Epoch {epoch}",
                    dirname=dirname+"/sequences",
                    filename=f"prediction_accuracy_epoch_{epoch}_seq_{b}.png"
                )

                plot_signals(
                    time_axis[:len(u_truth_sample)], 
                    [u_truth_sample, u_pred_sample],
                    labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
                    xlabel="Time", ylabel="Control Signal",
                    title=f"Mamba Prediction Accuracy",
                    dirname=dirname+"/sequences",
                    filename=f"prediction_accuracy.png"
                )

    # --- FINAL AGGREGATION & SAVING ---
    master_df = pd.concat(all_data_frames, ignore_index=True)
    save_df_to_csv(master_df, dirname=dirname, filename="all_training_data_summary")

    
    
    # Passing the whole config for tracking
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="trained_controller")

    # The x-axis is now the total number of sequences trained on
    df_loss = pd.DataFrame({
        "sequence_index": range(1, len(sequence_loss_history) + 1), 
        "loss": sequence_loss_history
    })
    save_df_to_csv(df_loss, dirname=dirname, filename="sequence_loss_history")
    
    plot_signals(
        df_loss["sequence_index"].values, [df_loss["loss"].values],
        labels=["Relative Huber Loss"], 
        xlabel="Total Sequences Trained", 
        ylabel="Loss",
        title="Learning Curve (Per Sequence)",
        dirname=dirname, filename="sequence_loss_plot"
    )

    # --- SAVE D_CENTER HISTORY ---
    # Create a DataFrame for D_center history (per epoch)
    d_center_df = pd.DataFrame({
        "Sequence": range(1, epochs*batch_size + 1),
        "D_center": sequence_D_centers
    })
    save_df_to_csv(
        d_center_df,
        dirname=dirname,
        filename="D_center_history.csv"
    )

    return loss_history


@track_resources
def GPUtrain_controller_from_disk(model, dataset_path, hyperparam_config, dirname="name_directory"):
    # --- EXTRACT HYPERPARAMETERS ---
    train_cfg = hyperparam_config["train"]
    epochs = train_cfg["epochs"]
    device = train_cfg["device"]
    lr = train_cfg["lr"]
    dt = hyperparam_config["signal"]["dt"]

    # --- INITIALIZE ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = relative_huber_loss
    sequence_loss_history = []
    
    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path) 
    batches_x = dataset["x"]
    batches_y = dataset["y"]

    model.to(device)

    for epoch in range(epochs):
        for b_idx, (x_batch, y_batch) in enumerate(zip(batches_x, batches_y)):
            # Move batch to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size = x_batch.shape[0]

            model.train()

            for s_idx in range(batch_size):
                optimizer.zero_grad()
                
                # 1. Prediction (Forward Pass)
                u_pred_single = model(x_batch[s_idx:s_idx+1, :, :])
                u_train_single = y_batch[s_idx:s_idx+1, :, :]
                
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

                # plot_signals(
                #         t_axis, 
                #         [u_t_np, u_p_np], # Pass the flattened numpy arrays
                #         labels=["Ground Truth (u)", "Mamba Prediction (u_hat)"],
                #         xlabel="Time", ylabel="Control Signal",
                #         title=f"Mamba Prediction Accuracy Epoch {epoch} Seq {s_idx}",
                #         dirname=dirname+"/sequences",
                #         filename=f"prediction_accuracy"
                #     )

        print(f"🚀 Epoch {epoch+1}/{epochs} Finished | Final Seq Loss: {loss.item():.6f}")

    # Final model and loss plots
    save_model(model, dirname=dirname, hyperparam_config=hyperparam_config, filename="trained_controller_disk")
    
    # Save master loss history
    loss_df = pd.DataFrame({"sequence_index": range(len(sequence_loss_history)), "loss": sequence_loss_history})
    save_df_to_csv(loss_df, dirname=dirname, filename="total_sequence_loss")

    plot_signals(
        loss_df["sequence_index"].values, [loss_df["loss"].values],
        labels=["Relative Huber Loss"], 
        xlabel="Total Sequences Trained", 
        ylabel="Loss",
        title="Learning Curve (Per Sequence) - From Disk",
        dirname=dirname, filename="sequence_loss_from_disk_plot"
    )

    return sequence_loss_history



