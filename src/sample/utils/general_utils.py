# Import standard libraries
import os
import sys
import numpy as np
import pandas as pd

# import machine learning modules
import joblib
from src.sample.decorators.general_decorators import *
from src.sample.utils.saving_utils import *
from src.sample.config import *
from src.sample.utils.plotting_utils import plot_signals
import torch
import torch.nn as nn

### 00
def train_controller(model, plant, epochs, seq_len, dt, model_config,device='cuda', dirname="name_directory"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    loss_history = []
    all_training_pairs = []

    print(f"Training Mamba on {plant.__class__.__name__}...")

    for epoch in range(epochs):
        # 1. Randomize parameters if the plant supports it
        if hasattr(plant, 'reset_trajectory'):
            plant.reset_trajectory()
            
        state = plant.get_initial_state()
        epoch_raw_history = []
        
        # --- Data Collection Phase ---
        for t_idx in range(seq_len):
            t = t_idx * dt
            u_signal = plant.generate_random_u(t)
            
            # Pass 't' to get_y to account for time-dependent Volume (V)
            y_t = plant.get_y(state, t) 
            
            # Step plant (ensure plant.step also uses/returns correct y_next)
            state_next, y_next = plant.step(state, u_signal, t, dt)
            
            record = {
                "t": t,
                "y_t": y_t,
                "y_t+dt": y_next,
                "u_t": u_signal,
                **plant.parse_state(state)
            }
            epoch_raw_history.append(record)
            state = state_next

        # Create DataFrame for this epoch
        df_epoch = pd.DataFrame(epoch_raw_history)
        
        # Optional: Save every X epochs to save disk space if training is long
        if (epoch + 1) % 10 == 0 or epoch == 0:
            save_df_to_csv(
                df_epoch, 
                dirname=f"{dirname}/training_episodes", 
                filename=f"train_series_epoch_{epoch+1:04d}"
            )

        # --- Training Phase ---
        # 2. Prepare Inverse Mapping [y_t, y_t+dt] -> u_t
        inputs = df_epoch[["y_t", "y_t+dt"]].values
        targets = df_epoch["u_t"].values
        
        # Use torch.from_numpy and np.array to avoid the "slow tensor creation" warning
        x_tensor = torch.from_numpy(np.array([inputs])).float().to(device)
        y_target = torch.from_numpy(np.array([targets])).float().unsqueeze(-1).to(device)

        model.train()
        optimizer.zero_grad()
        
        # Forward pass through Mamba
        u_pred = model(x_tensor)
        
        loss = criterion(u_pred, y_target)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        
     

        # Summary logging
        summary_slice = df_epoch[["y_t", "y_t+dt", "u_t"]].copy()
        summary_slice["epoch"] = epoch + 1
        all_training_pairs.append(summary_slice)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    # --- Final Post-Training Steps ---
    df_loss = pd.DataFrame({"epoch": range(1, epochs + 1), "loss": loss_history})
    df_train_full = pd.concat(all_training_pairs, ignore_index=True)
    
    save_df_to_csv(df_loss, dirname=dirname, filename="training_loss_history")
    save_df_to_csv(df_train_full, dirname=dirname, filename="training_data_full")
    save_df_to_csv(df_train_full.describe().reset_index(), dirname=dirname, filename="training_data_stats")

    plot_signals(
        df_loss["epoch"].values, [df_loss["loss"].values],
        labels=["MSE Loss"], xlabel="Epoch", ylabel="Loss",
        title=f"Convergence ({plant.__class__.__name__})",
        dirname=dirname, filename="training_loss_plot"
    )

    # --- Save the Model Weights ---
    save_model(model, dirname=dirname, model_config=model_config, filename="trained_controller")

    return()




import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time

def GPUtrain_controller(model, plant, epochs, seq_len, dt, model_config, batch_size, device='cuda', dirname="name_directory"):
    """
    Vectorized training: 128 parallel simulations are performed per epoch.
    The model learns the inverse mapping: (y_t, y_t+1) -> u_t
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loss_history = []
    model.to(device)

    print(f"🚀 Training Mamba on {plant.__class__.__name__}")
    print(f"   Batch Size: {batch_size} | Device: {device}")

    start_time = time.time()

    for epoch in range(epochs):
        # 1. Diversity: randomize trajectories if the plant supports it
        if hasattr(plant, 'reset_trajectory'):
            plant.reset_trajectory()
            
        # 2. Reset starting state for the entire batch [batch_size, state_dim]
        state = plant.get_initial_state() 
        
        all_y_t = []
        all_y_next = []
        all_u = []

        # --- SIMULATION PHASE (Data Generation) ---
        # We use torch.no_grad() because we are just collecting data from the physics engine
        with torch.no_grad():
            for t_idx in range(seq_len):
                t = t_idx * dt
                
                # Generate random control inputs (Glucose feed) for exploration
                u_signal = torch.rand((batch_size, 1), device=device) * plant.U_MAX
                
                # Current output (Growth rate)
                y_t = plant.get_y(state, t) 
                
                # Physics Step
                state_next, y_next = plant.step(state, u_signal, t, dt)
                
                # Store
                all_y_t.append(y_t)
                all_y_next.append(y_next)
                all_u.append(u_signal)
                
                # Transition - DETACH to stop gradient flow through simulation time
                state = state_next.detach()

        # --- TRAINING PHASE (Gradient Descent) ---
        # Stack lists into [Batch, Seq, Dim]
        y_t_seq = torch.stack(all_y_t, dim=1)      
        y_next_seq = torch.stack(all_y_next, dim=1) 
        y_target = torch.stack(all_u, dim=1) 
        
        # Input to model: pair of current and resulting growth rate
        x_tensor = torch.cat([y_t_seq, y_next_seq], dim=-1) 

        model.train()
        optimizer.zero_grad()
        
        # Mamba predicts the 'u' that caused the change from y_t to y_next
        u_pred = model(x_tensor) 
        
        loss = criterion(u_pred, y_target)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 100 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:04d}/{epochs} | Loss: {loss.item():.6f} | Time: {elapsed:.2f}s")

    # --- SAVE & PLOT ---
    df_loss = pd.DataFrame({"epoch": range(1, epochs + 1), "loss": loss_history})
    
    # Import locally to avoid circular dependencies
    from src.sample.utils.general_utils import save_model, save_df_to_csv, plot_signals
    
    save_model(model, dirname=dirname, model_config=model_config, filename="trained_controller")
    save_df_to_csv(df_loss, dirname=dirname, filename="training_loss_history")

    plot_signals(
        df_loss["epoch"].values, [df_loss["loss"].values],
        labels=["MSE Loss"], xlabel="Epoch", ylabel="Loss",
        title=f"Convergence ({plant.__class__.__name__})",
        dirname=dirname, filename="training_loss_plot"
    )

    return loss_history

def load_controller(model_class, filepath, device='cuda'):
    """
    Loads a MambaInverseController and its configuration from a saved checkpoint.
    
    Args:
        model_class: The class name (MambaInverseController).
        filepath: Full path to the .pt file.
        device: 'cuda' or 'cpu'.
    """
    # 1. Load the checkpoint dictionary from disk
    # map_location ensures it works even if you move from GPU to CPU
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    
    # 2. Extract the saved config
    model_config = checkpoint['model_config']
    print(f"Loading model with config: {model_config}")
    
    # 3. Reconstruct the architecture using the saved hyperparams
    model = model_class(**model_config)
    
    # 4. Load the learned weights (state_dict) into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 5. Move to device and set to evaluation mode
    model.to(device)
    model.eval() 
    
    return model, model_config

def simulate_control(model, plant, reference_signal, duration, dt, device, dirname):
    model.eval()
    state = plant.get_initial_state()
    history = []
    current_context = []
    steps = int(duration / dt)

    print(f"Starting generic simulation for {duration}h...")

    for i in range(steps):
        t = i * dt
        y_meas = plant.get_y(state, t)
        
        # 1. Reference Logic
        r_t = reference_signal[i] if isinstance(reference_signal, np.ndarray) else reference_signal

        # 2. Normalization & Inference
        y_norm = y_meas / plant.Y_MAX
        r_norm = r_t / plant.Y_MAX

        step_input = np.array([[y_norm, r_norm]])
        current_context.append(step_input)
        
        # Maintain a sliding window or full history for Mamba context
        input_tensor = torch.tensor(np.array(current_context), dtype=torch.float32).transpose(0, 1).to(device)

        with torch.no_grad():
            u_norm = float(model(input_tensor)[0, -1, 0])
        
        u_phys = np.clip(u_norm * plant.U_MAX, 0, plant.U_MAX)

        # 3. Physics Step
        next_state, _ = plant.step(state, u_phys, t, dt)
        
        # 4. Data Collection
        # Combines general control signals with plant-specific internal states
        record = {
            "t": t, 
            "y": y_meas, 
            "r": r_t, 
            "u": u_phys, 
            "error": r_t - y_meas,
            **plant.parse_state(state) 
        }
        history.append(record)
        state = next_state

    # --- Data Processing & Saving ---
    df_sim = pd.DataFrame(history)
    save_df_to_csv(df_sim, dirname=dirname, filename=f"{plant.__class__.__name__}_sim_data")

    # --- Automated Plotting ---
    t_data = df_sim["t"].values
    
    # Use the plant's own config to decide what to plot
    if hasattr(plant, 'get_plot_config'):
        plot_configs = plant.get_plot_config()
    else:
        raise NotImplementedError("Plant class must implement get_plot_config() to specify plotting configuration.")
    
    for idx, config in enumerate(plot_configs):
        signals = [df_sim[col].values for col in config["cols"]]
        
        plot_signals(
            t_data, 
            signals,
            labels=config["labels"],
            xlabel="Time (h)",
            ylabel=config["ylabel"],
            dirname=dirname,
            filename=f"plot_{idx}_{config['title'].lower().replace(' ', '_')}"
        )

    print(f"Simulation finished. Data and {len(plot_configs)} plots saved to {dirname}.")

    def calculate_metrics(df):
        error = df["r"] - df["y"]
        u_diff = df["u"].diff().abs().sum() # Total Variation
        
        metrics = {
            "RMSE": np.sqrt((error**2).mean()),
            "MAE": error.abs().mean(),
            "Max_Error": error.abs().max(),
            "Control_Total_Variation": u_diff,
            "Mean_Control_Signal": df["u"].mean(),
            "Standard_Deviation_Error": error.std()
        }
        return metrics

    # In your simulate_control function:
    perf_metrics = calculate_metrics(df_sim)
    df_perf = pd.DataFrame([perf_metrics])
    save_df_to_csv(df_perf, dirname=dirname, filename="performance_metrics")
    return()

import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    """
    Seeds all relevant libraries to ensure reproducible results.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    
    # Critical for CUDA reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def load_model_complete(model_class, filepath, device='cuda'):
    """
    Loads a model without needing to manually provide params.
    """
    # 1. Load the checkpoint
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    
    # 2. Extract the config and reconstruct the architecture
    model_config = checkpoint['model_config']
    model = model_class(**model_config)
    
    # 3. Load the weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded with config: {model_config}")
    return model, model_config


#=== FUNCTION TO CHOOSE NICE TICK STEP ===#
def get_tick_step(data_min, data_max, n_ticks=6):
    """
    Determine a 'nice' tick step for axis labeling based on the data range 
    and the desired number of ticks.

    Parameters:
    - data_min (float): The minimum value of the data range.
    - data_max (float): The maximum value of the data range.
    - n_ticks (int, optional): The desired number of ticks (default is 6).

    Returns:
    - float: A 'nice' tick step value that ensures well-distributed axis labels.
    
    The method works by computing the raw step size, then adjusting it to the 
    nearest 'nice' value (such as 1, 2, 5, or 10 multiplied by a power of 10), 
    ensuring clear and readable axis divisions.
    """
    raw_step = (data_max - data_min) / (n_ticks - 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    residual = raw_step / magnitude
    if residual < 1.5:
        nice_step = 1 * magnitude
    elif residual < 3:
        nice_step = 2 * magnitude
    elif residual < 7:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude
    return nice_step





#=== FUNCTION TO ADD CLASS LABELS DI/DF TO DATAFRAME ===#
@log_execution_time
def addLabelsToDataframe(df, constraints, label):
    """
    Assigns binary class labels to a DataFrame based on constraint satisfaction.

    This function checks whether each row satisfies all specified constraints—
    specifically, whether all values in the given columns are less than or equal to zero.
    If a row meets all conditions, it is labeled with a `1` (positive class); otherwise, `0` (negative class).

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        constraints (list of str): List of column names to evaluate.
        label (str): Name of the new column that stores assigned class labels.

    Returns:
        pd.DataFrame: The updated DataFrame with a new column containing binary class labels.

    Example:
        >>> df = pd.DataFrame({'A': [1, -2], 'B': [0, -1]})
        >>> addLabelsToDataframe(df, ['A', 'B'], 'label')
           A  B  label
        0  1  0      0
        1 -2 -1      1
    """
    # Reset the index to ensure it is continuous
    df = df.reset_index(drop=True)  
    df[label] = (df[constraints].le(0).all(axis=1)).astype(int)
    return df


#=== FUNCTION TO ADD LABELS ABOUT MISCLASSIFICATION TO DATAFRAME ===#
@log_execution_time
def addLabelsToDataFrame_clf(df, col_class, col_predicted, col_classification_result, col_classification_type):
    """
    Adds a classification result column to indicate whether a data point was correctly classified
    and a column to indicate if it is a False Positive, False Negative, True Positive, or True Negative.

    The function compares the "Predicted_Label" column with the actual "class" column.
    If a prediction matches the true label, the classification result is set to `1` (correctly classified).
    If the prediction is incorrect, it is set to `-1` (misclassified).

    Parameters:
        - df (pd.DataFrame): DataFrame containing classification results with columns
                           "Predicted_Label" (predicted category) and "class" (true category).

    Returns:
        pd.DataFrame: The updated DataFrame with additional "classification_result" and "classification_type" columns.
                      - `1` indicates correct classification.
                      - `-1` indicates misclassification.
                      - "classification_type" indicates FP, FN, TP, or TN.
    """
    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)
    df[col_classification_result] = [0] * len(df.index)
    df[col_classification_type] = [""] * len(df.index)

    for i in range(len(df.index)):
        predicted = df.loc[i, col_predicted]
        actual = df.loc[i, col_class]

        if predicted == actual:
            df.loc[i, col_classification_result] = 1
            if predicted == 1:  # Assuming 1 is the positive class
                df.loc[i, col_classification_type] = "TP"
            else:
                df.loc[i, col_classification_type] = "TN"
        else:
            df.loc[i, col_classification_result] = -1
            if predicted == 1:  # Assuming 1 is the positive class
                df.loc[i, col_classification_type] = "FP"
            else:
                df.loc[i, col_classification_type] = "FN"

    return df





#=== FUNCTION TO DETERMINE SET OF DOMINATED VECTORS ===#    
@log_execution_time
def is_dominated_by_k(costs, k, return_mask=True):
    """
    Identify points in a cost space that are dominated by exactly 'k' other points.

    Parameters:
    - costs (array-like): An array of shape (n_points, n_costs), where each row represents 
      a point and each column corresponds to a cost dimension.
    - k (int): The number of dominating points a data point must have.
    - return_mask (bool, optional): 
      - If True, returns a boolean mask indicating dominated points.
      - If False, returns an array of indices for dominated points.

    Returns:
    - numpy.ndarray:
      - If `return_mask` is True: A boolean array of shape (n_points,) marking dominated points.
      - If `return_mask` is False: An integer array containing indices of dominated points.

    The function computes the number of times each point is dominated by others in 
    the cost space, then selects points that are dominated exactly `k` times.
    """
    costs = np.array(costs)
    n_points = costs.shape[0]
    domination_counts = np.zeros(n_points, dtype=int)

    # Count how many times each point is dominated
    for i, point in enumerate(costs):
        domination_counts[i] = np.sum(np.all(costs < point, axis=1))

    # Get the mask or indices of points dominated by exactly k others
    if return_mask:
        return domination_counts == k
    else:
        return np.where(domination_counts == k)[0]


#=== FUNCTION TO GET UNION OF DOMINATED POINTS UP TO K ===#
@log_execution_time
def is_dominated_by_up_to_k(costs, k, return_mask=True):
    """
    Identify points in a cost space that are dominated by up to 'k' others.

    Parameters:
    - costs (array-like): An array of shape (n_points, n_costs), where each row represents 
      a point and each column corresponds to a cost dimension.
    - k (int): The maximum number of dominating points allowed.
    - return_mask (bool, optional): 
      - If True, returns a boolean mask indicating selected points.
      - If False, returns an array of the actual subset of selected points.

    Returns:
    - numpy.ndarray:
      - If `return_mask` is True: A boolean array of shape (n_points,) marking selected points.
      - If `return_mask` is False: A (n_selected_points, n_costs) array containing the selected points.

    The function computes how many times each point is dominated by others in the cost space. 
    It then selects points that are dominated at most `k` times.
    """
    costs = np.array(costs)
    n_points = costs.shape[0]
    domination_counts = np.zeros(n_points, dtype=int)

    # Count how many times each point is dominated
    for i, point in enumerate(costs):
        domination_counts[i] = np.sum(np.all(costs < point, axis=1))

    # Create mask for points dominated by up to 'k' others
    mask = domination_counts <= k

    return mask if return_mask else costs[mask]




#=== FUNCTION TO FIND UNIQUE STRUCTURES CORRESPONDING TO PARETO POINTS ===#
@log_execution_time
def find_frequent_combinations_with_min_max(ds, df, group_columns, value_column, second_value_column, threshold=1):
    """
    Identify frequent value combinations in specified columns, calculate the 
    minimum and maximum values for two target columns, and return the results 
    sorted by the minimum value of the first target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - group_columns (list of str): List of column names to group by (e.g., ['A', 'C', 'D']).
    - value_column (str): Name of the first column for which to find min and max values (e.g., 'F').
    - second_value_column (str): Name of the second column for which to find min and max values.
    - threshold (int, optional): Minimum frequency for a combination to be considered frequent (default is 1).

    Returns:
    - pd.DataFrame: DataFrame containing the grouped combinations, their min/max values for 
      both specified columns, and count of occurrences.

    The function groups the data by `group_columns`, counts occurrences, and filters 
    combinations that appear more than `threshold` times. For each frequent combination, 
    it calculates min/max values of `value_column` and `second_value_column`, then sorts the results 
    by the growing order of the minimum value in `value_column`.
    """
    # Group by all columns in 'group_columns' and count occurrences
    combinations = df.groupby(group_columns).size().reset_index(name='Count')

    # Filter only the combinations that occur more than the threshold
    frequent_combinations = combinations[combinations['Count'] > threshold]

    # For each frequent combination, find the min and max of the specified columns
    result = []
    for _, row in frequent_combinations.iterrows():
        # Dynamically filter using group_columns
        condition = (df[group_columns] == row[group_columns].values).all(axis=1)
        subset = df[condition]
        min_value = subset[value_column].min()
        max_value = subset[value_column].max()
        min_value2 = subset[second_value_column].min()
        max_value2 = subset[second_value_column].max()
        result.append({
            **row.to_dict(),
            'Min_' + value_column: min_value,
            'Max_' + value_column: max_value,
            'Min_' + second_value_column: min_value2,
            'Max_' + second_value_column: max_value2
        })

    # Create the results as a DataFrame and order by the 'Min_value_column'
    result_df = pd.DataFrame(result)
    if not result_df.empty:
        result_df = result_df.sort_values(by='Min_' + value_column).reset_index(drop=True)
        return result_df
    else:
        return df



#=== FUNCTION TO CALCULATE PERCENTAGES OF UNFULFILLED CONSTRAINTS ===#
@log_execution_time
def calculate_unfulfilled_constraints_per_constraint(df, constraints):
    """
    Compute the percentage of unfulfilled constraints for each constraint column.

    Parameters:
    - df (pandas.DataFrame): The dataset containing constraint columns.
    - constraints (list of str): List of column names representing constraints.

    Returns:
    - dict: A dictionary where keys are constraint names and values are the percentage 
      of rows where the constraint is unfulfilled (i.e., the value > 0).

    The function iterates over each constraint column and calculates the percentage 
    of rows where the constraint is violated. The result helps evaluate the frequency 
    of unfulfilled constraints in the dataset.
    """
    unfulfilled_percentages = {}
    
    for constraint in constraints:
        # Calculate the percentage of rows where the constraint is unfulfilled (value > 0)
        unfulfilled_count = (df[constraint] > 0).sum()
        total_rows = len(df)
        unfulfilled_percentages[constraint] = (unfulfilled_count / total_rows) * 100

    return unfulfilled_percentages





#=== FUNCION TO SPLIT DATAFRAME INTO SUBDATAFRAMES WITH UNIQUE COMBINATIONS OF COLUMN VALUES ===#
@log_execution_time
def split_dataframe(df, group_columns):
    """
    Split a DataFrame into multiple sub-DataFrames based on unique combinations 
    of values in the specified columns.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be split.
    - group_columns (list of str): Column names used for grouping.

    Returns:
    - dict: A dictionary where keys are tuples representing unique value combinations 
      in `group_columns`, and values are the corresponding sub-DataFrames.

    The function groups the DataFrame by the specified columns, then creates 
    separate sub-DataFrames for each unique combination of values. The results 
    are stored in a dictionary, allowing easy access to different subsets.
    """
    # Create a dictionary to store the split DataFrames
    split_dfs = {}

    # Group the DataFrame by the specified columns
    grouped = df.groupby(group_columns)

    # Iterate over each group
    for group_values, group_df in grouped:
        # Convert group values to a concatenated string of numbers
        key = ''.join(map(str, group_values))
        # Retain all columns in the resulting DataFrame
        split_dfs[key] = group_df.reset_index(drop=True)

    return split_dfs

#=== FUNCTION TO DETERMINE MODEL TYPE OF A MACINE LEARNING OBJECT ===#
def get_model_type(model):
    """
    Identify whether the given machine learning model is a classifier or a regressor.

    Parameters:
    - model (object): A machine learning model instance, typically from scikit-learn.

    Returns:
    - str: A string indicating the model type:
      - `"classification"` if the model is a classifier.
      - `"regression"` if the model is a regressor.
      - `"unknown"` if the model type cannot be determined.

    The function checks the `_estimator_type` attribute, which is commonly found in 
    scikit-learn models, to determine whether the model is intended for classification 
    or regression.
    """
    # Check if the model has the '_estimator_type' attribute
    if hasattr(model, '_estimator_type'):
        estimator_type = model._estimator_type
        if estimator_type == 'classifier':
            return "classification"
        elif estimator_type == 'regressor':
            return "regression"
        else:
            return "unknown"
    else:
        return "unknown"







#=== FUNCTION FOR LATIN HYPERCUBE SAMPLING WITH MIXED VARIABLE TYPES ===#
@log_execution_time
def latin_hypercube_sampling_with_bounds_and_types(num_samples, lower_bounds, upper_bounds, var_types, seed=None):
    """
    Perform Latin Hypercube Sampling (LHS) for mixed variable types (both float and integer).

    Parameters:
    - num_samples (int): The number of samples to generate.
    - lower_bounds (list or numpy.ndarray): Lower bounds for each variable.
    - upper_bounds (list or numpy.ndarray): Upper bounds for each variable.
    - var_types (list of str): Specifies whether each variable is `"int"` or `"float"`.
    - seed (int, optional): Random seed for reproducibility (default is None).

    Returns:
    - numpy.ndarray: A sampled array of shape (num_samples, n_vars), where 
      integer variables are rounded appropriately.

    The function first generates samples using Latin Hypercube Sampling, scales 
    them based on the provided bounds, and rounds integer variables while 
    keeping float variables unchanged.
    """
    from scipy.stats import qmc
    import numpy as np

    n_vars = len(lower_bounds)
    sampler = qmc.LatinHypercube(d=n_vars, seed=seed)
    samples = sampler.random(n=num_samples)
    scaled = samples * (np.array(upper_bounds) - np.array(lower_bounds)) + np.array(lower_bounds)
    for i, vtype in enumerate(var_types):
        if vtype == "int":
            scaled[:, i] = np.round(scaled[:, i]).astype(int)
    return scaled