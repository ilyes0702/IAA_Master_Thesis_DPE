import os
import joblib
# IMPORT SYSTEM MODULES
import sys
import os
import seaborn as sns
import torch

from src.sample.config import *
#=== FUNCTION TO SAVE TRAINED MODEL ===#
def save_model(model, model_config, dirname, filename):
    """
    Saves weights AND hyperparameters into one file.
    
    Parameters:
    - model: The trained model object.
    - model_config (dict): Dictionary of hyperparams (e.g., {'d_model': 16, 'n_layers': 2})
    """
    full_dir_path = os.path.join("models", date, date_and_time, dirname)
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)
    
    full_file_path = os.path.join(full_dir_path, f"{date_and_time}_{filename}.pt")

    # Combine everything into one dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        # You can even save training state:
        'date': date,
        'date_and_time': date_and_time
    }
    
    torch.save(checkpoint, full_file_path)
    print(f"Complete model package saved to: {full_file_path}")


#=== FUNCTION FOR SAVING DATAFRAME AS CSV IN SPECIFIED DIRECTORY ===#
def save_df_to_csv(df, dirname, filename, max_path_length=255):
    """
    Save a pandas DataFrame as a CSV file in a specified directory, ensuring 
    that the directory exists before writing the file.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be saved.
    - dirname (str): The directory path where the CSV file will be stored.
    - filename (str): The name of the CSV file (including the .csv extension).

    Returns:
    - None: The function saves the DataFrame but does not return anything.

    This function rounds all float columns to two decimal places before saving. 
    If the specified directory does not exist, it is created automatically.
    """
    # Round all float columns to 2 decimal places
    df = df.round(2)
    
     # Check full path length
    dirname = f"reports/{date}/{date_and_time}/{dirname}/"
    filename = f"{date_and_time}_{filename}.csv"
    os.makedirs(dirname, exist_ok=True)
    full_path = os.path.join(dirname, filename)
    if len(full_path) > max_path_length:
        basename, ext = os.path.splitext(filename)
        # Calculate how much to trim
        allowed_len = max_path_length - len(os.path.join(dirname, ext))
        new_basename = basename[:allowed_len]
        filename = new_basename + ext
        print(f"Filename was too long, truncated to: {filename}")
    df.to_csv(dirname+ filename, index=False)