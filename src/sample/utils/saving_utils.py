import os
import joblib
# IMPORT SYSTEM MODULES
import sys
import os
import seaborn as sns
import torch
import pandas as pd
import json


from src.sample.config import *
#=== FUNCTION TO SAVE TRAINED MODEL ===#
import torch
import os

import torch
import os

import os
import torch

def save_model(model, dirname, hyperparam_config, filename="trained_controller"):
    """
    Saves the model weights and config using the models/date/date_and_time/ structure.
    """
    # Assuming 'date' and 'date_and_time' are defined globally 
    # or extracted from your training session context
    global_date = date # e.g., "2026-04-29"
    timestamp = date_and_time # e.g., "2026-04-29_11-22"

    # Construct the directory and filename logic
    model_dir = f"models/{global_date}/{timestamp}/{dirname}/"
    save_filename = f"{timestamp}_{filename}.pt"
    
    os.makedirs(model_dir, exist_ok=True)
    full_path = os.path.join(model_dir, save_filename)
    
    # Check path length (keeping consistent with your CSV function)
    max_path_length = 255
    if len(full_path) > max_path_length:
        basename, ext = os.path.splitext(save_filename)
        allowed_len = max_path_length - len(os.path.join(model_dir, ext))
        save_filename = basename[:allowed_len] + ext
        full_path = os.path.join(model_dir, save_filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': hyperparam_config,
    }
    
    torch.save(checkpoint, full_path)
    print(f"💾 Model saved to: {full_path}")


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




def save_to_json(data, dirname, filename, max_path_length=255):
    """
    Saves a dictionary or pandas DataFrame as a JSON file with pretty formatting.
    """
    # 1. Handle the input type
    # If it's a DataFrame, convert to a list of dicts (records)
    if isinstance(data, pd.DataFrame):
        data = data.round(2)
        data_to_save = data.to_dict(orient="records")
    else:
        data_to_save = data

    # 2. Construct paths (using your specific global date variables)
    # Ensure these variables (date, date_and_time) are defined in your script
    dirname = f"reports/{date}/{date_and_time}/{dirname}/"
    filename = f"{date_and_time}_{filename}.json"
    
    os.makedirs(dirname, exist_ok=True)
    full_path = os.path.join(dirname, filename)

    # 3. Handle Path Length Truncation
    if len(full_path) > max_path_length:
        basename, ext = os.path.splitext(filename)
        allowed_len = max_path_length - len(os.path.join(dirname, ext))
        filename = basename[:allowed_len] + ext
        print(f"⚠️ Filename truncated to: {filename}")

    final_path = os.path.join(dirname, filename)

    # 4. Write the file
    with open(final_path, 'w', encoding='utf-8') as f:
        # indent=4 makes the JSON readable (pretty-print)
        # ensure_ascii=False handles special characters correctly
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)
    
    print(f"✅ JSON saved: {final_path}")