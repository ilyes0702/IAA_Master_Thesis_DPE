import os
import torch
import pandas as pd
import json

import os
from PIL import Image

from src.sample.config import date as default_date
from src.sample.config import date_and_time as default_date_and_time

import pickle

#=== FUNCTION TO SAVE TRAINED MODEL ===#
def save_model(model, dirname, hyperparam_config, filename="trained_controller"):
    """
    Saves the model weights and configuration using the models/date/date_and_time/ directory structure.

    Parameters:
    - model: The neural network model whose weights will be saved.
    - dirname (str): The subdirectory path where the model will be stored.
    - hyperparam_config: The hyperparameter configuration dictionary to be saved with the model.
    - filename (str): The name of the model file (default: "trained_controller").

    Returns:
    - None: The function saves the model but does not return anything.

    The model weights and configuration are saved as a PyTorch checkpoint file (.pt) in the
    models/<date>/<date_and_time>/<dirname>/ directory. If the file path exceeds the maximum
    allowed length (255 characters), the filename is automatically truncated.
    """
    # Assuming 'date' and 'date_and_time' are defined globally 
    # or extracted from your training session context
    global_date = default_date # e.g., "2026-04-29"
    timestamp = default_date_and_time # e.g., "2026-04-29_11-22"

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


#=== FUNCTION TO SAVE DATAFRAME AS CSV IN SPECIFIED DIRECTORY ===#
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
    df = df.round(4)
    
     # Check full path length
    dirname = f"results/{default_date}/{default_date_and_time}/{dirname}/reports/"
    filename = f"{default_date_and_time}_{filename}.csv"
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



#=== FUNCTION TO SAVE JSON FILE IN SPECIFIED DIRECTORY ===#
def save_to_json(data, dirname, filename, max_path_length=255):
    """
    Saves a dictionary or pandas DataFrame as a JSON file with pretty formatting.

    Parameters:
    - data (dict or pandas.DataFrame): The data to be saved as JSON. If a DataFrame is provided,
      it will be converted to a list of dictionaries (records format).
    - dirname (str): The directory path where the JSON file will be stored.
    - filename (str): The name of the JSON file (without the .json extension).
    - max_path_length (int): The maximum allowed path length in characters (default: 255).

    Returns:
    - None: The function saves the data but does not return anything.

    The file is saved in the results/<date>/<date_and_time>/<dirname>/reports/ directory with
    pretty formatting (indent=4). If the specified directory does not exist, it is created
    automatically. Float columns in DataFrames are rounded to two decimal places before saving.
    If the full path exceeds max_path_length, the filename is automatically truncated.
    """
    # 1. Handle the input type
    # If it's a DataFrame, convert to a list of dicts (records)
    if isinstance(data, pd.DataFrame):
        data = data.round(4)
        data_to_save = data.to_dict(orient="records")
    else:
        data_to_save = data

    # 2. Construct paths (using your specific global date variables)
    # Ensure these variables (date, date_and_time) are defined in your script
    dirname = f"results/{default_date}/{default_date_and_time}/{dirname}/reports/"
    filename = f"{default_date_and_time}_{filename}.json"
    
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



#=== FUNCTION TO SAVE PLOT IMAGE IN SPECIFIED DIRECTORY ===#
def save_plot_image(image, filename, dirname):
    """
    Saves a PIL Image object to a structured directory path.

    Parameters:
    - image (PIL.Image.Image): The PIL Image object to be saved.
    - filename (str): The name of the image file (without the .png extension).
    - dirname (str): The subdirectory path where the image will be stored.

    Returns:
    - None: The function saves the image but does not return anything.

    The image is saved in the results/<date>/<date_and_time>/<dirname>/plots/ directory as
    <date_and_time>_<filename>.png. If the specified directory does not exist, it is created
    automatically. The image parameter must be a valid PIL Image object; a TypeError is raised
    if it is not.
    """
    # 1. Validation
    if not isinstance(image, Image.Image):
        raise TypeError("The 'image' argument must be a PIL Image object.")
    
    # 2. Construct the directory path
    # Using your specific format: results/{date}/{date_and_time}/{dirname}/plots
    path = f"results/{default_date}/{default_date_and_time}/{dirname}/plots"
    
    # 3. Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # 4. Define full file path
    full_path = f"{path}/{default_date_and_time}_{filename}"
    if not full_path.endswith(".png"):
        full_path += ".png"
    
    # 5. Save and Log
    image.save(full_path)
    print(f"Image successfully saved to: {full_path}")
    
    return()

#=== FUNCTION TO SAVE TRAINING DATASET TENSORS ===#
def save_training_dataset(data_dict, dirname, filename="training_data"):
    """
    Saves the training tensors to a PyTorch .pt file using structured directory logic.

    Parameters:
    - data_dict (dict): A dictionary containing the training tensors (e.g., 'x' and 'y' keys).
    - dirname (str): The subdirectory path where the dataset file will be stored.
    - filename (str): The name of the dataset file (default: "training_data").

    Returns:
    - None: The function saves the data but does not return anything.

    The dataset tensors are saved in the results/<date>/<date_and_time>/<dirname>/dataset/
    directory as <date_and_time>_<filename>.pt. If the specified directory does not exist,
    it is created automatically. If the full path exceeds 255 characters, the filename is
    automatically truncated to ensure compatibility with the filesystem.
    """
    # Construct directory logic consistent with your other functions
    target_dir = f"results/{default_date}/{default_date_and_time}/{dirname}/dataset/"
    save_filename = f"{default_date_and_time}_{filename}.pt"
    
    os.makedirs(target_dir, exist_ok=True)
    full_path = os.path.join(target_dir, save_filename)
    
    # Path length safety check (255 chars)
    max_path_length = 255
    if len(full_path) > max_path_length:
        basename, ext = os.path.splitext(save_filename)
        allowed_len = max_path_length - len(os.path.join(target_dir, ext))
        save_filename = basename[:allowed_len] + ext
        full_path = os.path.join(target_dir, save_filename)

    # Save the dictionary containing the tensors
    torch.save(data_dict, full_path)
    print(f"📦 Dataset Tensors saved to: {full_path}")


    #=== FUNCTION TO SAVE SCALER OBJECTS IN SPECIFIED DIRECTORY ===#
def save_scaler_object(scaler, dirname, filename, max_path_length=255):
    """
    Saves a scikit-learn or custom scaler object to a pickle (.pkl) file.

    Parameters:
    - scaler: The scaler object (e.g., MinMaxScaler, StandardScaler) to be saved.
    - dirname (str): The subdirectory path where the scaler will be stored.
    - filename (str): The name of the file (without or with the .pkl extension).
    - max_path_length (int): The maximum allowed path length in characters (default: 255).

    Returns:
    - None: The function saves the object but does not return anything.

    The scaler is saved in the results/<date>/<date_and_time>/<dirname>/scalers/
    directory as <date_and_time>_<filename>.pkl. If the specified directory does not exist,
    it is created automatically. If the full path exceeds max_path_length, the filename is 
    automatically truncated.
    """
    # 1. Standardize file extension
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    # 2. Construct paths using your specific global variables
    target_dir = f"results/{default_date}/{default_date_and_time}/{dirname}/scalers/"
    save_filename = f"{default_date_and_time}_{filename}"
    
    os.makedirs(target_dir, exist_ok=True)
    full_path = os.path.join(target_dir, save_filename)

    # 3. Handle Path Length Truncation
    if len(full_path) > max_path_length:
        basename, ext = os.path.splitext(save_filename)
        allowed_len = max_path_length - len(os.path.join(target_dir, ext))
        save_filename = basename[:allowed_len] + ext
        full_path = os.path.join(target_dir, save_filename)
        print(f"⚠️ Filename too long, truncated to: {save_filename}")

    # 4. Write the binary pickle file
    with open(full_path, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"💾 Scaler successfully saved to: {full_path}")