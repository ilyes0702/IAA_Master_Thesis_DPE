import os
import joblib
# IMPORT SYSTEM MODULES
import sys
import os
import seaborn as sns
# ADD PROJECT DIRECTORY TO SYSTEM PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 
                    "D:/Stage_IAA_LRGP")))
from src.sample.config import *
#=== FUNCTION TO SAVE TRAINED MODEL ===#
def save_model(model, dirname, filename):
    """
    Save a trained machine learning model to a specified file path, ensuring that
    the target directory exists.

    Parameters:
    - model (object): The trained model to be saved (e.g., a scikit-learn model).
    - trained_model_path_name (str): Full path, including filename, where the model should be saved.
    - path_name_mod (str): Directory path where the model file should be stored.

    Returns:
    - None: The function saves the model file but does not return anything.

    This function first checks whether the target directory exists. If it does not, 
    it creates the directory before saving the model using joblib.
    """
    dirname = f"models/{date}/{date_and_time}/{dirname}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = f"{date_and_time}_{filename}"
    joblib.dump(model, dirname+filename)


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
    dirname = f"reports/{date}/{date_and_time}/{dirname}"
    filename = f"{date_and_time}_{filename}"
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