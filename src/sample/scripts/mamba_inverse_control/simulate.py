import matplotlib.pyplot as plt


plt.style.use("src/sample/style.mplstyle")
import torch
import pandas as pd

# Choose your plant model here - swap between different process models
from src.sample.classes.PenicilinFermentationProcessTropophase import FermentationProcess, GPUFermentationProcess
from src.sample.classes.SimpleLinearPlant import GPUSimpleLinearPlant
from src.sample.utils.saving_utils import save_df_to_csv

from src.sample.classes.MambaInverseController import MambaInverseController
from src.sample.utils.general_utils import GPUSimulateControl, GPUSimulateControl_new
from src.sample.utils.general_utils import load_controller

from src.hyperparam_config import hyperparam_config


# --- 1. Device Configuration --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    
    batch_size = 200
    #plant = GPUFermentationProcessFFT(batch_size=batch_size, device=device)
    plant = GPUSimpleLinearPlant(batch_size=batch_size, hyperparam_config=hyperparam_config, device=device)    
    dirname = plant.__class__.__name__

    controlller_path = f"models/2026-04-28/2026-04-28_14-16-07/GPUSimpleLinearPlant/2026-04-28_14-16-07_trained_controller.pt"
    loaded_controller, config = load_controller(MambaInverseController, controlller_path, device)

    # 2. Simulation
    GPUSimulateControl_new(
    model=loaded_controller, 
    plant=plant, 
    hyperparam_config=hyperparam_config,           # The central dictionary
    dirname=plant.__class__.__name__
    )
    
    flattened_dict = {}
    for section, params in hyperparam_config.items():
        if isinstance(params, dict):
            for key, val in params.items():
                flattened_dict[f"{section}_{key}"] = val
        else:
            flattened_dict[section] = params

    # 2. Convert to a DataFrame (one row)
    df_hyperparams = pd.DataFrame([flattened_dict])

    # 3. Save using your custom function
    save_df_to_csv(
        df=df_hyperparams, 
        dirname=plant.__class__.__name__, 
        filename="hyperparameters"
    )