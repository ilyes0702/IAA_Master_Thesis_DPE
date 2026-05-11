import matplotlib.pyplot as plt
from src.sample.config import date, date_and_time
from src.sample.config import get_run_description, log_message
plt.style.use("src/sample/style.mplstyle")
import torch

# Choose your plant model here - swap between different process models
from src.sample.classes.PenicilinFermentationProcessTropophase import GPUFermentationProcessFFT
from src.sample.classes.SimpleLinearPlant import GPUSimpleLinearPlant
from src.sample.classes.MambaInverseController import MambaInverseController
from src.sample.utils.general_utils import GPUSimulateTracking
from src.sample.utils.general_utils import load_model
from src.hyperparam_config import hyperparam_config
from src.sample.classes.GPUChemostatPlant import GPUChemostatPlant

# --- 1. Device Configuration --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    # 0. Log Run Description
    run_description = get_run_description()
    log_message(f"RUN DESCRIPTION: {run_description}")
    
    # 1. Initialize plant
    #plant = GPUSimpleLinearPlant(hyperparam_config=hyperparam_config)  
    plant = GPUChemostatPlant(hyperparam_config=hyperparam_config)
    #plant = GPUFermentationProcessFFT(hyperparam_config=hyperparam_config)  

    # 2. Load trained controller
    #controller_path = f"models/2026-05-08/2026-05-08_17-43-14/GPUChemostatPlant_training/2026-05-08_17-43-14_trained_controller.pt" # best one so far


    controller_path = "models/2026-05-11/2026-05-11_09-20-24/GPUChemostatPlant_training/2026-05-11_09-20-24_trained_controller.pt"
    loaded_controller = load_model(controller_path)

    # 3. Simulation
    # GPUSimulateControl_new_ma(
    #     model=loaded_controller, 
    #     plant=plant, 
    #     hyperparam_config=hyperparam_config,          
    #     dirname=plant.__class__.__name__
    # )

    GPUSimulateTracking(
        model=loaded_controller,
        plant=plant,
        hyperparam_config=hyperparam_config,
        dirname=plant.__class__.__name__
    )
