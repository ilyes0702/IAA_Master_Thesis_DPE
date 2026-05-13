import matplotlib.pyplot as plt
from src.sample.config import date, date_and_time
from src.sample.config import get_run_description, log_message
plt.style.use("src/sample/style.mplstyle")
import torch

# Choose your plant model here - swap between different process models
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
    #controller_path = f"models/2026-05-08/2026-05-08_17-43-14/GPUChemostatPlant_training/2026-05-08_17-43-14_trained_controller.pt" # best one so far, D_center = 0.25


    #controller_path = "models/2026-05-11/2026-05-11_10-49-41/GPUChemostatPlant_training/2026-05-11_10-49-41_trained_controller.pt" # D_center = 0.4

    #controller_path = "models/2026-05-11/2026-05-11_14-12-30/GPUChemostatPlant_training/2026-05-11_14-12-30_trained_controller.pt"
    # D_center randomized, trained on 300 epochs

    #controller_path = "models/2026-05-11/2026-05-11_15-44-54/GPUChemostatPlant_training/2026-05-11_15-44-54_trained_controller.pt"
    # D_center randomzed, trained on 30 epochs

    #controller_path= "models/2026-05-12/2026-05-12_14-46-16/GPUChemostatPlant_training/2026-05-12_14-46-16_trained_controller_disk.pt"
    # Huber loss function

    #controller_path = "models/2026-05-12/2026-05-12_16-49-43/GPUChemostatPlant_training/2026-05-12_16-49-43_trained_controller_disk.pt"
    # trained with randomized D_cnter and larger p, good results, trained on 5000 sequences


    controller_path = "models/2026-05-13/2026-05-13_12-18-55/GPUChemostatPlant_training_None/2026-05-13_12-18-55_trained_controller_disk.pt"
    # trained on 500 sequences
    
    loaded_controller = load_model(controller_path)


    GPUSimulateTracking(
        model=loaded_controller,
        plant=plant,
        hyperparam_config=hyperparam_config,
        dirname=plant.__class__.__name__
    )
