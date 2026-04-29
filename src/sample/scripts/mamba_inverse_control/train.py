import torch
from src.hyperparam_config import hyperparam_config
from src.sample.classes.PenicilinFermentationProcessTropophase import GPUFermentationProcessFFT
from src.sample.classes.MambaInverseController import MambaInverseController
from src.sample.utils.general_utils import GPUtrain_controllerFFT, seed_everything
from src.sample.classes.SimpleLinearPlant import GPUSimpleLinearPlant
from src.sample.config import *
from src.sample.utils.saving_utils import save_to_json

# --- 1. Device Configuration --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

if __name__ == "__main__":
    # 0. Log Run Description
    run_description = get_run_description()
    log_message(f"RUN DESCRIPTION: {run_description}")

    # 1. Initialize controller
    controller = MambaInverseController(hyperparam_config=hyperparam_config).to(device)
    
    # Initialize plant    
    plant = GPUSimpleLinearPlant(hyperparam_config=hyperparam_config)    
    
    # 3. Pre-generate the training trajectory for the plant
    plant.reset_trajectory()
    
    # 4. Run Training    
    GPUtrain_controllerFFT(
        model=controller, 
        plant=plant, 
        hyperparam_config=hyperparam_config, 
        dirname=plant.__class__.__name__ + "_training"
    )

    save_to_json(hyperparam_config, dirname=plant.__class__.__name__ + "_training", filename="hyperparameters")