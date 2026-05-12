import torch
from Archive.GPUChemostatPlant_with_delay import GPUChemostatPlant_with_delay
from src.sample.classes.GPUChemostatPlant import GPUChemostatPlant
from src.hyperparam_config import hyperparam_config
from src.sample.classes.PenicilinFermentationProcessTropophase import GPUFermentationProcessFFT
from src.sample.classes.MambaInverseController import MambaInverseController
from src.sample.utils.training_utils import GPUtrain_controller_from_disk, GPUtrain_controllerFFT
from src.sample.classes.SimpleLinearPlant import GPUSimpleLinearPlant
from src.sample.config import *
from src.sample.utils.saving_utils import save_to_json

# --- 1. Device Configuration --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # 0. Log Run Description
    run_description = get_run_description()
    log_message(f"RUN DESCRIPTION: {run_description}")
    

    # 1. Initialize controller
    controller = MambaInverseController(hyperparam_config=hyperparam_config).to(device)
    
    # Initialize plant    
    #plant = GPUSimpleLinearPlant(hyperparam_config=hyperparam_config)  
    plant = GPUChemostatPlant(hyperparam_config=hyperparam_config) 
    #plant = GPUFermentationProcessFFT(hyperparam_config=hyperparam_config) 
    dirname=plant.__class__.__name__ + "_training"
    save_to_json(hyperparam_config, dirname, filename="hyperparameters")
    # 3. Pre-generate the training trajectory for the plant    
    # 4. Run Training    
    # GPUtrain_controllerFFT(
    #     model=controller, 
    #     plant=plant, 
    #     hyperparam_config=hyperparam_config, 
    #     dirname=plant.__class__.__name__ + "_training"
    # )

    dataset_path = "results/2026-05-12/2026-05-12_16-48-07/GPUChemostatPlant_training_ata/dataset/2026-05-12_16-48-07_training_data.pt"

    GPUtrain_controller_from_disk(controller, dataset_path, hyperparam_config, dirname)
    