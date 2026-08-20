import matplotlib.pyplot as plt
plt.style.use("src/sample/style.mplstyle")
import torch

# Choose your plant model here - swap between different process models
from src.sample.classes.PenicilinFermentationProcessTropophase import FermentationProcess, GPUFermentationProcess
from seqControl.sample.classes.plants.SimpleLinearPlant import SimpleLinearPlant

from seqControl.sample.classes.controllers.MambaInverseController import MambaInverseController
from seqControl.sample.utils.general_utils import train_controller, seed_everything, GPUtrain_controller

# --- 1. Device Configuration --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
seed_everything(42)  # Set a global seed for reproducibility


if __name__ == "__main__":
    dt = 0.01
    
    # 1. Define model parameters (The "Skeleton" info)
    model_config = {
        "d_model": 16,
        "d_state": 32,
    }

    # 2. Initialize Model and Plant
    controller = MambaInverseController(**model_config).to('cuda')
    plant = GPUFermentationProcess()  # Set a seed for reproducibility
    dirname = plant.__class__.__name__
    # 3. Run Training
    # The function now saves weights + mamba_config into one .pt file
    # train_controller(
    #     model=controller,
    #     plant=plant,
    #     epochs=8000,
    #     seq_len=10000,
    #     dt=dt,
    #     model_config=model_config, # Pass it here!
    #     dirname=dirname
    # )

    GPUtrain_controller(
        model=controller,
        plant=plant,
        epochs=100,         # High epoch count is fine now because it's faster
        seq_len=10000,       # Length of each fermentation run
        dt=dt,
        model_config=model_config,
        batch_size=256,      # <--- Start with 128, increase if you have >8GB VRAM
        device='cuda',
        dirname=dirname
    )