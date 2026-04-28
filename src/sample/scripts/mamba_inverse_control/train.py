import torch
from src.sample.classes.PenicilinFermentationProcessTropophase import GPUFermentationProcessFFT
from src.sample.classes.MambaInverseController import MambaInverseController
from src.sample.utils.general_utils import GPUtrain_controllerFFT, seed_everything
from src.sample.classes.SimpleLinearPlant import GPUSimpleLinearPlant

# --- 1. Device Configuration --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

if __name__ == "__main__":
    dt = 0.01
    seq_len = 10000
    
    # model parameters
    model_config = {
        "d_model": 16,
        "d_state": 32,
    }

    # 2. Initialize Model and the FFT-based Plant
    controller = MambaInverseController(**model_config).to(device)
    
    # Use the FFT version of the plant for Canaday's method
    batch_size = 500
    #plant = GPUFermentationProcessFFT(batch_size=batch_size, device=device)
    plant = GPUSimpleLinearPlant(batch_size=batch_size, device=device)  # For testing with a simpler model
    
    # 3. Pre-generate the Canaday Training Signal (v_train)
    # lambda (5.0) is the frequency cutoff, p (0.4) is the perturbation magnitude
    plant.reset_trajectory(seq_len=seq_len, dt=dt, lambd=5.0, p=0.4)
    
    dirname = plant.__class__.__name__

    # 4. Run Training
    # Note: We use the existing GPUtrain_controller. 
    # To use the pre-generated 'u_buffer' from reset_trajectory, 
    # we ensure the training loop inside GPUtrain_controller pulls from it.
    
    GPUtrain_controllerFFT(
        model=controller,
        plant=plant,
        epochs=1000,
        seq_len=seq_len,
        dt=dt,
        model_config=model_config,
        batch_size=batch_size,
        device=device,
        dirname=dirname
    )