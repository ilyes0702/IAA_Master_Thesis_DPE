import torch
from src.sample.classes.GPUChemostatPlant import GPUChemostatPlant
from src.hyperparam_config import hyperparam_config
from src.sample.classes.MambaInverseController import MambaInverseController
from src.sample.utils.training_utils import GPUtrain_controller_from_disk
from src.sample.classes.SimpleLinearPlant import GPUSimpleLinearPlant
from src.sample.config import *
from src.sample.utils.saving_utils import save_to_json

# --- 1. Device Configuration --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # 0. Log Run Description
    #run_description = get_run_description()
    #log_message(f"RUN DESCRIPTION: {run_description}")
    

    # Initialize controller
    #controller = MambaInverseController(hyperparam_config=hyperparam_config).to(device)
    
    # Initialize plant    
    plant = GPUChemostatPlant(hyperparam_config=hyperparam_config) 
  
    dataset_path = "results/2026-05-13/2026-05-13_11-33-29/GPUChemostatPlant_training_ata/dataset/2026-05-13_11-33-29_training_data.pt"



    # dirname_100 =plant.__class__.__name__ + "_training_100"
    # dirname_200 =plant.__class__.__name__ + "_training_200"
    # save_to_json(hyperparam_config, dirname_100, filename="hyperparameters_100")
    # save_to_json(hyperparam_config, dirname_200, filename="hyperparameters_200")


    # GPUtrain_controller_from_disk(controller, dataset_path, hyperparam_config, dirname_100, num_sequences_to_use=100)
    # GPUtrain_controller_from_disk(controller, dataset_path, hyperparam_config, dirname_200, num_sequences_to_use=200)
    

    # Define the data steps you want to test
#data_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
data_steps = [None]

# If you want to re-initialize the model for each run (recommended for benchmarking)
def get_fresh_model():
    # Replace this with your actual model initialization code
    return MambaInverseController(hyperparam_config=hyperparam_config).to(device)

for count in data_steps:
    print(f"\n--- Starting Training Experiment with {count} sequences ---")
    
    # 1. Create a specific directory name
    dirname = f"{plant.__class__.__name__}_training_{count}"
    
    # 2. Save the config for this run
    save_to_json(hyperparam_config, dirname, filename=f"hyperparameters_{count}")
    
    # 3. Get a fresh model (otherwise the 200-run starts with 100-run weights)
    current_controller = get_fresh_model()
    
    # 4. Train
    loss_history = GPUtrain_controller_from_disk(
        current_controller, 
        dataset_path, 
        hyperparam_config, 
        dirname=dirname, 
        num_sequences_to_use=count
    )
    
    print(f"✅ Finished training for {count} sequences.")