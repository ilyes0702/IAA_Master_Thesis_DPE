import torch
from src.sample.classes.ChemostatPlant import ChemostatPlant
from src.hyperparam_config import hyperparam_config
from src.sample.classes.MambaInverseController import MambaInverseController
from src.sample.utils.training_utils import *
from src.sample.config import *
from src.sample.utils.saving_utils import *

# --- 1. Device Configuration --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # 0. Log Run Description
    #run_description = get_run_description()
    #log_message(f"RUN DESCRIPTION: {run_description}")
    

    # Initialize controller
    #controller = MambaInverseController(hyperparam_config=hyperparam_config).to(device)
    
    # Initialize plant    
    plant = ChemostatPlant(hyperparam_config=hyperparam_config) 
  
    #dataset_path = "results/2026-05-13/2026-05-13_11-33-29/GPUChemostatPlant_training_ata/dataset/2026-05-13_11-33-29_training_data.pt"    

    dataset_path = "results/2026-05-19/2026-05-19_10-22-09/ChemostatPlant_training_data/dataset/2026-05-19_10-22-09_training_data.pt"

    dataset_path = "results/2026-05-19/2026-05-19_16-58-14/ChemostatPlant_training_data/dataset/2026-05-19_16-58-14_training_data.pt"

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
        metadata_df = pd.DataFrame({
            "experiment_sequences": [str(count)],
            "source_dataset_path": [dataset_path]
        })
        
        # Saves into: results/.../ChemostatPlant_training_None/reports/..._dataset_source_log.csv
        save_df_to_csv(metadata_df, dirname=dirname, filename=f"dataset_source_log")
        print(f"📝 Logged source dataset configuration path.")

        # 3. Get a fresh model (otherwise the 200-run starts with 100-run weights)
        current_controller = get_fresh_model()
        
        # 4. Train
        train_controller_lr_decay(
            current_controller, 
            dataset_path, 
            hyperparam_config, 
            dirname=dirname, 
            num_sequences_to_use=count
        )
        
        print(f"✅ Finished training for {count} sequences.")