import time
import pandas as pd
from seqControl.sample.config import *
#from dictionaries.variable_names import axis_label_mapping

from seqControl.sample.utils.saving_and_loading_utils import save_df_to_csv
import functools
import torch

#=== DECORATOR TO TRACK GPU RESOURCES ===#
def track_resources(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs), 0.0
            
        # 1. Prepare GPU
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats() # Reset the "high-water mark"
        start_time = time.perf_counter()
        
        # 2. Execute Training
        result = func(*args, **kwargs)
        
        # 3. Finalize
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 4. Extract Metrics
        total_sec = end_time - start_time
        gpu_min = total_sec / 60
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_gb = peak_bytes / (1024**3) # Convert bytes to Gigabytes
        
        print("\n" + "🚀" + " ="*20)
        print(f"RESOURCE REPORT: {func.__name__}")
        print(f"⏱️  Time Used:  {gpu_min:.4f} GPU-minutes")
        print(f"💾 Peak VRAM:  {peak_gb:.2f} GB")
        print(" ="*20 + "\n")
        
        # Return results + a dictionary of metrics for easy logging
        metrics = {
            "gpu_minutes": gpu_min,
            "peak_vram_gb": peak_gb
        }

        
        resource_df = pd.DataFrame([metrics])
        csv_filename = kwargs.get("resource_filename", f"{func.__name__}_resource_stats")
        csv_dirname = kwargs.get("resource_dirname", "resource_stats")
        save_df_to_csv(resource_df, filename=csv_filename, dirname=csv_dirname)

        return result, metrics
        
    return wrapper

#=== DECORATOR TO LOG EXECUTION TIME ===#
def log_execution_time(func):
    """
    A decorator that measures and logs the execution time of a function.

    The execution time is recorded in seconds and logged using `log_message()`. 
    The log is saved in a file named `'execution_log.txt'`, providing insights 
    into function performance.

    Parameters:
        func (callable): The function being decorated. It can take any arguments and return any value.

    Returns:
        callable: A wrapped version of `func` that logs its execution time.

    Raises:
        None

    Additional Details:
        - Uses `time.time()` to capture start and end times.
        - Formats timestamps using `time.strftime("%Y-%m-%d %H:%M:%S")` for readability.
        - Logs execution time in a structured message format: 
          `"function_name executed in X.XXXXX seconds"`.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start timing
        log_message(f"STARTED {func.__name__}")
        result = func(*args, **kwargs)  # Execute the original function
        end_time = time.time()  # End timing
        execution_time = end_time - start_time
        
        # Get the current time in a readable format
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        
        # Log the execution time to a file with the time format
        log_message(f"ENDED {func.__name__} executed in {execution_time:.5f} seconds")
        
        return result  # Return the original function's result
    return wrapper
