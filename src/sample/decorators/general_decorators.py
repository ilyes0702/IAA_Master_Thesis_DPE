from functools import wraps
from PIL import Image
import time
import pandas as pd
from datetime import datetime
import os
import sys
# ADD PROJECT DIRECTORY TO SYSTEM PATH
if '__file__' in globals():
    base_dir = os.path.dirname(os.path.abspath(__file__))
else:
    # fallback to current working directory
    base_dir = os.getcwd()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *
import matplotlib.pyplot as plt
#from dictionaries.variable_names import axis_label_mapping


#=== DECORATOR TO SAVE IMAGE ===#
def save_image_decorator(func):
    """
    A decorator that saves the image returned by a function to a specified filename.

    The decorated function must return a PIL Image object and must be called with 
    'filename' and 'dirname' as keyword arguments to specify the storage location.

    Parameters:
        func (callable): The function being decorated. It must return a PIL Image object.

    Returns:
        callable: The wrapped function that saves the image before returning it.

    Raises:
        ValueError: If the decorated function is not called with 'filename' or 'dirname'.
        TypeError: If the decorated function does not return a PIL Image object.

    Additional Details:
        - Creates the specified directory if it does not exist.
        - Saves the image in the format `<dirname>/<date>/<date_and_time>/<filename>.png`.
        - Logs a success message upon saving the image.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the original function
        image = func(*args, **kwargs)
        
        # Check if 'filename' argument is provided
        if 'filename' in kwargs:
            filename = kwargs['filename']
        else:
            raise ValueError("The function must be called with a 'filename' argument specifying the image save name.")
        
        if "dirname" in kwargs:
                    dirname = kwargs['dirname']
        else:
            raise ValueError("The function must be called with a 'dirname' argument specifying the image save directory name.")

        # Save the image
        if isinstance(image, Image.Image):
            path = f"plots/{date}/{date_and_time}/{dirname}"
            os.makedirs(path, exist_ok=True)
            image.save(f"{path}/{date_and_time}_{filename}.png")
            log_message(f"Image successfully saved as '{filename}'")
        else:
            raise TypeError("The function must return a PIL Image object.")
        
        return image
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

#=== DECORATOR TO SAVE DATAFRAME TO CSV (NOT IN USE CURRENTLY)===#
def save_dataframe_to_csv():
    """
    Decorator to save a DataFrame returned by a function as a CSV file.

    Args:
        filename (str): The name of the CSV file to save the DataFrame to.

    Returns:
        Wrapper function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the original function to generate the image
            result = func(*args, **kwargs)

            # Check if 'filename' argument is provided
            if 'filename' in kwargs:
                filename = kwargs['filename']
            else:
                raise ValueError("The function must be called with a 'filename' argument specifying the csv save name.")
            
            if "dirname" in kwargs:
                    dirname = kwargs['dirname']
            else:
                raise ValueError("The function must be called with a 'dirname' argument specifying the image save directory name.")


            # Check if the result is a pandas DataFrame
            if isinstance(result, pd.DataFrame):
                date = datetime.now().strftime("%Y-%m-%d")
                date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
                path_name = "./reports/"+date+"/"+dirname+"/"+filename+"_"+date_and_time+".csv"
                
                if os.path.exists("./reports/"+date+"/"+dirname):
                    result.to_csv(path_name)
                else:
                    os.makedirs("./reports/"+date+"/"+dirname)
                    result.to_csv(path_name)
            
                log_message(f"Datarame successfully saved as '{filename}'")

            else:
                raise TypeError("The function must return a pandas DataFrame object.")
            
            return result  # Return the original result
        return wrapper
    return decorator