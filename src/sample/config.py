from datetime import datetime
import os
import logging

import pytz

# === Load or initialize run_id === #
tz = pytz.timezone('Europe/Berlin')
date = datetime.now(tz).strftime("%Y-%m-%d")
date_and_time = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")



path_name = "logs/" + date + "/"

if not os.path.exists(path_name):
    os.makedirs(path_name)

exlog_path_name = path_name + date_and_time + "_execution_log.txt"


# Configure logging to display ERROR and above
logging.basicConfig(filename=exlog_path_name, 
                    level=logging.INFO, 
                    format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

def log_message(message):
    """
    Log a message to the log file.
    """
    logging.info(message)

# --- Prompt for run description ---
print("\n" + "="*30)
run_description = input("Describe this run: ")
print("="*30 + "\n")

# Log the description immediately so it's the first thing in the file
log_message(f"RUN DESCRIPTION: {run_description}")



