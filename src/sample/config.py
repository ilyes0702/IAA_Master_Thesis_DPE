from datetime import datetime
import os
import logging



# === Load or initialize run_id === #

date = datetime.now().strftime("%Y-%m-%d")
date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



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



