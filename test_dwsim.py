import os

# --- CRITICAL FIX FOR LINUX ---
# You must tell pythonnet to use the DotNet Core runtime 
# before importing 'clr'
from pythonnet import load
try:
    load("coreclr")
except Exception as e:
    # If already loaded, this might throw an error, which we can ignore
    print(f"Runtime info: {e}")

import clr
import sys

# Now continue with your existing path setup...
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DWSIM_PATH = os.path.join(BASE_DIR, "dwsim_bin")
sys.path.append(DWSIM_PATH)

print(f"--- DWSIM Integration Test ---")
print(f"Base Directory: {BASE_DIR}")
print(f"Looking for DLLs in: {DWSIM_PATH}")

# --- 2. LOAD LIBRARIES ---
try:
    # Explicitly load the core automation libraries
    clr.AddReference("DWSIM.Automation")
    clr.AddReference("DWSIM.Interfaces")
    
    from DWSIM.Automation import Automation
    from DWSIM.Interfaces import IFlowsheet
    print("✓ DWSIM Libraries loaded successfully.")
except Exception as e:
    print(f"X Error loading DWSIM DLLs: {e}")
    print("Ensure all files from the portable version are in 'dwsim_bin'.")
    sys.exit(1)

# --- 3. RUN SIMULATION ---
try:
    # Initialize the Automation Manager
    interf = Automation()
    
    if not os.path.exists(SIM_PATH):
        print(f"X Simulation file not found at: {SIM_PATH}")
        sys.exit(1)

    print(f"Opening simulation: {os.path.basename(SIM_PATH)}...")
    sim = interf.LoadFlowsheet(SIM_PATH)
    
    if sim is not None:
        print("✓ Flowsheet loaded into memory.")
        
        # Trigger the solver
        print("Calculating...")
        interf.CalculateFlowsheet(sim, None)
        
        if sim.Solved:
            print("✓ SUCCESS: Flowsheet solved!")
            
            # Example: Extract one piece of data to prove it worked
            # (Assuming you have a stream named 'MSTR-001' in your file)
            try:
                obj = sim.GetFlowsheetSimulationObject("MSTR-001")
                if obj:
                    print(f"Stream 'MSTR-001' Pressure: {obj.GetPressure()} Pa")
            except:
                pass # Stream name might be different
        else:
            print(f"X Solve failed. Error: {sim.ErrorMessage}")
    else:
        print("X Failed to initialize flowsheet object.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")