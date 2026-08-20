"""
Generate training data for inverse-control tracking
Sonnleitner yeast fermentation
"""

import numpy as np
import pandas as pd
from sample.utils.saving_and_loading_utils import save_df_to_csv
# ===============================
# Parameters
# ===============================
mu_max = 0.4
K_S = 0.1
S_in = 20.0
dt = 0.01

T = 200
t = np.arange(0, T, dt)

# ===============================
# Reference trajectory
# ===============================
X_ref = 2.0 + 0.5 * np.sin(0.2 * t)
dX_ref = np.gradient(X_ref, dt)

# “If I want biomass X(t) to follow a desired reference trajectory, what dilution rate D(t) would be required according to an approximate inverse model?”
# Consistent growth-rate estimate
mu_est = mu_max * S_in / (K_S + S_in)

D_ref = mu_est - dX_ref / X_ref

#That line limits the computed D_ref values so they stay within the range 0.01 to 0.6.

# If any value in D_ref is less than 0.01, it becomes 0.01
# If any value is greater than 0.6, it becomes 0.6
# Values already between those bounds stay unchanged
# So it “clips” the dilution rate to a realistic range, preventing extreme values that might be infeasible in practice.
D_ref = np.clip(D_ref, 0.01, 0.6)

# ===============================
# Store as CSV
# ===============================
df = pd.DataFrame({
    "time": t,
    "X_ref": X_ref,
    "dX_ref": dX_ref,
    "D_ref": D_ref,
})

save_df_to_csv(df, dirname="training_data", filename="train_inverse_tracking")
save_df_to_csv(df.describe(), dirname="training_data_info", filename="train_inverse_tracking_info")
print("Training data saved to CSV.")