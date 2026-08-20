from reservoirpy.nodes import Reservoir, Ridge
import numpy as np
import matplotlib.pyplot as plt
from seqControl.sample.utils.plotting_utils import plot_signals

# Initialize ESN configuration
reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = Ridge(ridge=1e-7)
esn_model = reservoir >> readout

# Synthesize baseline signal
X = np.sin(np.linspace(0, 6 * np.pi, 100)).reshape(-1, 1)
X_train = X[:50]
Y_train = X[1:51]

X_large = np.sin(np.linspace(0, 6 * np.pi, 1000)).reshape(-1, 1)

# Fit structural weights analytically
esn_model = esn_model.fit(X_train, Y_train, warmup=10)
print(f"Reservoir Initialized: {reservoir.initialized} | Readout Initialized: {readout.initialized}")
esn_model.reset()
# Execute future prediction rollout sequence (Returns length 50)
Y_pred = esn_model.run(X_large[700:])

# --- 🟢 FIXED & COMPATIBLE PLOT_SIGNALS CALL ---

# 1. Align the target truth slice length with the shifted horizon length (49 items)
target_truth = X_large[701:].flatten()
prediction_aligned = Y_pred[:len(target_truth)].flatten()

# 2. Match the horizontal timeline dimension exactly to the aligned signal length (49 items)
time_steps = np.arange(len(target_truth))

# 3. Call your custom processing layout function safely
final_image = plot_signals(
    t=time_steps,
    signals=[prediction_aligned, target_truth],
    labels=["Predicted $\sin(t+1)$", "Real $\sin(t+1)$"],
    title="A sine wave and its future.",
    xlabel="Time step ($t$)",
    ylabel="Signal Magnitude",
    show=True,                       # Triggers standard plt.show() inside your function
    filename="esn_sine_prediction",  # Passes variables downstream to your save_plot_image utility
    dirname="results/plots"
)