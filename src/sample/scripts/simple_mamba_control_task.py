import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba
import numpy as np
import matplotlib.pyplot as plt
from src.sample.utils.plotting_utils import plot_signals
plt.style.use('src/sample/style.mplstyle')

# ----------------------------
# 1. Toy plant (unknown system)
# ----------------------------
# Simple nonlinear system:
# y(t+1) = y(t) + 0.1 * (u(t) - y(t)^3)

def plant_step(y, u):
    return y + 0.1 * (u - y**3)


# ----------------------------
# 2. Generate dataset
# ----------------------------
import numpy as np
import torch

def generate_bandlimited_signal(T, dt, cutoff_freq, amplitude=1.0):
    """
    Generate smooth random signal with limited frequency content
    """

    # --- 1. random signal ---
    signal = np.random.randn(T)

    # --- 2. FFT ---
    fft_signal = np.fft.rfft(signal)

    freqs = np.fft.rfftfreq(T, d=dt)

    # --- 3. low-pass filter ---
    fft_signal[freqs > cutoff_freq] = 0

    # --- 4. inverse FFT ---
    filtered = np.fft.irfft(fft_signal, n=T)

    # --- 5. normalize and scale ---
    filtered = filtered / np.max(np.abs(filtered))
    filtered *= amplitude

    return torch.tensor(filtered).float()


def generate_data(T=1000, seq_len=10, dt=1.0):
    y = torch.zeros(T)
    u = torch.zeros(T)
    r = torch.zeros(T)

    # --- reference (sinusoid) ---
    for t in range(T):
        r[t] = math.sin(0.02 * t)

    # --- NEW: smooth control signal ---
    u_train = generate_bandlimited_signal(
        T=T,
        dt=dt,
        cutoff_freq=0.001,   # 🔥 key hyperparameter
        amplitude=1.0
    )

    # --- simulate plant ---
    for t in range(T - 1):
        u[t] = u_train[t]
        y[t+1] = plant_step(y[t], u[t])

    # --- build dataset ---
    X, Y = [], []
    for i in range(T - seq_len - 1):
        seq = []
        for j in range(seq_len):
            seq.append([y[i+j].item(), r[i+j].item()])
        X.append(seq)
        Y.append([u[i+seq_len].item()])

    return torch.tensor(X).float(), torch.tensor(Y).float(), y, r, u


# ----------------------------
# 3. Mamba Controller
# ----------------------------
class MambaController(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, 2)
        x = self.input_proj(x)
        x = self.mamba(x)

        # take last timestep
        x = x[:, -1, :]
        u = self.output_proj(x)

        return u.unsqueeze(1)  # (B, 1, 1)


# ----------------------------
# 4. Training
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, Y, y, r, u = generate_data(T=1500, seq_len=100)

t = np.arange(len(y))

plot_signals(
    t=t,
    signals=[y.numpy(), r.numpy(), u.numpy()],
    labels=["y (plant output)", "r (reference)", "u (control)"],
    title="Training Data",
    filename="training_data",
    dirname="plots"
)

X = X.to(device)
Y = Y.to(device)

model = MambaController().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print("Training...")

loss_history = []


for epoch in range(100):
    model.train()

    pred = model(X)
    loss = loss_fn(pred.squeeze(), Y.squeeze())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


t_epochs = np.arange(len(loss_history))

plot_signals(
    t=t_epochs,
    signals=[np.array(loss_history)],
    labels=["Training Loss"],
    title="Training Loss Curve",
    xlabel="Epoch",
    ylabel="MSE Loss",
    dirname="plots",
    filename="training_loss"
)

# ----------------------------
# 5. Closed-loop test
# ----------------------------
print("\nTesting closed-loop control...")

T_test = 300
seq_len = 15

# buffers
y_hist = torch.zeros(seq_len).to(device)
r_hist = torch.zeros(seq_len).to(device)

y = torch.zeros(T_test)
r = torch.zeros(T_test)
u = torch.zeros(T_test)

for t in range(T_test):
    r[t] = math.sin(0.02 * t)

model.eval()

for t in range(T_test - 1):
    # update history
    if t > 0:
        y_hist = torch.roll(y_hist, -1)
        r_hist = torch.roll(r_hist, -1)

        y_hist[-1] = y[t]
        r_hist[-1] = r[t]

    # create input
    x_input = torch.stack([y_hist, r_hist], dim=1).unsqueeze(0)  # (1,T,2)

    x_input = x_input.to(device)

    with torch.no_grad():
        u_pred = model(x_input)

    u[t] = u_pred.item()

    # apply control to plant
    y[t+1] = plant_step(y[t], u[t])


t_test = np.arange(len(y))

plot_signals(
    t=t_test,
    signals=[y.numpy(), r.numpy()],
    labels=["y (output)", "r (reference)"],
    title="Closed-Loop Tracking (Test)",
    filename="test_tracking",
    dirname="plots"
)

plot_signals(
    t=t_test,
    signals=[u.numpy()],
    labels=["u (control signal)"],
    title="Control Signal (Test)",
    filename="test_control",
    dirname="plots"
)
# ----------------------------
# 6. Results (simple print)
# ----------------------------
print("\nFirst 20 test values:")
for i in range(20):
    print(f"t={i:3d} | y={y[i]:+.3f} | r={r[i]:+.3f} | u={u[i]:+.3f}")
