# ============================================================
# 0. Imports
# ============================================================
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.style.use('src/sample/style.mplstyle')


from mamba_ssm import Mamba
from src.sample.utils.plotting_utils import plot_signals

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ============================================================
# 1. Chemostat simulator
# ============================================================
class Chemostat:
    def __init__(self, dt=0.1):
        # Parameters
        self.mu_max = 1.0
        self.Ks = 0.1
        self.Y = 0.5
        self.S_in = 10.0
        self.dt = dt

    def mu(self, S):
        return self.mu_max * S / (self.Ks + S + 1e-8)

    def step(self, X, S, D):
        μ = self.mu(S)
        dX = (μ - D) * X
        dS = D * (self.S_in - S) - (1 / self.Y) * μ * X

        Xn = torch.clamp(X + self.dt * dX, min=0.0)
        Sn = torch.clamp(S + self.dt * dS, min=0.0)
        return Xn, Sn


# ============================================================
# 2. Mamba forward model (learned dynamics)
# ============================================================
class MambaModel(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.mamba(x)
        return self.output_proj(x[:, -1])


# ============================================================
# 3. Rolling prediction with Mamba
# ============================================================
def mamba_rollout(model, history, u_seq):
    """
    history: (1, L, 2) -> [X, D]
    u_seq: (H, 1)
    """
    seq = history.clone()
    preds = []

    for u in u_seq:
        X_pred = model(seq)
        preds.append(X_pred)

        next_in = torch.cat([X_pred, u.view(1, 1)], dim=1)
        seq = torch.cat([seq[:, 1:], next_in.unsqueeze(1)], dim=1)

    return torch.cat(preds, dim=1)


# ============================================================
# 4. Mamba-based MPC controller
# ============================================================
def mamba_mpc(model, history, X_ref, D_prev,
              H=10, iters=25, lr=0.05):

    D = torch.zeros(H, 1, device=device, requires_grad=True)
    opt = optim.Adam([D], lr=lr)

    D_min, D_max = 0.0, 1.0
    dD_max = 0.05

    for _ in range(iters):
        opt.zero_grad()

        X_preds = mamba_rollout(model, history, D)
        tracking = ((X_preds - X_ref) ** 2).mean()
        effort = (D ** 2).mean()

        # Soft constraint penalties
        rate_penalty = ((D[0] - D_prev).abs() - dD_max).clamp(min=0).mean()
        bound_penalty = (D.clamp(D_min, D_max) - D).abs().mean()

        loss = tracking + 0.01 * effort + 10.0 * rate_penalty + 50.0 * bound_penalty
        loss.backward()
        opt.step()

    D0 = torch.clamp(D[0], D_min, D_max).detach()
    return D0


# ============================================================
# 5. PID baseline controller
# ============================================================
class PID:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt = dt
        self.I = 0.0
        self.e_prev = 0.0

    def step(self, X, X_ref):
        e = (X_ref - X).item()
        self.I += e * self.dt
        D = (e - self.e_prev) / self.dt
        self.e_prev = e

        u = self.Kp * e + self.Ki * self.I + self.Kd * D
        return torch.clamp(torch.tensor([[u]], device=device), 0.0, 1.0)


# ============================================================
# 6. Closed-loop simulation
# ============================================================
def simulate(controller, model=None, T=300):
    plant = Chemostat()
    X = torch.tensor([[0.5]], device=device)
    S = torch.tensor([[5.0]], device=device)

    X_hist, D_hist = [], []
    X_ref = torch.tensor([[1.0]], device=device)

    L = 20
    history = torch.zeros(1, L, 2, device=device)
    D_prev = torch.tensor([[0.0]], device=device)

    for _ in range(T):
        if controller == "mamba":
            D = mamba_mpc(model, history, X_ref, D_prev)
        else:
            D = pid.step(X, X_ref)

        X, S = plant.step(X, S, D)

        history = torch.cat([
            history[:, 1:],
            torch.tensor([[[X.item(), D.item()]]], device=device)
        ], dim=1)

        X_hist.append(X.item())
        D_hist.append(D.item())
        D_prev = D

    return X_hist, D_hist


# ============================================================
# 7. Run comparison
# ============================================================
model = MambaModel().to(device)
pid = PID(Kp=3.0, Ki=0.5, Kd=0.1, dt=0.1)

X_mamba, D_mamba = simulate("mamba", model=model)
X_pid, D_pid = simulate("pid")

# ============================================================
# 8. Plot results
# ============================================================
t = [i * 0.1 for i in range(len(X_mamba))]

plot_signals(
    t,
    [X_mamba, X_pid],
    labels=["Mamba-MPC", "PID"],
    title="Biomass X",
    xlabel="Time",
    ylabel="Biomass X",
    figsize=(12, 4),
    show=True,
    filename="biomass_comparison_mamba_pid",
    dirname="mamba_chemostat",
)

plot_signals(
    t,
    [D_mamba, D_pid],
    labels=["Mamba-MPC", "PID"],
    title="Dilution rate D",
    xlabel="Time",
    ylabel="Dilution rate D",
    figsize=(12, 4),
    show=True,
    filename="dilution_rate_comparison_mamba_pid",
    dirname="mamba_chemostat",
)
