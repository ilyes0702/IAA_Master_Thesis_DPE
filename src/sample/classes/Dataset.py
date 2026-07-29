"""
PlantSignalDataset: a torch Dataset wrapping the plant simulation pipeline
from data_generation_utils, with per-channel signal-generation stages
exposed for inspection/debugging.

This file adds *new* functions (generate_signal_single_with_stages,
generate_signals_with_stages, generate_training_batch_with_stages) that
mirror the existing generate_signal_single / generate_signals /
generate_training_batch functions exactly, but additionally capture the
intermediate tensors at each stage of the signal-generation pipeline:

    1. raw_uniform  -- uniform samples in [-1, 1]
    2. fft_raw      -- rFFT of the raw signal (before cutoff)
    3. fft_filtered -- rFFT spectrum after zeroing frequencies above cutoff
    4. band_limited -- time-domain signal after inverse rFFT (v_train)
    5. normalized   -- signal after min-max normalization to [-1, 1] (v_norm)
    6. u_center     -- per-sequence baseline center for this channel
    7. adaptive_p   -- per-sequence (possibly shrunk) amplitude
    8. u_buffer     -- final signal (same values as the plain u_buffer)

The original generate_signal_single / generate_signals / generate_training_batch
functions are untouched -- these are additive, so anything already calling
them keeps working exactly as before.
"""

import torch
import numpy as np

from src.sample.utils.data_generation_utils import generate_training_batch
from src.sample.utils.plotting_utils import plot_stacked


# =====================================================================
# Stage-capturing counterparts of the existing generation functions
# =====================================================================

def generate_signal_single_with_stages(hyperparam_config, channel_idx=1):
    """
    Same computation as generate_signal_single, but also returns a dict
    of the intermediate pipeline tensors for this channel.

    Returns:
        u_buffer, u_center, stages (dict[str, Tensor | float])
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]
    plant_cfg = hyperparam_config["plant"]

    batch_size = train_cfg["batch_size"]
    seq_len = sig_cfg["seq_len"]
    device = train_cfg["device"]

    lambd = sig_cfg.get(f"u_{channel_idx}_lambd")
    if lambd is None:
        lambd = sig_cfg["lambd"]

    configured_p = sig_cfg.get(f"u_{channel_idx}_p")
    if configured_p is None:
        configured_p = sig_cfg["p"]

    # --- Stage 1: uniform sampling in [-1, 1] ---
    raw = torch.rand((batch_size, seq_len), device=device) * 2 - 1

    # --- Stage 2: forward FFT ---
    fft_sig = torch.fft.rfft(raw, dim=1)
    freqs = torch.fft.rfftfreq(seq_len, d=sig_cfg["dt"])

    # --- Stage 3: zero out frequencies above cutoff ---
    cutoff = 1.0 / lambd
    fft_filtered = fft_sig.clone()
    fft_filtered[:, freqs > cutoff] = 0

    # --- Stage 4: inverse FFT -> band-limited time-domain signal ---
    v_train = torch.fft.irfft(fft_filtered, n=seq_len, dim=1)

    # --- Stage 5: normalize to [-1, 1] ---
    v_min = v_train.min(dim=1, keepdim=True)[0]
    v_max = v_train.max(dim=1, keepdim=True)[0]
    v_norm = 2 * (v_train - v_min) / (v_max - v_min + 1e-8) - 1

    c_min = plant_cfg[f"u_{channel_idx}_D_center_min"]
    c_max = plant_cfg[f"u_{channel_idx}_D_center_max"]
    u_hard_min = plant_cfg.get(f"u_{channel_idx}_hard_min")
    u_hard_max = plant_cfg.get(f"u_{channel_idx}_hard_max")

    # --- Stage 6: baseline center per sequence ---
    u_center = torch.rand((batch_size, 1), device=device) * (c_max - c_min) + c_min

    # --- Stage 7: active-shielding amplitude ---
    dist_to_max = u_hard_max - u_center
    dist_to_min = u_center - u_hard_min
    max_safe_p = torch.minimum(dist_to_max, dist_to_min)
    adaptive_p = torch.minimum(torch.full_like(max_safe_p, configured_p), max_safe_p * 0.98)

    # --- Stage 8: final signal ---
    u_buffer = u_center + (v_norm * adaptive_p)
    u_buffer = torch.clamp(u_buffer, u_hard_min, u_hard_max)

    stages = {
        "raw_uniform": raw.detach(),
        "freqs": freqs.detach(),
        "cutoff": cutoff,
        "lambd": lambd,
        "configured_p": configured_p,
        "fft_raw": fft_sig.detach(),
        "fft_filtered": fft_filtered.detach(),
        "band_limited": v_train.detach(),
        "normalized": v_norm.detach(),
        "u_center": u_center.detach(),
        "adaptive_p": adaptive_p.detach(),
        "u_buffer": u_buffer.detach(),
    }

    return u_buffer, u_center, stages


def generate_signals_with_stages(hyperparam_config):
    """
    Same as generate_signals, but also returns a per-channel stages dict:
        stages[channel_idx] = { ...stage tensors for that channel... }

    NOTE: mirrors generate_signals' channel count (plant_cfg["input_dim"]),
    not generate_signals_mix's (mamba_cfg["output_dim"]) -- the two differ
    in the source file. Confirm this matches your intended u-channel count.
    """
    plant_cfg = hyperparam_config["plant"]
    input_dim = plant_cfg["input_dim"]

    u_buffer_list = []
    D_center_list = []
    stages = {}

    for i in range(input_dim):
        channel_idx = i + 1
        u_single, D_center, ch_stages = generate_signal_single_with_stages(
            hyperparam_config, channel_idx=channel_idx
        )
        u_buffer_list.append(u_single.unsqueeze(-1))
        D_center_list.append(D_center)
        stages[channel_idx] = ch_stages

    u_buffer = torch.cat(u_buffer_list, dim=-1)
    D_center = torch.stack(D_center_list, dim=-1)

    return u_buffer, D_center, stages


def generate_training_batch_with_stages(plant, hyperparam_config):
    """
    Same as generate_training_batch, but also returns the per-channel
    signal-generation stages captured by generate_signals_with_stages.

    Returns:
        raw_u, raw_y, raw_states, D_center, stages
    """
    sig_cfg = hyperparam_config["signal"]
    train_cfg = hyperparam_config["train"]

    seq_len = int(sig_cfg["seq_len"])
    dt = sig_cfg["dt"]
    device = train_cfg["device"]

    state = plant.get_initial_state(train_cfg["batch_size"])
    u_buffer, D_center, stages = generate_signals_with_stages(hyperparam_config)

    raw_y_history = []
    raw_u_history = []
    raw_state_history = []

    for t_idx in range(seq_len):
        t = t_idx * dt
        u_signal = u_buffer[:, t_idx, :]
        y_t = plant.get_y(state, t)

        raw_y_history.append(y_t)
        raw_u_history.append(u_signal)
        raw_state_history.append(state.clone())

        state, _ = plant.step(state, u_signal, t, dt)
        state = state.detach()

    raw_u = torch.stack(raw_u_history, dim=1).to(device)
    raw_y = torch.stack(raw_y_history, dim=1).to(device)
    raw_states = torch.stack(raw_state_history, dim=1).to(device)

    return raw_u, raw_y, raw_states, D_center, stages


# =====================================================================
# Dataset class
# =====================================================================

class PlantSignalDataset(torch.utils.data.Dataset):
    """
    torch Dataset wrapping the plant control-signal / simulation pipeline.

    Each item is one simulated sequence: control input (u), plant output
    (y), and internal state trajectory (states).

    Attributes:
        dt, seq_len              : simulation timing config
        lambd_per_channel (dict) : {channel_idx: lambda} bandwidth per u-channel
        p_per_channel (dict)     : {channel_idx: p} amplitude per u-channel
        u, y, states, D_center   : generated tensors, shape [B, T, C] / [B, 1, C]
        stages (dict)            : {channel_idx: {stage_name: tensor}} --
                                    intermediate signal-generation tensors,
                                    only populated if capture_stages=True

    This wraps the *raw*, unfiltered simulation (equivalent to
    generate_training_batch) -- it does NOT run the bounds/correlation
    filtering that generate_and_save_dataset applies. That's intentional:
    filtering discards sequences based on a separate random draw, so the
    captured stages wouldn't correspond 1:1 to a filtered dataset's
    surviving sequences. If you need the filtered/validated dataset too,
    generate it separately with generate_and_save_dataset and load its
    saved tensors alongside this one.
    """

    def __init__(self, plant, hyperparam_config, capture_stages=True):
        self.plant = plant
        self.hyperparam_config = hyperparam_config
        self.capture_stages = capture_stages

        sig_cfg = hyperparam_config["signal"]
        plant_cfg = hyperparam_config["plant"]

        self.dt = sig_cfg["dt"]
        self.seq_len = sig_cfg["seq_len"]

        input_dim = plant_cfg["input_dim"]
        self.lambd_per_channel = {}
        self.p_per_channel = {}
        for i in range(input_dim):
            ch = i + 1
            self.lambd_per_channel[ch] = sig_cfg.get(f"u_{ch}_lambd", sig_cfg["lambd"])
            self.p_per_channel[ch] = sig_cfg.get(f"u_{ch}_p", sig_cfg["p"])

        self.u = None
        self.y = None
        self.states = None
        self.D_center = None
        self.stages = {}

        self._generate()

    def _generate(self):
        if self.capture_stages:
            u, y, states, D_center, stages = generate_training_batch_with_stages(
                self.plant, self.hyperparam_config
            )
            self.stages = stages
        else:
            u, y, states, D_center = generate_training_batch(
                self.plant, self.hyperparam_config
            )

        self.u = u
        self.y = y
        self.states = states
        self.D_center = D_center

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        return {
            "u": self.u[idx],
            "y": self.y[idx],
            "states": self.states[idx],
            "D_center": self.D_center[idx],
        }

    def get_channel_stage(self, channel_idx, stage_name):
        """
        Convenience accessor, e.g.:
            dataset.get_channel_stage(1, "band_limited")
            dataset.get_channel_stage(2, "u_buffer")

        Available stage_name values: raw_uniform, freqs, cutoff, lambd,
        configured_p, fft_raw, fft_filtered, band_limited, normalized,
        u_center, adaptive_p, u_buffer.
        """
        return self.stages[channel_idx][stage_name]

    def plot_sequence(self, idx, dirname=None, filename=None, show=True):
        """
        Plot one sequence (u, y, and states) from this dataset.

        Mirrors the show_plots branch of generate_and_save_dataset: control
        inputs (u) and outputs (y) are stacked together in one figure via
        plot_stacked, and state variables get their own separate stacked
        figure. Axis labels are pulled from self.plant.get_plot_config()
        when available, falling back to generic u_i / y_i / x_i labels
        otherwise -- same fallback behavior as generate_and_save_dataset.

        Args:
            idx: index of the sequence in this dataset to plot (0-based)
            dirname: directory to save plot images to; if None, nothing
                     is saved to disk (only shown/returned)
            filename: base filename for saved images; defaults to
                      f"sequence_{idx}"
            show: forwarded to plot_stacked -- whether to display inline

        Returns:
            (io_fig, state_fig): return values of plot_stacked for the
            u/y figure and the states figure. state_fig is None if this
            dataset has no state trajectories.
        """
        u = self.u[idx].detach().cpu().numpy()      # [seq_len, n_u_channels]
        y_t = self.y[idx].detach().cpu().numpy()    # [seq_len, n_y_channels]
        states = (
            self.states[idx].detach().cpu().numpy()
            if self.states is not None else None
        )

        time_axis = np.arange(u.shape[0]) * self.dt
        filename_base = filename if filename is not None else f"sequence_{idx}"

        plot_configs = self.plant.get_plot_config()
        u_config = next((c for c in plot_configs if any(col.startswith("u") for col in c["cols"])), None)
        y_config = next((c for c in plot_configs if any(col.startswith("y") for col in c["cols"])), None)
        state_config = next((c for c in plot_configs if any(col.startswith("x") for col in c["cols"])), None)

        # --- control inputs (u) + outputs (y), stacked in one figure ---
        signals_to_plot, labels_to_plot, ylabels_to_plot = [], [], []

        for i in range(u.shape[1]):
            signals_to_plot.append(u[:, i])
            labels_to_plot.append([None])
            ylabels_to_plot.append(u_config["labels"][i] if u_config else rf"Input $u_{{{i+1}}}$")

        for i in range(y_t.shape[1]):
            signals_to_plot.append(y_t[:, i])
            labels_to_plot.append([None])
            ylabels_to_plot.append(y_config["labels"][i] if y_config else rf"Output $y_{{{i+1}}}$")

        io_asp = [0.33] * len(signals_to_plot)
        io_fig = plot_stacked(
            t=time_axis,
            signals=signals_to_plot,
            labels=labels_to_plot,
            xlabel=rf"$t \; / \; \mathrm{{h}}$",
            ylabel=ylabels_to_plot,
            asp=io_asp,
            dirname=dirname,
            filename=f"{filename_base}_plot.png",
            show=show,
        )

        # --- state variables, separate stacked figure (no legends needed) ---
        state_fig = None
        if states is not None:
            state_signals, state_labels, state_ylabels = [], [], []
            for i in range(states.shape[1]):
                state_signals.append(states[:, i])
                state_labels.append([None])
                if state_config and i < len(state_config["labels"]):
                    state_ylabels.append(state_config["labels"][i])
                else:
                    state_ylabels.append(rf"State $x_{{{i+1}}}$")

            state_asp = [0.33] * len(state_signals)
            state_fig = plot_stacked(
                t=time_axis,
                signals=state_signals,
                labels=state_labels,
                xlabel=rf"$t \; / \; \mathrm{{h}}$",
                ylabel=state_ylabels,
                asp=state_asp,
                dirname=dirname,
                filename=f"{filename_base}_states_plot.png",
                show=show,
            )

        return io_fig, state_fig


from src.sample.classes.plants.ChemostatPlant import ChemostatPlant, hyperparam_config_ChemostatPlant


plant = ChemostatPlant(hyperparam_config=hyperparam_config_ChemostatPlant)

dataset = PlantSignalDataset(
    plant=plant,
    hyperparam_config=hyperparam_config_ChemostatPlant,
    capture_stages=True,
)

print(dataset.dt)                  # 0.1
print(dataset.seq_len)              # 1001
print(dataset.lambd_per_channel)    # {1: 15}
print(dataset.p_per_channel)        # {1: 0.15}
print(dataset.u.shape)              # [1000, 1001, 1]  (batch_size, seq_len, u-channels)
print(dataset.y.shape)              # [1000, 1001, 1]
print(dataset.states.shape)         # [1000, 1001, 2]  (biomass, substrate)

# inspect the pipeline for channel 1 (Chemostat only has one u channel: D)
band_limited = dataset.get_channel_stage(1, "band_limited")
final_signal = dataset.get_channel_stage(1, "u_buffer")