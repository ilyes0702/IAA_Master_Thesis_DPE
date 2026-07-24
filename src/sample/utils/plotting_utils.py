from src.sample.utils.saving_utils import save_plot_image
# ADD PROJECT DIRECTORY TO SYSTEM PATH
import optuna
# IMPORT DATA PROCESSING MODULES
import numpy as np
# IMPORT VISUALIZATION MODULES
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

import torch

#=== FUNCTION TO PLOT SIGNALS ===#
def plot_signals(
    t,
    signals,
    labels=None,
    title=None,
    xlabel="Time",
    ylabel="Value",
    figsize=(5, 5),
    save_path=None,
    show=False,
    filename=None,
    dirname=None,
    asp=1.0
):
    """
    Generic plotting function for time-series or x-y signals.

    Parameters:
    - t (numpy.ndarray or list): The independent variable timeline array for the horizontal x-axis.
    - signals (list of numpy.ndarray): Collection of dependent signal arrays to map along the vertical y-axis.
    - labels (list of str, optional): Text labels matched by index sequence to identify each unique signal.
    - title (str, optional): Overarching header title text displayed at the top of the plot grid.
    - xlabel (str): Explicit label tracking the horizontal x-axis context (default: "Time").
    - ylabel (str): Explicit label tracking the vertical y-axis context (default: "Value").
    - figsize (tuple): Specific scale layout dimensions for the graphic asset canvas (default: (5, 5)).
    - save_path (str, optional): A targeted physical file path to write the initial vector plot graphic.
    - show (bool): Toggle flag which forces standard Matplotlib UI canvas rendering if True (default: False).
    - filename (str, optional): Defined target image label for processing custom raster export sequences.
    - dirname (str, optional): System subfolder location targeted for writing custom raster image files.

    Returns:
    - image (PIL.Image.Image): A high-resolution raster image object version of the finalized signal canvas.

    The function acts as a uniform parsing layout for multi-trace system tracking data. It sets up 
    the canvas grid framework, sequentially maps input arrays onto a scatter axis topology, and forces 
    a strict normalized 1:1 rectangular aspect ratio overlay. It maps peripheral aesthetic labels, 
    serializes raw canvas elements into an in-memory high-density byte stream buffer, and wraps the result 
    into a PIL image construct for flexible external logging or automated directory routing.
    """

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    # Iterate and render each signal track onto the common subplot grid
    for i, sig in enumerate(signals):
        if labels is not None:
            ax.plot(t, sig, label=labels[i])
        else:
            ax.plot(t, sig)

    # === FIXED ASPECT RATIO === #
    # Standardize scale boundaries to maintain geometric proportions
    x_range = np.diff(ax.get_xlim())[0]
    y_range = np.diff(ax.get_ylim())[0]
    range_ratio = x_range / y_range
    ax.set_aspect(asp * range_ratio)

    # Apply axis descriptors and grid annotations
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if labels:
        ax.legend()

    # Handle standard initial vector export if path properties are configured
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)

    if show:
        plt.show()

    # plt.tight_layout()

    # Serialize the complete plot canvas into an active memory buffer stream
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()

    # Convert the memory stream into an editable PIL Image representation
    image = Image.open(buf)
    
    # Process custom external tracking image storage
    save_plot_image(image=image, filename=filename, dirname=dirname)
    
    return image

def plot_clustered_signals_pil(
    t,
    X,
    cluster_labels,
    centroids=None,
    title="Clustered Time Series",
    xlabel="Time",
    ylabel="Value",
    figsize=(6, 4),
    show=False,
    filename=None,
    dirname=None,
    asp=0.5
):
    """
    Plots a dataset of time series colored by cluster assignment using the 
    PIL-buffer architecture of `plot_signals`.

    Parameters:
    - t (numpy.ndarray or list): Time axis array [Seq_Len].
    - X (torch.Tensor or numpy.ndarray): Signals array of shape [N, Seq_Len, 1] or [N, Seq_Len].
    - cluster_labels (list or numpy.ndarray): Cluster IDs assigned to each sequence [N].
    - centroids (numpy.ndarray, optional): Cluster barycenters of shape [n_clusters, Seq_Len, 1].
    - title, xlabel, ylabel, figsize, show, filename, dirname, asp: Forwarded to plot_signals architecture.

    Returns:
    - image (PIL.Image.Image): High-resolution raster image object.
    """
    # 1. Convert input tensors/arrays to standard 2D NumPy array [N, Seq_Len]
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = np.array(X)
        
    X_2d = np.squeeze(X_np)  # Ensures shape is [N, Seq_Len]
    cluster_labels = np.array(cluster_labels)
    unique_clusters = np.unique(cluster_labels)

    # 2. Set up colormap across distinct cluster IDs
    cmap = plt.cm.get_cmap("tab10", max(len(unique_clusters), 1))
    
    # 3. Build signals and labels list matching `plot_signals` input structure
    signals_list = []
    labels_list = []
    
    # We maintain a tracker to ensure each Cluster ID only creates ONE legend entry
    added_to_legend = set()

    # Pack individual time series traces
    for seq, label in zip(X_2d, cluster_labels):
        signals_list.append(seq)
        
        if label not in added_to_legend:
            labels_list.append(f"Cluster {label}")
            added_to_legend.add(label)
        else:
            labels_list.append(None)  # Hides duplicate trace entries in legend

    # 4. Optional: Append Centroid trajectories (thick dashed lines)
    if centroids is not None:
        centroids_2d = np.squeeze(centroids)
        for c_idx in unique_clusters:
            if c_idx < len(centroids_2d):
                signals_list.append(centroids_2d[c_idx])
                labels_list.append(f"Centroid {c_idx}")

    # 5. Build Matplotlib canvas using your buffer layout
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    num_traces = len(X_2d)
    
    # Render individual sequence curves
    for i in range(num_traces):
        cluster_id = cluster_labels[i]
        color = cmap(cluster_id % 10)
        label_text = labels_list[i]
        
        ax.plot(t, signals_list[i], color=color, alpha=0.45, linewidth=1.2, label=label_text)

    # Render centroid lines if provided
    if centroids is not None:
        for i in range(num_traces, len(signals_list)):
            label_text = labels_list[i]
            ax.plot(t, signals_list[i], color="black", linestyle="--", linewidth=2.5, label=label_text)

    # === FIXED ASPECT RATIO (Identical to your plot_signals engine) === #
    x_range = np.diff(ax.get_xlim())[0]
    y_range = np.diff(ax.get_ylim())[0]
    range_ratio = x_range / y_range if y_range != 0 else 1.0
    ax.set_aspect(asp * range_ratio)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    # Build clean legend without duplicates
    handles, legend_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    ax.grid(True, linestyle=":", alpha=0.6)

    if show:
        plt.show()

    # === MEMORY BUFFER & PIL CONVERSION === #
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()

    image = Image.open(buf)
    
    # Process custom storage if save_plot_image is available in scope
    if "save_plot_image" in globals():
        save_plot_image(image=image, filename=filename, dirname=dirname)
    elif filename and dirname:
        import os
        os.makedirs(dirname, exist_ok=True)
        image.save(os.path.join(dirname, filename))

    return image

def plot_param_heatmap(
    study,
    param_x,
    param_y,
    title=None,
    figsize=(6, 6),
    filename=None,
    dirname=None,
    asp=1.0,
    show=False,
):
    """Generates a 2D objective surface heatmap for ANY pair of hyperparameters

    from an Optuna study, following the project canvas rendering pipeline.

    Parameters:
    - study (optuna.study.Study): Completed or ongoing Optuna study object.
    - param_x (str): Name of the hyperparameter mapped to the horizontal x-axis.
    - param_y (str): Name of the hyperparameter mapped to the vertical y-axis.
    - title (str, optional): Custom title header (defaults to "Loss Heatmap:
    {param_y} vs {param_x}").
    - figsize (tuple): Canvas size layout dimensions.
    - filename (str, optional): Output image name (defaults to
    "{param_y}_vs_{param_x}_heatmap").
    - dirname (str, optional): Directory path for saving the raster image.
    - asp (float): Aspect ratio scalar modifier.
    - show (bool): Toggle UI canvas display.
    """
    # 1. Extract all unique sampled values for param_x and param_y
    x_vals = set()
    y_vals = set()

    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if param_x in trial.params and param_y in trial.params:
                x_vals.add(trial.params[param_x])
                y_vals.add(trial.params[param_y])

    if not x_vals or not y_vals:
        raise ValueError(
            f"No completed trials found containing both hyperparameters: '{param_x}' and '{param_y}'."
        )

    # Sort tick labels for clean axis ordering
    x_ticks = sorted(list(x_vals))
    y_ticks = sorted(list(y_vals))

    x_to_idx = {val: idx for idx, val in enumerate(x_ticks)}
    y_to_idx = {val: idx for idx, val in enumerate(y_ticks)}

    # 2. Initialize grid with NaNs
    grid = np.full((len(y_ticks), len(x_ticks)), np.nan)

    # 3. Populate matrix (aggregating via minimum if duplicates exist)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            px = trial.params.get(param_x)
            py = trial.params.get(param_y)
            val = trial.value

            if px in x_to_idx and py in y_to_idx and val is not None:
                row_idx = y_to_idx[py]
                col_idx = x_to_idx[px]
                if np.isnan(grid[row_idx, col_idx]):
                    grid[row_idx, col_idx] = val
                else:
                    grid[row_idx, col_idx] = min(grid[row_idx, col_idx], val)

    # 4. Render canvas
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    cax = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Mean Validation Loss")

    # Set dynamic axis tick labels
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.set_yticklabels([str(y) for y in y_ticks])

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)

    if title is None:
        ax.set_title(f"Optuna Loss Heatmap: {param_y} vs {param_x}")
    elif title:
        ax.set_title(title)

    # Overlay numerical cell values
    mean_val = np.nanmean(grid)
    for i in range(len(y_ticks)):
        for j in range(len(x_ticks)):
            val = grid[i, j]
            if not np.isnan(val):
                # Format floating point numbers cleanly
                text_val = f"{val:.4f}" if val < 1.0 else f"{val:.2f}"
                ax.text(
                    j,
                    i,
                    text_val,
                    ha="center",
                    va="center",
                    color="white" if val > mean_val else "black",
                    fontsize=8,
                )

    # Dynamic aspect ratio scaling based on grid dimensions
    range_ratio = len(x_ticks) / len(y_ticks)
    ax.set_aspect(asp * range_ratio)

    if show:
        plt.show()

    # 5. Pipeline export via PIL buffer
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()

    image = Image.open(buf)

    out_filename = (
        filename if filename else f"{param_y}_vs_{param_x}_heatmap"
    )
    save_plot_image(image=image, filename=out_filename, dirname=dirname)

    return image


def plot_stacked(
    t,
    signals,
    labels=None,         
    title=None,
    xlabel="Time",
    ylabel="Value",         
    save_path=None,
    show=False,
    filename=None,
    dirname=None,
    asp=0.33,            # Can be a single float (applied to all) or a list/tuple
    hspace=0.05          
):
    """
    Generic plotting function that stacks various subplots on top of each other.
    """
    num_subplots = len(signals)
    
    fig, axes = plt.subplots(
        nrows=num_subplots, 
        ncols=1, 
        sharex=True
    )
    
    if num_subplots == 1:
        axes = [axes]

    fig.subplots_adjust(hspace=hspace)

    # Iterate and render each subplot row
    for i, sig_group in enumerate(signals):
        ax = axes[i]
        
        is_multi = isinstance(sig_group, (list, tuple)) and not isinstance(sig_group[0], (int, float))
        curves = sig_group if is_multi else [sig_group]
        row_labels = labels[i] if labels is not None else None
        
        if row_labels is not None:
            if not isinstance(row_labels, (list, tuple)):
                row_labels = [row_labels]
        
        for j, sig in enumerate(curves):
            lbl = row_labels[j] if row_labels is not None and j < len(row_labels) else None
            ax.plot(t, sig, label=lbl)
            
        if row_labels is not None and any(lbl is not None for lbl in row_labels):
            ax.legend(loc="upper right")

        # === ADJUSTABLE ASPECT RATIO === #
        if isinstance(asp, (list, tuple)) and len(asp) == num_subplots:
            current_asp = asp[i]
        else:
            current_asp = asp if isinstance(asp, (int, float)) else 0.33

        x_range = np.diff(ax.get_xlim())[0]
        y_range = np.diff(ax.get_ylim())[0]
        if y_range > 0:  
            range_ratio = x_range / y_range
            ax.set_aspect(current_asp * range_ratio)

        # Handle y-axis labels
        if isinstance(ylabel, list) and len(ylabel) == num_subplots:
            ax.set_ylabel(ylabel[i])
        else:
            ax.set_ylabel(ylabel if isinstance(ylabel, str) else f"Signal {i+1}")

        if hspace < 0.15 and i < num_subplots - 1:
            ax.label_outer()

    axes[-1].set_xlabel(xlabel)

    if title:
        axes[0].set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)

    if show:
        plt.show()

    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=600, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    image = Image.open(buf)
    if 'save_plot_image' in globals():
        save_plot_image(image=image, filename=filename, dirname=dirname)
        
    return image
    

from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def plot_heatmap(
    matrix,
    title=None,
    xlabel="Timestep",
    ylabel="State Dimension",
    cmap="viridis",
    figsize=(12, 6),
    save_path=None,
    show=False,
    filename=None,
    dirname=None,
    vmin=None,
    vmax=None,
    colorbar_label=None,
):
    """
    Generic plotting function for heatmaps of 2D matrices (e.g., B, C, or attention weights).

    Parameters:
    - matrix (numpy.ndarray or torch.Tensor): 2D matrix to plot as a heatmap (shape: [d_state, L] or [L, d_state]).
    - title (str, optional): Title for the heatmap.
    - xlabel (str): Label for the x-axis (default: "Timestep").
    - ylabel (str): Label for the y-axis (default: "State Dimension").
    - cmap (str): Colormap for the heatmap (default: "viridis").
    - figsize (tuple): Figure size (default: (12, 6)).
    - save_path (str, optional): Path to save the heatmap image.
    - show (bool): Whether to display the plot (default: False).
    - filename (str, optional): Filename for saving the heatmap.
    - dirname (str, optional): Directory to save the heatmap.
    - vmin (float, optional): Minimum value for the colormap.
    - vmax (float, optional): Maximum value for the colormap.
    - colorbar_label (str, optional): Label for the colorbar.

    Returns:
    - image (PIL.Image.Image): High-resolution raster image of the heatmap.
    """

    # Convert torch.Tensor to numpy if needed
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Handle saving to a custom path
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)

    if show:
        plt.show()

    # Serialize the plot into a memory buffer
    buf = BytesIO()
    plt.savefig(buf, format="PNG", dpi=600)
    buf.seek(0)
    plt.close()

    # Convert to PIL Image
    image = Image.open(buf)

    # Save to custom directory if specified
    if filename and dirname:
        save_path = Path(dirname) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)

    return image