from src.sample.utils.saving_utils import save_plot_image
# ADD PROJECT DIRECTORY TO SYSTEM PATH

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