from src.sample.utils.saving_utils import save_plot_image
# ADD PROJECT DIRECTORY TO SYSTEM PATH

# IMPORT DATA PROCESSING MODULES
import numpy as np
# IMPORT VISUALIZATION MODULES
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO



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
    dirname=None
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
            ax.scatter(t, sig, label=labels[i])
        else:
            ax.scatter(t, sig)

    # === FIXED ASPECT RATIO === #
    # Standardize scale boundaries to maintain geometric proportions
    x_range = np.diff(ax.get_xlim())[0]
    y_range = np.diff(ax.get_ylim())[0]
    ax.set_aspect(x_range / y_range)

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




