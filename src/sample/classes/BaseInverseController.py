"""
Base class for all inverse controller models.
Defines the interface that all controllers must implement.
"""

import torch
import torch.nn as nn


class BaseInverseController(nn.Module):
    """
    Abstract base class for inverse controllers.
    All controller implementations should inherit from this class
    to ensure compatibility with training and simulation pipelines.
    """

    def __init__(self, input_dim=2, output_dim=1):
        """
        Initialize the base controller.

        Args:
            input_dim (int): Dimension of input (e.g., reference vs actual values)
            output_dim (int): Dimension of output (e.g., control signal)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, y_seq):
        """
        Forward pass through the controller.

        Args:
            y_seq: Input sequence tensor of shape (..., input_dim)

        Returns:
            Control output tensor of shape (..., output_dim)
            Values should be normalized (e.g., [0, 1] for dilution rates)
        """
        raise NotImplementedError("Subclasses must implement forward()")
