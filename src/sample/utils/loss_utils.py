
import torch

def relative_huber_loss(u_pred, u_truth, delta=1.0):
    """
    Relative Huber loss for control signals.
    Args:
        u_pred: Predicted control sequence (torch.Tensor)
        u_truth: Ground truth control sequence (torch.Tensor)
        delta: Threshold for Huber loss (default: 1.0)
    Returns:
        loss: Scalar tensor
    """
    error = (u_pred - u_truth) / (u_truth.abs() + 1e-8)  # Normalize by u_truth magnitude
    huber_loss = torch.where(
        torch.abs(error) < delta,
        0.5 * error ** 2,
        delta * (torch.abs(error) - 0.5 * delta)
    )
    return torch.mean(huber_loss)