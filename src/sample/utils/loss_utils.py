
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

def mape_loss(u_pred, u_truth):
    """
    Penalizes the percentage difference. 
    Useful if u values span different orders of magnitude.
    """
    error = (u_pred - u_truth).abs() / (u_truth.abs() + 1e-8)
    return torch.mean(error)


def sobolev_loss(u_pred, u_truth, alpha=0.5):
    """
    Combines MSE with a penalty on the difference in derivatives.
    alpha: weighting factor for the derivative penalty.
    """
    # Standard MSE
    mse = torch.mean((u_pred - u_truth)**2)
    
    # Calculate derivatives along the time dimension (dim=1)
    # Assumes shape [batch, seq_len, features]
    du_pred = u_pred[:, 1:, :] - u_pred[:, :-1, :]
    du_truth = u_truth[:, 1:, :] - u_truth[:, :-1, :]
    
    deriv_loss = torch.mean((du_pred - du_truth)**2)
    
    return mse + alpha * deriv_loss


def log_cosh_loss(u_pred, u_truth):
    """
    A smooth approximation of Huber loss that is easier to optimize 
    with second-order methods.
    """
    error = u_pred - u_truth
    return torch.mean(torch.log(torch.cosh(error + 1e-12)))


def cosine_shape_loss(u_pred, u_truth):
    """
    Focuses on the shape/alignment of the curves.
    Returns 1 - similarity (so 0 is perfect alignment).
    """
    cos = torch.nn.CosineSimilarity(dim=1) # Across the time dimension
    # Flattening to [batch, seq_len * features]
    sim = cos(u_pred.flatten(1), u_truth.flatten(1))
    return 1.0 - torch.mean(sim)


def hybrid_control_loss(u_pred, u_truth):
    # 1. Accuracy (Huber)
    l1 = torch.nn.functional.huber_loss(u_pred, u_truth, delta=0.5)
    
    # 2. Smoothness (Derivative)
    du_pred = u_pred[:, 1:, :] - u_pred[:, :-1, :]
    du_truth = u_truth[:, 1:, :] - u_truth[:, :-1, :]
    l2 = torch.mean((du_pred - du_truth)**2)
    
    # 3. Shape (Cosine)
    l3 = cosine_shape_loss(u_pred, u_truth)
    
    return l1 + 0.5 * l2 + 0.1 * l3