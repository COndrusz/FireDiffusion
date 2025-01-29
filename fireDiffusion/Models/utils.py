"""
Christopher Ondrusz
GitHub: acse_cro23
"""
import torch
import numpy as np


def compute_alpha_sigma(lambda_tau):
    """
    Compute the alpha and sigma values used in the diffusion process from the
    given lambda values.

    Parameters:
    -----------
    lambda_tau : torch.Tensor
        A tensor containing lambda values, which are used to calculate the
        alpha and sigma values.

    Returns:
    --------
    tuple of torch.Tensor
        A tuple containing two tensors:
        - `alpha_tau`: The calculated alpha values, which are used in the
        diffusion process.
        - `sigma_tau`: The calculated sigma values, representing the standard
        deviation of the noise added during diffusion.

    Example:
    --------
    >>> lambda_tau = torch.tensor([0.5, 1.0, 1.5])
    >>> alpha_tau, sigma_tau = compute_alpha_sigma(lambda_tau)
    >>> print(alpha_tau)
    >>> print(sigma_tau)
    """
    sigma_tau = torch.sqrt(1. / (torch.exp(lambda_tau) + 1.))
    alpha_tau = sigma_tau * torch.exp(lambda_tau / 2.)
    return alpha_tau, sigma_tau


def variance_preserving_diffusion(x_0, t, alpha_tau, sigma_tau):
    """
    Apply variance-preserving diffusion to an input tensor at a specific
    timestep.

    Parameters:
    -----------
    x_0 : torch.Tensor
        The original input tensor, representing the initial data
        (e.g., an image).
    t : int
        The timestep at which to apply the diffusion process.
    alpha_tau : torch.Tensor
        A tensor containing precomputed alpha values for each timestep.
    sigma_tau : torch.Tensor
        A tensor containing precomputed sigma values for each timestep.

    Returns:
    --------
    torch.Tensor
        The diffused tensor `z_t`, which is the result of applying the
        diffusion process to `x_0` at timestep `t`.

    Example:
    --------
    >>> x_0 = torch.randn(1, 3, 32, 32)
    >>> t = 10
    >>> alpha_tau = torch.linspace(0.1, 0.9, 50)
    >>> sigma_tau = torch.linspace(0.9, 0.1, 50)
    >>> z_t = variance_preserving_diffusion(x_0, t, alpha_tau, sigma_tau)
    >>> print(z_t.shape)
    """
    epsilon = torch.randn_like(x_0)*255
    z_t = alpha_tau[t] * x_0 + sigma_tau[t] * epsilon
    return z_t


def surrogate_target(x_t, t, alpha, sigma):
    """
    Calculate the surrogate target used in the training of diffusion models.

    Parameters:
    -----------
    x_t : torch.Tensor
        The diffused tensor at timestep `t`.
    t : int
        The current timestep.
    alpha : torch.Tensor
        A tensor containing alpha values for each timestep.
    sigma : torch.Tensor
        A tensor containing sigma values for each timestep.

    Returns:
    --------
    torch.Tensor
        The surrogate target tensor, which is used for model training to
        approximate the noise added during diffusion.

    Example:
    --------
    >>> x_t = torch.randn(1, 3, 32, 32)
    >>> t = 10
    >>> alpha = torch.linspace(0.1, 0.9, 50)
    >>> sigma = torch.linspace(0.9, 0.1, 50)
    >>> target = surrogate_target(x_t, t, alpha, sigma)
    >>> print(target.shape)
    """
    epsilon = torch.tensor(np.random.normal(0, 255, x_t.shape),
                           dtype=torch.float32).to(x_t.device)
    alpha_t = alpha[t].view(-1, 1, 1, 1)
    sigma_t = sigma[t].view(-1, 1, 1, 1)
    return (alpha_t * epsilon) - (x_t * sigma_t)


def noise_schedule(timesteps, schedule_type='linear'):
    """
    Generate a noise schedule based on the specified type and number of
    timesteps.

    Parameters:
    -----------
    timesteps : int
        The number of timesteps for which to generate the noise schedule.
    schedule_type : str, optional
        The type of noise schedule to generate. Supported types are 'linear'
        and 'cosine'.
        Default is 'linear'.

    Returns:
    --------
    torch.Tensor
        A tensor containing the noise schedule values for each timestep.

    Raises:
    -------
    ValueError
        If the `schedule_type` is not 'linear' or 'cosine'.

    Example:
    --------
    >>> timesteps = 50
    >>> schedule = noise_schedule(timesteps, schedule_type='cosine')
    >>> print(schedule)
    """
    if schedule_type == 'linear':
        return torch.linspace(1, 0, timesteps)
    elif schedule_type == 'cosine':
        return torch.cos((torch.linspace(0, np.pi / 2, timesteps)))
    else:
        raise ValueError("Unknown schedule type")


def loss_function(v_tau, v_hat, lambda_tau, t):
    """
    Compute the loss for training a diffusion model, considering the predicted
    and true noise.

    Parameters:
    -----------
    v_tau : torch.Tensor
        The true noise tensor used in the diffusion process.
    v_hat : torch.Tensor
        The predicted noise tensor by the model.
    lambda_tau : torch.Tensor
        A tensor containing lambda values for each timestep.
    t : torch.Tensor
        A tensor of timesteps at which the loss is computed.

    Returns:
    --------
    torch.Tensor
        The computed loss value, which is used to optimize the diffusion model
        during training.

    Example:
    --------
    >>> v_tau = torch.randn(1, 3, 32, 32)
    >>> v_hat = torch.randn(1, 3, 32, 32)
    >>> lambda_tau = torch.linspace(0.1, 0.9, 50)
    >>> t = torch.tensor([10])
    >>> loss = loss_function(v_tau, v_hat, lambda_tau, t)
    >>> print(loss.item())
    """
    w_tau = torch.exp(-lambda_tau / 2)
    w_tau_t = w_tau[t.long()].view(-1, 1, 1, 1)
    lambda_tau_t = lambda_tau[t.long()].view(-1, 1, 1, 1)

    lambda_tau_grad = torch.autograd.grad(
        outputs=lambda_tau_t,
        inputs=lambda_tau,
        grad_outputs=torch.ones_like(lambda_tau_t),
        create_graph=True
    )[0]

    lambda_tau_grad_t = lambda_tau_grad[t.long()].view(-1, 1, 1, 1)
    loss = w_tau_t * (torch.abs(lambda_tau_grad_t) *
                      (torch.exp(-lambda_tau_t) + 1)**-1) * (v_tau - v_hat)**2
    return loss.mean()
