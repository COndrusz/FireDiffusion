"""
Christopher Ondrusz
GitHub: acse_cro23
"""
import torch


def threshold(image, value=0):
    """
    Apply a threshold to an image tensor, converting all pixel values to
    either 1 or -1 based on the threshold value.

    Parameters:
    -----------
    image : torch.Tensor
        The input image as a PyTorch tensor. This tensor can have any shape,
        but typically it will be a 2D or 3D tensor representing an image.
    value : int or float, optional
        The threshold value to apply. Any pixel value in the input image
        greater than or equal to this value will be set to 1, and any pixel
        value below this value will be set to -1.
        The default threshold value is 0.

    Returns:
    --------
    torch.Tensor
        A tensor with the same shape as the input image, where each pixel is
        either 1 or -1 depending on the thresholding.

    Example:
    --------
    >>> import torch
    >>> image = torch.tensor([[0.5, -0.2], [1.2, 0.0]])
    >>> thresholded_image = threshold(image, value=0.1)
    >>> print(thresholded_image)
    (tensor([[ 1, -1],
             [ 1, -1]]),)"""
    image = torch.where(image >= value,
                        torch.tensor(1, dtype=image.dtype,
                                     device=image.device),
                        torch.tensor(-1, dtype=image.dtype,
                                     device=image.device))
    return image


def calculate_matching_percentage(tensor1, tensor2):
    """
    Calculate the percentage of elements that have the same value between two
    tensors.

    Parameters:
    -----------
    tensor1 : torch.Tensor
        The first input tensor. This tensor can have any shape, but it must
        match the shape of `tensor2`.
    tensor2 : torch.Tensor
        The second input tensor, which must have the same shape as `tensor1`.

    Returns:
    --------
    float
        The percentage of elements that match between the two tensors. This is
        calculated as the number of matching elements divided by the total
        number of elements, multiplied by 100.

    Raises:
    -------
    ValueError
        If the input tensors do not have the same shape.

    Example:
    --------
    >>> import torch
    >>> tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> tensor2 = torch.tensor([[1, 2, 0], [4, 5, 6]])
    >>> percentage = calculate_matching_percentage(tensor1, tensor2)
    >>> print(f"Percentage of matching elements: {percentage}%")
    Percentage of matching elements: 83.33333333333334%
    """
    # Ensure both tensors have the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same shape")

    # Perform element-wise comparison
    matching_pixels = (tensor1 == tensor2).sum()

    # Calculate the total number of pixels (elements)
    total_pixels = tensor1.numel()

    # Calculate the percentage of matching pixels
    percentage_matching = (matching_pixels.float() / total_pixels) * 100

    return percentage_matching.item()
