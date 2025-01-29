from fireDiff.Utils import calculate_matching_percentage, threshold
import torch


def test_threshold():
    image = torch.tensor([[0.5, -0.2], [1.2, 0.0]], dtype=float)
    thresholded_image = threshold(image, value=0.1)
    target_image = torch.tensor([[1, -1], [1, -1]], dtype=float)
    assert torch.allclose(thresholded_image, target_image)


def test_percentage():
    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor2 = torch.tensor([[0, 4, 0], [4, 5, 6]])
    percentage = calculate_matching_percentage(tensor1, tensor2)
    assert percentage == 50.0
