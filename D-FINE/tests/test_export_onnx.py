import torch

from tools.deployment.export_onnx import make_example_inputs


def test_example_inputs_support_batch_size_at_960():
    images, sizes = make_example_inputs(batch_size=2, input_size=960)

    assert images.shape == (2, 3, 960, 960)
    assert images.dtype == torch.float32
    assert sizes.tolist() == [[960, 960], [960, 960]]
    assert sizes.dtype == torch.int64
