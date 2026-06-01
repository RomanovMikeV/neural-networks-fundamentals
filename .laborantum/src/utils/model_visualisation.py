import torch
from torchview import draw_graph


def draw_forward_kernel_graph(
        model,
        input_shape=(3, 28, 28),
        graph_name='SimpleFCNN',
        expand_nested=True):
    class _ForwardKernelAdapter(torch.nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.wrapped_model = wrapped_model
            kernel_name = f'_{wrapped_model.__class__.__name__}__forward_kernel'
            self.forward_kernel = getattr(wrapped_model, kernel_name)

        def forward(self, signal):
            return self.forward_kernel(signal)

    adapter = _ForwardKernelAdapter(model)
    return draw_graph(
        adapter,
        input_data=torch.randn(*input_shape),
        expand_nested=expand_nested,
        graph_name=graph_name)
