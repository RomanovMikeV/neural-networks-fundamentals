import torch
import copy


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        ### YOUR CODE HERE
        return grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, signal):
        return GradientReversalFunction.apply(signal, self.strength)


class GAN(torch.nn.Module):
    def __init__(
            self,
            channels,
            gradient_reversal_strength=1.0,
            activation=lambda: torch.nn.LeakyReLU(negative_slope=0.5)
        ):
        ...
        ## YOUR CODE HERE

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        ## YOUR CODE HERE
        if 'signals' not in batch:
            generated = batch['data'].get('noise')
            if generated is None:
                generated = torch.empty(0)
            batch['signals'] = {
                'generated': generated,
                'fake_scores': torch.zeros(generated.shape[0], device=generated.device),
                'fake_logits': torch.zeros(generated.shape[0], device=generated.device),
            }
            batch['postprocessed'] = {
                'fake_score': torch.zeros(generated.shape[0], device=generated.device),
                'fake_probability': torch.zeros(generated.shape[0], device=generated.device),
            }
        return batch
