import torch

class Sampler(torch.nn.Module):
    def __init__(self, channels):
        self.mu_regressor = torch.nn.Linear(channels, channels)
        self.logvar_regressor = torch.nn.Linear(channels, channels)


    def __call__(self, signal):
        res = signal
        ## YOUR CODE HERE
        return res


class VAE(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE

    def __call__(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        res = res.reshape(input_shape)
        return res
