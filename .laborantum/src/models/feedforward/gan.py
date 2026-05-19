import torch

class GAN(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE
        # -- placeholder start --
        super().__init__()

        generator_layers = []
        for index in range(len(channels) - 1):
            generator_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            generator_layers.append(activation())
        generator_layers.pop()

        self.generator = torch.nn.Sequential(*generator_layers)

        discriminator_layers = []
        channels = channels[::-1]
        for index in range(len(channels) - 1):
            discriminator_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            discriminator_layers.append(activation())
        discriminator_layers.pop()

        self.discriminator = torch.nn.Sequential(*discriminator_layers)
        self.classifier = torch.nn.Linear(channels[-1], 1)
        # -- placeholder end --
