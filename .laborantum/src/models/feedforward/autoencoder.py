import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE
        # -- placeholder start --
        super().__init__()

        encoder_layers = []
        for index in range(len(channels) - 1):
            encoder_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            encoder_layers.append(activation())
        encoder_layers.pop()

        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        channels = channels[::-1]
        for index in range(len(channels) - 1):
            decoder_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            decoder_layers.append(activation())
        decoder_layers.pop()

        self.decoder = torch.nn.Sequential(*decoder_layers)
        # -- placeholder end --

    def __call__(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        # -- placeholder start --
        res = res.reshape([res.shape[0], -1])
        res = self.encoder(res)
        res = self.decoder(res)
        # -- placeholder end --
        res = res.reshape(input_shape)
        return res


class Sampler(torch.nn.Module):
    def __init__(self, channels):
        ...
        ## YOUR CODE HERE
        # -- placeholder start --
        super().__init__()
        self.mu_regressor = torch.nn.Linear(channels, channels)
        self.logvar_regressor = torch.nn.Linear(channels, channels)
        # -- placeholder end --


    def __call__(self, signal):
        res = signal
        mu = signal
        sigma = signal

        ## YOUR CODE HERE
        # -- placeholder start --
        mu = self.mu_regressor(signal)
        logvar = self.logvar_regressor(signal)

        sigma = (logvar / 2).exp()

        if self.training:
            noise = torch.randn_like(mu)
            res = noise * sigma + mu
        else:
            res = mu
        # -- placeholder end --
        return res, mu, sigma


class VAE(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE
        # -- placeholder start --
        super().__init__()

        encoder_layers = []
        for index in range(len(channels) - 1):
            encoder_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            encoder_layers.append(activation())
        encoder_layers.pop()

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.sampler = Sampler(channels[-1])

        decoder_layers = []
        channels = channels[::-1]
        for index in range(len(channels) - 1):
            decoder_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            decoder_layers.append(activation())
        decoder_layers.pop()

        self.decoder = torch.nn.Sequential(*decoder_layers)
        # -- placeholder end --

    def __call__(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        # -- placeholder start --
        res = res.reshape([res.shape[0], -1])
        res = self.encoder(res)
        res, mu, sigma = self.sampler(res)
        res = self.decoder(res)
        # -- placeholder end --
        res = res.reshape(input_shape)
        return res, mu, sigma
