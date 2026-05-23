import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE
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

    def __call__(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        res = res.reshape([res.shape[0], -1])
        res = self.encoder(res)
        res = self.decoder(res)
        res = res.reshape(input_shape)
        return res


class Sampler(torch.nn.Module):
    def __init__(self, channels):
        ...
        ## YOUR CODE HERE
        super().__init__()
        self.mu_regressor = torch.nn.Linear(channels, channels)
        self.logvar_regressor = torch.nn.Linear(channels, channels)


    def __call__(self, signal):
        res = signal
        mu = signal
        sigma = signal

        ## YOUR CODE HERE
        mu = self.mu_regressor(signal)
        logvar = self.logvar_regressor(signal)

        sigma = (logvar / 2).exp()

        if self.training:
            noise = torch.randn_like(mu)
            res = noise * sigma + mu
        else:
            res = mu
        return res, mu, sigma


class VAE(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE
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

    def __call__(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        res = res.reshape([res.shape[0], -1])
        res = self.encoder(res)
        res, mu, sigma = self.sampler(res)
        res = self.decoder(res)
        res = res.reshape(input_shape)
        return res, mu, sigma
