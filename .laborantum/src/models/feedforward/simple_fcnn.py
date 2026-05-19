import torch

class SimpleFCNN(torch.nn.Module):
    def __init__(
            self, 
            channels=[],
            n_classes=10,
            activation=torch.nn.ReLU):
        ...
        ## YOUR CODE HERE
        # -- placeholder start --
        super().__init__()
        modules = []
        for i in range(len(channels) - 1):
            modules.append(torch.nn.Linear(channels[i], channels[i+1]))
            modules.append(activation())

        self.backbone = torch.nn.Sequential(*modules)
        self.classifier = torch.nn.Linear(channels[-1], n_classes)
        # -- placeholder end --

    def __call__(self, signal):
        res = signal.reshape([signal.shape[0], -1])
        ## YOUR CODE HERE
        # -- placeholder start --
        res = self.backbone(res)
        res = self.classifier(res)
        # -- placeholder end --
        return res