import torch


# class LinAct(torch.nn.Module):
#     def __init__(
#             self, 
#             *args,
#             activation=torch.nn.ReLU
#             **kwargs):
        
#         self.module = torch.nn.Sequential([
#             torch.nn.Linear(*args, **kwargs),
#             activation()])

#     def __call__(self, x):
#         return self.module(x)


# class LinNormAct(torch.nn.Module):
#     def __init__(
#             self, 
#             *args,
#             activation=torch.nn.ReLU
#             **kwargs):

#         self.module = torch.nn.Linear(*arg)
#         self.channels = channels
#         self.norm = torch.nn.BatchNorm1d(channels)
#         self.act = torch.nn.ReLU()

#     def __call__(self, x):
#         return self.act(self.norm(x))


# class Residual(torch.nn.Module):
#     def __init__(self, module):
#         self.module = module

#     def __call__(self, x):
#         return self.module(x) + x


# class Bottleneck(torch.nn.Module):
#     def __init__(self, channels, expansion=4):
#         self.channels = channels
#         self.expansion = expansion
#         self.lin1 = torch.nn.Linear(channels[0], channels[0] * expansion)
#         self.lin2 = torch.nn.Linear(channels[0] * expansion, channels[1])
#         self.norm = torch.nn.BatchNorm1d(channels[0] * expansion)
#         self.act = torch.nn.ReLU()

#     def __call__(self, x):
#         x = self.lin1(x)
#         x = self.norm(x)
#         x = self.act(x)
#         x = self.lin2(x)
#         return x

# class FullyConnectedNN(torch.nn.Module):
#     def __init__(
#             self, 
#             channels, 
#             block=LinAct):
#         pass


class FullyConnectedNN(torch.nn.Module):
    def __init__(
            self, 
            channels,
            n_classes=10,
            activation=torch.nn.ReLU):
        super().__init__()
        
        modules = []
        for i in range(len(channels) - 1):
            modules.append(torch.nn.Linear(channels[i], channels[i+1]))
            modules.append(actiavtion())

        self.modules = torch.nn.Sequential(*modules)
        self.classifier = torch.nn.Linear(channels[-1], n_classes)

    def __call__(self, x):
        return self.network(x)