import torchvision.datasets

class MNISTSimpleDataset:
    def __init__(self, train=True):
        ...
        ## Load MNIST dataset here
        ## YOUR CODE HERE
        # -- placeholder start --
        dataset = torchvision.datasets.MNIST(root='~/', train=train, download=True)
        self.X = dataset.data
        self.y = dataset.targets
        # -- placeholder end --


    def __len__(self):
        res = 0
        ## Return number of items that is there in the dataset
        ## YOUR CODE HERE
        # -- placeholder start --
        res = len(self.y)
        # -- placeholder end --
        return res


    def __getitem__(self, index):
        sample = {}

        ## Return a sample of the dataset that correponds to the input index
        ## YOUR CODE HERE
        # -- placeholder start --
        sample['image'] = self.X[index, :, :].float() / 255.0 * 2 - 1
        sample['label'] = self.y[index].long()
        # -- placeholder end --
        
        return sample 