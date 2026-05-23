import torchvision.datasets

class MNISTSimpleDataset:
    def __init__(self, train=True):
        ...
        ## Load MNIST dataset here
        ## YOUR CODE HERE
        dataset = torchvision.datasets.MNIST(root='~/', train=train, download=True)
        self.X = dataset.data
        self.y = dataset.targets


    def __len__(self):
        res = 0
        ## Return number of items that is there in the dataset
        ## YOUR CODE HERE
        res = len(self.y)
        return res


    def __getitem__(self, index):
        sample = {}

        ## Return a sample of the dataset that correponds to the input index
        ## YOUR CODE HERE
        sample['image'] = self.X[index, :, :].float() / 255.0 * 2 - 1
        sample['label'] = self.y[index].long()
        
        return sample 