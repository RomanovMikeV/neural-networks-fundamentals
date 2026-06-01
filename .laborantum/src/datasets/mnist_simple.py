import torchvision.datasets

class MNISTSimpleDataset:
    def __init__(self, train=True):
        ...
        ## Load MNIST dataset here
        ## YOUR CODE HERE


    def __len__(self):
        res = 0
        ## Return number of items that is there in the dataset
        ## YOUR CODE HERE
        return res


    def __getitem__(self, index):
        sample = {}

        ## Return a sample of the dataset that corresponds to the input index
        ## YOUR CODE HERE
        
        return sample 
