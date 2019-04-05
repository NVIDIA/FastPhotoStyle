import torch
import torch.utils.data
import multiprocessing

class CachedDataset(torch.utils.data.Dataset):
    """Multiprocessing-safe wrapper which caches a dataset in memory.
    Arguments:
        dataset (Dataset): a pytorch dataset
        cache (dict): An optional initial cache
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = multiprocessing.Manager().dict()
    
    def __getitem__(self, index):
        if index in self.cache != None:
            return self.cache[index]
        item = self.dataset[index]
        self.cache[index] = item
        return item

    def __len__(self):
        return len(self.dataset)
    
    
