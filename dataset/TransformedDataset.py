import torch
import torch.utils.data

class TransformedDataset(torch.utils.data.Dataset):
    """Dataset wrapping datasets with a transform (function).
    Each sample will be retrieved by indexing the underlying dataset and mapping the result through the transform function.
    Arguments:
        transform (callable): the function through which to map the underlying dataset.
        dataset (Dataset): a pytorch dataset
    """
    def __init__(self, transform, dataset):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        return self.transform(self.dataset[index])

    def __len__(self):
        return len(self.dataset)