from torch.utils.data import Dataset
import numpy as np

class DataframeDataset(Dataset):
    def __init__(self, df, xcol, ycol=None, weight_col=None, xloader=None, yloader=None, classes=None, xtransform=None, ytransform=None):
        """Creates a dataframe-backed dataset.
        Args:
            df (pandas.DataFrame): The dataframe.
            xcol (string): The column containing the samples.
            ycol (string): The column containing the target classnames (strings not ints!).
            xloader (callable): A function to load a sample given its dataframe value (e.g. a file path or URL).
            yloader (callable): A function to load a target value given its dataframe value (e.g. a file path or URL).
            xtransform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            ytransform (callable, optional): A function/transform that takes
                in the target and transforms it.
         Attributes:
            classes (list): List of the class names.
            class_to_idx (dict): Dict with items (class_name, class_index).
            samples (list): List of (sample path, class_index) tuples
        """

        self.samples = df.as_matrix(columns=(xcol, ycol, weight_col))
        if len(self.samples) == 0:
            raise(RuntimeError("Found 0 samples in columns ({}, {}).".format(xcol, ycol)))

        # self.df = df
        # self.xcol = xcol
        # self.ycol = ycol
        self.xloader = xloader
        self.yloader = yloader

        if classes:
            self.classes = classes
        elif ycol:
            self.classes = df[ycol].unique()
        else:
            self.classes = []
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.xtransform = xtransform
        self.ytransform = ytransform
        
        if ycol != None and weight_col != None:
            X, Y, W = zip(*self.samples)
            Y = [self.class_to_idx[y] for y in Y]
            self.samples = list(zip(X, Y, W))
        elif ycol:
            X, Y = zip(*self.samples)
            Y = [self.class_to_idx[y] for y in Y]
            self.samples = list(zip(X, Y))
        else:
            X = np.array(self.samples).flatten()
            Y = [-1]*len(self.samples)
            self.samples = list(zip(X, Y))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target, *others = self.samples[index]
        
        if self.xloader is not None:
            sample = self.xloader(sample)
        if self.xtransform is not None:
            sample = self.xtransform(sample)

        if self.yloader is not None:
            sample = self.yloader(target)
        if self.ytransform is not None:
            target = self.ytransform(target)

        return (sample, target, *others)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.xtransform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.ytransform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
                  
                  
                  