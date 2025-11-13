import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device=None,
        dtype=None,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
      ### BEGIN YOUR SOLUTION
      self.index = 0
      n = len(self.dataset)
      idx = np.arange(n)
      if self.shuffle:
          np.random.shuffle(idx)
      self.ordering = np.array_split(idx, range(self.batch_size, n, self.batch_size))
      return self
      ### END YOUR SOLUTION

    def __next__(self):
      ### BEGIN YOUR SOLUTION
      if self.index >= len(self.ordering):
        raise StopIteration

      idx = self.ordering[self.index]
      self.index += 1

      batch = self.dataset[idx]                 # may be x or (x, y, ...)
      if not isinstance(batch, tuple):
          batch = (batch,)

      tensor_kwargs = {}
      if self.device is not None:
          tensor_kwargs["device"] = self.device
      if self.dtype is not None:
          tensor_kwargs["dtype"] = self.dtype

      return tuple(
          x
          if isinstance(x, Tensor)
          else Tensor(x, requires_grad=False, **tensor_kwargs)
          for x in batch
      )


      ### END YOUR SOLUTION

