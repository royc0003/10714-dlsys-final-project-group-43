import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        # Initialize the parent Dataset class with transforms
        super().__init__(transforms=transforms)
        
        # Load data based on train flag
        X_list = []
        y_list = []
        
        if train:
            # Load training batches (data_batch_1 through data_batch_5)
            batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            # Load test batch
            batch_files = ['test_batch']
        
        for batch_file in batch_files:
            file_path = os.path.join(base_folder, batch_file)
            with open(file_path, 'rb') as f:
                batch_dict = pickle.load(f, encoding='bytes')
            
            # Extract data and labels
            # Note: CIFAR-10 pickled files use bytes keys in Python 3
            if b'data' in batch_dict:
                data = batch_dict[b'data']
                labels = batch_dict[b'labels']
            else:
                # Fallback for non-bytes keys
                data = batch_dict['data']
                labels = batch_dict['labels']
            
            X_list.append(data)
            y_list.extend(labels)
        
        # Concatenate all batches
        X = np.concatenate(X_list, axis=0)
        
        # Reshape from (N, 3072) to (N, 3, 32, 32)
        # CIFAR-10 stores data as: 1024 R values + 1024 G values + 1024 B values
        # So we need to reshape: (N, 3, 1024) then (N, 3, 32, 32)
        N = X.shape[0]
        X = X.reshape(N, 3, 32, 32)
        
        # Normalize pixel values to [0, 1] range
        X = X.astype(np.float32) / 255.0
        
        # Store as attributes
        self.X = X
        self.y = np.array(y_list, dtype=np.int32)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        # Get image and label
        img = self.X[index]  # May be (3, 32, 32) or (batch, 3, 32, 32) if index is array
        label = self.y[index]
        
        # Check if we got a batch (when index is array/list) or single image
        # If img has 4 dimensions, it's a batch; if 3 dimensions, it's a single image
        if img.ndim == 4:
            # Batch mode: img shape is (batch, 3, 32, 32)
            # Apply transforms to each image in the batch
            batch_size = img.shape[0]
            transformed_imgs = []
            for i in range(batch_size):
                single_img = img[i]  # (3, 32, 32)
                # Transform from (C, H, W) to (H, W, C) for transforms
                single_img = single_img.transpose(1, 2, 0)  # (3, 32, 32) -> (32, 32, 3)
                # Apply transforms
                single_img = self.apply_transforms(single_img)
                # Transform back to (C, H, W) format
                single_img = single_img.transpose(2, 0, 1)  # (32, 32, 3) -> (3, 32, 32)
                transformed_imgs.append(single_img)
            img = np.stack(transformed_imgs, axis=0)  # (batch, 3, 32, 32)
        else:
            # Single image mode: img shape is (3, 32, 32)
            # Apply transforms if any (transforms expect H x W x C, so transpose)
            # Transform from (C, H, W) to (H, W, C) for transforms
            img = img.transpose(1, 2, 0)  # (3, 32, 32) -> (32, 32, 3)
            
            # Apply transforms
            img = self.apply_transforms(img)
            
            # Transform back to (C, H, W) format
            img = img.transpose(2, 0, 1)  # (32, 32, 3) -> (3, 32, 32)
        
        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
