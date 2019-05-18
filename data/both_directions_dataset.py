import os.path
from data.base_dataset import BaseDataset, get_transform, make_dataset
from PIL import Image
import random


class BothDirectionsDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A
    '/path/to/data/trainA' and from domain B '/path/to/data/trainB' respectively.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)

        # create paths '/path/to/data/trainA' and '/path/to/data/trainB'
        self.dir_A = os.path.join(opt.dataroot, 'trainA')
        self.dir_B = os.path.join(opt.dataroot, 'trainB')

        # load images from '/path/to/data/trainA' and '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))

        # get the size of dataset A and B
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # make sure index is within then range
        A_path = self.A_paths[index % self.A_size]

        # randomize the index for domain B to avoid fixed pairs.
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
