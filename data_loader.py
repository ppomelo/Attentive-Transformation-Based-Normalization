import os

from utils.dataset import Dataset
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class DataLoaderGenerator():
    def __init__(self, main_args):
        # signature unrelated magic nums
        self.test_size = 0.2
        self.random_state = 41
        self.image_path_suffix = 'Image/*'
        self.train_mask_path_suffix = 'Mask/*' 

        # variables
        self.main_args = main_args
        
        
    def get_slice_path(self, number_path_list, mode=False):
        train_img_paths = []
        for temp_path in number_path_list:
            temp_slice_path = os.listdir(temp_path)
            for temp_slice_slice_path in temp_slice_path:
                slice_path = temp_path +'/' + temp_slice_slice_path
                train_img_paths.append(slice_path)
        train_img_paths.sort()
        return train_img_paths

    def generate_random_train_and_valid_subsets_paths(self):
        original_image_path = self.main_args.original_image_path

        img_paths = sorted(glob(original_image_path + self.image_path_suffix))
        mask_paths = sorted(glob(original_image_path + self.train_mask_path_suffix))

        train_img_paths_file_path, val_img_paths_file_path, train_mask_paths_file_path, val_mask_paths_file_path = \
        train_test_split(img_paths, mask_paths, test_size=self.test_size, random_state=self.random_state)

        return self.get_slice_path(train_img_paths_file_path), self.get_slice_path(val_img_paths_file_path), \
            self.get_slice_path(train_mask_paths_file_path), self.get_slice_path(val_mask_paths_file_path)

    def generate(self):
        # Randomly generate train and valid subsets paths via sklearn encapsulated function
        train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
            self.generate_random_train_and_valid_subsets_paths()

        print("train_num:%s"%str(len(train_img_paths)))
        print("val_num:%s"%str(len(val_img_paths)))

        # Generate dataset
        
        train_dataset = Dataset(self.main_args, train_img_paths, train_mask_paths, self.main_args.aug)
        val_dataset = Dataset(self.main_args, val_img_paths, val_mask_paths)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.main_args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.main_args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False)
        
        return train_loader, val_loader

