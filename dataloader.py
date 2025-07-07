import torch
import os
from PIL import Image

def get_image_list(raw_image_path, clear_image_path=None):
    raw_image_list = sorted(os.listdir(raw_image_path))
    
    if clear_image_path is not None:
        clear_image_list = sorted(os.listdir(clear_image_path))
        assert len(raw_image_list) == len(clear_image_list), "Number of raw and clear images must be same"
    else:
        clear_image_list = [None] * len(raw_image_list)

    image_list = []
    for raw_img_name, clear_img_name in zip(raw_image_list, clear_image_list):
        raw_img_path = os.path.join(raw_image_path, raw_img_name)
        clear_img_path = os.path.join(clear_image_path, clear_img_name) if clear_img_name is not None else None
        image_list.append([raw_img_path, clear_img_path, raw_img_name])
    return image_list


class myDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path, clear_image_path=None, transform=None, mode="train"):
        """
        mode: 'train', 'val', or 'test'
        - 'train' => input + GT required
        - 'val'   => only input image, GT = dummy
        - 'test'  => input + GT required
        """
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.mode = mode
        self.transform = transform
        self.image_list = get_image_list(self.raw_image_path, self.clear_image_path)

    def __getitem__(self, index):
        raw_path, clear_path, image_name = self.image_list[index]

        # Load and transform input image
        raw_img = Image.open(raw_path).convert('RGB')
        if self.transform:
            raw_img = self.transform(raw_img)

        # Handle GT image
        if self.mode == "val":
            # No GT during validation
            clear_img = torch.zeros_like(raw_img)
        else:
            if clear_path is None:
                raise ValueError(f"GT image not found for mode={self.mode}")
            clear_img = Image.open(clear_path).convert('RGB')
            if self.transform:
                clear_img = self.transform(clear_img)

        return raw_img, clear_img, image_name

    def __len__(self):
        return len(self.image_list)
