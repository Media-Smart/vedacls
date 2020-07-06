from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def image_process(self, image):
        if self.transform:
            image = self.transform(image=image)['image']

        return image

    def target_process(self, label):
        if self.target_transform:
            label = self.target_transform(label=label)['label']

        return label
