from PIL import Image
from torch.utils.data import Dataset


class Mydataset(Dataset):
    def __init__(self, leaves_data, path, transform=None):
        self.data = leaves_data
        self.transform = transform
        self.path = path

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        img = Image.open(self.path / self.data["image"][item])
        label = self.data["label"][item]
        if self.transform is not None:
            img = self.transform(img)

        return img, label