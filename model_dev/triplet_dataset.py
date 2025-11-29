from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch

class TripletDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        anchor_path = self.data.iloc[idx, 0]
        positive_path = self.data.iloc[idx, 1]
        negative_path = self.data.iloc[idx, 2]

        anchor = self.load_image(anchor_path)
        positive = self.load_image(positive_path)
        negative = self.load_image(negative_path)

        return anchor, positive, negative

    def __len__(self):
        return len(self.data)
