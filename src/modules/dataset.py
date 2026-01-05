# dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # ISIC 2019 class columns
        self.class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]

        # Convert one-hot → class index
        self.df["label"] = self.df[self.class_names].values.argmax(axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["image"]
        label = int(self.df.iloc[idx]["label"])

        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label