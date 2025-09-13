import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class GalaxyZooDataset(Dataset):
    def __init__(self, image_dir, labels_df=None, transform=None, galaxy_ids=None):
        """
        image_dir : path to folder with images
        labels_df : DataFrame containing GalaxyIDs and labels (for training/validation)
        transform : torchvision transforms
        galaxy_ids: DataFrame containing GalaxyIDs to use (for test/prediction subset)
        """
        self.image_dir = image_dir
        self.transform = transform

        if labels_df is None and galaxy_ids is None:
            raise ValueError("Either labels_df or galaxy_ids must be provided")

        if galaxy_ids is not None:
            # Selecting only the requested GalaxyIDs (test/prediction subset)
            if isinstance(galaxy_ids, list):
                self.galaxy_ids = galaxy_ids
            else:
                self.galaxy_ids = galaxy_ids["GalaxyID"].values
            self.labels = None
        else:
            self.galaxy_ids = labels_df["GalaxyID"].values
            # Exclude GalaxyID column for labels
            self.labels = labels_df.drop(columns=["GalaxyID"]).values.astype('float32')

    def __len__(self):
        return len(self.galaxy_ids)

    def __getitem__(self, idx):
        gid = self.galaxy_ids[idx]
        image_path = os.path.join(self.image_dir, f"{gid}.jpg")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            target = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, target
        else:
            # For test dataset, just return the image and its GalaxyID
            return image, gid
