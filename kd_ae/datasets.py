import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import tensorflow as tf


class OpenImagesDataset(Dataset):
    """
    Custom Dataset class to wrap the TensorFlow dataset for PyTorch
    """
    def __init__(self, tfds_dataset, transform=None):
        self.dataset = tfds_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert the image from a TensorFlow tensor to a NumPy array (HWC format)
        image = image.numpy()

        # If a transform is defined, apply it
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define image transformations (resize, normalization, etc.)
# transform = transforms.Compose([
#     transforms.ToPILImage(),              # Convert from NumPy to PIL
#     transforms.Resize((224, 224)),         # Resize to 224x224 (or another size)
#     transforms.ToTensor(),                # Convert to PyTorch tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for pretrained models
# ])