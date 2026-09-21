import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

class CIFAR10Dataset:
    def __init__(self, batch_size=12, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers 
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
    def train_loader(self):
        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=self.train_transform,
        )
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        
    def test_loader(self):
        dataset = datasets.CIFAR10(
            root="./data",
            train=False, 
            download=False,
            transform=self.test_transform,
            )
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            )