import torch
import torch.nn as nn
from .model import SqueezeNet
from .dataset import CIFAR10Dataset

BATCH_SIZE = 64
EPOCHS = 55
LR = 0.001
WEIGHT_DECAY = 0.01
SEED = 1
NUM_CLASSES = 10
MODEL_NAME = "SqueezeNet"

torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    torch.cuda.manual_seed(SEED)

class Trainer:
    def __init__(
        self,
        batch_size=64,
        epochs=55,
        lr=0.001,
        weight_decay=0.01,
    ):
        self.epochs = epochs
        self.device = device
        data = CIFAR10Dataset(batch_size=batch_size)
        self.train_loader = data.train_loader()
        self.test_loader = data.test_loader()
        self.model = SqueezeNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.best_accuracy = 0.0

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images).flatten(1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        return loss, accuracy

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images).flatten(1)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        loss = total_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        return loss, accuracy

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            train_loss, train_accuracy = self.train_epoch()
            test_loss, test_accuracy = self.evaluate()
            print(
                f"Epoch {epoch:02d}/{self.epochs} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_accuracy:.2f}% | "
                f"Test Loss {test_loss:.4f} | "
                f"Test Acc {test_accuracy:.2f}%"
            )
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy
                torch.save(
                    self.model.state_dict(),
                    "squeezenet.pth"
                )
        print(f"Best Accuracy: {self.best_accuracy:.2f}%")

if __name__ == "__main__":
    print(f"Using Device: {device}")

    trainer = Trainer(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    trainer.fit()