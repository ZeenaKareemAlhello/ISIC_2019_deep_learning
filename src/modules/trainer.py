from tqdm import tqdm


class Trainer:
    def __init__(self, model, device, optimizer, criterion):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(loader), correct / total
