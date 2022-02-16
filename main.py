""" used as reference:
https://github.com/kuangliu/pytorch-cifar
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch
import torchvision.models as models
from torchvision.models.resnet import BasicBlock
from torch import nn, optim
from tqdm import tqdm

from cifar_dataloader import build_dataloader
from resnet_s import resnet32


dataset = 'cifar10'
lr = 0.1
min_lr = 0.0001
epochs = 200
train_batchsize = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()

# model = models.ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10)  # resnet34
model = resnet32()
print(model)

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0002)
train_loader, test_loader = build_dataloader(dataset=dataset, batch_size=128)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, lr, min_lr, epochs)


def train():
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    scheduler.step()
    return correct / total


def evaluate():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total


def main():
    for epoch in range(0, epochs):
        train_acc = train()
        test_acc = evaluate()
        print(
            f"\r[{epoch + 1}/{epochs}] train_acc: {train_acc * 100 :3.2f}%, test_acc: {test_acc * 100:3.2f}%, lr: {optimizer.param_groups[0]['lr'] :.4f}")


if __name__ == '__main__':
    main()



# 100ep: best test loss: [100/100] train_acc: 46.45%, test_acc: 78.03%, lr: 0.0000
# 200ep: best test loss: [192/200] train_acc: 99.86%, test_acc: 89.74%, lr: 0.0004