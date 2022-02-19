""" used as reference:
https://github.com/kuangliu/pytorch-cifar
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import copy

import numpy as np
import properscoring as ps
import torch
import torch.nn.functional
import torchvision.models as models
from torchvision.models.resnet import BasicBlock
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from cifar_dataloader import build_dataloader
from resnet_s import resnet32

dataset = 'cifar10'
lr = 0.1
min_lr = 0.0001
epochs = 200
batchsize = 128
mcbn_iters = 64
mcbn_test_sample_count = 250  # mcbn evaluation is extremely slow, reducing the sample count can speed it up

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()

# model = models.ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10)  # resnet34
model = resnet32()
print(model)

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0002)
train_loader, test_loader = build_dataloader(dataset=dataset, batch_size=batchsize)
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


def evaluate_mcbn():
    eval_model = copy.deepcopy(model)
    eval_model.eval()

    # set batchnorm layer to per mini batch mode
    for name, module in eval_model.named_modules():
        if 'bn' in name:
            module.track_running_stats = False

    mcbn_train_dataloader = DataLoader(train_loader.dataset, batch_size=batchsize - 1, shuffle=True, num_workers=1, drop_last=True)
    mcbn_test_dataloader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False, num_workers=1)

    # There is not much benefit in drawing new batches of the training data every time (given large enough batch sizes and mcbn_iter count)
    # and no difference at all in relation to individual samples.
    # So I take all required batches here and will reuse them for every test sample.
    mcbn_train_dataloader_iter = iter(mcbn_train_dataloader)
    train_samples = [next(mcbn_train_dataloader_iter) for i in range(mcbn_iters)]
    # performance could be increased a lot by doing one forward pass per sample with only the bn layers in training mode,
    # saving the running mean and var and loading them per mcbn sample, instead of passing the train_samples along the test sample
    # through the net. BN layers have then to be used in regular eval mode (track_running_stats = True).
    # This would avoid the irrelevant calculations on the train samples and allow passing multiple test samples per batch

    mcbn_samples_count = min(len(mcbn_test_dataloader), mcbn_test_sample_count)
    mcbn_results = torch.zeros((mcbn_samples_count, mcbn_iters, 10))  # this will collect the raw mcbn_samples tensors for all frames
    mcbn_targets = torch.zeros(mcbn_samples_count)
    for batch_idx, (eval_input, eval_target) in enumerate(tqdm(mcbn_test_dataloader)):
        mcbn_samples = torch.zeros((mcbn_iters, 10))
        for i in range(mcbn_iters):
            # generate batch
            train_inputs, train_targets = train_samples[i]
            inputs = torch.concat([eval_input, train_inputs])
            targets = torch.concat([eval_target, train_targets])
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            mcbn_samples[i] = torch.softmax(outputs[0],
                                            dim=0)  # only first element in batch is relevant, other other samples are train samples

        mcbn_results[batch_idx] = mcbn_samples
        mcbn_targets[batch_idx] = eval_target

        if batch_idx == mcbn_samples_count - 1:
            # early stopping if mcbn_test_sample_count is less than the test sample count
            break

    # calculate metrics
    # [mcbn_results[x].max(1)[1].eq(mcbn_targets[x]) for x in range(len(mcbn_results))]  # check for all mcbn_iter of each sample if prediction was correct

    summed_probs = torch.zeros((mcbn_samples_count, 10))
    for i in range(mcbn_iters):
        tp = 0
        NLL = 0
        for j in range(mcbn_samples_count):
            # calc.py metrics (https://github.com/icml-mcbn/mcbn/blob/master/code/cifar10)
            summed_probs[j] += mcbn_results[j][i]

            probs_cur = summed_probs / (i + 1)
            pred = probs_cur[j].topk(1)[1].item()
            tp += pred == mcbn_targets[j]
            NLL -= np.log(probs_cur[j][int(mcbn_targets[j])].item())

        # crps & gaussian_nll
        mean = np.array([sample[:i + 1].mean(axis=0).numpy() for sample in mcbn_results])
        if i > 0:  # cant calculate std for only one sample
            std = np.array([sample[:i + 1].std(axis=0).numpy() for sample in mcbn_results])
            std = np.maximum(std, 1e-20)  # prevent division by zero in ps.crps_gaussian
            crps = np.mean(ps.crps_gaussian([[i] for i in mcbn_targets], mu=mean, sig=std))
            gaussian_nll1 = nn.GaussianNLLLoss()(torch.tensor([mean[i][int(mcbn_targets[i].item())] for i in range(len(mean))]),
                                                 1,
                                                 torch.tensor([std[i][int(mcbn_targets[i].item())] for i in range(len(std))]))
            gaussian_nll2 = nn.GaussianNLLLoss()(torch.tensor(mean),
                                                 torch.stack([torch.nn.functional.one_hot(i.long(), 10) for i in mcbn_targets]),
                                                 torch.tensor(std))
        else:
            crps, gaussian_nll1, gaussian_nll2 = 0, 0, 0

        print(f"mcbn_iters: {i + 1}: "
              f"tp: {(tp / mcbn_samples_count * 100) : .2f}%, "
              f"NLL sum: {NLL : .2f}, "
              f"NLL avg: {NLL/mcbn_samples_count : .2f}, "
              f"CRPS: {crps : .2f}, "
              f"gaussian NLL expected class: {gaussian_nll1 : .2f} "  # nll only for expected element of onehot vector
              f"gaussian NLL over all classes: {gaussian_nll2 : .2f}")  # nll for all elements of onehot vector

    return tp / mcbn_samples_count


def main():
    for epoch in range(0, epochs):
        train_acc = train()
        if epoch +1 == epochs or (epoch + 1) % 10 == 0:
            test_acc = evaluate_mcbn()
        else:
            test_acc = evaluate()
        print(f"\r[{epoch + 1}/{epochs}] "
              f"train_acc: {train_acc * 100 :3.2f}%, "
              f"test_acc: {test_acc * 100:3.2f}%, "
              f"lr: {optimizer.param_groups[0]['lr'] :.4f}")


if __name__ == '__main__':
    main()

# 100ep: best test loss: [100/100] train_acc: 46.45%, test_acc: 78.03%, lr: 0.0000
# 200ep: best test loss: [192/200] train_acc: 99.86%, test_acc: 89.74%, lr: 0.0004
