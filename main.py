""" used as reference:
https://github.com/kuangliu/pytorch-cifar
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import copy
import math
import random
import sys

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
epochs = 10
batchsize = 32
mcbn_iters = 128
mcbn_test_sample_count = 10000  # mcbn evaluation is extremely slow, reducing the sample count can speed it up

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


_bn_std_var_buffer = []


def get_bn_params(eval_model):
    if len(_bn_std_var_buffer) == 0:
        print("collect bn parameters for each train batch")

        # set batchnorm layer to per mini batch mode
        for name, module in eval_model.named_modules():
            if 'bn' in name:
                module.train()
                module.reset_running_stats()
                module.momentum = 1.  # only one batch is processed, and it should have full impact on "tracked" values

        mcbn_train_dataloader = DataLoader(train_loader.dataset, batch_size=batchsize, shuffle=True, num_workers=8, drop_last=True)
        for batch_idx, (train_input, train_target) in enumerate(tqdm(mcbn_train_dataloader)):
            train_input = train_input.cuda()
            eval_model(train_input)

            bn_params = []
            for name, module in eval_model.named_modules():
                if 'bn' in name:
                    bn_params.append(copy.deepcopy((module.running_mean, module.running_var)))
                    module.reset_running_stats()
            _bn_std_var_buffer.append(bn_params)

        eval_model.eval()

    return _bn_std_var_buffer


def evaluate_mcbn():
    eval_model = copy.deepcopy(model)
    eval_model.eval()

    mcbn_test_dataloader = DataLoader(test_loader.dataset, batch_size=batchsize, shuffle=False, num_workers=4, drop_last=True)
    bn_std_var_buffer = get_bn_params(eval_model)
    mcbn_samples_count = min(len(mcbn_test_dataloader) * batchsize, mcbn_test_sample_count)
    mcbn_samples_count = (int(mcbn_samples_count / batchsize)) * batchsize  # adjust to full batch amount
    if mcbn_samples_count == 0:
        print("too low sample count (mcbn_test_sample_count)")
        sys.exit(1)
    mcbn_results = torch.zeros((mcbn_samples_count, mcbn_iters, 10))  # this will collect the raw mcbn_samples tensors for all frames
    mcbn_targets = torch.zeros(mcbn_samples_count)
    for batch_idx, (eval_input, eval_target) in enumerate(tqdm(mcbn_test_dataloader)):
        mcbn_samples = torch.zeros((batchsize, mcbn_iters, 10))
        for batch_iter in range(mcbn_iters):
            # set bn mean/std to the values of a random precalculated values
            bn_params = random.choice(bn_std_var_buffer)
            bn_params_index = 0
            for name, module in eval_model.named_modules():
                if 'bn' in name:
                    module.running_mean = bn_params[bn_params_index][0]
                    module.running_var = bn_params[bn_params_index][1]
                    bn_params_index += 1

            inputs = eval_input
            targets = eval_target
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                outputs = eval_model(inputs)

            for sample_of_cur_batch in range(len(outputs)):
                mcbn_samples[sample_of_cur_batch][batch_iter] = torch.softmax(outputs[sample_of_cur_batch], dim=0)

        for batch_iter in range(batchsize):
            total_iter = batch_idx * batchsize + batch_iter
            mcbn_results[total_iter] = mcbn_samples[batch_iter]
            mcbn_targets[total_iter] = eval_target[batch_iter]

        if (batch_idx + 1) * batchsize == mcbn_samples_count:
            # early stopping if mcbn_test_sample_count is less than the test sample count
            break

    # calculate metrics
    summed_probs = torch.zeros((mcbn_samples_count, 10))
    for batch_iter in range(mcbn_iters):
        tp = 0
        NLL = 0
        for sample_of_cur_batch in range(mcbn_samples_count):
            # calc.py metrics (https://github.com/icml-mcbn/mcbn/blob/master/code/cifar10)
            summed_probs[sample_of_cur_batch] += mcbn_results[sample_of_cur_batch][batch_iter]

            probs_cur = summed_probs / (batch_iter + 1)
            pred = probs_cur[sample_of_cur_batch].topk(1)[1].item()
            tp += pred == mcbn_targets[sample_of_cur_batch]
            NLL -= np.log(probs_cur[sample_of_cur_batch][int(mcbn_targets[sample_of_cur_batch])].item())

        # reduce logging a bit (logging each iteration results in log spam and provides minor additional value)
        # the gaussian calculations part is not required to iterate over every single iteration like the clac.py metrics
        # so they can be skipped if they are not printed
        # Will be executed at iteration 2^iteration and last iteration (1,2,4,8,...)
        if (batch_iter + 1) in [2 ** x for x in range(math.ceil(math.log2(mcbn_iters)))] + [mcbn_iters]:
            # crps & gaussian_nll
            mean = np.array([sample[:batch_iter + 1].mean(axis=0).numpy() for sample in mcbn_results])
            if batch_iter > 0:  # cant calculate std for only one sample
                std = np.array([sample[:batch_iter + 1].std(axis=0).numpy() for sample in mcbn_results])
                std = np.maximum(std, 1e-20)  # prevent division by zero in ps.crps_gaussian
                crps = np.mean(ps.crps_gaussian(
                    torch.stack([torch.nn.functional.one_hot(i.long(), 10) for i in mcbn_targets]),  # labels onehot encoded
                    mu=mean,
                    sig=std))
                gaussian_nll1 = nn.GaussianNLLLoss()(torch.tensor([mean[i][int(mcbn_targets[i].item())] for i in range(len(mean))]),
                                                     1,
                                                     torch.tensor([std[i][int(mcbn_targets[i].item())] for i in range(len(std))]))
                gaussian_nll2 = nn.GaussianNLLLoss()(torch.tensor(mean),
                                                     torch.stack([torch.nn.functional.one_hot(i.long(), 10) for i in mcbn_targets]),
                                                     torch.tensor(std))
            else:
                crps, gaussian_nll1, gaussian_nll2 = 0, 0, 0

            print(f"mcbn_iters: {batch_iter + 1}: "
                  f"tp: {(tp / mcbn_samples_count * 100) : .2f}%, "
                  f"NLL sum: {NLL : .2f}, "
                  f"NLL avg: {NLL / mcbn_samples_count : .2f}, "
                  f"CRPS: {crps : .2f}, "
                  f"gaussian NLL expected class: {gaussian_nll1 : .2f} "  # nll only for expected element of onehot vector
                  f"gaussian NLL over all classes: {gaussian_nll2 : .2f}")  # nll for all elements of onehot vector

    return tp / mcbn_samples_count


def main():
    for epoch in range(0, epochs):
        train_acc = train()
        test_acc = evaluate()
        print(f"\r[{epoch + 1}/{epochs}] "
              f"train_acc: {train_acc * 100 :3.2f}%, "
              f"test_acc: {test_acc * 100:3.2f}%, "
              f"lr: {optimizer.param_groups[0]['lr'] :.4f}")
        if epoch + 1 == epochs or (epoch + 1) % 10 == 0:
            test_acc = evaluate_mcbn()
            print(f"\r[{epoch + 1}/{epochs}] "
                  f"train_acc: {train_acc * 100 :3.2f}%, "
                  f"test_acc: {test_acc * 100:3.2f}%, "
                  f"lr: {optimizer.param_groups[0]['lr'] :.4f}")


if __name__ == '__main__':
    main()

# concept of faster bn approach
# for each training sample:
#     run one batch with bn in training mode
#     collect bn.running_mean and bn.running_var
#     bn.reset_running_stats()
# set bn to eval()
# for each processed batch of test data
#     get one collected entry
#     set bn.running_mean and bn.running_var
#     process batch
#
# Advantages:
# - much faster (for bs 128 its 128 times faster)
# - same results if looking on individual samples
# Disadvantages:
# - if looking on multiple samples they are calculated based on the same bn layer values.
#   While it doesn't impact general quality its
#   1) more important that the chosen bn params are good (less problematic if mcbn iters is high enough)
#   2) not the approach of the original paper
