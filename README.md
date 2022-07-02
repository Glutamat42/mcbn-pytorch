State at the time of submission: [here](https://github.com/Glutamat42/mcbn-pytorch/tree/e0195368a4a3236ec5e680ca36f59126707eeed5)

# About this project
Implementation of the CIFAR10 example from [Bayesian Uncertainty Estimation for Batch Normalized Deep Networks](https://arxiv.org/abs/1802.06455) with Pytorch. 

The implementation is close to the paper's implementation but has some differences. Also I can't guarantee that the calculated metrics match the exact calculations from the paper because [the provided code](https://github.com/icml-mcbn/mcbn/tree/master/code/cifar10) is probably outdated. To convert NLL to PLL, multiply the NLL value by "-1".   

[UNCERTAINTY ESTIMATION VIA STOCHASTIC BATCH NORMALIZATION](https://arxiv.org/abs/1802.04893) describes a similar concept but also addresses computational cost, a huge problem of the paper implemented here.

# Speed improved mcbn algorithm
I implemented a modified version of the algorithm, which results in similar results (and even nearly identical results if looking only at individual samples) but is way faster (for batch size 128, it should be around 128 times faster). On my notebook, I can calculate mcbn for all 10k test samples with 64 iterations in under 2 minutes. 

The [authors also suggest this approach](https://arxiv.org/abs/1802.06455), but to reduce memory requirements: "One can store BN statistics, instead of batches, to reduce memory issues."

## concept of the faster bn approach
```
set bn to train()
for all training samples:
    run one batch through net
    collect bn.running_mean and bn.running_var of all bn layers
    bn.reset_running_stats()
set bn to eval()
for each processed batch of test data
    get one collected entry
    set bn.running_mean and bn.running_var of the corresponding bn layers
    process batch
```

## Advantages:
 - much faster (for bs 128, its 128 times faster)
 - same results if looking at individual samples
## Disadvantages:
 - If looking at multiple samples, they are calculated based on the same bn layer values if processed in the same batch.
   While it doesn't impact general quality, its ...
    1) more important that the chosen bn params are good (less problematic if mcbn iters are high enough)
    2) not the approach of the original paper

# Reproduction of paper results
| Number of stochastic forward passes | 1    | 2    | 4    | 8    | 16   | 32   | 64   | 128  | baseline |
|-------------------------------------|------|------|------|------|------|------|------|------|----------|
| PLL paper                           | -.36 | -.32 | -.30 | -.29 | -.29 | -.28 | -.28 | -.28 | -.32     |
| PLL this implementation             | -.37 | -.33 | -.30 | -.29 | -.28 | -.27 | -.27 | -.27 | -.33     |

Full log is provided in [run.log](run.log)


