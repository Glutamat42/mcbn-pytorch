Implementation of the CIFAR10 example from the paper "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" with Pytorch.

Computational cost: every single test example will be combined with a batch of train samples (eg 1 test sample and 31 train samples). Many of these combinations will be ran through the network per single test example. This results in very poor performance.

"UNCERTAINTY ESTIMATION VIA STOCHASTIC BATCH NORMALIZATION" describes the same concept, but also addresses computational cost, which is a huge problem of the first paper.

