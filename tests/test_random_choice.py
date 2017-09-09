import time
import numpy as np
from mowgly.sampling import random_choice_with_cdf


def test():
    print('\nTesting mowgly.sampling.random_choice_with_cdf ...')
    minibatchsize = 256
    N = 500000
    pdf = np.random.rand(N)
    pdf /= pdf.sum()
    cdf = np.cumsum(pdf)
    arange = np.arange(0,N,dtype=np.int32)
    
    iters = 10000
    rng = np.random.RandomState(3)
    start = time.time()
    for i in range(iters):
        rng.choice(arange, size=minibatchsize, replace=True)
    elapsed = (time.time() - start)/iters
    print('np.random.choice with replacement took {0:2.6f} microseconds '.format(elapsed*1e6))
    iters = 1000
    rng = np.random.RandomState(3)
    start = time.time()
    for i in range(iters):
        np.random.choice(arange, size=minibatchsize, replace=False)
    elapsed = (time.time() - start)/iters
    print('np.random.choice without replacement took {0:2.6f} microseconds '.format(elapsed*1e6))
    rng = np.random.RandomState(3)
    start = time.time()
    for i in range(iters):
        rng.choice(arange, size=minibatchsize, p=pdf, replace=True)
    elapsed = (time.time() - start)/iters
    print('np.random.choice with replacement and pdf took {0:2.6f} microseconds '.format(elapsed*1e6))
    rng = np.random.RandomState(3)
    start = time.time()
    for i in range(iters):
        rng.choice(arange, size=minibatchsize, p=pdf, replace=False)
    elapsed = (time.time() - start)/iters
    print('np.random.choice without replacement and pdf took {0:2.6f} microseconds '.format(elapsed*1e6))
    rng = np.random.RandomState(3)
    iters = 10000
    start = time.time()
    for i in range(iters):
        random_choice_with_cdf(arange, size=minibatchsize, cdf=cdf, random_state=rng)
    elapsed = (time.time() - start)/iters
    print('random_choice_with_cdf with replacement {0:2.6f} microseconds '.format(elapsed*1e6))