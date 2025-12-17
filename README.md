# Program To Optimize Stepping Stones
PTOSS is designed to efficiently compute terms in the sequence [A337663](https://oeis.org/A337663) on the GPU.

## Build instructions

Makefile included, designed to be built with Cuda compilation tools release 13.0.

## Running

The program has two operating modes. Given one integer argument it finds the optimal board for allowing that many ones to be placed using multithreading on the GPU. Alternatively if the first argument is `CPU` then the second argument is used as the number of ones to place and the computation is performed on a single CPU thread instead.
```
PTOSS [N]
PTOSS CPU [N]
```