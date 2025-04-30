Fp Implementation written in CUDA/PTX
-----
This repo contains a highly optimized Fp implementation in CUDA PTX. No 
parallelisation is used, meaning that every thread is doing the same. The 
reduction is implemented as the Montgomery reduction. For the multiplication 
either the schoolbook or karatsuba implementation can be used. Also combinations
of both can be used. Additionally a everything is implemented either with 32- 
or 64 bit limbs.

This is the backbone implementation of [this paper](https://eprint.iacr.org/2022/1464)

The following functionallities are implemented:
	- mult
	- add
	- sub
	- sqr
	- mod

Build:
----

Run the following to prepare your system:
## Arch:
```bash
sudo pacman -S make cmake cuda
````

## Nix:
```bash
nix-shell
```

Then run the following to setup the code:
```bash
git clone https://github.com/FloydZ/cuda_fp_arithmetic
mkdir -p build
cd build
cmake ..
make
```

Usage:
---

You can either test the code via:
```bash
./test_limb2
./test_limb3
./test_limb4
./test_limb5
./test_limb6
./test_limb7
```

or run benchmarks via:
```bash
cd build
make bench
./bench
```

You can also generate test cases on you own via the script `python test_gen.py`.
Its possible to tune the number of variables via `-t XXX`, the bitlength of the
underlying limbs (either 32- or 64-bits limbs) and its possible to tune the
number of limbs (\in [2, ..., 8]).
Example:
```bash
python test_gen.py -l 6 -t 80 -b 32
```
will generate 80 different tests for 6 limb arithmetic based on 32bit limbs
